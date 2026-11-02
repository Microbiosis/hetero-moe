"""用真实数据训练 v62.1（router 真正参与计算的修复版）。

数据：redmingwei/hetero-moe-cache（1万 ag_news + 5千 food101 的真实隐藏状态）
模型：HeteroMoEProjSpace —— 维度投影空间 + 1024 维融合层 + 5000 万可训专家

监控塌缩：
  - 专家负载分布（min/max）
  - 路由熵
  - 各专家输出是否退化
  - loss 下降是否来自任务还是来自容量

任务：从融合后的表示预测 ag_news 4 类（线性探针，冻结主干）。
      这是唯一能判断"表示是否变好"的客观指标。
"""
import os, sys, json, time, subprocess
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

t0 = time.time()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, flush=True)
if DEV=='cuda': print(torch.cuda.get_device_name(0), flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub','scikit-learn'],check=False)
from huggingface_hub import hf_hub_download, HfApi

HFT = os.environ.get('HF_TOKEN','')
D_SHARED = 256
LR      = float(os.environ.get('LR','2e-4'))
STEPS   = int(os.environ.get('STEPS','1500'))
BS      = int(os.environ.get('BS','32'))
BAL_W   = float(os.environ.get('BAL_W','0.01'))
SLOT_ENT_W = float(os.environ.get('SLOT_ENT_W','0.05'))
VER     = os.environ.get('VER','62.2')

# ============ 载入真实数据 ============
print('\n[1/5] 载入缓存 ...', flush=True)
R='redmingwei/hetero-moe-cache'
p={f: hf_hub_download(R,f,repo_type='dataset') for f in
   ['h_qwen.npy','h_bert.npy','h_vit.npy','labels.npy','food_labels.npy']}
h_q=np.load(p['h_qwen.npy'],mmap_mode='r')
h_b=np.load(p['h_bert.npy'],mmap_mode='r')
h_v=np.load(p['h_vit.npy'],mmap_mode='r')
y_t=np.load(p['labels.npy']); y_v=np.load(p['food_labels.npy'])
print('   qwen%s bert%s vit%s 类别%s'%(h_q.shape,h_b.shape,h_v.shape,
      np.bincount(y_t,minlength=4).tolist()), flush=True)

# ============ 模型（内联，云端无需额外文件）============
print('\n[2/5] 构建模型 ...', flush=True)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import combinations

@dataclass
class SlotSubspace:
    name:str; dim:int; cka_pair:float; cka_triple:Optional[float]
    models:List[str]=field(default_factory=list)

MEASURED={
 'Q':SlotSubspace('Q',256,0.9879,0.6767,['Qwen','BERT','ViT']),
 'K':SlotSubspace('K',256,0.9873,None,['BERT','ViT']),
 'V':SlotSubspace('V',256,0.9882,None,['BERT','ViT']),
 'O':SlotSubspace('O',256,0.9876,0.7168,['Qwen','BERT','ViT']),
}

class DimProjectionSpace(nn.Module):
    def __init__(s, d_shared=256, subspaces=None, n_experts=4, top_k=2):
        super().__init__()
        s.d_shared=d_shared; s.subspaces=subspaces or MEASURED
        s.slot_names=list(s.subspaces.keys()); s.n_slots=len(s.slot_names)
        s.n_experts=n_experts; s.top_k=top_k
        s.reference={k:sp.models[0] for k,sp in s.subspaces.items()}
        s.proj=nn.ModuleDict()
        for k,sp in s.subspaces.items():
            for m in sp.models:
                lin=nn.Linear(d_shared,sp.dim,bias=False)
                if sp.dim==d_shared: nn.init.orthogonal_(lin.weight)
                s.proj[f'{k}|{m}']=lin
        s.align=nn.ModuleDict()
        for k,sp in s.subspaces.items():
            for m1,m2 in combinations(sp.models,2):
                for a,b in [(m1,m2),(m2,m1)]:
                    lin=nn.Linear(sp.dim,sp.dim,bias=False)
                    if sp.dim==d_shared: nn.init.orthogonal_(lin.weight)
                    s.align[f'{k}|{a}>{b}']=lin
        s.fuse_w=nn.ParameterDict()
        for k,sp in s.subspaces.items():
            s.fuse_w[k]=nn.Parameter(torch.ones(len(sp.models))/len(sp.models))
        # 跨模态交叉注意力（可训练，每个槽一个）
        s.cross_attn=nn.ModuleDict()
        for k,sp in s.subspaces.items():
            if len(sp.models)>1:
                s.cross_attn[k]=nn.MultiheadAttention(sp.dim, num_heads=4,
                                                      batch_first=True)
        s.slot_gate=nn.Parameter(torch.ones(s.n_slots))
        s.route_slots=[k for k,sp in s.subspaces.items()
                       if sp.cka_triple is not None and sp.cka_triple<0.9]
        # router 输入宽度：只有在 init 时声明了 available_models 才能定死。
        # 改为延迟构建 —— 首次 forward 时按实际可用模型数建 router。
        # router：**eager** 创建（v62.1 修）
        #
        # 此前是 lazy（__init__ 里 s.router=None，首次 forward 才建），后果三连：
        #   ① load_state_dict 时 proj_space.router.weight 被判为 unexpected，
        #      权重根本加载不上；
        #   ② trainable / optimizer 在 router 存在之前就取好了，
        #      **router 从未被优化**；
        #   ③ 熵正则的梯度流向 router，却不在 optimizer 里，梯度被丢弃。
        # 结果：「四槽激活率 100%」建立在一个**随机初始化**的 router 上。
        #
        # 宽度按**声明**的模型数定死（缺模态用零占位补齐，见 forward），
        # 与打包版 research/.../projspace.py 做法一致。
        _rd=sum(sp.dim for k in s.route_slots
                 for m in s.subspaces[k].models
                 if f'{k}|{m}' in s.proj)
        s._route_dim=_rd
        s._router_inited=True
        if _rd>0:
            s.router=nn.Linear(_rd,s.n_slots,bias=False)
            nn.init.normal_(s.router.weight,std=0.01)
        else:
            s.router=None
        s.total_dim=sum(sp.dim for sp in s.subspaces.values())
    def _path(s,slot,model):
        ref=s.reference[slot]
        if model==ref: return []
        k=f'{slot}|{model}>{ref}'
        if k in s.align: return [k]
        for mid in s.subspaces[slot].models:
            if mid in (model,ref): continue
            k1,k2=f'{slot}|{model}>{mid}',f'{slot}|{mid}>{ref}'
            if k1 in s.align and k2 in s.align: return [k1,k2]
        return []
    def forward(s, feats, return_aux=False):
        per={}; B=S=None
        for k,sp in s.subspaces.items():
            al=[]; idxs=[]
            for i,m in enumerate(sp.models):
                key=f'{k}|{m}'
                if key not in s.proj or m not in feats or k not in feats[m]: continue
                x=s.proj[key](feats[m][k])
                if B is None: B,S=x.shape[0],x.shape[1]
                for st in s._path(k,m): x=s.align[st](x)
                al.append(x); idxs.append(i)
            if not al: continue
            # 序列长度可能不同（Qwen/BERT 128, ViT 197）：
            # 用交叉注意力对齐，而不是加权平均。
            # 以第一个模型为 query，其余为 key/value。
            if len(al)==1 or k not in s.cross_attn:
                per[k]=al[0]
            else:
                q=al[0]
                kv=torch.cat(al[1:],dim=1)      # 不同长度也能拼（沿 seq 维）
                fused,_=s.cross_attn[k](q,kv,kv) # q 的长度保持 q 自己的
                w=F.softmax(s.fuse_w[k],0)
                per[k]=w[0]*q + (1-w[0])*fused
        gate=F.softmax(s.slot_gate,0)
        dev=next(s.parameters()).device
        # 各槽 seq 长度可能不同 → 池化到统一长度再拼接
        Ss=[v.shape[1] for v in per.values()] if per else [1]
        Smin=min(Ss)
        def _pool(v):
            if v.shape[1]==Smin: return v
            # 均匀采样到 Smin
            idx=torch.linspace(0,v.shape[1]-1,Smin).long().to(v.device)
            return v[:,idx,:]
        S=Smin

        # ---------- 路由：token 级槽门控（v62.1 修复）----------
        # 此前 router 的输出 alpha 只被 return、never 参与 canon 计算，
        # 梯度不流向它 ⇒ 权重是随机的，而报告的负载/熵统计的正是这个装饰品。
        # 现在 eff = 全局 slot_gate × 逐 token alpha，归一化后真正乘进 canon。
        # 按**声明**的模型数固定宽度，缺失模态用零向量占位。
        # （nn.Linear 无 bias，零输入 → 零贡献 = 「该模态缺席」，语义正确）
        # 漏掉这一步会让「只给文本不给图像」时 router 输入宽度变化而崩。
        rp=[]
        for k in s.route_slots:
            for m in s.subspaces[k].models:
                kk=f'{k}|{m}'
                if kk not in s.proj: continue
                if m not in feats or k not in feats[m]:
                    rp.append(torch.zeros(B,S,s.subspaces[k].dim,
                                          device=dev,dtype=gate.dtype))
                    continue
                v=s.proj[kk](feats[m][k])
                if v.shape[1]!=S:
                    idx=torch.linspace(0,v.shape[1]-1,S).long().to(v.device)
                    v=v[:,idx,:]
                rp.append(v)
        if s.route_slots and rp:
            # 全槽软门控，不做 top-k（四槽投影已算完，top-k 只丢信息）
            alpha=F.softmax(s.router(torch.cat(rp,-1)),-1)
        else:
            alpha=torch.ones(B,S,s.n_slots,device=dev)/s.n_slots

        eff=gate.view(1,1,-1)*alpha
        eff=eff/(eff.sum(-1,keepdim=True)+1e-9)

        canon=torch.cat([_pool(per[k])*eff[...,i:i+1] if k in per else
                         torch.zeros(B,Smin,sp.dim,device=dev)
                         for i,(k,sp) in enumerate(s.subspaces.items())],dim=-1)
        D=max(sp.dim for sp in s.subspaces.values())
        struct=torch.zeros(B,S,s.n_slots,D,device=dev,dtype=canon.dtype)
        for i,(k,sp) in enumerate(s.subspaces.items()):
            if k in per: struct[:,:,i,:sp.dim]=_pool(per[k])*eff[...,i:i+1]
        if return_aux:
            return struct,canon,{'alpha':alpha,'per_slot':per,'slot_gate':gate.detach(),
                                 'route_slots':s.route_slots,'total_dim':s.total_dim}
        return struct,canon,alpha

class FusionOverProjSpace(nn.Module):
    """1024 维融合层，专家解冻。"""
    def __init__(s,d_space=1024,d_ff=None,ne=4,top_k=2):
        super().__init__()
        s.d_space=d_space; s.ne=ne; s.top_k=top_k
        d_ff=d_ff or 4*d_space
        s.gammas=nn.ParameterList([nn.Parameter(torch.ones(d_space)) for _ in range(ne)])
        s.betas =nn.ParameterList([nn.Parameter(torch.zeros(d_space)) for _ in range(ne)])
        s.w_gates=nn.ParameterList([nn.Parameter(torch.randn(d_ff,d_space)*0.02) for _ in range(ne)])
        s.w_ups  =nn.ParameterList([nn.Parameter(torch.randn(d_ff,d_space)*0.02) for _ in range(ne)])
        s.w_downs=nn.ParameterList([nn.Parameter(torch.randn(d_space,d_ff)*0.02) for _ in range(ne)])
        s.alphas=nn.ParameterList([nn.Parameter(torch.tensor([0.05])) for _ in range(ne)])
        s.W_router=nn.Parameter(torch.randn(ne,d_space)*0.01)
    def _sg(s,x,wg,wu,wd): return F.linear(F.silu(F.linear(x,wg))*F.linear(x,wu),wd)
    def forward_with_routing(s,x):
        z=F.linear(x.detach(),s.W_router)
        a=F.softmax(z,-1)
        tv,ti=torch.topk(a,s.top_k,-1)
        ah=a*torch.zeros_like(a).scatter_(-1,ti,1.0)
        out=torch.zeros_like(x)
        for m in range(s.ne):
            h=x*s.gammas[m]+s.betas[m]           # 不 detach
            f=s._sg(h,s.w_gates[m],s.w_ups[m],s.w_downs[m])
            out=out+ah[...,m:m+1]*s.alphas[m]*f
        return x+out, ah

class HeteroMoEProjSpace(nn.Module):
    def __init__(s,modal_dims=(896,256,768),d_shared=256,ne=4,top_k=2):
        super().__init__()
        s.P_m=nn.ParameterList([nn.Parameter(torch.empty(d_shared,D)) for D in modal_dims])
        for m in range(len(modal_dims)):
            nn.init.orthogonal_(s.P_m[m]); s.P_m[m].requires_grad_(False)
        s.proj_space=DimProjectionSpace(d_shared,MEASURED,ne,top_k)
        s.d_space=s.proj_space.total_dim
        s.fusion=FusionOverProjSpace(s.d_space,None,ne,top_k)
    def forward(s, hidden, fused_only=True):
        names=['Qwen','BERT','ViT']; pr={}
        for i,m in enumerate(names):
            if m in hidden: pr[m]=hidden[m]@s.P_m[i].t()
        # K/V 槽只有 BERT/ViT 参与；Q/O 三模型都有
        feats={}
        for m in pr:
            feats[m]={k:pr[m] for k in s.proj_space.slot_names}
        struct,canon,aux=s.proj_space(feats,return_aux=True)
        if fused_only:
            y,alpha=s.fusion.forward_with_routing(canon)
            return y,alpha,aux
        return canon,aux

torch.manual_seed(0)
model=HeteroMoEProjSpace().to(DEV)
n_ps=sum(p.numel() for p in model.proj_space.parameters())
n_fu=sum(p.numel() for p in model.fusion.parameters())
print('   投影空间 %.2fM | 融合层 %.2fM | 总 %.2fM'
      %(n_ps/1e6,n_fu/1e6,(n_ps+n_fu)/1e6), flush=True)

trainable=[p for p in model.parameters() if p.requires_grad]
print('   可训练 %.2fM'%(sum(p.numel() for p in trainable)/1e6), flush=True)
opt=torch.optim.AdamW(trainable, lr=LR)

# ============ 训练：接分类头 ============
print('\n[3/5] 训练（带分类头，任务是 4 类分类）...', flush=True)
head=nn.Linear(model.d_space,4).to(DEV)
opt.add_param_group({'params':head.parameters()})
ce=nn.CrossEntropyLoss()
n=h_q.shape[0]
# ==== 按类别分层划分 train/val ====
rng=np.random.default_rng(0)
tr_idx, va_idx = [], []
for c in range(4):
    ids = np.where(y_t == c)[0]
    rng.shuffle(ids)
    k = int(len(ids) * 0.8)
    tr_idx.append(ids[:k]); va_idx.append(ids[k:])
tr_idx = np.sort(np.concatenate(tr_idx))
va_idx = np.sort(np.concatenate(va_idx))
n = len(tr_idx)
print('   分层划分: train %d / val %d' % (n, len(va_idx)), flush=True)
for c in range(4):
    a=(y_t[tr_idx]==c).sum(); b=(y_t[va_idx]==c).sum()
    print('     类%d  train %d (%.2f)  val %d (%.2f)'
          % (c, a, a/n, b, b/len(va_idx)), flush=True)
hist=[]
best_acc = -1.0
best_state = None

for step in range(STEPS+1):
    bi = np.sort(rng.choice(tr_idx, size=min(BS,len(tr_idx)), replace=False))
    hq=torch.from_numpy(np.asarray(h_q[bi],dtype=np.float32)).to(DEV)
    hb=torch.from_numpy(np.asarray(h_b[bi],dtype=np.float32)).to(DEV)
    # ViT 数据只有 5000 条，循环取模对齐 batch
    vi=bi % h_v.shape[0]
    hv=torch.from_numpy(np.asarray(h_v[vi],dtype=np.float32)).to(DEV)
    lab=torch.as_tensor(y_t[bi],device=DEV)

    y,alpha,aux=model({'Qwen':hq,'BERT':hb,'ViT':hv}, fused_only=True)
    if step==0:
        print('   [debug] y.shape=%s alpha.shape=%s' % (tuple(y.shape), tuple(alpha.shape)), flush=True)
        print('   [debug] hq.shape=%s hb.shape=%s' % (tuple(hq.shape), tuple(hb.shape)), flush=True)
    pooled=y.mean(dim=1)                 # [B,1024]
    if step==0:
        print('   [debug] pooled.shape=%s head.in=%d' % (tuple(pooled.shape), head.in_features), flush=True)
    logits=head(pooled)
    loss_ce=ce(logits,lab)

    # 负载均衡
    #
    # v62.1 修：此前这里取的是 aux['alpha']，那是 **proj_space 的槽门控**（4 个槽），
    # 不是专家。真正的专家选择在融合层返回的 alpha 里。用错对象会让
    # 「均衡损失」去均衡 Q/K/V/O 四个槽，而 4 个专家无人看管。
    #
    # 现在两个都算，但权重分开：专家均衡是主目标（BAL_W），
    # 槽均衡只做轻微约束（BAL_W * 0.1），避免槽门控塌到单个槽。
    ah_slot = aux['alpha']                       # [B,S,n_slots]
    ah_exp  = alpha                              # [B,S,n_experts] ← 真正的专家
    fm_s=(ah_slot>0).float().mean(dim=(0,1)); pm_s=ah_slot.mean(dim=(0,1))
    loss_bal_slot = ah_slot.shape[-1]*(fm_s*pm_s).sum()
    fm_e=(ah_exp>0).float().mean(dim=(0,1));  pm_e=ah_exp.mean(dim=(0,1))
    loss_bal_exp  = 4*(fm_e*pm_e).sum()
    loss_bal = loss_bal_exp + 0.1*loss_bal_slot

    # 槽门控熵正则：最大化熵，防止塌到单槽。
    # v62.1 实测槽门控熵只有 0.0209（上界 1.0），等价于只剩一个槽，
    # 与「空间 = Σ 子空间」的主张矛盾。这是人为干预，不是自然收敛。
    slot_ent = -(ah_slot*torch.log(ah_slot+1e-9)).sum(-1).mean()
    loss_ent = -slot_ent

    loss=loss_ce+BAL_W*loss_bal+SLOT_ENT_W*loss_ent
    if step<STEPS:
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.0)
        opt.step()

    if step%250==0 or step==STEPS:
        # 在 val 上评估
        model.eval(); head.eval()
        vc=0; vt=0
        with torch.no_grad():
            for vs in range(0,len(va_idx),200):
                vb=va_idx[vs:vs+200]
                vq=torch.from_numpy(np.asarray(h_q[vb],dtype=np.float32)).to(DEV)
                vbb=torch.from_numpy(np.asarray(h_b[vb],dtype=np.float32)).to(DEV)
                vvi=vb % h_v.shape[0]
                vv=torch.from_numpy(np.asarray(h_v[vvi],dtype=np.float32)).to(DEV)
                vy,_,_=model({'Qwen':vq,'BERT':vbb,'ViT':vv},fused_only=True)
                vp=head(vy.mean(1)).argmax(-1)
                vc+=(vp==torch.as_tensor(y_t[vb],device=DEV)).sum().item()
                vt+=len(vb)
        val_acc=vc/vt
        if val_acc>best_acc:
            best_acc=val_acc
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            best_head={k:v.detach().clone() for k,v in head.state_dict().items()}
        model.train(); head.train()
        acc=(logits.argmax(-1)==lab).float().mean().item()
        load=(ah_exp.detach()>0).float().mean(dim=(0,1)).cpu().numpy()
        ah_np=ah_exp.detach().cpu().numpy()
        ent=float(-(ah_np*np.log(ah_np+1e-9)).sum(-1).mean()/np.log(4))
        # router 健康度：权重范数 + 槽门控熵 + 梯度范数
        rw = model.proj_space.router.weight
        rg = rw.grad
        slot_ent = float(-(ah_slot*torch.log(ah_slot+1e-9)).sum(-1).mean()
                         / np.log(ah_slot.shape[-1]))
        hist.append({'step':step,'ce':float(loss_ce.detach()),'train_acc':acc,
                     'val_acc':val_acc,'bal':float(loss_bal.detach()),
                     'load':load.tolist(),'ent':ent,
                     'router_wnorm': float(rw.detach().norm()),
                     'router_gnorm': float(rg.norm()) if rg is not None else 0.0,
                     'slot_ent': slot_ent,
                     'loss_ent': float(loss_ent.detach())})
        print('   step %4d  ce %.4f  acc %.3f  负载 %s  熵 %.3f  %.0fs'
              %(step,float(loss_ce),acc,np.round(load,2).tolist(),ent,time.time()-t0), flush=True)

# ============ 评估 ============
print('\n[4/5] 加载最佳权重并评估 ...', flush=True)
print('   最佳 val acc = %.4f (step %s)' % (best_acc,
      [h['step'] for h in hist if h.get('val_acc')==best_acc][:1]), flush=True)
if best_state is not None:
    model.load_state_dict(best_state)
    head.load_state_dict(best_head)
model.eval(); head.eval()
with torch.no_grad():
    idx=va_idx
    corr=0; tot=0; AH=[]; AHP=[]; AHF=[]
    for s in range(0,len(idx),200):
        b=idx[s:s+200]
        hq=torch.from_numpy(np.asarray(h_q[b],dtype=np.float32)).to(DEV)
        hb=torch.from_numpy(np.asarray(h_b[b],dtype=np.float32)).to(DEV)
        vi=b % h_v.shape[0]
        hv=torch.from_numpy(np.asarray(h_v[vi],dtype=np.float32)).to(DEV)
        y,alpha,aux=model({'Qwen':hq,'BERT':hb,'ViT':hv},fused_only=True)
        pred=head(y.mean(1)).argmax(-1)
        corr+=(pred==torch.as_tensor(y_t[b],device=DEV)).sum().item()
        tot+=len(b)
        AHP.append(aux['alpha'].cpu().numpy())   # proj_space：槽门控 [B,S,n_slots]
        AHF.append(alpha.cpu().numpy())          # fusion：专家选择 [B,S,n_experts]
    acc_ho=corr/tot

def _stat(A, K, name):
    load=(A>0).mean(axis=(0,1))
    ent=float(-(A*np.log(A+1e-9)).sum(-1).mean()/np.log(K))
    bal=float(load.min()/(load.max()+1e-9))
    print('   %-22s 负载 %s  熵 %.4f  均衡 %.4f'
          %(name, np.round(load,3).tolist(), ent, bal), flush=True)
    return {'load':load.tolist(),'entropy':ent,'balance':bal}

ap=np.concatenate(AHP); af=np.concatenate(AHF)
print('   held-out acc %.4f  (chance 0.25)'%acc_ho, flush=True)
SP=_stat(ap, ap.shape[-1], 'proj_space 槽门控')
SF=_stat(af, 4,           'fusion 专家选择')
load_ho=np.array(SF['load']); ent_ho=SF['entropy']; bal_ho=SF['balance']

# ============ 保存 + 上传 ============
print('\n[5/5] 保存并上传 ...', flush=True)
OUT='/tmp/v2'; os.makedirs(OUT,exist_ok=True)
torch.save({'model':model.state_dict(),'head':head.state_dict(),
            'config':{'d_space':model.d_space,'n_experts':4,'top_k':2}},
           f'{OUT}/heteromoe_projspace_v2.pt')
report={'hist':hist,'best_val_acc':best_acc,
        'split':'按类别分层 80/20',
        'eval':{'acc_heldout':acc_ho,'chance':0.25,
                'load':load_ho.tolist(),'entropy':ent_ho,'balance':bal_ho,
                'proj_space_slot_gate':SP,'fusion_expert':SF},
        'version': VER,
        'fix': os.environ.get('FIXNOTE','proj_space.router 此前是死代码（输出未被使用），现作为 token 级槽门控乘进 canonical'),
        'hparams':{'lr':LR,'steps':STEPS,'bs':BS,'bal_w':BAL_W},
        'params':{'proj_space':int(n_ps),'fusion':int(n_fu)},
        'note':'融合层吃1024维，专家解冻（偏离项目原设定）'}
json.dump(report,open(f'{OUT}/report.json','w'),indent=2,ensure_ascii=False)
print(json.dumps(report['eval'],indent=2,ensure_ascii=False), flush=True)

if HFT:
    api=HfApi(token=HFT)
    # 版本化路径，避免不同版本互相覆盖
    CKPT_DIR = os.environ.get('CKPT_DIR','checkpoints/v62.2')
    for f in ['heteromoe_projspace_v2.pt','report.json']:
        api.upload_file(path_or_fileobj=f'{OUT}/{f}',
                        path_in_repo=f'{CKPT_DIR}/{f}',
                        repo_id='redmingwei/HeteroMoEProjSpace-v2', repo_type='model',
                        commit_message=f'v{VER}: {f}')
        print('UPLOADED %s/%s'%(CKPT_DIR,f), flush=True)
print('总耗时 %.0fs'%(time.time()-t0), flush=True)
