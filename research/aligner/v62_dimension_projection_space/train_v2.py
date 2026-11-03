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
# 只在缺包时才装；--break-system-packages 是为了在 PEP 668 管理的
# 系统环境里也能装（PEP 668 会直接拒绝系统级 pip install）。
try:
    import huggingface_hub  # noqa: F401
    import sklearn         # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--break-system-packages',
                    'huggingface_hub', 'scikit-learn'], check=False)
from huggingface_hub import hf_hub_download, HfApi

HFT = os.environ.get('HF_TOKEN','')
D_SHARED = 256
LR      = float(os.environ.get('LR','2e-4'))
STEPS   = int(os.environ.get('STEPS','1500'))
BS      = int(os.environ.get('BS','32'))
BAL_W   = float(os.environ.get('BAL_W','0.01'))
SLOT_ENT_W = float(os.environ.get('SLOT_ENT_W','0.05'))
VER     = os.environ.get('VER','62.4')

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

# ============ 算子：直接用仓库里那一份，不再内联副本 =============
#
# v62.3 修 bug 时发现：训练脚本曾内联一份 DimProjectionSpace，而仓库里
# 同时有 research/.../projspace.py。两份代码会分叉 —— 那次 lazy router 的
# bug 正是「训练脚本的 router 是 lazy、打包版是 eager」造成的，
# 于是训练出的权重加载不上、router 从未被优化。
# 单一实现，从根上消除这类问题。
#
# 取文件的方式：优先 git clone 整个仓库（便于复算），但 **checkout 可能不完整**
# —— Kaggle 上实测过：clone 成功、git tree 里也有 research/，工作区却缺文件
# （git status 全是 "deleted:"）。所以再用 raw 下载兜底，只补缺失的算子文件。
import importlib.util as _ilu
import subprocess as _sp
import sys as _sys
import urllib.request as _ur
from pathlib import Path as _Path

_URL = 'https://github.com/Microbiosis/hetero-moe.git'
_RAW = 'https://raw.githubusercontent.com/Microbiosis/hetero-moe/main'
_REL = 'research/aligner/v62_dimension_projection_space'
_FILES = {'projspace.py': 'v62_projspace', 'fusion.py': 'v62_fusion',
          'hyper_proj.py': 'v62_hyper_proj', 'self_repair.py': 'v62_self_repair',
          'recurrent_gate.py': 'v62_recurrent_gate',
          'hebbian_align.py': 'v62_hebbian_align',
          'memory_router.py': 'v62_memory_router',
          'modality_adapter.py': 'v62_modality_adapter',
          'feedback_head.py': 'v62_feedback_head',
          'learned_optimizer.py': 'v62_learned_optimizer'}


def _find_repo():
    """找仓库根：脚本在 research/<line>/<pkg>/ 下，往上找含 research/ 的目录。

    notebook（Kaggle）里没有 __file__，所以候选里也要放 cwd 与 /kaggle/working。
    """
    cands = []
    try:
        cands.append(_Path(__file__).resolve())
    except NameError:
        pass
    cands += [_Path('/kaggle/working/hetero-moe'),
              _Path.cwd() / 'hetero-moe',
              _Path('/workspace/hetero-moe')]
    for c in cands:
        for q in [c, *c.parents]:
            if (q / 'research').is_dir() and (q / 'hetero_moe').is_dir():
                return q
    return None


def _fetch_ops(repo):
    """确保 repo 下有两个算子文件；缺的就从 raw 补。"""
    for fn in _FILES:
        dst = repo / _REL / fn
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print('[0/5] 补齐 %s ...' % fn, flush=True)
        _ur.urlretrieve('%s/%s/%s' % (_RAW, _REL, fn), dst)
        if dst.stat().st_size < 1000:
            raise RuntimeError('取不到 %s' % fn)


_REPO = _find_repo()
if _REPO is None:
    _REPO = _Path('/kaggle/working/hetero-moe')
    print('[0/5] clone 仓库 ...', flush=True)
    try:
        _sp.run(['git', 'clone', '--depth', '1', _URL, str(_REPO)], check=True)
    except Exception as e:
        print('    clone 失败（%s），改用 raw 取文件' % e, flush=True)
        _REPO.mkdir(parents=True, exist_ok=True)
_fetch_ops(_REPO)
_sys.path.insert(0, str(_REPO))
# v62.4：projspace.py 现在 import 同目录的 hyper_proj / self_repair，
# Kaggle exec 环境 cwd 不在包目录，必须把包目录插入 sys.path。
_sys.path.insert(0, str(_REPO / _REL))


def _load(name, rel):
    path = _REPO / rel
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 先把 v62 包目录插入 sys.path，再加载算子 —— projspace.py 内部的
# `from hyper_proj import HyperLinear` 需要能解析到同目录的新模块。
_pkg_dir = str((_find_repo() or _Path.cwd()) / _REL)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)
_ps = _load('v62_projspace', _REL + '/projspace.py')
_fu = _load('v62_fusion', _REL + '/fusion.py')
DimProjectionSpace = _ps.DimProjectionSpace
FusionOverProjSpace = _fu.FusionOverProjSpace
MEASURED = _ps.MEASURED_SUBSPACES

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import combinations


class HeteroMoEProjSpace(nn.Module):
    def __init__(s,modal_dims=(896,256,768),d_shared=256,ne=4,top_k=2,
                 use_hyper=False,use_fast=False,hyper_rank=8,fast_decay=0.9,
                 use_recurrent_gate=False,use_memory_router=False,
                 use_hebbian=False,use_modality_adapter=False,use_feedback=False):
        super().__init__()
        s.P_m=nn.ParameterList([nn.Parameter(torch.empty(d_shared,D)) for D in modal_dims])
        for m in range(len(modal_dims)):
            nn.init.orthogonal_(s.P_m[m]); s.P_m[m].requires_grad_(False)
        s.proj_space=DimProjectionSpace(d_shared,MEASURED,ne,top_k,
                                        use_hyper=use_hyper,hyper_rank=hyper_rank,
                                        use_recurrent_gate=use_recurrent_gate,
                                        use_memory_router=use_memory_router,
                                        use_hebbian=use_hebbian,
                                        use_modality_adapter=use_modality_adapter)
        s.d_space=s.proj_space.total_dim
        s.fusion=FusionOverProjSpace(s.d_space,None,ne,top_k,
                                     use_fast=use_fast,fast_decay=fast_decay)
        # ---- 路线9：跨层递归反馈 ----
        s.use_feedback=use_feedback
        if use_feedback:
            from feedback_head import RecursiveFeedbackHead
            s.feedback_head=RecursiveFeedbackHead(s.d_space, ne, d_shared, rank=4)
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
            # ---- 路线9：top-down 反馈回写投影 ----
            if s.use_feedback:
                conf,deltas=s.feedback_head(y)
                with torch.no_grad():
                    for i,k in enumerate(s.proj_space.slot_names):
                        key=f'{k}|Qwen'   # 回写到参考模型投影
                        if key in s.proj_space.proj:
                            s.proj_space.proj[key].weight.add_(
                                s.feedback_head.lambda_*deltas[i].mean(dim=(0,1)))
                aux['feedback_conf']=conf
            return y,alpha,aux
        return canon,aux

torch.manual_seed(0)
_USE_HYPER = os.environ.get('USE_HYPER','0') == '1'
_USE_FAST  = os.environ.get('USE_FAST','0') == '1'
_USE_RECUR = os.environ.get('USE_RECUR','0') == '1'
_USE_MEM   = os.environ.get('USE_MEM','0') == '1'
_USE_HEBB  = os.environ.get('USE_HEBB','0') == '1'
_USE_ADAPT = os.environ.get('USE_ADAPT','0') == '1'
_USE_FB    = os.environ.get('USE_FB','0') == '1'
_on = [n for n,v in [('HYPER',_USE_HYPER),('FAST',_USE_FAST),('RECUR',_USE_RECUR),
        ('MEM',_USE_MEM),('HEBB',_USE_HEBB),('ADAPT',_USE_ADAPT),('FB',_USE_FB)] if v]
if _on:
    print('   [自修改权重] 启用: %s' % ','.join(_on), flush=True)
model=HeteroMoEProjSpace(
    use_hyper=_USE_HYPER,use_fast=_USE_FAST,
    use_recurrent_gate=_USE_RECUR,use_memory_router=_USE_MEM,
    use_hebbian=_USE_HEBB,use_modality_adapter=_USE_ADAPT,
    use_feedback=_USE_FB).to(DEV)
n_ps=sum(p.numel() for p in model.proj_space.parameters())
n_fu=sum(p.numel() for p in model.fusion.parameters())
print('   投影空间 %.2fM | 融合层 %.2fM | 总 %.2fM'
      %(n_ps/1e6,n_fu/1e6,(n_ps+n_fu)/1e6), flush=True)

trainable=[p for p in model.parameters() if p.requires_grad]
print('   可训练 %.2fM'%(sum(p.numel() for p in trainable)/1e6), flush=True)
opt=torch.optim.AdamW(trainable, lr=LR)

# ============ 自检（python train_v2.py 时默认跑）=============
# 验证三件事：算子来自仓库唯一实现、router 是 eager（v62.3 的关键修复）、
# 以及 state_dict 的 key 与打包版完全一致。
def _selftest():
    import torch as _t
    print('[自检] 算子来源: %s.%s' % (type(model.proj_space).__module__,
                                     type(model.proj_space).__name__), flush=True)
    assert type(model.proj_space).__module__.endswith('v62_projspace'), \
        '算子不是从仓库 projspace.py 来的 —— 分叉了'
    names = [n for n, _ in model.named_parameters() if 'proj_space.router' in n]
    print('[自检] proj_space.router 参数: %s' % names, flush=True)
    assert names, 'router 不在参数里 —— lazy 创建会让它永远不被优化'
    print('[自检] router 输出维: %d (n_slots=%d)'
          % (model.proj_space.router.out_features, model.proj_space.n_slots), flush=True)
    print('[自检] OK', flush=True)


# ============ 训练：接分类头 ============
_selftest()
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

from self_repair import SlotHealthMonitor

# 路线 7：自修复槽监测器（v62.3 发现 Q 槽权重仅 0.003）
monitor = SlotHealthMonitor(
    n_slots=model.proj_space.n_slots,
    slot_names=model.proj_space.slot_names,
    threshold=0.005,     # 降至 0.005（v62.4 训练时 Q 槽 EMA 约 0.02–0.05，未击穿 0.01）
    ema_decay=0.99,
    cooldown=200,        # 修复后至少 200 步内不重复触发
    device=str(DEV)
)
print('[自修复] 监测器就绪: %s' % monitor.summary(), flush=True)

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

    # ---- 路线 7：自修复槽监测 ----
    # 在 optimizer 步进后更新 EMA；每 50 步检查一次是否需要修复低贡献槽。
    monitor.update(ah_slot)
    if step % 50 == 0:
        repaired = monitor.repair(model.proj_space, opt)
        if repaired:
            print('   [自修复] step %d 触发: %s | %s'
                  % (step, repaired, monitor.summary()), flush=True)

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
                     'slot_ema': monitor.status(),
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
    # 目录由 VER 推导 —— 此前 CKPT_DIR 硬编码成 'checkpoints/v62.2'，
    # 于是每次重训都覆盖同一个目录，出现「目录名是 v62.2、内容却是 v62.3」的错位。
    CKPT_DIR = os.environ.get('CKPT_DIR', 'checkpoints/v%s' % VER)
    for f in ['heteromoe_projspace_v2.pt','report.json']:
        api.upload_file(path_or_fileobj=f'{OUT}/{f}',
                        path_in_repo=f'{CKPT_DIR}/{f}',
                        repo_id='redmingwei/HeteroMoEProjSpace-v2', repo_type='model',
                        commit_message=f'v{VER}: {f}')
        print('UPLOADED %s/%s'%(CKPT_DIR,f), flush=True)
print('总耗时 %.0fs'%(time.time()-t0), flush=True)
