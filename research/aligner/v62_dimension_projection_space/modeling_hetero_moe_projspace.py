# coding=utf-8
"""HeteroMoEProjSpace — 维度投影空间完整模型（v62.3，自包含单文件）。

由 research/aligner/v62_dimension_projection_space/ 的
projspace.py / fusion.py / model.py 内联生成。**只依赖 torch + transformers**。

版本史
------
v62.0  proj_space.router 是死代码：输出只被 return，从未参与计算。
v62.1  router 作为 token 级槽门控真正乘进 canonical。但仍在**槽**上做 top-k=2
       ⇒ 每 token 最大权重均值 0.9933、熵 0.0209，等价于只剩一个槽。
v62.2  槽门控改为**全槽软门控**（四槽投影在门控前已算完，top-k 不省算力、只丢信息）
       + 训练加熵正则。
       ⚠️ 但 v62.0~v62.2 的 router 是 **lazy 创建**的：既没被 load_state_dict 加载上，
       也从未进入 optimizer ⇒ 槽门控全由随机权重决定，**结论一律作废**。
v62.3  router 改为 **eager** 创建（能加载权重、也进 optimizer），真正被优化。
       槽熵 0.647；模型**学会了几乎不用 Q 槽**（平均权重 [0.003,0.171,0.434,0.392]）。
       训练脚本改为直接 import 仓库这份 projspace.py，不再内联副本 ——
       此前两份实现会分叉，正是 lazy/eager 不一致的根源。
       top-k 只保留在融合层的专家选择上 —— 那里真的省算力。

    from transformers import AutoModel
    m = AutoModel.from_pretrained('redmingwei/HeteroMoEProjSpace-v2',
                                  trust_remote_code=True)
    out = m(qwen_input_ids=..., qwen_attention_mask=...,
            bert_input_ids=..., bert_attention_mask=..., bert_token_type_ids=...,
            pixel_values=...)

    out.y          # [B, S, 1024]
    out.slot_alpha # [B, S, 4]  槽门控 —— 四个槽各贡献多少（全槽软分布，和=1）
    out.alpha      # [B, S, 4]  专家选择 —— 4 选 top-2（和≠1，是 top-k 掩码后的权重）

    ⚠️ slot_alpha 与 alpha 都是 4 维，但语义完全不同：
    前者是**槽**门控（选 Q/K/V/O 哪个子空间），后者是**专家**选择。
    要看「四个槽是否都在用」请读 slot_alpha。
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (AutoConfig, AutoModel, PretrainedConfig,
                          PreTrainedModel, ViTModel)
from transformers.modeling_outputs import ModelOutput


QWEN_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
BERT_ID = 'google/bert_uncased_L-4_H-256_A-4'
VIT_ID = 'google/vit-base-patch16-224-in21k'


class HeteroMoEProjSpaceConfig(PretrainedConfig):
    """v62.3 配置。moe_top_k 不能叫 top_k（会和生成参数冲突）。"""
    model_type = 'hetero_moe_projspace'

    def __init__(self, qwen_id=QWEN_ID, bert_id=BERT_ID, vit_id=VIT_ID,
                 d_shared=256, n_experts=4, moe_top_k=2, d_space=1024,
                 modal_dims=(896, 256, 768),
                 use_hyper=False, hyper_rank=8,
                 use_fast=False, fast_decay=0.9, **kw):
        self.qwen_id = qwen_id; self.bert_id = bert_id; self.vit_id = vit_id
        self.d_shared = d_shared; self.n_experts = n_experts
        self.moe_top_k = moe_top_k; self.d_space = d_space
        self.modal_dims = list(modal_dims)
        self.use_hyper = use_hyper; self.hyper_rank = hyper_rank
        self.use_fast = use_fast; self.fast_decay = fast_decay
        super().__init__(**kw)


# ============================================================ projspace
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 包内相对 import 优先 (research.aligner.v62_dimension_projection_space.projspace);
# 平铺 import 兜底 (独立脚本 / Kaggle 训练 / 审计流程: 本目录已在 sys.path)。
if __package__:
    from .hyper_proj import HyperLinear
else:
    from hyper_proj import HyperLinear


@dataclass
class SlotSubspace:
    """一个功能槽贡献的子空间。"""
    name: str
    dim: int                    # 该子空间维数（实测：等维 256 无损）
    cka_pair: float             # 两两对齐实测 CKA
    cka_triple: Optional[float] # 三模型同时对齐实测 CKA（None = 只有两模型参与）
    models: List[str] = field(default_factory=list)


#: 实测规格（来源 results/metrics.json: functional_slots / alignment / rank_sweep）
MEASURED_SUBSPACES: Dict[str, SlotSubspace] = {
    # cka_pair 取该槽两两对齐的**最小值**（保守）；cka_triple 取三模型同时对齐
    'Q': SlotSubspace('Q', 256, cka_pair=0.9878892898559570,
                      cka_triple=0.6766622265179952,
                      models=['Qwen', 'BERT', 'ViT']),
    'K': SlotSubspace('K', 256, cka_pair=0.9872695207595825, cka_triple=None,
                      models=['BERT', 'ViT']),
    'V': SlotSubspace('V', 256, cka_pair=0.9881786108016968, cka_triple=None,
                      models=['BERT', 'ViT']),
    'O': SlotSubspace('O', 256, cka_pair=0.9876090884208679,
                      cka_triple=0.7168068885803222,
                      models=['Qwen', 'BERT', 'ViT']),
}


class DimProjectionSpace(nn.Module):
    """维度投影空间：把多个模型、多个功能槽合并成结构化 token。

    输出形状 [B, S, n_slots, D]：每个 token 沿 slot 轴有多个分量。
      沿 slot 轴切片 = 分离（separate）
      跨 token 拼接   = 重组（recombine）

    token 数 S 保持不变 —— 合并发生在**模型轴**，不是 token 轴。
    """

    def __init__(self, d_shared: int = 256,
                 subspaces: Optional[Dict[str, SlotSubspace]] = None,
                 n_experts: int = 4, top_k: int = 2,
                 use_hyper: bool = False, hyper_rank: int = 8,
                 use_recurrent_gate: bool = False,
                 use_memory_router: bool = False,
                 use_hebbian: bool = False,
                 use_modality_adapter: bool = False):
        super().__init__()
        self.d_shared = d_shared
        self.subspaces = subspaces or MEASURED_SUBSPACES
        self.slot_names = list(self.subspaces.keys())
        self.n_slots = len(self.slot_names)
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_hyper = use_hyper
        self.hyper_rank = hyper_rank
        self.use_recurrent_gate = use_recurrent_gate
        self.use_memory_router = use_memory_router
        self.use_hebbian = use_hebbian
        self.use_modality_adapter = use_modality_adapter
        self.reference = {k: sp.models[0] for k, sp in self.subspaces.items()}

        # ---- 槽轴 × 模型轴：投影 ----
        # use_hyper=True 时用 HyperLinear（slow 正交冻结 + hyper 低秩 delta），
        # 否则用标准 nn.Linear（v62.3 行为）。
        self.proj = nn.ModuleDict()
        for k, sp in self.subspaces.items():
            for m in sp.models:
                if use_hyper:
                    self.proj[f'{k}|{m}'] = HyperLinear(
                        d_shared, sp.dim, rank=hyper_rank, slow_trainable=False)
                else:
                    lin = nn.Linear(d_shared, sp.dim, bias=False)
                    if sp.dim == d_shared:
                        nn.init.orthogonal_(lin.weight)   # 等维 → 保距
                    self.proj[f'{k}|{m}'] = lin

        # ---- 对齐轴：成对（可复合）----
        self.align = nn.ModuleDict()
        for k, sp in self.subspaces.items():
            for m1, m2 in combinations(sp.models, 2):
                for a, b in ((m1, m2), (m2, m1)):
                    lin = nn.Linear(sp.dim, sp.dim, bias=False)
                    # align 恒为方阵 (sp.dim, sp.dim)，无论 sp.dim 是否等于 d_shared 均应正交保距初始化
                    nn.init.orthogonal_(lin.weight)
                    self.align[f'{k}|{a}>{b}'] = lin

        # ---- 融合权重（参考帧下）----
        self.fuse_w = nn.ParameterDict()
        for k, sp in self.subspaces.items():
            self.fuse_w[k] = nn.Parameter(torch.ones(len(sp.models)) / len(sp.models))

        # ---- 跨模态交叉注意力（序列长度不同时用它对齐）----
        self.cross_attn = nn.ModuleDict()
        for k, sp in self.subspaces.items():
            if len(sp.models) > 1:
                self.cross_attn[k] = nn.MultiheadAttention(
                    sp.dim, num_heads=4, batch_first=True)

        # ---- 槽门控 ----
        self.slot_gate = nn.Parameter(torch.ones(self.n_slots))

        # ---- 路由：只作用于三模型对齐失败的槽 ----
        self.route_slots = [k for k, sp in self.subspaces.items()
                            if sp.cka_triple is not None and sp.cka_triple < 0.9]
        _rd = sum(self.subspaces[k].dim for k in self.route_slots
                  for m in self.subspaces[k].models
                  if f'{k}|{m}' in self.proj)
        # 输出是 **逐 token 的槽门控**（n_slots 维），不是专家选择。
        # v62.1 前这里写成 n_experts，且结果未被使用（死代码），已修。
        if _rd > 0:
            self.router = nn.Linear(_rd, self.n_slots, bias=False)
            nn.init.normal_(self.router.weight, std=0.01)
        else:
            self.router = None
        self._route_dim = _rd

        self.total_dim = sum(sp.dim for sp in self.subspaces.values())

        # ---- 路线2/4/6/8：自修改权重模块（默认关闭）----
        if __package__:
            from .recurrent_gate import RecurrentSlotGate
            from .hebbian_align import HebbianSlotAlignment
            from .memory_router import MemoryAugmentedRouter
            from .modality_adapter import ModalityAdapterBank
        else:
            from recurrent_gate import RecurrentSlotGate
            from hebbian_align import HebbianSlotAlignment
            from memory_router import MemoryAugmentedRouter
            from modality_adapter import ModalityAdapterBank

        # 路线2：递归槽门控（替换静态 slot_gate）
        if use_recurrent_gate:
            self.recurrent_gate = RecurrentSlotGate(self.total_dim, self.n_slots)
        # 路线8：记忆增强路由（替换 self.router）
        if use_memory_router:
            self.memory_router = MemoryAugmentedRouter(self._route_dim or d_shared,
                                                       self.n_slots)
        # 路线4：Hebbian 模态耦合
        if use_hebbian:
            self.hebbian = HebbianSlotAlignment()
        # 路线6：模态条件 adapter（按每槽实际输出维 sp.dim 构造，兼容 pm_reconcile d_shared != sp.dim）
        if use_modality_adapter:
            self.adapter_bank = ModalityAdapterBank(
                slots=self.slot_names,
                d=d_shared,
                rank=8,
                slot_dims={k: sp.dim for k, sp in self.subspaces.items()},
            )

    # ------------------------------------------------------------------
    def _path(self, slot: str, model: str) -> List[str]:
        """模型到参考模型的对齐路径（支持两步复合）。"""
        ref = self.reference[slot]
        if model == ref:
            return []
        direct = f'{slot}|{model}>{ref}'
        if direct in self.align:
            return [direct]
        for mid in self.subspaces[slot].models:
            if mid in (model, ref):
                continue
            k1 = f'{slot}|{model}>{mid}'
            k2 = f'{slot}|{mid}>{ref}'
            if k1 in self.align and k2 in self.align:
                return [k1, k2]
        return []

    def build_slot_feats(
        self,
        projs: Dict[str, torch.Tensor],
        slot_acts: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        strict_real_slots: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """构造四槽特征：优先使用真实注意力 Q/K/V/O 激活，缺省时显式标记为启发式回退。

        修复缺陷: 此前 slot_acts=None 时静默用启发式变换冒充真实 Q/K/V/O 槽。
        现记录 self.last_slot_source ('real_slots' | 'mixed' | 'heuristic_fallback')，
        且 strict_real_slots=True 时缺失真实槽激活将显式报错。
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        n_real = 0
        n_heur = 0
        for m, h in projs.items():
            out[m] = {}
            for k in self.slot_names:
                if slot_acts and m in slot_acts and k in slot_acts[m]:
                    out[m][k] = slot_acts[m][k]
                    n_real += 1
                else:
                    if strict_real_slots:
                        raise ValueError(
                            f'strict_real_slots=True 但缺少真实槽激活: model={m!r}, slot={k!r}'
                        )
                    n_heur += 1
                    if k == 'Q':
                        c = h - h.mean(dim=1, keepdim=True)
                        out[m][k] = F.layer_norm(c, (h.shape[-1],))
                    elif k == 'K':
                        scale = torch.sigmoid(h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5))
                        out[m][k] = h * scale
                    elif k == 'O' and h.shape[1] >= 3:
                        ctx = F.avg_pool1d(
                            F.pad(h.transpose(1, 2), (1, 1), mode='replicate'),
                            kernel_size=3, stride=1,
                        ).transpose(1, 2)
                        out[m][k] = 0.5 * h + 0.5 * ctx
                    else:
                        out[m][k] = h
        if n_real > 0 and n_heur == 0:
            self.last_slot_source = 'real_slots'
        elif n_real > 0:
            self.last_slot_source = 'mixed'
        else:
            self.last_slot_source = 'heuristic_fallback'
        return out

    def forward(self, feats: Dict[str, Dict[str, torch.Tensor]],
                return_aux: bool = False):
        """feats: {model: {slot: [B,S,d_shared]}}

        returns:
          struct:    [B,S,n_slots,D]   结构化 token（槽轴可分/可重组）
          canonical: [B,S,Σdim_s]      扁平拼接，喂给融合层
          alpha:     [B,S,E]           路由权重
        """
        # ---- 路线4：Hebbian 更新（仅训练态更新，避免 eval 推理态漂移；聚合全部可用槽）----
        if self.use_hebbian and hasattr(self, 'hebbian') and self.training:
            with torch.no_grad():
                agg_feats = {}
                for m in feats:
                    avail = [feats[m][s] for s in self.slot_names if s in feats.get(m, {})]
                    if avail:
                        agg_feats[m] = torch.stack([v.mean(dim=1) for v in avail], dim=0).mean(dim=0, keepdim=True)
                if agg_feats:
                    self.hebbian(agg_feats)

        per_slot: Dict[str, torch.Tensor] = {}
        B = S = None

        def _apply_proj(slot_name: str, model_name: str, inp: torch.Tensor) -> torch.Tensor:
            key = f'{slot_name}|{model_name}'
            mod = self.proj[key]
            use_adapt = (
                self.use_modality_adapter and hasattr(self, 'adapter_bank')
                and slot_name in self.adapter_bank.INJECTION.get(model_name, [])
            )
            if not use_adapt:
                return mod(inp)
            _mid = {'Qwen': 0, 'BERT': 1, 'ViT': 2}.get(model_name, 0)
            # 若同时启用 use_hyper (HyperLinear)，先叠加低秩动态权重 delta，避免绕过 HyperLinear
            if hasattr(mod, 'delta'):
                W_base = mod.weight.unsqueeze(0) + mod.delta(inp)
                ad = self.adapter_bank.get_adapter(slot_name, model_name)
                A_ad, B_ad = ad(_mid)
                W_eff = W_base + (ad.scale * (A_ad @ B_ad)).unsqueeze(0)
                if inp.dim() == 2:
                    return torch.bmm(W_eff, inp.unsqueeze(-1)).squeeze(-1)
                return torch.einsum('bsd,bod->bso', inp, W_eff)
            W_eff = self.adapter_bank.get_adapter(slot_name, model_name).adapt(mod.weight, _mid)
            return F.linear(inp, W_eff)

        for k, sp in self.subspaces.items():
            aligned: List[torch.Tensor] = []
            present_idx: List[int] = []
            for m_idx, m in enumerate(sp.models):
                key = f'{k}|{m}'
                if key not in self.proj:
                    continue
                if m not in feats or k not in feats[m]:
                    continue
                x = _apply_proj(k, m, feats[m][k])
                if B is None:
                    B, S = x.shape[0], x.shape[1]
                for step in self._path(k, m):
                    x = self.align[step](x)
                aligned.append(x)
                present_idx.append(m_idx)
            if not aligned:
                continue
            if len(aligned) == 1 or k not in self.cross_attn:
                per_slot[k] = aligned[0]
            else:
                # 序列长度可能不同（Qwen/BERT 128, ViT 197）→ 交叉注意力对齐
                q = aligned[0]
                idx_t = torch.tensor(present_idx, dtype=torch.long, device=self.fuse_w[k].device)
                w_base = F.softmax(self.fuse_w[k].index_select(0, idx_t), dim=0)
                # ---- 路线4：Hebbian 耦合矩阵调制融合权重 ----
                if self.use_hebbian and hasattr(self, 'hebbian'):
                    slot_feats = {m: feats[m][k] for m in sp.models
                                  if m in feats and k in feats.get(m, {})}
                    w_hebb = self.hebbian.gate_for_slot(slot_feats).to(
                        w_base.device)
                    w = w_base * w_hebb[:len(w_base)]        # 可学习 × 动态耦合
                    w = w / (w.sum() + 1e-9)
                else:
                    w = w_base
                if len(aligned) > 2:
                    w_rest = w[1:] / (w[1:].sum() + 1e-9) * (len(aligned) - 1)
                    kv = torch.cat([aligned[1 + idx_m] * w_rest[idx_m]
                                    for idx_m in range(len(aligned) - 1)], dim=1)
                else:
                    kv = torch.cat(aligned[1:], dim=1)
                fused, _ = self.cross_attn[k](q, kv, kv)
                per_slot[k] = w[0] * q + (1 - w[0]) * fused

        if not per_slot:
            raise ValueError('没有任何有效输入')

        ref_tensor = next(iter(per_slot.values()))
        dev = ref_tensor.device
        dtype = ref_tensor.dtype
        # 各槽 seq 长度可能不同 → 自适应平均池化到统一长度再拼接（保留全部视觉 patch 信息）
        s_min = min(v.shape[1] for v in per_slot.values())

        def _pool(v: torch.Tensor) -> torch.Tensor:
            if v.shape[1] == s_min:
                return v
            return F.adaptive_avg_pool1d(v.transpose(1, 2), s_min).transpose(1, 2)

        S = s_min
        # ---- 路线2：递归槽门控 或 静态 slot_gate ----
        if self.use_recurrent_gate and hasattr(self, 'recurrent_gate'):
            # 用当前 canonical（近似：per_slot 拼接）驱动 GRU
            gate_in = torch.cat(
                [_pool(per_slot[k]) if k in per_slot
                 else torch.zeros(B, S, self.subspaces[k].dim, device=dev, dtype=dtype)
                 for k in self.subspaces], dim=-1)      # [B,S,total_dim]
            gate = self.recurrent_gate(gate_in)         # [B,S,n_slots]
            gate_global = gate.mean(dim=1)              # [B,n_slots] 兜底
        else:
            gate = F.softmax(self.slot_gate, dim=0)     # 全局槽权重 [n_slots]

        # ---------- 路由：token 级槽门控 ----------
        #
        # v62.1 修复：此前 router 的输出 alpha 只被返回、从未参与计算，
        # canonical 只乘了全局的 slot_gate ⇒ router 是死代码，
        # 梯度不会流向它，训练出的权重无意义，报告里的「专家负载/路由熵」
        # 统计的也是这个装饰品。
        #
        # 现在 router 的输出作为 **token 级槽门控** 真正乘进 canonical：
        #   slot_gate  : 全局，哪些槽整体更重要（[n_slots]）
        #   alpha      : 逐 token，这个 token 更该看哪几个槽（[B,S,n_slots]，top-k 稀疏）
        # 有效门控 = 全局 × 逐 token，再归一化。
        #
        # 语义边界：router 只读 route_slots（三模型对齐失败的槽）的投影，
        # 但门控作用在**全部槽**上 —— 它学的是「哪些槽值得看」。
        rp: List[torch.Tensor] = []
        for k in self.route_slots:
            for m in self.subspaces[k].models:
                key = f'{k}|{m}'
                if key not in self.proj:
                    continue
                if m not in feats or k not in feats[m]:
                    rp.append(torch.zeros(B, S, self.subspaces[k].dim,
                                          device=dev, dtype=dtype))
                    continue
                v = _apply_proj(k, m, feats[m][k])
                if v.shape[1] != S:
                    v = F.adaptive_avg_pool1d(v.transpose(1, 2), S).transpose(1, 2)
                rp.append(v)

        if self.route_slots and rp and self.router is not None:
            xin = torch.cat(rp, dim=-1)
            assert xin.shape[-1] == self._route_dim, (
                'router 输入宽度 %d != 声明 %d' % (xin.shape[-1], self._route_dim))
            # ---- 路线8：记忆增强路由 或 标准 router ----
            if self.use_memory_router and hasattr(self, 'memory_router'):
                alpha = F.softmax(self.memory_router(xin), dim=-1)
            else:
                alpha = F.softmax(self.router(xin), dim=-1)
        else:
            alpha = torch.ones(B, S, self.n_slots, device=dev, dtype=dtype) / self.n_slots

        # 有效门控 = 全局槽权重 × 逐 token 槽激活，归一化到和为 1
        # 路线2 时 gate 已是 [B,S,n_slots]（GRU 逐 token 输出），直接用；
        # 否则 gate 是 [n_slots]，广播成 [1,1,n_slots]。
        if gate.dim() == 3:
            eff = gate * alpha
        else:
            eff = gate.view(1, 1, -1) * alpha
        eff = eff / (eff.sum(-1, keepdim=True) + 1e-9)

        canonical = torch.cat(
            [_pool(per_slot[k]) * eff[..., i:i + 1] if k in per_slot
             else torch.zeros(B, S, self.subspaces[k].dim, device=dev, dtype=dtype)
             for i, k in enumerate(self.subspaces)],
            dim=-1)

        # 结构化 token（同样按有效门控缩放，保证两条出口一致）
        D = max(sp.dim for sp in self.subspaces.values())
        struct = torch.zeros(B, S, self.n_slots, D, device=dev, dtype=canonical.dtype)
        for i, (k, sp) in enumerate(self.subspaces.items()):
            if k in per_slot:
                struct[:, :, i, :sp.dim] = _pool(per_slot[k]) * eff[..., i:i + 1]

        if return_aux:
            return struct, canonical, {
                'alpha': alpha,
                'per_slot': per_slot,
                'slot_gate': gate.detach(),
                'route_slots': self.route_slots,
                'total_dim': self.total_dim,
                'struct': struct,
            }
        return struct, canonical, alpha

    # ------------------------------------------------------------------
    # 分离 / 重组
    # ------------------------------------------------------------------
    def separate(self, struct: torch.Tensor, slot: str) -> torch.Tensor:
        """从结构化 token 取出某个槽的分量。"""
        i = self.slot_names.index(slot)
        return struct[:, :, i, :self.subspaces[slot].dim]

    def recombine(self, struct: torch.Tensor, recipe: Dict[str, int]) -> torch.Tensor:
        """跨 token 重组：recipe = {slot: token_index}。

        例：{'K': 0, 'V': 3} → 用 token0 的 K 分量 + token3 的 V 分量拼新 token。
        """
        B, S, _, _ = struct.shape
        out = torch.zeros_like(struct)
        for slot, idx in recipe.items():
            i = self.slot_names.index(slot)
            # 兼容负索引（如 idx=-1），避免 -1:0 产生空切片
            out[:, :, i, :] = struct.select(1, idx)[:, i, :].unsqueeze(1).expand(-1, S, -1)
        return out

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def consistency_loss(self) -> torch.Tensor:
        """环路一致性：A_{a→c} ≡ A_{b→c} ∘ A_{a→b}。

        这是「三模型对齐失败」的直接修复项 —— 多条对齐路径必须自洽。
        """
        losses = []
        for k, sp in self.subspaces.items():
            ms = sp.models
            if len(ms) < 3:
                continue
            for a, b, c in [(x, y, z) for x in ms for y in ms for z in ms
                            if len({x, y, z}) == 3]:
                k1, k2, k3 = f'{k}|{a}>{b}', f'{k}|{b}>{c}', f'{k}|{a}>{c}'
                if not all(key in self.align for key in (k1, k2, k3)):
                    continue
                w1 = self.align[k1].weight
                w2 = self.align[k2].weight
                w3 = self.align[k3].weight
                losses.append(F.mse_loss(w2 @ w1, w3))
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(losses).mean()

    def slot_entropy_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """槽门控熵的**负值** —— 最小化它等于最大化熵，防止门控塌到单槽。

        训练时用法： total = ce + BAL_W * bal + SLOT_ENT_W * op.slot_entropy_loss(alpha)

        ⚠️ 这是**人为干预**，不是自然收敛的结果。若加了正则仍不展开，
        说明该任务下四槽分工本来就不成立，那是应当如实报告的负结果。
        """
        h = -(alpha * torch.log(alpha + 1e-9)).sum(-1).mean()
        return -h

    def alignment_loss(
        self,
        pairs: Dict[Tuple[str, Tuple[str, str]], Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """对齐监督：pairs = {(slot, (m1, m2)): (X_m1, X_m2)}，X 已在该槽子空间内。"""
        total, n = 0.0, 0
        for (slot, (m1, m2)), (x1, x2) in pairs.items():
            key = f'{slot}|{m1}>{m2}'
            if key not in self.align:
                continue
            total = total + F.mse_loss(self.align[key](x1), x2)
            n += 1
        if n == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        return total / n

    # ------------------------------------------------------------------
    def describe(self) -> str:
        lines = ['维度投影空间（多子空间）']
        lines.append('  空间 = Σ_s S_s , dim = %d , n_slots = %d'
                     % (self.total_dim, self.n_slots))
        lines.append('  %-3s %-6s %-14s %-9s %-9s %s'
                     % ('槽', 'dim', 'models', 'CKA对', 'CKA三', '对齐路径'))
        for k, sp in self.subspaces.items():
            paths = []
            for m in sp.models:
                p = self._path(k, m)
                paths.append('%s→%s(%d)' % (m, self.reference[k], len(p)) if p
                             else '%s=ref' % m)
            lines.append('  %-3s %-6d %-14s %-9.4f %-9s %s'
                         % (k, sp.dim, ','.join(sp.models), sp.cka_pair,
                            ('%.4f' % sp.cka_triple) if sp.cka_triple else '—',
                            ' '.join(paths)))
        lines.append('  路由槽（三模型对齐失败）: %s' % (self.route_slots or '无'))
        lines.append('  router 输出: %d 维 token 级槽门控（全槽软门控，非 top-k）' % self.n_slots)
        lines.append('  参数: %d' % sum(p.numel() for p in self.parameters()))
        return '\n'.join(lines)



# ============================================================ fusion
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionOverProjSpace(nn.Module):
    """输入 [B,S,d_space]，输出 [B,S,d_space] + 路由权重。"""

    def __init__(self, d_space: int = 1024, d_ff: int | None = None,
                 num_experts: int = 4, top_k: int = 2,
                 use_fast: bool = False, fast_decay: float = 0.9):
        super().__init__()
        self.d_space = d_space
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_fast = use_fast
        self.fast_decay = fast_decay
        d_ff = d_ff or 4 * d_space

        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_space)) for _ in range(num_experts)])
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_space)) for _ in range(num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_space) * 0.02) for _ in range(num_experts)])
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_space) * 0.02) for _ in range(num_experts)])
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_space, d_ff) * 0.02) for _ in range(num_experts)])
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])
        self.W_router = nn.Parameter(torch.randn(num_experts, d_space) * 0.01)

        # ---- 路线 3：快速权重专家（Fast-Weight FFN）----
        # 真正实现：用 key-value outer product 累积 ΔW，推理后遗忘。
        # 每专家有 fast_u/fast_v 生成低秩 ΔW = outer(u(x), v(x))，
        # 动态修改 FFN 第一层权重。默认关闭（use_fast=False）。
        self.use_fast = use_fast
        if use_fast:
            self.fast_u = nn.ParameterList(
                [nn.Parameter(torch.randn(d_space, d_ff) * 0.02)
                 for _ in range(num_experts)])
            self.fast_v = nn.ParameterList(
                [nn.Parameter(torch.randn(d_space, d_space) * 0.02)
                 for _ in range(num_experts)])
            self.fast_scale = nn.Parameter(torch.tensor([0.05]))

    @staticmethod
    def _swiglu(x: torch.Tensor, wg, wu, wd) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)

    def forward_with_routing(self, x: torch.Tensor):
        # 路由只看输入分布，不该被专家梯度污染
        logits = F.linear(x.detach(), self.W_router)
        a = F.softmax(logits, dim=-1)
        self.last_router_probs = a
        _, ti = torch.topk(a, self.top_k, dim=-1)
        ah = a * torch.zeros_like(a).scatter_(-1, ti, 1.0)

        # ---- 路线 3：快速权重 ΔW_b = outer(u(x_b), v(x_b)) ----
        # 逐样本沿序列维求均值生成低秩因子 (u_b, v_b)，通过分解式
        # h @ ΔW_b^T = (h @ v_b) * u_b 计算，既避免跨样本信息串扰，又无需物化 [B,d_ff,d_space]。
        fast_uv = None
        if self.use_fast:
            mu = x.detach().mean(dim=1)                       # [B, d_space] 逐样本均值
            fast_uv = []
            for m in range(self.num_experts):
                u = mu @ self.fast_u[m]                       # [B, d_ff]
                v = mu @ self.fast_v[m]                       # [B, d_space]
                fast_uv.append((u, v))

        out_flat = torch.zeros_like(x.reshape(-1, self.d_space))
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        ah_flat = ah.reshape(B * S, self.num_experts)
        counts = []
        for m in range(self.num_experts):
            w_m = ah_flat[:, m]
            active = torch.nonzero(w_m > 0, as_tuple=False).squeeze(-1)
            n_act = int(active.numel())
            counts.append(n_act)
            if n_act == 0:
                continue
            x_act = x_flat.index_select(0, active)
            h = x_act * self.gammas[m] + self.betas[m]   # 不 detach：专家参与训练，仅计算活跃 token
            if self.use_fast and fast_uv is not None:
                u_b, v_b = fast_uv[m]
                b_idx = torch.div(active, S, rounding_mode='floor')
                u_act = u_b.index_select(0, b_idx)       # [N_act, d_ff]
                v_act = v_b.index_select(0, b_idx)       # [N_act, d_space]
                proj_v = (h * v_act).sum(dim=-1, keepdim=True)
                gate_lin = F.linear(h, self.w_gates[m]) + self.fast_scale * proj_v * u_act
                f = F.linear(F.silu(gate_lin) * F.linear(h, self.w_ups[m]), self.w_downs[m])
            else:
                f = self._swiglu(h, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            contrib = w_m.index_select(0, active).unsqueeze(-1) * self.alphas[m] * f
            out_flat = out_flat.index_add(0, active, contrib)
        self.last_expert_token_counts = counts
        return x + out_flat.reshape(B, S, D), ah

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_routing(x)[0]

    def describe(self) -> str:
        d_ff = self.w_gates[0].shape[0]
        n = sum(p.numel() for p in self.parameters())
        n_frozen = 0
        return ('FusionOverProjSpace\n'
                '  d_space=%d → d_ff=%d → d_space=%d\n'
                '  experts=%d  top_k=%d\n'
                '  参数=%d（专家解冻，全部可训）'
                % (self.d_space, d_ff, self.d_space,
                   self.num_experts, self.top_k, n))



# ============================================================ model
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers import (
    AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel, ViTModel,
)
from transformers.modeling_outputs import ModelOutput



QWEN_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
BERT_ID = 'google/bert_uncased_L-4_H-256_A-4'
VIT_ID = 'google/vit-base-patch16-224-in21k'

MODAL_NAMES = ['Qwen', 'BERT', 'ViT']


def remap_vit_state(sd: Dict[str, torch.Tensor], to_hf: bool = True) -> Dict[str, torch.Tensor]:
    """在旧（BERT 风格）/ 新（HF ViT）命名之间转换 vit.* 张量。

    to_hf=True  : encoder.layer.N.attention.attention.query → layers.N.attention.q_proj
    to_hf=False : 反向
    """
    # 旧 → 新（有序：先具体后一般，否则 .attention.attention. 会吃掉子规则）
    fwd = [
        ('.attention.attention.query.', '.attention.q_proj.'),
        ('.attention.attention.key.', '.attention.k_proj.'),
        ('.attention.attention.value.', '.attention.v_proj.'),
        ('.attention.attention.', '.attention.'),
        ('.attention.output.dense.', '.attention.o_proj.'),
        ('.attention.output.LayerNorm.', '.layernorm_before.'),
        ('.intermediate.dense.', '.mlp.fc1.'),
        ('.output.dense.', '.mlp.fc2.'),
        ('.output.LayerNorm.', '.layernorm_after.'),
    ]
    # 新 → 旧必须**显式**给出：reversed(fwd) 会让 `.attention.` → `.attention.attention.`
    # 先执行，把后面的 q_proj/k_proj 规则全部挡掉（实测踩过）。
    bwd = [
        ('.attention.q_proj.', '.attention.attention.query.'),
        ('.attention.k_proj.', '.attention.attention.key.'),
        ('.attention.v_proj.', '.attention.attention.value.'),
        ('.attention.o_proj.', '.attention.output.dense.'),
        ('.layernorm_before.', '.attention.output.LayerNorm.'),
        ('.mlp.fc1.', '.intermediate.dense.'),
        ('.mlp.fc2.', '.output.dense.'),
        ('.layernorm_after.', '.output.LayerNorm.'),
    ]
    pairs = fwd if to_hf else bwd

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        if to_hf:
            if k.startswith('vit.') and ('.encoder.layer.' in k or '.encoder.layers.' in k):
                nk = (k.replace('.encoder.layer.', '.layers.')
                       .replace('.encoder.layers.', '.layers.'))
                for a, b in pairs:
                    nk = nk.replace(a, b)
        else:
            if k.startswith('vit.') and '.layers.' in k:
                nk = k.replace('.layers.', '.encoder.layer.')
                for a, b in pairs:
                    nk = nk.replace(a, b)
        out[nk] = v
    return out




@dataclass
class ProjSpaceOutput(ModelOutput):
    y: torch.FloatTensor = None
    alpha: torch.FloatTensor = None
    slot_alpha: torch.FloatTensor = None


class HeteroMoEProjSpaceModel(PreTrainedModel):
    config_class = HeteroMoEProjSpaceConfig
    base_model_prefix = 'hetero_moe_projspace'
    _tied_weights_keys: List[str] = []
    all_tied_weights_keys: Dict[str, str] = {}   # transformers 5.x 内部会访问

    def __init__(self, config: HeteroMoEProjSpaceConfig):
        super().__init__(config)
        self.config = config

        # 骨干用 from_config 建骨架：不能在 __init__ 里嵌套 from_pretrained，
        # 会和 transformers 的 meta-device 加载流程冲突。
        qc = AutoConfig.from_pretrained(config.qwen_id)
        bc = AutoConfig.from_pretrained(config.bert_id)
        vc = AutoConfig.from_pretrained(config.vit_id)
        self.qwen = AutoModel.from_config(qc)
        self.bert = AutoModel.from_config(bc)
        self.vit = AutoModel.from_config(vc)

        # 各模态 → 共享空间（沿用 v33 的 P_m，冻结）
        self.P_m = nn.ParameterList(
            [nn.Parameter(torch.empty(config.d_shared, D))
             for D in config.modal_dims])
        for i in range(len(config.modal_dims)):
            nn.init.orthogonal_(self.P_m[i])
            self.P_m[i].requires_grad_(False)

        self.proj_space = DimProjectionSpace(
            config.d_shared, MEASURED_SUBSPACES,
            config.n_experts, config.moe_top_k,
            use_hyper=getattr(config, 'use_hyper', False),
            hyper_rank=getattr(config, 'hyper_rank', 8))
        self.fusion = FusionOverProjSpace(
            config.d_space, None, config.n_experts, config.moe_top_k,
            use_fast=getattr(config, 'use_fast', False),
            fast_decay=getattr(config, 'fast_decay', 0.9))

    @classmethod
    def from_scratch(cls, config: HeteroMoEProjSpaceConfig):
        """首次构建：真正加载三个预训练骨干（不走 PreTrainedModel 的加载流程）。"""
        obj = cls(config)
        obj.qwen = AutoModel.from_pretrained(config.qwen_id)
        obj.bert = AutoModel.from_pretrained(config.bert_id)
        obj.vit = ViTModel.from_pretrained(config.vit_id)
        return obj

    def load_packed_state(self, sd: Dict[str, torch.Tensor],
                          remap_vit: bool = True):
        """加载打包权重：先做 ViT 命名转换再灌。返回 (missing, unexpected)。"""
        if remap_vit:
            sd = remap_vit_state(sd, to_hf=True)
        res = self.load_state_dict(sd, strict=False)
        return res.missing_keys, res.unexpected_keys

    @staticmethod
    def _find_slot_modules(backbone: nn.Module, model_name: str) -> Dict[str, nn.Module]:
        """定位末层真实注意力 Q/K/V/O 投影模块（兼容 transformers 4.x/5.x ViT 命名）。"""
        slot_suffixes = {
            'Qwen': {'Q': ('self_attn.q_proj',), 'K': ('self_attn.k_proj',),
                     'V': ('self_attn.v_proj',), 'O': ('self_attn.o_proj',)},
            'BERT': {'Q': ('attention.self.query',), 'K': ('attention.self.key',),
                     'V': ('attention.self.value',), 'O': ('attention.output.dense',)},
            'ViT':  {'Q': ('attention.q_proj', 'attention.attention.query'),
                     'K': ('attention.k_proj', 'attention.attention.key'),
                     'V': ('attention.v_proj', 'attention.attention.value'),
                     'O': ('attention.o_proj', 'attention.output.dense')},
        }.get(model_name, {})
        found: Dict[str, nn.Module] = {}
        for name, mod in backbone.named_modules():
            for slot, sufs in slot_suffixes.items():
                if any(name.endswith(suf) for suf in sufs):
                    found[slot] = mod          # 遍历至末层，自然保留最后一层
        return found

    def _project_slot_act(self, act: torch.Tensor, base_p: torch.Tensor) -> torch.Tensor:
        """将真实槽激活 [B,S,D_slot] 投影至共享空间 d_shared（兼容 GQA K/V 窄于 d_model 的情况）。"""
        d_slot = act.shape[-1]
        if d_slot == base_p.shape[1]:
            return act @ base_p.t()
        # GQA 等非方阵槽（如 Qwen K/V 维 128 < 896）: 截断或零填充列后正交投影
        if d_slot < base_p.shape[1]:
            return act @ base_p[:, :d_slot].t()
        return act[..., :base_p.shape[1]] @ base_p.t()

    def forward(self, qwen_input_ids=None, qwen_attention_mask=None,
                bert_input_ids=None, bert_attention_mask=None,
                bert_token_type_ids=None, pixel_values=None,
                use_real_slots: bool = True):
        hs: Dict[str, torch.Tensor] = {}
        raw_slots: Dict[str, Dict[str, torch.Tensor]] = {}

        def _run_with_hooks(m_name: str, backbone: nn.Module, call_fn):
            captured: Dict[str, torch.Tensor] = {}
            handles = []
            if use_real_slots:
                for slot, mod in self._find_slot_modules(backbone, m_name).items():
                    def _hook(_mod, _inp, out, _s=slot):
                        t = out[0] if isinstance(out, tuple) else out
                        captured[_s] = t
                    handles.append(mod.register_forward_hook(_hook))
            try:
                out_h = call_fn()
            finally:
                for hdl in handles:
                    hdl.remove()
            if captured:
                raw_slots[m_name] = captured
            return out_h

        if qwen_input_ids is not None:
            hs['Qwen'] = _run_with_hooks(
                'Qwen', self.qwen,
                lambda: self.qwen(
                    input_ids=qwen_input_ids,
                    attention_mask=qwen_attention_mask).last_hidden_state,
            )
        if bert_input_ids is not None:
            hs['BERT'] = _run_with_hooks(
                'BERT', self.bert,
                lambda: self.bert(
                    input_ids=bert_input_ids,
                    attention_mask=bert_attention_mask,
                    token_type_ids=bert_token_type_ids).last_hidden_state,
            )
        if pixel_values is not None:
            hs['ViT'] = _run_with_hooks(
                'ViT', self.vit,
                lambda: self.vit(pixel_values=pixel_values).last_hidden_state,
            )
        if not hs:
            raise ValueError('至少需要一个模态的输入')

        projs: Dict[str, torch.Tensor] = {}
        slot_acts: Dict[str, Dict[str, torch.Tensor]] = {}
        for i, m in enumerate(MODAL_NAMES):
            if m in hs:
                projs[m] = hs[m] @ self.P_m[i].t()
                if m in raw_slots:
                    slot_acts[m] = {
                        k: self._project_slot_act(v, self.P_m[i])
                        for k, v in raw_slots[m].items()
                    }

        feats = self.proj_space.build_slot_feats(projs, slot_acts=slot_acts or None)
        _struct, canonical, _aux = self.proj_space(feats, return_aux=True)
        y, alpha = self.fusion.forward_with_routing(canonical)
        # alpha      : 专家选择 [B,S,n_experts]，top-k 真稀疏分派（仅计算活跃 token）
        # slot_alpha : 槽门控   [B,S,n_slots]  ，全槽软分布（四个槽各贡献多少）
        return ProjSpaceOutput(y=y, alpha=alpha, slot_alpha=_aux['alpha'])

    def describe(self) -> str:
        n = lambda mod: sum(p.numel() for p in mod.parameters())
        total = n(self)
        return ('HeteroMoEProjSpaceModel\n'
                '  总参数 %s (%.2fM)\n'
                '    qwen      %.2fM\n'
                '    bert      %.2fM\n'
                '    vit       %.2fM\n'
                '    proj_space %.2fM\n'
                '    fusion    %.2fM\n'
                '%s'
                % ('{:,}'.format(total), total / 1e6,
                   n(self.qwen) / 1e6, n(self.bert) / 1e6, n(self.vit) / 1e6,
                   n(self.proj_space) / 1e6, n(self.fusion) / 1e6,
                   '\n'.join('  ' + ln for ln in self.proj_space.describe().splitlines())))
