"""维度投影空间算子。

设计依据全部来自实测（见 results/metrics.json），不是设想：

  1. 功能槽确实存在，且形式各异
       Q  Qwen:self_attn.q_proj  BERT:self.query        ViT:attention.q_proj
       K  Qwen:self_attn.k_proj  BERT:self.key          ViT:attention.k_proj
       V  Qwen:self_attn.v_proj  BERT:self.value        ViT:attention.v_proj
       O  Qwen:o_proj            BERT:output.dense      ViT:o_proj
     （FFN: Qwen 是 SwiGLU 两路，BERT/ViT 是单路 GELU —— 形式不同）

  2. 同功能的权重谱跨架构显著更相似
       同功能 JS 0.0302 / 不同功能 0.0401 / 置换检验 p=0.0002 (n=1728)
       但效应量中等 (Cohen d=0.252)，且各槽通用性差异很大：
       contract 0.0087 < O 0.0140 < expand 0.0245 < V 0.0406 < Q 0.0431 < K 0.0504

  3. 两两对齐都能到 0.987+，但三模型共用一个变换会失败
       Q 两两 0.9879~0.9893 → 三模型 0.6767
       O 两两 0.9876~0.9898 → 三模型 0.7168
       K/V 只有两模型参与，所以「看起来」更容易
     ⇒ 失败的原因不是 Qwen 的 GQA 结构，是**一个变换对 3 个模型过定**

  4. 环路一致性约束修复了它
       consistency_loss 让 A_ac ≡ A_bc ∘ A_ab
       训练后 三模型对齐 0.9925，直达与复合路径差 0.0038

  5. 等维对齐，不压也不升
       rank   CKA     保真度
       8      0.9981  0.060
       24     0.9982  0.165     ← 压维丢 83% 信息
       128    0.9982  0.712
       256    0.9982  0.9998    ← 等维无损
       384+   0.998   1.0000    ← 升维无额外收益
     ⇒ CKA 与维度无关；保真度才是维度的函数

  6. 对照实验排除了「对齐是假的」
       随机表示对齐后 CKA 0.057（真对齐 0.987+）
       假对齐 K×V / Q×O 秩几乎不降（真对齐 168→30）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d_shared = d_shared
        self.subspaces = subspaces or MEASURED_SUBSPACES
        self.slot_names = list(self.subspaces.keys())
        self.n_slots = len(self.slot_names)
        self.n_experts = n_experts
        self.top_k = top_k
        self.reference = {k: sp.models[0] for k, sp in self.subspaces.items()}

        # ---- 槽轴 × 模型轴：投影 ----
        self.proj = nn.ModuleDict()
        for k, sp in self.subspaces.items():
            for m in sp.models:
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
                    if sp.dim == d_shared:
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

    def forward(self, feats: Dict[str, Dict[str, torch.Tensor]],
                return_aux: bool = False):
        """feats: {model: {slot: [B,S,d_shared]}}

        returns:
          struct:    [B,S,n_slots,D]   结构化 token（槽轴可分/可重组）
          canonical: [B,S,Σdim_s]      扁平拼接，喂给融合层
          alpha:     [B,S,E]           路由权重
        """
        per_slot: Dict[str, torch.Tensor] = {}
        B = S = None

        for k, sp in self.subspaces.items():
            aligned: List[torch.Tensor] = []
            for m in sp.models:
                key = f'{k}|{m}'
                if key not in self.proj:
                    continue
                if m not in feats or k not in feats[m]:
                    continue
                x = self.proj[key](feats[m][k])
                if B is None:
                    B, S = x.shape[0], x.shape[1]
                for step in self._path(k, m):
                    x = self.align[step](x)
                aligned.append(x)
            if not aligned:
                continue
            if len(aligned) == 1 or k not in self.cross_attn:
                per_slot[k] = aligned[0]
            else:
                # 序列长度可能不同（Qwen/BERT 128, ViT 197）→ 交叉注意力对齐
                q = aligned[0]
                kv = torch.cat(aligned[1:], dim=1)
                fused, _ = self.cross_attn[k](q, kv, kv)
                w = F.softmax(self.fuse_w[k], dim=0)
                per_slot[k] = w[0] * q + (1 - w[0]) * fused

        if not per_slot:
            raise ValueError('没有任何有效输入')

        dev = next(self.parameters()).device
        # 各槽 seq 长度可能不同 → 池化到统一长度再拼接
        s_min = min(v.shape[1] for v in per_slot.values())

        def _pool(v: torch.Tensor) -> torch.Tensor:
            if v.shape[1] == s_min:
                return v
            idx = torch.linspace(0, v.shape[1] - 1, s_min).long().to(v.device)
            return v[:, idx, :]

        S = s_min
        gate = F.softmax(self.slot_gate, dim=0)          # 全局槽权重 [n_slots]

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
                                          device=dev, dtype=next(self.parameters()).dtype))
                    continue
                v = self.proj[key](feats[m][k])
                if v.shape[1] != S:
                    idx = torch.linspace(0, v.shape[1] - 1, S).long().to(v.device)
                    v = v[:, idx, :]
                rp.append(v)

        if self.route_slots and rp and self.router is not None:
            xin = torch.cat(rp, dim=-1)
            assert xin.shape[-1] == self._route_dim, (
                'router 输入宽度 %d != 声明 %d' % (xin.shape[-1], self._route_dim))
            # 槽门控用**全槽软门控**，不做 top-k。
            #
            # top-k 的用途是省算力（专家只算被选中的那些）。但四个槽的投影
            # 在进入这一步之前**已经全部算完了** —— 在这里 top-k 省不下任何
            # 计算，只会丢信息，并且会把门控压成「单槽独占」。
            #
            # v62.1 实测（top-k=2）：每 token 最大权重均值 0.9933、中位 1.0000，
            # 熵 0.0209（top-2 的理论上界才 0.5）。等价于只剩一个槽在工作，
            # 而 v62 的核心主张正是「空间 = Σ 子空间」，四槽必须都有贡献。
            alpha = F.softmax(self.router(xin), dim=-1)
        else:
            alpha = torch.ones(B, S, self.n_slots, device=dev) / self.n_slots

        # 有效门控 = 全局槽权重 × 逐 token 槽激活，归一化到和为 1
        eff = gate.view(1, 1, -1) * alpha                        # [B,S,n_slots]
        eff = eff / (eff.sum(-1, keepdim=True) + 1e-9)

        canonical = torch.cat(
            [_pool(per_slot[k]) * eff[..., i:i + 1] if k in per_slot
             else torch.zeros(B, S, self.subspaces[k].dim, device=dev)
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
            out[:, :, i, :] = struct[:, idx:idx + 1, i, :].expand(-1, S, -1)
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
