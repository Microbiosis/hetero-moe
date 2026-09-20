"""v10.0 — AlignedFusionLayer (用 SemanticAligner 替换 P_m)。

设计: 与 v9 的 CentralAugmentedFusionLayer 类似, 但用 Aligner 替代线性 P_m.
关键差异:
    v9: x_shared = h_m · P_m^T           (每个底座独立投影)
    v10: x_shared = Aligner([h_A, h_B, h_C]) (跨底座联合对齐)

其余部分 (双路由 + 中枢 gate-style) 复用 v9.0 修正 C.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.router import SparseRouterSTE
from research._primitives.attention import AttnPool
from research._primitives.central_mechanism import GateStyleWorkspace

from .aligner import SemanticAligner


def _swiglu(x, w_gate, w_up, w_down):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


class AlignedFusionLayer(nn.Module):
    """v10.0: 用 SemanticAligner 替代 v6/v9 的线性 P_m.

    注意: 此 layer 需要**所有底座**的隐藏态同时输入, 不能像 v9 那样逐个处理.
    输入: h_list = [h_A, h_B, h_C], 每个 [B, S, D_m]
    输出: y [B, S, d_shared]

    C-1 中枢: 用 v9_gate_central.GateStyleWorkspace (gate-style).
    """

    def __init__(
        self,
        d_shared: int,
        d_ff: int,
        attn_pools: List[AttnPool],
        modal_dims: List[int],
        aligner: SemanticAligner,
        top_k_attn: int = 1,
        top_k_ffn: int = 1,
        quant_bits: int = 4,
        quant_group_size: int = 128,
        ema_decay: float = 0.9,
        c1_alpha: float = 0.1,
        c1_mode: str = "gate",
    ):
        super().__init__()
        self.d_shared = d_shared
        self.aligner = aligner
        self.num_experts_attn = len(attn_pools)
        self.top_k_attn = top_k_attn
        self.num_experts_ffn = len(modal_dims)
        self.top_k_ffn = top_k_ffn
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.c1_mode = c1_mode
        assert c1_mode in ("gate", "broadcast"), f"c1_mode 必须为 gate/broadcast, 实测 {c1_mode}"

        # attn 路由
        self.attn_pools = nn.ModuleList(attn_pools)
        self.W_router_attn = nn.Parameter(torch.randn(self.num_experts_attn, d_shared) * 0.01)
        self.alphas_attn = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(self.num_experts_attn)]
        )

        # ffn 路由 (用 aligner 内部的 down_projection 替代 v6/v9 的 P_m)
        self.W_router_ffn = nn.Parameter(torch.randn(self.num_experts_ffn, d_shared) * 0.01)
        self.V_coop = nn.Parameter(torch.eye(d_shared) * 0.1)

        # FFN 权重 (冻结)
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02) for _ in range(self.num_experts_ffn)]
        )
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02) for _ in range(self.num_experts_ffn)]
        )
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_shared, d_ff) * 0.02) for _ in range(self.num_experts_ffn)]
        )
        for pl in (self.w_gates, self.w_ups, self.w_downs):
            for p in pl:
                p.requires_grad = False
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_shared)) for _ in range(self.num_experts_ffn)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_shared)) for _ in range(self.num_experts_ffn)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(self.num_experts_ffn)]
        )

        # 中枢 (默认 gate-style, v9 修正 C, 通过 v9_gate_central.make_workspace 工厂)
        # c1_mode="broadcast" 时使用 BroadcastWorkspace (v8.0 原版, 已知 H1 失败)
        from research._primitives.central_mechanism import make_workspace
        self.cw_attn = make_workspace(
            mode=c1_mode, d_shared=d_shared,
            num_experts=self.num_experts_attn,
            ema_decay=ema_decay, alpha=c1_alpha,
        )
        self.cw_ffn = make_workspace(
            mode=c1_mode, d_shared=d_shared,
            num_experts=self.num_experts_ffn,
            ema_decay=ema_decay, alpha=c1_alpha,
        )
        self._ema_enabled = True

    def disable_ema(self):
        self._ema_enabled = False

    def _align_all(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        """调用 aligner, 输出 [B, S, d_shared] 统一表示"""
        return self.aligner(h_list)

    def _ffn_path(self, x_shared: torch.Tensor, attn_sum: torch.Tensor):
        """与 v9 类似, 但 x_shared 已经是对齐后的统一表示"""
        x_det = x_shared.detach()
        coop_ctx = F.linear(attn_sum.detach(), self.V_coop)
        z_f = F.linear(x_det + coop_ctx, self.W_router_ffn)
        z_f = self.cw_ffn.augment_router_logits(z_f)
        alpha_f = F.softmax(z_f, dim=-1)
        ahat_f = SparseRouterSTE.apply(alpha_f, self.top_k_ffn)
        ffn_sum = torch.zeros_like(x_shared)
        expert_outs = []
        for m in range(self.num_experts_ffn):
            h_a = x_det * self.gammas[m] + self.betas[m]
            f_m = _swiglu(h_a, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_q = FakeQuantSTE.apply(f_m, self.quant_bits, self.quant_group_size)
            expert_outs.append(f_q.detach())
            ffn_sum = ffn_sum + ahat_f[..., m:m+1] * self.alphas[m] * f_q
        return ffn_sum, ahat_f, expert_outs

    def _attn_path(self, x_shared: torch.Tensor):
        x_det = x_shared.detach()
        z_a = F.linear(x_det, self.W_router_attn)
        z_a = self.cw_attn.augment_router_logits(z_a)
        alpha_a = F.softmax(z_a, dim=-1)
        ahat_a = SparseRouterSTE.apply(alpha_a, self.top_k_attn)
        attn_sum = torch.zeros_like(x_shared)
        expert_outs = []
        for m in range(self.num_experts_attn):
            a_out = self.attn_pools[m](x_shared)
            expert_outs.append(a_out.detach())
            attn_sum = attn_sum + ahat_a[..., m:m+1] * self.alphas_attn[m] * a_out
        return attn_sum, ahat_a, expert_outs

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        """h_list: [h_A, h_B, h_C], 每个 [B, S, D_m]"""
        x_shared = self._align_all(h_list)
        attn_sum, _, attn_outs = self._attn_path(x_shared)
        ffn_sum, _, ffn_outs = self._ffn_path(x_shared, attn_sum)
        if self._ema_enabled and self.training:
            self.cw_attn.ema_update(attn_outs)
            self.cw_ffn.ema_update(ffn_outs)
        return x_shared + attn_sum + ffn_sum