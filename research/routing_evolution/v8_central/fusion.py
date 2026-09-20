"""v8.0 — C-2 attn→ffn 协同融合层。

相对 v7.0 CrossArchAttnFFNFusionLayer 的关键扩展:
    - 集成 CentralWorkspace (C-1), 路由 logits 受中枢广播
    - attn 通路输出影响 ffn 路由 (attn→ffn 协同)
    - 其他 (P_m 投影, SwiGLU, INT4, K=1 STE) 与 v7 一致

forward 流程:
    1. attn 通路:
        z_a = CW.augment(W_router_attn · x)
        α̂_a = SparseRouterSTE(softmax(z_a), K)
        attn_sum = Σ α̂_a · α_a · attn_pool_m(x)
    2. ffn 通路 (受 attn 影响):
        z_f = CW.augment(W_router_ffn · (x + attn_sum))     ← attn→ffn 协同
        α̂_f = SparseRouterSTE(softmax(z_f), K)
        ffn_sum = Σ α̂_f · α_f · ffn_pool_m(x)
    3. y = x + attn_sum + ffn_sum
    4. cw.ema_update([attn_pool_outputs, ffn_pool_outputs])
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.router import SparseRouterSTE
from research._primitives.attention import AttnPool

from .central import CentralWorkspace


def _swiglu(x, w_gate, w_up, w_down):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


class CentralAugmentedFusionLayer(nn.Module):
    """v8.0 C-2: attn→ffn 协同 + 中枢广播。

    输入:  x_shared: [B, S, D_shared]
    输出:  y: [B, S, D_shared]
    额外:  expert_outputs 列表 (供 c.ema_update)
    """

    def __init__(
        self,
        d_shared: int,
        d_ff: int,
        attn_pools: List[AttnPool],
        modal_dims: List[int],
        top_k_attn: int = 1,
        top_k_ffn: int = 1,
        quant_bits: int = 4,
        quant_group_size: int = 128,
        ema_decay: float = 0.9,
        central_init_scale: float = 0.01,
    ):
        super().__init__()
        self.d_shared = d_shared
        # ---- attn 通路 (复用 v7 接口) ----
        self.attn_pools = nn.ModuleList(attn_pools)
        self.num_experts_attn = len(attn_pools)
        self.top_k_attn = top_k_attn
        self.W_router_attn = nn.Parameter(torch.randn(self.num_experts_attn, d_shared) * 0.01)
        self.alphas_attn = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(self.num_experts_attn)]
        )
        # ---- ffn 通路 ----
        self.num_experts_ffn = len(modal_dims)
        self.top_k_ffn = top_k_ffn
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.P_m = nn.ParameterList(
            [nn.Parameter(torch.empty(d_shared, Dm)) for Dm in modal_dims]
        )
        for p in self.P_m:
            nn.init.orthogonal_(p)
            p.requires_grad_(False)
        self.W_router_ffn = nn.Parameter(torch.randn(self.num_experts_ffn, d_shared) * 0.01)
        # 协同矩阵 V: d_shared × d_shared, 把 attn 输出投影后加到 ffn 路由 logits
        self.V_coop = nn.Parameter(torch.eye(d_shared) * 0.1)
        # 冻结 FFN 权重
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
        # ---- C-1 中央广播中枢 (attn + ffn 各自一个, v8 broadcast 模式) ----
        # 注意: v8 broadcast 是 H1 已知失败模式, 仅保留作历史对照.
        # 新代码请用 v9_gate_central.GateStyleWorkspace.
        self.cw_attn = CentralWorkspace(d_shared, self.num_experts_attn, ema_decay=ema_decay,
                                         init_scale=central_init_scale)
        self.cw_ffn  = CentralWorkspace(d_shared, self.num_experts_ffn,  ema_decay=ema_decay,
                                         init_scale=central_init_scale)

        # EMA 是否启用 (供 trainer 控制)
        self._ema_enabled = True

    def disable_ema(self) -> None:
        self._ema_enabled = False

    def _attn_path(self, x_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """返回 (attn_sum, alpha_hat, expert_outputs_for_ema)"""
        x_det = x_shared.detach()
        z_a = F.linear(x_det, self.W_router_attn)
        z_a = self.cw_attn.augment_router_logits(z_a)      # C-1 中枢广播
        alpha_a = F.softmax(z_a, dim=-1)
        ahat_a = SparseRouterSTE.apply(alpha_a, self.top_k_attn)
        attn_sum = torch.zeros_like(x_shared)
        expert_outs = []
        for m in range(self.num_experts_attn):
            a_out = self.attn_pools[m](x_shared)
            expert_outs.append(a_out.detach())
            attn_sum = attn_sum + ahat_a[..., m:m+1] * self.alphas_attn[m] * a_out
        return attn_sum, ahat_a, expert_outs

    def _ffn_path(
        self, x_shared: torch.Tensor, attn_sum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """attn→ffn 协同: ffn 路由 logits 受 attn 输出影响 (V_coop)"""
        x_det = x_shared.detach()
        # 协同上下文: attn 输出经 V_coop 投影后加入路由 logits 输入
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

    def forward(self, x_shared: torch.Tensor) -> torch.Tensor:
        """前向 + (可选) EMA 中枢更新"""
        attn_sum, _, attn_outs = self._attn_path(x_shared)
        ffn_sum, _, ffn_outs = self._ffn_path(x_shared, attn_sum)
        if self._ema_enabled and self.training:
            self.cw_attn.ema_update(attn_outs)
            self.cw_ffn.ema_update(ffn_outs)
        return x_shared + attn_sum + ffn_sum

    def forward_with_routing(self, x_shared: torch.Tensor):
        """返回 (y, alpha_hat_attn, alpha_hat_ffn)"""
        attn_sum, ahat_a, _ = self._attn_path(x_shared)
        ffn_sum, ahat_f, _ = self._ffn_path(x_shared, attn_sum)
        y = x_shared + attn_sum + ffn_sum
        return y, ahat_a, ahat_f

    def freeze_p_m(self) -> None:
        for p in self.P_m:
            p.requires_grad_(False)
