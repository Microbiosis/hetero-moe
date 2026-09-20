"""v7.0 — CrossArchAttnFFNFusionLayer 双路由主块。

相对 v6.0 CrossModalFusionLayer 的关键扩展:
    - 新增 attn 通路: 每 token 选 1 attn 专家 (M_attn 个专家池)
    - ffn 通路: 与 v6 一致 (M_ffn 个专家池)
    - 两条路由独立选择 (W_router_attn / W_router_ffn)
    - 输出: y = x_shared + attn_sum + ffn_sum

与 v6.0 共享:
    - P_m 矩形投影 (D_m → D_shared), 冻结
    - SwiGLU + INT4 FakeQuantSTE (FFN 路径)
    - SparseRouterSTE (K=1)
    - SwiGLU 中间 FFN 权重冻结
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.router import SparseRouterSTE

from research._primitives.attention import AttnPool


def _swiglu(x, w_gate, w_up, w_down):
    gate = F.linear(x, w_gate)
    up = F.linear(x, w_up)
    return F.linear(F.silu(gate) * up, w_down)


class CrossArchAttnFFNFusionLayer(nn.Module):
    """跨架构族 Attn + FFN 双路由融合层 (v7.0)。

    输入: x_shared: [B, S, D_shared] (已由外部完成 P_m 矩形投影)
    输出: y: [B, S, D_shared]
    """

    def __init__(
        self,
        d_shared: int,
        d_ff: int,
        attn_pools: List[AttnPool],     # 已构造好的 M_attn 个 attn 专家
        modal_dims: List[int],          # M_ffn 个底座维度 (与 attn_pools 数量可不同)
        top_k_attn: int = 1,
        top_k_ffn: int = 1,
        quant_bits: int = 4,
        quant_group_size: int = 128,
    ):
        super().__init__()
        assert len(attn_pools) >= 1
        assert len(modal_dims) >= 1
        # ---- attn 通路 ----
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
        self.d_shared = d_shared
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        # P_m: D_shared × D_m, 正交初始化, 冻结
        self.P_m = nn.ParameterList(
            [nn.Parameter(torch.empty(d_shared, Dm)) for Dm in modal_dims]
        )
        for p in self.P_m:
            nn.init.orthogonal_(p)
            p.requires_grad_(False)

        self.W_router_ffn = nn.Parameter(torch.randn(self.num_experts_ffn, d_shared) * 0.01)
        # 每专家 SwiGLU FFN (冻结, 沿用 v6 的随机初始化)
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
        # 对齐参数 γ/β (可训练)
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_shared)) for _ in range(self.num_experts_ffn)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_shared)) for _ in range(self.num_experts_ffn)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(self.num_experts_ffn)]
        )

    def _attn_path(self, x_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """attn 通路, 返回 (attn_sum, alpha_hat_attn)"""
        x_det = x_shared.detach()
        z_a = F.linear(x_det, self.W_router_attn)
        alpha_a = F.softmax(z_a, dim=-1)
        ahat_a = SparseRouterSTE.apply(alpha_a, self.top_k_attn)
        attn_sum = torch.zeros_like(x_shared)
        for m in range(self.num_experts_attn):
            a_out = self.attn_pools[m](x_shared)
            attn_sum = attn_sum + ahat_a[..., m:m+1] * self.alphas_attn[m] * a_out
        return attn_sum, ahat_a

    def _ffn_path(self, x_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ffn 通路, 返回 (ffn_sum, alpha_hat_ffn)"""
        x_det = x_shared.detach()
        z_f = F.linear(x_det, self.W_router_ffn)
        alpha_f = F.softmax(z_f, dim=-1)
        ahat_f = SparseRouterSTE.apply(alpha_f, self.top_k_ffn)
        ffn_sum = torch.zeros_like(x_shared)
        for m in range(self.num_experts_ffn):
            h_a = x_det * self.gammas[m] + self.betas[m]
            f_m = _swiglu(h_a, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_q = FakeQuantSTE.apply(f_m, self.quant_bits, self.quant_group_size)
            ffn_sum = ffn_sum + ahat_f[..., m:m+1] * self.alphas[m] * f_q
        return ffn_sum, ahat_f

    def forward(self, x_shared: torch.Tensor) -> torch.Tensor:
        """x_shared: [B, S, D_shared] -> y: [B, S, D_shared]"""
        attn_sum, _ = self._attn_path(x_shared)
        ffn_sum, _ = self._ffn_path(x_shared)
        return x_shared + attn_sum + ffn_sum

    def forward_with_routing(self, x_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (y, alpha_hat_attn, alpha_hat_ffn), 供分析与基准对照"""
        attn_sum, ahat_a = self._attn_path(x_shared)
        ffn_sum, ahat_f = self._ffn_path(x_shared)
        y = x_shared + attn_sum + ffn_sum
        return y, ahat_a, ahat_f

    def freeze_p_m(self) -> None:
        """冻结所有 P_m (v6/v7 一致)"""
        for p in self.P_m:
            p.requires_grad_(False)
