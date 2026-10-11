"""v33.0 — 规模异构融合层 (复用 v6.0 CrossModalFusionLayer)。

每模态独立 P_m: D_m -> D_shared (矩形投影), 共享 W_router 在 D_shared 空间
做 token 级路由。规模异构体现在 modal_dims 不等 (2560 vs 1024)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.router import SparseRouterSTE


class CrossModalFusionLayer(nn.Module):
    """V6.0 跨模态融合层 (v33.0 原样复用)。

    每模态独立 P_m: D_m -> D_shared (矩形投影).
    共享 W_router 在 D_shared 空间做 token 级路由.
    """

    def __init__(self, d_shared, d_ff, num_experts, top_k, modal_dims,
                 quant_bits=4, quant_group_size=128):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.top_k = top_k
        self.modal_dims = modal_dims

        self.P_m = nn.ParameterList([nn.Parameter(torch.empty(d_shared, D_m)) for D_m in modal_dims])
        for m in range(len(modal_dims)):
            nn.init.orthogonal_(self.P_m[m])

        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02) for _ in range(num_experts)])
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02) for _ in range(num_experts)])
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_shared, d_ff) * 0.02) for _ in range(num_experts)])
        for pl in (self.w_gates, self.w_ups, self.w_downs):
            for p in pl:
                p.requires_grad = False

        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_shared)) for _ in range(num_experts)])
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_shared)) for _ in range(num_experts)])
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])

        self.W_router = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)

    def _swiglu(self, x, w_gate, w_up, w_down):
        gate = F.linear(x, w_gate)
        up = F.linear(x, w_up)
        return F.linear(F.silu(gate) * up, w_down)

    def _compute_output(self, x_shared):
        x_det = x_shared.detach()
        z = F.linear(x_det, self.W_router)
        alpha = F.softmax(z, dim=-1)
        alpha_hat = SparseRouterSTE.apply(alpha, self.top_k)

        out_sum = torch.zeros_like(x_shared)
        for m in range(self.num_experts):
            h_aligned = x_det * self.gammas[m] + self.betas[m]
            f_m = self._swiglu(h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_m_q = FakeQuantSTE.apply(f_m, 4, 128)
            out_sum = out_sum + alpha_hat[..., m:m+1] * self.alphas[m] * f_m_q
        return x_shared + out_sum, alpha_hat

    def forward(self, x_shared):
        y, _ = self._compute_output(x_shared)
        return y

    def forward_with_routing(self, x_shared):
        return self._compute_output(x_shared)
