"""v34.0 — 融合层 (v33 + DMN 双管注入: 用当前联想 r)。

双管注入:
    先验铺垫:  x' = x + λ·r
    路由调制:  z' = z + μ·g(r)   g: D→M
λ/μ 由构造参数给出 (默认 0.1), 与快照一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_moe.core.quant import FakeQuantSTE
from hetero_moe.core.router import SparseRouterSTE


class FusionLayer(nn.Module):
    def __init__(self, d_shared, d_ff, num_experts, top_k, modal_dims,
                 lam=0.1, mu=0.1):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.top_k = top_k
        self.lam = lam
        self.mu = mu
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
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(d_shared)) for _ in range(num_experts)])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(d_shared)) for _ in range(num_experts)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])
        self.W_router = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)

    def _swiglu(self, x, w_gate, w_up, w_down):
        return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)

    def forward(self, x_shared, dmn=None, r_inj=None):
        """前向. dmn+r_inj 非 None 时双管注入(先验+路由调制)."""
        x_proj = x_shared
        if dmn is not None and r_inj is not None:
            x_proj = x_proj + self.lam * r_inj          # 先验铺垫
        x_det = x_proj.detach()
        z = F.linear(x_det, self.W_router)
        if dmn is not None and r_inj is not None:
            z = z + self.mu * dmn.route_bias_proj(r_inj)  # 路由调制
        alpha = F.softmax(z, dim=-1)
        alpha_hat = SparseRouterSTE.apply(alpha, self.top_k)
        out_sum = torch.zeros_like(x_proj)
        for m in range(self.num_experts):
            h_aligned = x_det * self.gammas[m] + self.betas[m]
            f_m = self._swiglu(h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_m_q = FakeQuantSTE.apply(f_m, 4, 128)
            out_sum = out_sum + alpha_hat[..., m:m + 1] * self.alphas[m] * f_m_q
        return x_proj + out_sum

    def forward_single(self, x_shared, expert_idx, dmn=None, r_inj=None):
        x_proj = x_shared
        if dmn is not None and r_inj is not None:
            x_proj = x_proj + self.lam * r_inj
        x_det = x_proj.detach()
        h_a = x_det * self.gammas[expert_idx] + self.betas[expert_idx]
        f = self._swiglu(h_a, self.w_gates[expert_idx], self.w_ups[expert_idx], self.w_downs[expert_idx])
        return x_proj + self.alphas[expert_idx] * f

    def forward_avg(self, x_shared, dmn=None, r_inj=None):
        x_proj = x_shared
        if dmn is not None and r_inj is not None:
            x_proj = x_proj + self.lam * r_inj
        x_det = x_proj.detach()
        total_f = torch.zeros_like(x_proj)
        for m in range(self.num_experts):
            h_a = x_det * self.gammas[m] + self.betas[m]
            f = self._swiglu(h_a, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            total_f = total_f + f
        return x_proj + total_f / self.num_experts
