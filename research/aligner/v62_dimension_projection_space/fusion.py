"""1024 维融合层：直接工作在维度投影空间上。

与 v33 CrossModalFusionLayer 的区别
────────────────────────────────────
  v33 : d_shared=256，专家 256→1024→256，专家**冻结**且 x.detach()
  v62 : d_space=1024（= Σ 子空间维数），专家 1024→4096→1024，专家**解冻**

⚠️ 解冻专家偏离项目 v33 原设定。v48 SPEC 记载：v33 全流程真正被训练的只有
   W_router + 每专家 (γ, β, α) ≈ 2.3K 参数，专家权重要求 requires_grad=False。
   本代的端到端训练结果（held-out 0.9076，v62.3）是在解冻前提下取得的，
   **不能**与 v33 的 +26.7% 直接比较。

保留的原设计：每专家 (γ,β) 仿射 → SwiGLU → α 加权 → 残差。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionOverProjSpace(nn.Module):
    """输入 [B,S,d_space]，输出 [B,S,d_space] + 路由权重。"""

    def __init__(self, d_space: int = 1024, d_ff: int | None = None,
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d_space = d_space
        self.num_experts = num_experts
        self.top_k = top_k
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

    @staticmethod
    def _swiglu(x: torch.Tensor, wg, wu, wd) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)

    def forward_with_routing(self, x: torch.Tensor):
        # 路由只看输入分布，不该被专家梯度污染
        logits = F.linear(x.detach(), self.W_router)
        a = F.softmax(logits, dim=-1)
        _, ti = torch.topk(a, self.top_k, dim=-1)
        ah = a * torch.zeros_like(a).scatter_(-1, ti, 1.0)

        out = torch.zeros_like(x)
        for m in range(self.num_experts):
            h = x * self.gammas[m] + self.betas[m]           # 不 detach：专家参与训练
            f = self._swiglu(h, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            out = out + ah[..., m:m + 1] * self.alphas[m] * f
        return x + out, ah

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
