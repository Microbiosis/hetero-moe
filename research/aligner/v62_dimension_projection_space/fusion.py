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
