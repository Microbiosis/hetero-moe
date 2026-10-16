"""§2 — 异构融合层 (前向传播与梯度流定义)。

实现 §2.1 (分布对齐/输入截断) → §2.2 (SwiGLU + INT4 伪量化) → §2.3 (稀疏路由)
→ §2.4 (残差融合)。融合层同时承载训练路径与推理路径（通过 routed_weights 注入），
避免维护两份权重副本。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import FakeQuantSTE
from .router import SparseRouterSTE


def swiglu_forward(x, w_gate, w_up, w_down):
    """§2.2 SwiGLU: F = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down。

    权重形状遵循 F.linear 的 [out, in] 约定:
        w_gate: [d_ff, d_model], w_up: [d_ff, d_model], w_down: [d_model, d_ff]
    """
    h_gate = F.linear(x, w_gate)
    h_up = F.linear(x, w_up)
    h_act = F.silu(h_gate) * h_up
    return F.linear(h_act, w_down)


class HeteroFusionLayer(nn.Module):
    """异构微观融合层 (§2 完整定义)。

    可训练参数 (§4.1 初始化):
        W_router : [M, D]   ~ N(0, 0.01²)          —— 路由权重
        gamma_m  : [D]       恒等初始化 (全 1)      —— 逐通道缩放
        beta_m   : [D]       零初始化 (全 0)         —— 逐通道偏移
        alpha_m  : [1]       ~ Uniform(0.01, 0.1)   —— 正值初始化门控标量
    冻结参数 (底座 FFN):
        w_gates[m] / w_ups[m] / w_downs[m]          —— requires_grad=False

    内存优化 (§1.4): 输入经 detach 截断深层梯度，底座 FFN 中间激活不进自动微分图，
    显存由 O(M·L·D_ff) 降至 O(L·D + Adapter_Params)。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        quant_bits: int = 4,
        quant_group_size: int = 128,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k 不能超过专家数"
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size

        # ---- 冻结底座 FFN 权重 (§1.3 异构性仅存在于 FFN) ----
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)]
        )
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)]
        )
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model, d_ff) * 0.02) for _ in range(num_experts)]
        )
        for plist in (self.w_gates, self.w_ups, self.w_downs):
            for p in plist:
                p.requires_grad = False

        # ---- 可训练参数 (§4.1 初始化规范) ----
        self.W_router = nn.Parameter(torch.randn(num_experts, d_model) * 0.01)
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_model)) for _ in range(num_experts)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(num_experts)]
        )
        # alpha_m 正值初始化，独立 Parameter 包装避免切片非叶节点 (§4.1)
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.empty(1).uniform_(0.01, 0.1)) for _ in range(num_experts)]
        )

    @property
    def trainable_parameters(self):
        """仅可训练参数 (供优化器分组, §4.2/§4.3)。"""
        return [self.W_router, *self.gammas, *self.betas, *self.alphas]

    def forward(
        self,
        x: torch.Tensor,
        phase1: bool = False,
        routed_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x:              [B, S, D] 主干残差流。
            phase1:         True 则强制均匀路由 (§4.2)，W_router 不收梯度。
            routed_weights: [B, S, M] 预计算路由权重 (推理路径, §5)。提供时
                            跳过路由计算，直接用于融合。

        Returns:
            y:          [B, S, D] 残差融合输出 (§2.4: y = x + Σ α̂_m·α_m·F_m^noisy)
            z:          [B, S, M] 路由 logits (供 L_z, §3.3)
            alpha_hat:  [B, S, M] 实际使用的路由权重 (供 L_balance, §3.2)
        """
        x_residual = x  # §2.4 残差连接

        # §2.3 Router 基于未被噪声污染的原始 x_t 计算路由
        z = F.linear(x, self.W_router)
        alpha = F.softmax(z, dim=-1)

        if routed_weights is not None:
            # 推理路径: 使用外部容量路由结果 (§5)
            alpha_hat = routed_weights
        elif phase1:
            # §4.2 强制均匀路由 + detach 双重保险
            alpha_hat = (torch.ones_like(alpha) / self.num_experts).detach()
        else:
            # §2.3 训练期 Top-K 稀疏路由 + STE
            alpha_hat = SparseRouterSTE.apply(alpha, self.top_k)

        # ---- 逐专家计算并融合 (§2.1→§2.2→§2.4) ----
        out_sum = torch.zeros_like(x)
        x_detached = x.detach()  # §2.1 / §1.4 截断深层梯度
        for m in range(self.num_experts):
            # §2.1 分布对齐: h_aligned = stop_grad(x) ⊙ γ_m + β_m
            h_aligned = x_detached * self.gammas[m] + self.betas[m]
            # §2.2 SwiGLU 精确值
            f_m_exact = swiglu_forward(
                h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m]
            )
            # §2.2 / §6.2 INT4 伪量化噪声注入 (残差连接自然传播至下一层 Router)
            f_m_noisy = FakeQuantSTE.apply(
                f_m_exact, self.quant_bits, self.quant_group_size
            )
            # §2.4 残差融合: α̂_m · α_m · F_m^noisy
            out_sum = out_sum + alpha_hat[..., m : m + 1] * self.alphas[m] * f_m_noisy

        y = x_residual + out_sum
        return y, z, alpha_hat
