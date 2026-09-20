"""v31.0 — Gradient-Scaled Gate-Style 中枢。

方向 B: 在 tanh 前对 t_scalar 做梯度重缩放，避免饱和区梯度消失。

公式:
    t_scalar = W_t @ c
    t_scalar_scaled = t_scalar / (1.0 + |t_scalar|)
    temperature = 1.0 + alpha * tanh(t_scalar_scaled)
    z_aug = z_router / temperature

重缩放性质:
    - 当 |t_scalar| 小时，scaled ≈ t_scalar（保持原始行为）
    - 当 |t_scalar| 大时，scaled → sign(t_scalar) * 1.0（饱和在 ±1）
    - 梯度 ∂scaled/∂t_scalar 始终有下界（不会为 0）
"""

import torch
import torch.nn as nn


class GradientScaledGateStyleWorkspace(nn.Module):
    """Gradient-Scaled Gate-Style 中枢 (v31.0)。

    Args:
        d_shared:    共享维度
        num_experts: 专家数 (仅用于接口兼容)
        alpha:       温度调制幅度 (默认 0.1)
        ema_decay:   EMA 衰减系数 (默认 0.9)
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        alpha: float = 0.1,
        ema_decay: float = 0.9,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.alpha = alpha
        self.ema_decay = ema_decay

        self.c = nn.Parameter(torch.zeros(d_shared))
        self.W_t = nn.Parameter(torch.zeros(d_shared))

    def augment_router_logits(self, z_router: torch.Tensor) -> torch.Tensor:
        """z_router: [B, S, M] → 梯度缩放后的温度调制 logits。

        Returns:
            z_router / (1 + alpha * tanh(t_scalar / (1 + |t_scalar|)))
        """
        t_scalar = self.W_t @ self.c
        # 梯度重缩放：|t| 小时保持原值，|t| 大时饱和在 ±1
        t_scalar_scaled = t_scalar / (1.0 + t_scalar.abs())
        temperature = 1.0 + self.alpha * torch.tanh(t_scalar_scaled)
        return z_router / temperature

    @torch.no_grad()
    def ema_update(self, expert_outputs) -> None:
        """EMA 更新中枢 c。"""
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        self.c.data.mul_(self.ema_decay).add_(new_c, alpha=1.0 - self.ema_decay)

    def snapshot(self) -> torch.Tensor:
        """推理时取中枢快照。"""
        return self.c.detach().clone()


__all__ = ["GradientScaledGateStyleWorkspace"]
