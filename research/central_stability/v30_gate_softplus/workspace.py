"""v30.0 — Softplus Gate-Style 中枢。

方向 A: 用 softplus 替代 tanh，避免饱和区梯度消失。

公式:
    t_scalar = W_t @ c
    temperature = 1.0 + alpha * softplus(t_scalar)
    z_aug = z_router / temperature

softplus 性质:
    - 恒正，保证 temperature > 1
    - 对于大 |t_scalar|，softplus ≈ |t_scalar|（线性增长，不会饱和）
    - 梯度 ∂softplus/∂x = sigmoid(x)，永远不会为 0

相对 GateStyleWorkspace (v9):
    - 无 tanh 饱和问题
    - temperature 可无限增长（由 softplus 保证）
    - 梯度链永不消失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftplusGateStyleWorkspace(nn.Module):
    """Softplus Gate-Style 中枢 (v30.0)。

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
        """z_router: [B, S, M] → softplus 温度调制后的 logits。

        Returns:
            z_router / (1 + alpha * softplus(W_t · c))
        """
        t_scalar = self.W_t @ self.c
        temperature = 1.0 + self.alpha * F.softplus(t_scalar)
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


__all__ = ["SoftplusGateStyleWorkspace"]
