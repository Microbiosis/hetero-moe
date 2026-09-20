"""v28.0 — adaptive-ema 架构变体 (3 个).

v22 adaptive-ema 数学形态:
    augment: z_out = z + U @ c
    ema decay: warmup 期 base_decay=0.9; 之后 sigmoid(α·||c||) 趋近 1.0

v27 证明: 这是架构性问题, sigmoid decay 在训练后期饱和, c 冻结成训练分布快照,
导致 held-out 退步.

v28 提出 3 个架构变体, 修改 decay 或 augment 的形态:
    Variant 5: AdaptiveEMAClamped    - decay = min(sigmoid(...), max_decay=0.95)
    Variant 6: AdaptiveEMABounded    - decay = base_decay + 0.05 * sigmoid(...) ∈ [0.9, 0.95]
    Variant 7: AdaptiveEMAGate       - 用 W_t 替换 U, z / temperature (gate 风格)

可追溯性:
    AdaptiveEMACentral         -> v22 AdaptiveEMACentral (held-out 退化基线)
    AdaptiveEMAClamped         -> 变体 5 (clamp decay 上限)
    AdaptiveEMABounded         -> 变体 6 (bounded decay, 无硬边界)
    AdaptiveEMAGate            -> 变体 7 (gate 风格, 借用有界温度)

注意: 3 个变体共享 c 的存在与训练, 只改 augment 或 ema_update 形态.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from research.central_stability.v22_central_theory import CentralTheory


class _AdaptiveEMAVariantBase(CentralTheory):
    """adaptive-ema 变体基类.

    共享 v22 AdaptiveEMACentral 的:
        - c / U / adaptive_alpha 参数
        - _ema_enabled / _step_count 控制
        - warmup 期间的 base_decay 行为

    变体只改 ema_update 的 decay 计算 或 augment_router_logits 的形态.
    """

    def __init__(self, d_shared: int, num_experts: int, ema_decay: float = 0.9,
                 adaptive_warmup_steps: int = 10):
        nn.Module.__init__(self)
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = "adaptive-ema-variant"
        self.ema_decay = ema_decay
        self.adaptive_warmup_steps = adaptive_warmup_steps
        self._step_count = 0
        self._ema_enabled = True
        self.c = nn.Parameter(torch.zeros(d_shared))
        self.U = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.adaptive_alpha = nn.Parameter(torch.tensor(0.0))


# ---------------------------------------------------------------------------
# Variant 5: AdaptiveEMAClamped (clamp max decay)
# ---------------------------------------------------------------------------


class AdaptiveEMAClamped(_AdaptiveEMAVariantBase):
    """变体 5: decay = min(sigmoid(α·||c||), max_decay=0.95).

    解决: sigmoid 饱和 → c 冻结成训练分布快照 → held-out 失配.
    clamp 防止 decay 趋近 1.0, 强制 c 持续更新.
    """

    def __init__(self, d_shared: int, num_experts: int, ema_decay: float = 0.9,
                 adaptive_warmup_steps: int = 10, max_decay: float = 0.95):
        super().__init__(d_shared, num_experts, ema_decay, adaptive_warmup_steps)
        self.max_decay = max_decay

    @torch.no_grad()
    def ema_update(self, expert_outputs, base_decay: float = 0.9):
        if not self._ema_enabled or not self.training:
            return
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        if self._step_count < self.adaptive_warmup_steps:
            decay = base_decay
        else:
            adaptive_decay = torch.sigmoid(
                self.adaptive_alpha * self.c.norm()
            ).item()
            # 关键修改: clamp 上限
            decay = min(adaptive_decay, self.max_decay)
        self.c.data.mul_(decay).add_(new_c, alpha=1.0 - decay)
        self._step_count += 1

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.U @ self.c


# ---------------------------------------------------------------------------
# Variant 6: AdaptiveEMABounded (bounded linear decay)
# ---------------------------------------------------------------------------


class AdaptiveEMABounded(_AdaptiveEMAVariantBase):
    """变体 6: decay = base_decay + 0.05 * sigmoid(...) ∈ [0.9, 0.95].

    解决: 同上, 但用平滑的 bounded 函数, 无硬 clamp 边界.
    decay 始终在 [base_decay=0.9, 0.95] 范围内.
    """

    def __init__(self, d_shared: int, num_experts: int, ema_decay: float = 0.9,
                 adaptive_warmup_steps: int = 10, max_increment: float = 0.05):
        super().__init__(d_shared, num_experts, ema_decay, adaptive_warmup_steps)
        self.max_increment = max_increment

    @torch.no_grad()
    def ema_update(self, expert_outputs, base_decay: float = 0.9):
        if not self._ema_enabled or not self.training:
            return
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        if self._step_count < self.adaptive_warmup_steps:
            decay = base_decay
        else:
            adaptive_decay = torch.sigmoid(
                self.adaptive_alpha * self.c.norm()
            ).item()
            # 关键修改: 平滑 bounded
            decay = base_decay + self.max_increment * adaptive_decay
        self.c.data.mul_(decay).add_(new_c, alpha=1.0 - decay)
        self._step_count += 1

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.U @ self.c


# ---------------------------------------------------------------------------
# Variant 7: AdaptiveEMAGate (gate 风格: W_t 替换 U, z / temperature)
# ---------------------------------------------------------------------------


class AdaptiveEMAGate(_AdaptiveEMAVariantBase):
    """变体 7: 用 W_t 替换 U, z / temperature (gate 风格).

    解决: sigmoid 饱和 + U 矩阵过大导致过拟合.
    借用 gate 的有界温度保护 (W_t 单向量而非 M×D_shared 矩阵), 但保留 adaptive decay.

    注意: 变体 7 在 forward 期间使用 adaptive EMA decay 更新 c, 但 augment 时用
    gate 风格的 W_t 投影 (而非 U 矩阵), 提供有界温度.
    """

    def __init__(self, d_shared: int, num_experts: int, ema_decay: float = 0.9,
                 adaptive_warmup_steps: int = 10, alpha: float = 0.1):
        super().__init__(d_shared, num_experts, ema_decay, adaptive_warmup_steps)
        # gate 风格的 W_t: 单向量, 初始化为 0 → temperature = 1.0 起始
        self.W_t = nn.Parameter(torch.zeros(d_shared))
        self.alpha = alpha
        # U 仍存在 (供诊断用), 但不影响 augment 输出

    @torch.no_grad()
    def ema_update(self, expert_outputs, base_decay: float = 0.9):
        """保留 v22 adaptive EMA decay 逻辑."""
        if not self._ema_enabled or not self.training:
            return
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        if self._step_count < self.adaptive_warmup_steps:
            decay = base_decay
        else:
            adaptive_decay = torch.sigmoid(
                self.adaptive_alpha * self.c.norm()
            ).item()
            decay = min(adaptive_decay, 0.95)  # 顺便 clamp, 防止 c 完全冻结
        self.c.data.mul_(decay).add_(new_c, alpha=1.0 - decay)
        self._step_count += 1

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        """gate 风格: z / temperature (而非 z + bias)."""
        t_scalar = self.W_t @ self.c
        temperature = 1.0 + self.alpha * torch.tanh(t_scalar)
        return z / temperature


# 3 个变体的 mode 字符串
ADAPTIVE_EMA_VARIANT_MODES = [
    "adaptive-ema-clamped",
    "adaptive-ema-bounded",
    "adaptive-ema-gate",
]

ADAPTIVE_EMA_VARIANT_CLASSES = {
    "adaptive-ema-clamped": AdaptiveEMAClamped,
    "adaptive-ema-bounded": AdaptiveEMABounded,
    "adaptive-ema-gate": AdaptiveEMAGate,
}


def make_adaptive_ema_variant(mode: str, d_shared: int, num_experts: int):
    """工厂: 根据 mode 字符串创建对应变体."""
    if mode not in ADAPTIVE_EMA_VARIANT_CLASSES:
        raise ValueError(f"unknown adaptive-ema variant mode {mode!r}")
    return ADAPTIVE_EMA_VARIANT_CLASSES[mode](d_shared, num_experts)
