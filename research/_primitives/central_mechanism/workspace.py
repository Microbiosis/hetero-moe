"""v9.0 — 中枢实现统一封装 (路线 A 唯一实现).

设计:
    v8.0 的 CentralWorkspace (broadcast 模式) 已迁出 v8_central, 统一在此封装.
    v9_gate_central 现在是**所有中枢机制**的唯一模块, v8_central 不再导出
    CentralWorkspace / CentralAugmentedFusionLayer 之外的中枢相关符号.

三种中枢实现:

1. BroadcastWorkspace (v8.0 原版, 历史对照用):
       z += U @ c
   V8.1 根因分析确认会破坏 v7 路由 (H1 假设验证).
   仅保留作 baseline 对照, 不推荐用于新代码.

2. GateStyleWorkspace (v9.0 修正 C, 当前最优):
       z = z / (1 + alpha * tanh(W_t · c))
   初始: W_t=0, c=0 → temperature=1 → 完美退化 v7.
   验证: C-1 单独 +44.5%, C-3 +88.1%.

3. 修正 A / B: 函数式 patch, 用于 v8 broadcast 中枢的在线修复.

API:
    # 推荐用法
    cw = GateStyleWorkspace(d_shared=256, num_experts=3)
    z_aug = cw.augment_router_logits(z_router)        # [B, S, M]
    cw.ema_update(expert_outputs)

    # 修正 A: 冻结 BroadcastWorkspace 的 U
    cw_b = BroadcastWorkspace(d_shared=256, num_experts=3)
    freeze_broadcast_matrix(cw_b)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BroadcastWorkspace(nn.Module):
    """Broadcast 中枢 (v8.0 原版, 历史对照).

    公式: z += U @ c
    其中 c ∈ R^{D_shared} 是中枢 token, U ∈ R^{M × D_shared} 是广播矩阵.

    警告:
        V8.1 根因分析确认此模式会破坏 v7 路由分布 (H1 假设).
        仅用于对照实验. 新代码请用 GateStyleWorkspace.

    Args:
        d_shared:    共享维度
        num_experts: 专家数
        ema_decay:   EMA 衰减系数 (默认 0.9)
        init_scale:  U 初始化 scale (默认 0.01)
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        ema_decay: float = 0.9,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.ema_decay = ema_decay

        self.c = nn.Parameter(torch.zeros(d_shared))
        self.U = nn.Parameter(torch.randn(num_experts, d_shared) * init_scale)

    def augment_router_logits(self, z_router: torch.Tensor) -> torch.Tensor:
        """z_router: [B, S, M] → z_router + U @ c."""
        bias = self.U @ self.c  # [M]
        return z_router + bias

    @torch.no_grad()
    def ema_update(self, expert_outputs: List[torch.Tensor]) -> None:
        """EMA 更新中枢 c (与 v8 一致)."""
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        self.c.data.mul_(self.ema_decay).add_(new_c, alpha=1.0 - self.ema_decay)

    def snapshot(self) -> torch.Tensor:
        """推理时取中枢快照."""
        return self.c.detach().clone()


class GateStyleWorkspace(nn.Module):
    """Gate-Style 中枢 (v9.0 修正 C, 当前最优).

    公式: z = z / (1 + alpha * tanh(W_t · c))

    相对 BroadcastWorkspace:
        - 无 U 矩阵 (H1 风险天然消除)
        - 初始状态退化为 identity (temperature=1, 完美等价 v7)
        - 训练更稳定 (v22 给出数学解释: gradient monotonic)

    Args:
        d_shared:    共享维度
        num_experts: 专家数
        ema_decay:   EMA 衰减系数 (默认 0.9)
        alpha:       温度调制幅度 (默认 0.1, 越大中枢影响越强)
        freeze_c:    是否冻结 c (调试用, 默认 False)
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        ema_decay: float = 0.9,
        alpha: float = 0.1,
        freeze_c: bool = False,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.freeze_c = freeze_c

        self.c = nn.Parameter(torch.zeros(d_shared))
        self.W_t = nn.Parameter(torch.zeros(d_shared))

        if freeze_c:
            self.c.requires_grad_(False)

    def augment_router_logits(self, z_router: torch.Tensor) -> torch.Tensor:
        """z_router: [B, S, M] → 用温度调制后的 logits.

        Returns:
            z_router / (1 + alpha * tanh(W_t · c))

        初始状态: W_t=0, c=0 → temperature=1 → z_router 不变 (v7 行为).
        """
        t_scalar = self.W_t @ self.c
        temperature = 1.0 + self.alpha * torch.tanh(t_scalar)
        return z_router / temperature

    @torch.no_grad()
    def ema_update(self, expert_outputs: List[torch.Tensor]) -> None:
        """EMA 更新中枢 c (与 v8 一致)."""
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        self.c.data.mul_(self.ema_decay).add_(new_c, alpha=1.0 - self.ema_decay)

    def snapshot(self) -> torch.Tensor:
        """推理时取中枢快照."""
        return self.c.detach().clone()


# ---------------------------------------------------------------------------
# 修正 A / B — 函数式 patch (用于 BroadcastWorkspace 在线修复)
# ---------------------------------------------------------------------------


def freeze_broadcast_matrix(cw: nn.Module) -> None:
    """修正 A: 冻结 BroadcastWorkspace 的广播矩阵 U.

    一行代码: cw.U.requires_grad_(False).
    c 仍训练 + EMA, U 不再随梯度扰动.

    验证: BroadcastWorkspace C-1 单独从 -121% 翻正到 +21.7%.
    """
    if not hasattr(cw, "U"):
        raise TypeError(f"{type(cw).__name__} 没有 U 属性, 不能用修正 A. "
                        f"请用 GateStyleWorkspace (无需修正 A).")
    cw.U.requires_grad_(False)


def set_broadcast_init_scale(cw: nn.Module, scale: float = 0.001) -> None:
    """修正 B: 重新初始化 U 为更小 scale.

    只在 layer 创建时调用 (在 forward 之前). 重新初始化会丢失已学习的 U.

    验证: BroadcastWorkspace C-1 单独 -72% → -40%.
    """
    if not hasattr(cw, "U"):
        raise TypeError(f"{type(cw).__name__} 没有 U 属性, 不能用修正 B.")
    with torch.no_grad():
        cw.U.data = torch.randn_like(cw.U) * scale


# ---------------------------------------------------------------------------
# 工厂函数: 推荐入口
# ---------------------------------------------------------------------------


def make_workspace(
    mode: str = "gate",
    d_shared: int = 256,
    num_experts: int = 3,
    **kwargs,
) -> nn.Module:
    """统一工厂: 创建中枢 workspace.

    Args:
        mode: "gate" (默认, 推荐) | "broadcast" (历史对照)
        d_shared, num_experts: 中枢维度
        **kwargs: 透传给具体类的参数 (alpha / ema_decay / init_scale / freeze_c)

    Returns:
        nn.Module: GateStyleWorkspace 或 BroadcastWorkspace 实例
    """
    if mode == "gate":
        return GateStyleWorkspace(d_shared=d_shared, num_experts=num_experts, **kwargs)
    elif mode == "broadcast":
        return BroadcastWorkspace(d_shared=d_shared, num_experts=num_experts, **kwargs)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'gate' or 'broadcast'")
