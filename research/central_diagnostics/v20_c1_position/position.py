"""v20.0 — CentralBroadcaster (可配置广播位置的中央 token).

设计:
    5 种广播模式:
        NONE:   不广播 (基线)
        FFN:    c 加到 ffn 路由 (v8.0 默认)
        ATTN:   c 加到 attn 路由 (H5 假设)
        BOTH:   同时加到 attn 和 ffn 路由
        SIGNAL: c 作为 attn 路由的信号 (z_attn = W_attn · (x + U@c))
                而不是偏置 (z_attn = W_attn · x + U@c)

实现:
    委托给 v9_gate_central.BroadcastWorkspace, 加上 broadcast_position 维度.
    中枢实现 (c + U + EMA) 在 v9_gate_central, v20 只负责位置调度.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from research._primitives.central_mechanism import BroadcastWorkspace


class BroadcastPosition(Enum):
    """中央 token c 广播到路由的位置."""
    NONE = "none"                  # 不广播
    FFN = "ffn"                    # 加到 ffn 路由 (v8.0 默认)
    ATTN = "attn"                  # 加到 attn 路由 (H5 假设)
    BOTH = "both"                  # 同时加到 attn 和 ffn
    SIGNAL = "signal"              # 作为 attn 路由信号 (调制 x)


class CentralBroadcaster(nn.Module):
    """可配置广播位置的中央 token 容器.

    委托给 v9_gate_central.BroadcastWorkspace (单一中枢实现), 加上 broadcast_position 维度.

    Args:
        d_shared:    共享维度
        num_experts: 专家数
        ema_decay:   EMA 衰减系数 (默认 0.9)
        broadcast_position: "none"/"ffn"/"attn"/"both"/"signal" (默认 "ffn")
        init_scale:  U 初始化 scale (默认 0.01)

    接口:
        cb = CentralBroadcaster(d_shared=256, num_experts=3, broadcast_position="attn")
        z_attn = cb.augment_router_logits(z_attn, position="attn")
        cb.ema_update(expert_outputs)
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        ema_decay: float = 0.9,
        broadcast_position: str = "ffn",
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.position = BroadcastPosition(broadcast_position)
        # 委托给 v9_gate_central.BroadcastWorkspace (单一中枢实现)
        self.cw = BroadcastWorkspace(
            d_shared=d_shared,
            num_experts=num_experts,
            ema_decay=ema_decay,
            init_scale=init_scale,
        )
        # 兼容旧字段名 (供 trainer 或诊断用)
        self.c = self.cw.c
        self.U = self.cw.U
        self.ema_decay = ema_decay
        self._ema_enabled = True

    def augment_router_logits(
        self, z_router: torch.Tensor, position: Optional[str] = None,
    ) -> torch.Tensor:
        """为指定路由位置增加中枢 broadcast.

        Args:
            z_router: [B, S, M] 路由 logits
            position: "attn" | "ffn" | None (用 self.position)

        Returns:
            z_router + U @ c  (broadcast, 通过 v9_gate_central.BroadcastWorkspace)
            或 z_router 不变 (position=none)
        """
        pos = position if position is not None else self.position.value
        if pos == "none":
            return z_router
        # signal 模式: 实际是 broadcast (SIGNAL 在外层 forward 处理)
        return self.cw.augment_router_logits(z_router)

    @torch.no_grad()
    def ema_update(self, expert_outputs):
        if not self._ema_enabled:
            return
        self.cw.ema_update(expert_outputs)

    def snapshot(self):
        return self.cw.snapshot()
