"""v43_central_audit — 中枢机制的结构-容量-泄漏审计.

审计 v9 gate-style 中枢的 +44.5% 主张。三条: (1) 容量混杂 —— gate 的参数是
broadcast 的一半, 混杂方向与担心的相反; (2) 容量与机制**不可分离** ——
把 gate 展宽到 broadcast 的容量必须改用逐专家温度, 那会破坏保序性;
(3) 泄漏 —— v9 的 runner 在训练同源数据上评估。
"""
from .workspace_audit import (
    broadcast_params,
    capacity_ratio,
    gate_params,
    is_identity_at_init,
    order_break_rate,
    order_preserved,
    widened_break_rate,
)

__all__ = [
    "broadcast_params",
    "gate_params",
    "capacity_ratio",
    "is_identity_at_init",
    "order_preserved",
    "order_break_rate",
    "widened_break_rate",
]
__version__ = "43.0.0"
