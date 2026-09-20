"""v30.0 — Gate-Style 中枢 Softplus 变体。

方向 A: 用 softplus 替代 tanh，避免饱和区梯度消失。
"""

from research.central_stability.v30_gate_softplus.workspace import SoftplusGateStyleWorkspace
from research.central_stability.v30_gate_softplus.layer import make_v30_layer

__all__ = [
    "SoftplusGateStyleWorkspace",
    "make_v30_layer",
]
