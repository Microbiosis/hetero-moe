"""v31.0 — Gate-Style 中枢 Gradient-Scale 变体。

方向 B: 在 tanh 前对 t_scalar 做梯度重缩放，避免饱和区梯度消失。
"""

from research.central_stability.v31_gate_gradient_scale.workspace import GradientScaledGateStyleWorkspace
from research.central_stability.v31_gate_gradient_scale.layer import make_v31_layer

__all__ = [
    "GradientScaledGateStyleWorkspace",
    "make_v31_layer",
]
