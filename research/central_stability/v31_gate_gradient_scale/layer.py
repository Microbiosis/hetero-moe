"""v31.0 — Gradient-Scaled Gate-Style 中枢的轻量级层封装。"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def make_v31_layer(
    v8_layer_factory: Callable[..., nn.Module],
    d_shared: int,
    num_experts_attn: int,
    num_experts_ffn: int,
    alpha: float = 0.1,
    **kwargs: Any,
) -> nn.Module:
    """创建 v31 layer: C-1 用 GradientScaledGateStyleWorkspace 替代 broadcast。

    Args:
        v8_layer_factory: v8 的层构造工厂
        d_shared:         共享维度
        num_experts_attn: attn 通路专家数
        num_experts_ffn:  ffn 通路专家数
        alpha:            温度调制幅度
        **kwargs:         透传给 v8_layer_factory 的其他参数
    """
    from research.central_stability.v31_gate_gradient_scale.workspace import GradientScaledGateStyleWorkspace

    layer = v8_layer_factory(d_shared=d_shared, **kwargs)
    if hasattr(layer, "cw_attn"):
        layer.cw_attn = GradientScaledGateStyleWorkspace(
            d_shared, num_experts_attn, alpha=alpha,
        )
    if hasattr(layer, "cw_ffn"):
        layer.cw_ffn = GradientScaledGateStyleWorkspace(
            d_shared, num_experts_ffn, alpha=alpha,
        )
    return layer


__all__ = ["make_v31_layer"]
