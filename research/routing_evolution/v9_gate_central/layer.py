"""v9.0 — Gate-Style 中枢的轻量级层封装.

设计原则:
    v8 已经有完整的 CentralAugmentedFusionLayer (C-1 broadcast + C-2 + C-3).
    v9.0 不重写整个层, 而是:
      1. 暴露 BroadcastWorkspace / GateStyleWorkspace 作为中枢的标准实现
      2. 提供 make_v9_layer() 函数, 帮助从 v8 layer "升级" C-1 到 gate-style
      3. C-2 (V_coop) 和 C-3 (central_expert) 完全沿用 v8, 不重新实现

如果你想用完整的 v9 layer, 只需:
    from research.routing_evolution.v8_central import CentralAugmentedFusionLayer
    from research.routing_evolution.v9_gate_central import make_v9_layer
    layer = make_v9_layer(CentralAugmentedFusionLayer, d_shared=256, ...)
    # 此时 layer 的 C-1 中枢已自动替换为 GateStyleWorkspace.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from research._primitives.central_mechanism import BroadcastWorkspace


def make_v9_layer(
    v8_layer_factory: Callable[..., nn.Module],
    d_shared: int,
    num_experts_attn: int,
    num_experts_ffn: int,
    c1_alpha: float = 0.1,
    **kwargs: Any,
) -> nn.Module:
    """创建 v9.0 layer: C-1 用 GateStyleWorkspace 替代 broadcast.

    Args:
        v8_layer_factory: v8 的层构造工厂, 如 CentralAugmentedFusionLayer
                          或 HierarchicalCentralLayer
        d_shared:         共享维度
        num_experts_attn: attn 通路专家数 (M_attn)
        num_experts_ffn:  ffn 通路专家数 (M_ffn)
        c1_alpha:         gate-style alpha (默认 0.1)
        **kwargs:         透传给 v8_layer_factory 的其他参数

    Returns:
        v9 layer, 其中 cw_attn / cw_ffn 已被替换为 GateStyleWorkspace.
        C-2 (V_coop) / C-3 (central_expert) 保持 v8 原状.
    """
    from research.routing_evolution.v9_gate_central.workspace import GateStyleWorkspace

    layer = v8_layer_factory(d_shared=d_shared, **kwargs)
    if hasattr(layer, "cw_attn"):
        layer.cw_attn = GateStyleWorkspace(
            d_shared, num_experts_attn, alpha=c1_alpha,
        )
    if hasattr(layer, "cw_ffn"):
        layer.cw_ffn = GateStyleWorkspace(
            d_shared, num_experts_ffn, alpha=c1_alpha,
        )
    return layer


def diagnose_c1_root_cause(layer: nn.Module) -> dict:
    """诊断 v8 C-1 失败根因 (H1/H2 验证).

    Args:
        layer: 带 cw_attn 属性的 v8/v9 layer

    Returns:
        dict with:
            has_U_attr: bool
            U_trainable: bool
            U_init_scale: float
            ema_decay: float
            c_init_norm: float
            diagnosis: str  ("H1 (U 训练扰动)" / "H2 (EMA 过快)" / "OK")
    """
    cw = getattr(layer, "cw_attn", None)
    if cw is None:
        return {"has_U_attr": False, "diagnosis": "no cw_attn"}
    if not isinstance(cw, BroadcastWorkspace):
        return {
            "has_U_attr": False,
            "diagnosis": f"非 BroadcastWorkspace ({type(cw).__name__}), 无 U 矩阵, 无 H1 风险",
        }
    out = {
        "has_U_attr": True,
        "U_trainable": cw.U.requires_grad,
        "U_init_scale": float(cw.U.std().item()),
        "ema_decay": cw.ema_decay,
        "c_init_norm": float(cw.c.norm().item()),
    }
    if cw.U.requires_grad:
        out["diagnosis"] = "H1 (U 训练扰动破坏 v7 路由) — 建议 freeze_u=True (v9_gate_central.freeze_broadcast_matrix)"
    elif cw.ema_decay < 0.5:
        out["diagnosis"] = "H2 (EMA 过快, c 训练震荡) — 建议 ema_decay ≥ 0.8"
    else:
        out["diagnosis"] = "OK"
    return out
