"""v28.0 — diagnose_variant 诊断工具.

诊断每个 v28 变体的关键属性:
    - 变体类型 (router-norm 或 adaptive-ema)
    - 关键参数 (LayerNorm affine / RMSNorm / W_t 等)
    - 数学形态描述
    - 期望的 held-out 行为
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .router_norm_variants import (
    RouterNormAffineFalse,
    RouterNormZeroU,
    RouterNormRMS,
    RouterNormPure,
    ROUTER_NORM_VARIANT_MODES,
)
from .adaptive_ema_variants import (
    AdaptiveEMAClamped,
    AdaptiveEMABounded,
    AdaptiveEMAGate,
    ADAPTIVE_EMA_VARIANT_MODES,
)


def diagnose_variant(central: nn.Module) -> dict:
    """诊断 v28 变体的关键属性.

    Args:
        central: ArchitecturalFixLayer.cb_attn 或 cb_ffn 实例

    Returns:
        dict with:
            - variant_type: "router-norm-variant" | "adaptive-ema-variant" | "other"
            - math_form: 人类可读的数学形态描述
            - key_params: 关键参数状态
            - held_out_expectation: 期望的 held-out 行为
    """
    out = {
        "variant_type": "other",
        "math_form": "unknown",
        "key_params": {},
        "held_out_expectation": "unknown",
    }

    if isinstance(central, RouterNormAffineFalse):
        out["variant_type"] = "router-norm-variant"
        out["math_form"] = "LayerNorm_{affine=False}(z + U @ c)"
        out["key_params"] = {
            "has_LN_affine": False,
            "has_U": True,
            "U_init_scale": central._init_scale,
        }
        out["held_out_expectation"] = (
            "移除 6 个 LN scale/bias 参数, 减少过拟合."
            "预期: held-out 比 router-norm 改善."
        )
    elif isinstance(central, RouterNormZeroU):
        out["variant_type"] = "router-norm-variant"
        out["math_form"] = "LayerNorm_{affine=True}(z + U @ c), U=zeros init"
        out["key_params"] = {
            "has_LN_affine": True,
            "has_U": True,
            "U_init": "zeros",
        }
        out["held_out_expectation"] = (
            "初始 U=0 让 bias 起点干净, U 通过训练学习生长."
            "预期: held-out 比 router-norm 改善."
        )
    elif isinstance(central, RouterNormRMS):
        out["variant_type"] = "router-norm-variant"
        out["math_form"] = "RMSNorm_{affine=True}(z + U @ c)"
        out["key_params"] = {
            "normalizer": "RMSNorm",
            "has_U": True,
            "U_init_scale": central._init_scale,
        }
        out["held_out_expectation"] = (
            "RMSNorm 不强制 mean=0, 保留 router 方向信息."
            "预期: held-out 比 router-norm 改善."
        )
    elif isinstance(central, RouterNormPure):
        out["variant_type"] = "router-norm-variant"
        out["math_form"] = "RMSNorm_{affine=False}(z), 无 U"
        out["key_params"] = {
            "normalizer": "RMSNorm",
            "has_U": False,
        }
        out["held_out_expectation"] = (
            "完全删除 U, 只用 RMSNorm 稳定 router 输出."
            "预期: 可能改善但激进改动风险大."
        )
    elif isinstance(central, AdaptiveEMAClamped):
        out["variant_type"] = "adaptive-ema-variant"
        out["math_form"] = "z + U @ c, decay=min(sigmoid(α·||c||), max_decay)"
        out["key_params"] = {
            "max_decay": central.max_decay,
            "decay_range": f"[0.9, {central.max_decay}]",
        }
        out["held_out_expectation"] = (
            "clamp 防止 sigmoid 饱和到 1.0, c 持续更新."
            "预期: held-out 比 adaptive-ema 改善."
        )
    elif isinstance(central, AdaptiveEMABounded):
        out["variant_type"] = "adaptive-ema-variant"
        out["math_form"] = "z + U @ c, decay=base+δ·sigmoid(α·||c||)"
        out["key_params"] = {
            "max_increment": central.max_increment,
            "decay_range": f"[{central.ema_decay}, {central.ema_decay + central.max_increment}]",
        }
        out["held_out_expectation"] = (
            "bounded decay 平滑无硬边界."
            "预期: held-out 比 adaptive-ema 改善 (与 clamped 类似)."
        )
    elif isinstance(central, AdaptiveEMAGate):
        out["variant_type"] = "adaptive-ema-variant"
        out["math_form"] = "z / temperature (gate 风格, 借用有界温度)"
        out["key_params"] = {
            "uses_W_t": True,
            "alpha": central.alpha,
            "decay_clamped": True,
        }
        out["held_out_expectation"] = (
            "借用 gate 的有界温度保护, 同时保留 adaptive EMA decay (但 clamp 上限)."
            "预期: 接近 gate 水平, 可能成为最强变体."
        )
    else:
        # v22 原始 mode
        mode = getattr(central, "mode", "unknown")
        if mode == "broadcast":
            out["variant_type"] = "broadcast"
            out["math_form"] = "z + U @ c (v22 默认)"
        elif mode == "gate":
            out["variant_type"] = "gate"
            out["math_form"] = "z / (1 + α·tanh(W_t · c)) (v22 委托 v9 GateStyleWorkspace)"
        elif mode == "router-norm":
            out["variant_type"] = "router-norm"
            out["math_form"] = "LayerNorm_{affine=True}(z + U @ c) (v22 退化基线)"
        elif mode == "adaptive-ema":
            out["variant_type"] = "adaptive-ema"
            out["math_form"] = "z + U @ c, decay=sigmoid(α·||c||) (v22 退化基线)"
        else:
            out["math_form"] = f"unknown mode {mode}"
        out["held_out_expectation"] = "v22 原始模式 (基线对照)"

    return out


def is_v28_variant_mode(mode: str) -> bool:
    """判断 mode 是否是 v28 变体."""
    return mode in ROUTER_NORM_VARIANT_MODES + ADAPTIVE_EMA_VARIANT_MODES
