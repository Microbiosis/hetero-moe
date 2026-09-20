"""v28.0 — 架构层面修正 router-norm / adaptive-ema (第四个跨版本集成包).

设计:
    v27 证明 router-norm / adaptive-ema 的 held-out 退步是架构性问题,
    v28 修改其数学形态 (LayerNorm / RMSNorm / decay / W_t) 试图修正.

7 个架构变体:
    Router-norm 4 个:
        - RouterNormAffineFalse (变体 1): LayerNorm affine=False
        - RouterNormZeroU       (变体 2): 初始化 U=0
        - RouterNormRMS         (变体 3): 用 RMSNorm 替换 LayerNorm
        - RouterNormPure        (变体 4): 完全删除 U, 只做 RMSNorm

    Adaptive-EMA 3 个:
        - AdaptiveEMAClamped    (变体 5): clamp decay ≤ 0.95
        - AdaptiveEMABounded    (变体 6): bounded decay ∈ [0.9, 0.95]
        - AdaptiveEMAGate       (变体 7): gate 风格 (W_t + temperature)

可追溯性:
    ArchitecturalFixLayer    -> 继承 v25 RealCorpusTheoryLayer, 支持 11 mode
    diagnose_variant         -> 对每个变体给出关键属性和预期 held-out 行为
    ROUTER_NORM_VARIANT_MODES / ADAPTIVE_EMA_VARIANT_MODES
                              -> 7 个变体的 mode 字符串
    ALL_MODES                -> v22 4 + v28 7 = 11 个 mode 汇总

验证环境: 复用 v26 fully-real-corpus (text/code/image 全部真实)
端到端规模: 11 modes × 5 seeds = 55 run
"""
from .router_norm_variants import (
    RouterNormAffineFalse,
    RouterNormZeroU,
    RouterNormRMS,
    RouterNormPure,
    ROUTER_NORM_VARIANT_MODES,
    ROUTER_NORM_VARIANT_CLASSES,
    make_router_norm_variant,
)
from .adaptive_ema_variants import (
    AdaptiveEMAClamped,
    AdaptiveEMABounded,
    AdaptiveEMAGate,
    ADAPTIVE_EMA_VARIANT_MODES,
    ADAPTIVE_EMA_VARIANT_CLASSES,
    make_adaptive_ema_variant,
)

# 合并所有变体 mode
ALL_VARIANT_MODES = ROUTER_NORM_VARIANT_MODES + ADAPTIVE_EMA_VARIANT_MODES
from .layer import (
    ArchitecturalFixLayer,
    group_by_modal,
    V22_MODES,
    V28_VARIANT_MODES,
    ALL_MODES,
)
from .diagnostics import diagnose_variant, is_v28_variant_mode

__all__ = [
    # router-norm 变体
    "RouterNormAffineFalse",
    "RouterNormZeroU",
    "RouterNormRMS",
    "RouterNormPure",
    "ROUTER_NORM_VARIANT_MODES",
    "ROUTER_NORM_VARIANT_CLASSES",
    "make_router_norm_variant",
    # adaptive-ema 变体
    "AdaptiveEMAClamped",
    "AdaptiveEMABounded",
    "AdaptiveEMAGate",
    "ADAPTIVE_EMA_VARIANT_MODES",
    "ADAPTIVE_EMA_VARIANT_CLASSES",
    "make_adaptive_ema_variant",
    "ALL_VARIANT_MODES",
    # layer & diagnostics
    "ArchitecturalFixLayer",
    "group_by_modal",
    "V22_MODES",
    "V28_VARIANT_MODES",
    "ALL_MODES",
    "diagnose_variant",
    "is_v28_variant_mode",
]
__version__ = "28.0.0"
