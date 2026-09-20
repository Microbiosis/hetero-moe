"""research._primitives.central_theory — 共享原语: 中枢理论 (4 mode 基类 + patches)

来源: research/central_stability/v22_central_theory/theory.py
原 v22_central_theory 的 CentralTheory + patches 被 3 个研究包复用 (v25/v27/v28),
   提升为 _primitives 后, 这些研究包不再依赖 v22_central_theory.

依赖:
    from research._primitives.central_mechanism import (
        BroadcastWorkspace, GateStyleWorkspace, freeze_broadcast_matrix,
    )

API:
    CentralTheory           — 4 mode 中枢基类 (broadcast/gate/router-norm/adaptive-ema)
    RouterNormCentral       — v22 独有: router-norm 模式
    AdaptiveEMACentral      — v22 独有: adaptive-ema 模式
    freeze_u / freeze_c     — 函数式 patch (从 v9 移植)
    set_u_init_scale        — 函数式 patch
    diagnose_theory_layer   — 根因诊断工具
    gate_style_analysis     — 静态数学分析
"""
from .theory import (
    CentralTheory,
    RouterNormCentral,
    AdaptiveEMACentral,
    freeze_u,
    freeze_c,
    set_u_init_scale,
    diagnose_theory_layer,
    gate_style_analysis,
)

__all__ = [
    "CentralTheory",
    "RouterNormCentral",
    "AdaptiveEMACentral",
    "freeze_u",
    "freeze_c",
    "set_u_init_scale",
    "diagnose_theory_layer",
    "gate_style_analysis",
]
__research_line__ = "_primitives/central_theory"
__lifted_from__ = "research/central_stability/v22_central_theory"
__is_primitive__ = True

