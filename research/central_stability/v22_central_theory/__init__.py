"""v22.0 — Central Theory 实验的 thin re-export shim.

v22 的 CentralTheory + RouterNormCentral + AdaptiveEMACentral 已全部提升为共享原语
   (research._primitives.central_theory). 本包保留作为 thin re-export, 保持历史 import 兼容.

API (推荐 — 直接从 _primitives):
    from research._primitives.central_theory import (
        CentralTheory, RouterNormCentral, AdaptiveEMACentral,
        freeze_u, freeze_c, set_u_init_scale, diagnose_theory_layer, gate_style_analysis,
    )
"""
from research._primitives.central_theory import (
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
__version__ = "22.0.0"
__lifted_to__ = "research._primitives.central_theory"
__is_re_export_shim__ = True
