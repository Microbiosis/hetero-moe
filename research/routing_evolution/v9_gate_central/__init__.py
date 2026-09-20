"""v9.0 — gate-style 中枢 (实验).

原语 (BroadcastWorkspace, GateStyleWorkspace, make_workspace, freeze_broadcast_matrix,
     set_broadcast_init_scale) 已提升至 research._primitives.central_mechanism.
本包现在只持有实验性部件:
    - make_v9_layer           : 工厂入口 (v9 实例组装)
    - diagnose_c1_root_cause  : C-1 失败根因诊断工具

API (推荐 — 直接从 _primitives 拿原语):
    from research._primitives.central_mechanism import GateStyleWorkspace, make_workspace
    from research.routing_evolution.v9_gate_central import make_v9_layer, diagnose_c1_root_cause
"""
from research._primitives.central_mechanism import (
    BroadcastWorkspace,
    GateStyleWorkspace,
    make_workspace,
    freeze_broadcast_matrix,
    set_broadcast_init_scale,
)
from .layer import make_v9_layer, diagnose_c1_root_cause

__all__ = [
    "BroadcastWorkspace",
    "GateStyleWorkspace",
    "make_workspace",
    "freeze_broadcast_matrix",
    "set_broadcast_init_scale",
    "make_v9_layer",
    "diagnose_c1_root_cause",
]
__version__ = "9.0.0"
__lifted_to__ = "research._primitives.central_mechanism"
