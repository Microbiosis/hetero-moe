"""research._primitives.central_mechanism — 共享原语: 中枢机制 (broadcast + gate-style)

来源: research/routing_evolution/v9_gate_central/workspace.py
原 v9_gate_central 中的核心中枢机制被 6+ 个研究包复用 (v8历史/v10/v20/v22/v30/v31),
   提升为 _primitives 后, 这些研究包不再依赖 v9_gate_central.

API:
    BroadcastWorkspace        — v8.0 原版中枢 (历史对照)
    GateStyleWorkspace        — v9.0 修正 C (当前最优, 推荐)
    freeze_broadcast_matrix   — 修正 A (函数式 patch)
    set_broadcast_init_scale  — 修正 B (函数式 patch)
    EMA_DECAY                 — 默认 EMA 衰减
"""
from .workspace import (
    BroadcastWorkspace,
    GateStyleWorkspace,
    freeze_broadcast_matrix,
    set_broadcast_init_scale,
    make_workspace,
)

__all__ = [
    "BroadcastWorkspace",
    "GateStyleWorkspace",
    "freeze_broadcast_matrix",
    "set_broadcast_init_scale",
    "make_workspace",
]
__research_line__ = "_primitives/central_mechanism"
__lifted_from__ = "research/routing_evolution/v9_gate_central"
__is_primitive__ = True

