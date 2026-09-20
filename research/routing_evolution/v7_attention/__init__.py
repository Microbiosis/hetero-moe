"""v7.0 — 跨架构族 Attention + FFN 双路由 (实验).

原语 (AttnPool, manual_multi_head_attention) 已提升至 research._primitives.attention.
本包现在只持有实验性部件:
    - CrossArchAttnFFNFusionLayer : 双路由主块
    - V7Trainer / build_v7_param_groups / LR_* : per-expert 训练协议

API (推荐 — 直接从 _primitives 拿原语):
    from research._primitives.attention import AttnPool, manual_multi_head_attention
    from research.routing_evolution.v7_attention import CrossArchAttnFFNFusionLayer, V7Trainer
"""
from research._primitives.attention import (
    AttnPool,
    manual_multi_head_attention,
)
from .fusion import CrossArchAttnFFNFusionLayer
from .trainer import V7Trainer, build_v7_param_groups, LR_ROUTER, LR_ADAPTER, LR_ALPHA

__all__ = [
    "AttnPool",
    "manual_multi_head_attention",
    "CrossArchAttnFFNFusionLayer",
    "V7Trainer",
    "build_v7_param_groups",
    "LR_ROUTER",
    "LR_ADAPTER",
    "LR_ALPHA",
]
__version__ = "7.0.0"
__lifted_to__ = "research._primitives.attention"
