"""research._primitives.attention — 共享原语: 跨架构注意力池

来源: research/routing_evolution/v7_attention/attn_pool.py
原 v7_attention 包中的核心原语被 4 个研究包复用 (v8/v10/v12/v25),
   提升为 _primitives 后, 这些研究包不再依赖 v7_attention.

API:
    manual_multi_head_attention — 手动实现的 MHA
    AttnPool                    — P_m 投影 + MHA 封装
"""
from .attn_pool import (
    manual_multi_head_attention,
    AttnPool,
)

__all__ = ["manual_multi_head_attention", "AttnPool"]
__research_line__ = "_primitives/attention"
__lifted_from__ = "research/routing_evolution/v7_attention"
__is_primitive__ = True

