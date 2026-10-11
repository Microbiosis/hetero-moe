"""v12.0 — ComposedFusionLayer (v9 中枢 + v10 对齐器 组合).

设计:
    = v10 的 AlignedFusionLayer (CrossArchAttnAligner)
    + v9_gate_central.GateStyleWorkspace (gate-style 中枢, 修正 C)
    + v9 的 V_coop (attn→ffn 协同)

实际: 这是 v10 的标准配置, 我们给一个清晰的命名包装, 体现"v9 + v10 组合".
"""
from __future__ import annotations

from typing import List

from research._primitives.attention import AttnPool
from research.aligner.v10_embedding.aligner import CrossArchAttnAligner
from research.aligner.v10_embedding.fusion import AlignedFusionLayer


def ComposedFusionLayer(
    d_shared: int,
    d_ff: int,
    attn_pools: List[AttnPool],
    modal_dims: List[int],
    num_heads: int = 4,
    c1_alpha: float = 0.1,
    **kwargs,
) -> AlignedFusionLayer:
    """v9 + v10 组合: CrossArchAttnAligner + v9_gate_central.GateStyleWorkspace.

    默认参数:
        c1_alpha=0.1   (gate-style alpha, v9 修正 C 标准)
    """
    aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=d_shared, num_heads=num_heads)
    return AlignedFusionLayer(
        d_shared=d_shared, d_ff=d_ff,
        attn_pools=attn_pools, modal_dims=modal_dims,
        aligner=aligner,
        c1_alpha=c1_alpha,
        **kwargs,
    )