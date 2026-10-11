"""v33.0 — "1大+N小" 规模异构融合 (实验).

原语 (paths, encoders, first_existing, load_msa_big, load_small 等) 已提升至
   research._primitives.scale_hetero_loaders. 本包现在只持有实验性部件:
    - CrossModalFusionLayer : 矩形 P_m 投影 + 共享路由融合

API (推荐 — 直接从 _primitives):
    from research._primitives.scale_hetero_loaders import (
        MSA_DIR, EMB_DIR, RER_DIR, load_msa_big, load_small, load_tokenizer, encode,
    )
    from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer
"""
from research._primitives.scale_hetero_loaders import (
    MSA_DIR,
    EMB_DIR,
    RER_DIR,
    first_existing,
    load_msa_big,
    load_small,
    load_tokenizer,
    encode,
)
from .fusion import CrossModalFusionLayer

__all__ = [
    "MSA_DIR",
    "EMB_DIR",
    "RER_DIR",
    "first_existing",
    "load_msa_big",
    "load_small",
    "load_tokenizer",
    "encode",
    "CrossModalFusionLayer",
]
__version__ = "33.0.0"
__lifted_to__ = "research._primitives.scale_hetero_loaders"
