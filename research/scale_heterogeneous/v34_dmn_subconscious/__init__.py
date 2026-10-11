"""v34.0 — DMN 潜意识创造性联想层 (Default-Mode-Network Subconscious Association)。

可追溯性:
    DMNSubconscious    → 跨专家×跨时刻 cross-attention 联想 + 慢变 C + 路由调制 (dmn)
    FusionLayer        → v33 融合层 + 双管注入 (先验铺垫 + 路由调制) (fusion)
    load_msa_big 等    → 复用 v33_scale_hetero 的浅层加载 (前置 v33)

核心命题: 显式路由(意识层)之外加 DMN 联想层, 产生涌现表征 r (≠Σa_m),
双管注入下一步意识层, 使融合超过 v33 纯线性路由 ("灵光一现" > "理性选择")。

前置: v33.0 (规模异构融合)。本包复用 v33_scale_hetero 的底座加载与编码。
"""
from .dmn import DMNSubconscious
from .fusion import FusionLayer

# 复用 v33 的底座加载 (v34 前置 v33)
from research._primitives.scale_hetero_loaders import (
    MSA_DIR, EMB_DIR,
    load_msa_big, load_small, load_tokenizer, encode,
)

__all__ = [
    "DMNSubconscious",
    "FusionLayer",
    "MSA_DIR",
    "EMB_DIR",
    "load_msa_big",
    "load_small",
    "load_tokenizer",
    "encode",
]
__version__ = "34.0.0"
