"""v36.0 — α2 跨模态验证 (实验的 thin re-export shim).

v36 的大模型容器加载 + TabLDM 入口 + k-NN 评估 已全部提升为共享原语
   (research._primitives.scale_hetero_loaders). 本包保留作为 thin re-export,
   保持历史 import 兼容.

API (推荐 — 直接从 _primitives):
    from research._primitives.scale_hetero_loaders import (
        load_big_container, big_forward, get_tabldm_repr, encode_baseline_text,
        knn_acc, concat_norm_then_knn, MSA_DIR, TABLDM_CKPT, first_existing,
    )
"""
from research._primitives.scale_hetero_loaders import (
    MSA_DIR,
    TABLDM_CKPT,
    first_existing,
    load_big_container,
    big_forward,
    get_tabldm_repr,
    encode_baseline_text,
    D_BIG,
    B_DEFAULT,
    S_DEFAULT,
    knn_acc,
    concat_norm_then_knn,
)

__all__ = [
    "MSA_DIR",
    "TABLDM_CKPT",
    "first_existing",
    "load_big_container",
    "big_forward",
    "get_tabldm_repr",
    "encode_baseline_text",
    "D_BIG",
    "B_DEFAULT",
    "S_DEFAULT",
    "knn_acc",
    "concat_norm_then_knn",
]
__version__ = "36.0.0"
__lifted_to__ = "research._primitives.scale_hetero_loaders"
__is_re_export_shim__ = True
