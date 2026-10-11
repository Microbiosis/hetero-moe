"""v37.0 — β 自训练预测编码入口 (实验的 thin re-export shim).

v37 的 EntryEncoder + train_arm 已全部提升为共享原语
   (research._primitives.scale_hetero_loaders). 本包保留作为 thin re-export,
   保持历史 import 兼容.

API (推荐 — 直接从 _primitives):
    from research._primitives.scale_hetero_loaders import (
        EntryEncoder, gen_table_classification,
        train_arm, knn_acc,
        MSA_DIR, TABLDM_CKPT, first_existing,
    )
"""
from research._primitives.scale_hetero_loaders import (
    MSA_DIR,
    TABLDM_CKPT,
    first_existing,
    EntryEncoder,
    gen_table_classification,
    train_arm,
    knn_acc,
)

__all__ = [
    "MSA_DIR",
    "TABLDM_CKPT",
    "first_existing",
    "EntryEncoder",
    "gen_table_classification",
    "train_arm",
    "knn_acc",
]
__version__ = "37.0.0"
__lifted_to__ = "research._primitives.scale_hetero_loaders"
__is_re_export_shim__ = True
