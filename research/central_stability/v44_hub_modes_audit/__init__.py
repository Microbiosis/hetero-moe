"""v44_hub_modes_audit — 四种中枢模式的容量-泄漏审计.

审计 `paper.tex` 主表所依据的 v22 四模式比较(broadcast / gate / router-norm /
adaptive-ema)。gate 的中枢参数只有其余三种的一半, 而主表赢的是 adaptive-ema;
泄漏方面 v22 的 runner 与 v9 一样在训练同源数据上评估。
"""
from .audit import (
    MODES,
    capacity_table,
    gate_is_smallest,
    generalization_gap,
    mode_central_params,
    paired_gain,
)

__all__ = [
    "MODES",
    "mode_central_params",
    "capacity_table",
    "gate_is_smallest",
    "paired_gain",
    "generalization_gap",
]
__version__ = "44.0.0"
