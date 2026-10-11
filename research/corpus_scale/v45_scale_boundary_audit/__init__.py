"""v45_scale_boundary_audit — 语料规模边界的方向判定与样本外存活性.

审计 `paper.tex` 的理论主张(最优中枢模式随规模从 gate 切到 adaptive-EMA)与
它自己主表给出的数据(小语料 adaptive-ema 优、大语料 gate 优)之间的矛盾,
并检验该边界在 held-out 上是否仍在。
"""
from .boundary import (
    boundary_direction,
    boundary_survives_heldout,
    better_mode,
    scale_table,
)

__all__ = [
    "better_mode",
    "boundary_direction",
    "boundary_survives_heldout",
    "scale_table",
]
__version__ = "45.0.0"
