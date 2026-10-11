"""v40_dispatch — 分派式多教师蒸馏.

对照 v17 (加权和) 提供 per-sample 单教师分派族, 检验
``MANIFEST.json:root_cause_identified_later`` 的归因"失败源于同时多目标
回归的实现形式, 而非多教师本身不可行".

外部对照: MOPD (arXiv:2606.30406) 用 per-prompt 分派取代加权求和.
前置条件差异见 ``dispatch.py`` 模块 docstring.
"""
from .dispatch import (
    consensus_assign,
    cyclic_assign,
    dispatch_loss,
    oracle_assign,
    pairwise_disagreement,
    teacher_mean,
    teacher_target_mse,
    weighted_loss,
)

__all__ = [
    "weighted_loss",
    "dispatch_loss",
    "oracle_assign",
    "consensus_assign",
    "cyclic_assign",
    "teacher_mean",
    "pairwise_disagreement",
    "teacher_target_mse",
]
__version__ = "40.0.0"
