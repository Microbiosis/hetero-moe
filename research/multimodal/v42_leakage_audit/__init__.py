"""v42_leakage_audit — 多模态蒸馏的 β_h 泄漏审计.

v21 的学生损失是 ``0.7·MSE(y, teacher) + 0.3·MSE(y, target)``, 与 v16 的结构完全相同,
即 v32 已判定为泄漏源的那个系数 (β_h = 0.3)。本包把 v32 的连续过渡扫描移植到 v21,
判定 "学生比教师优 50.5%" 是线性插值还是阈值机制。
"""
from .beta_scan import (
    KINK_RATIO_THRESHOLD,
    MIN_USABLE_NOISE,
    beta_from_weight,
    blend_loss,
    linearity_report,
    linearity_verdict,
    weight_from_beta,
)

__all__ = [
    "blend_loss",
    "MIN_USABLE_NOISE",
    "KINK_RATIO_THRESHOLD",
    "beta_from_weight",
    "weight_from_beta",
    "linearity_report",
    "linearity_verdict",
]
__version__ = "42.0.0"
