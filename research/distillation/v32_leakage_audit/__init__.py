"""v32_leakage_audit — 目标泄漏审计代 (自包含子包).

v32 不引入新机制, 它审计 v16 蒸馏结论: 把损失里的 0.3 直接目标项 (目标泄漏)
移走后, 学生是否还能超越教师。答案是否定的 —— 表观收益全部归因于泄漏项.

本包自带**原始实验记录**, 因此结论可被独立重算, 而非只能相信 MANIFEST 的声明:

    >>> from research.distillation.v32_leakage_audit import leakage_verdict
    >>> leakage_verdict()["pure_beats_teacher"]
    False

API:
    load_records, paired_gap, variance_isolation_table, published_headline,
    beta_scan_curve, beta_scan_is_monotonic, super_deterministic_table,
    transfer_accuracy, leakage_verdict
"""
from research.distillation.v32_leakage_audit.audit import (
    PUBLISHED_CONDITION,
    RESULTS_DIR,
    beta_scan_curve,
    beta_scan_is_monotonic,
    leakage_verdict,
    load_records,
    paired_gap,
    published_headline,
    super_deterministic_table,
    transfer_accuracy,
    variance_isolation_table,
)

__all__ = [
    "RESULTS_DIR",
    "PUBLISHED_CONDITION",
    "load_records",
    "paired_gap",
    "variance_isolation_table",
    "published_headline",
    "beta_scan_curve",
    "beta_scan_is_monotonic",
    "super_deterministic_table",
    "transfer_accuracy",
    "leakage_verdict",
]
__version__ = "32.0.0"
