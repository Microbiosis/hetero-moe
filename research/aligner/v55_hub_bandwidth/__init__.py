"""V55.0 — 中枢带宽: 融合算子的秩判据 (理论 + 只读实测, 无训练).

权威正文: SPEC.md; 机器可读声明: MANIFEST.json; 实测: results/metrics.json
复现: python research/aligner/v55_hub_bandwidth/measure.py
"""
from .measure import (  # noqa: F401
    REL_TOL,
    _load_fusion,
    jac,
    run_one,
    spectrum_report,
)

__all__ = ["REL_TOL", "_load_fusion", "jac", "run_one", "spectrum_report"]
