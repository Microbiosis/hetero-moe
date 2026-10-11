"""v12.0 — v9 + v10 组合 (语义对齐器 + gate-style 中枢)。

设计动机:
    v9.0 修正 C (gate-style 中枢) 让 C-1 单独工作, fuse MSE +44.5%
    v10.0 语义对齐器让 attn < mean < linear, 比 v9 线性 P_m 改善 +67%
    两者独立验证有效. v12 验证"叠加增益": 用 v10 的 attn aligner +
    v9 的 gate-style 中枢, 是否比单独使用更好?

与 v8/v9/v10 的关系:
    v8   中枢探索               (广播中枢机制)
    v9   修正单独中枢           (gate-style 中枢)
    v10  更好的跨架构对齐       (语义对齐器)
    v12  v9 + v10 组合           (gate 中枢 + attn 对齐器)

    v12 不引入新机制, 只组合 v9 和 v10 已验证有效的部件.
"""
from .fusion import ComposedFusionLayer

__all__ = ["ComposedFusionLayer"]
__version__ = "12.0.0"