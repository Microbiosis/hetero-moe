"""v10.0 — 跨架构对齐器 — **转发壳**.

实现已于 2026-10-11 提升到 ``research._primitives.aligners``, 使
``corpus_scale`` 线 (v25 / v28) 可以复用它而不构成跨线引用。

本文件保留是为了让 ``research.aligner.v10_embedding`` 的既有 import 全部继续
可用 —— 包括 ``examples/`` 下 20 余处 runner。
"""
from research._primitives.aligners import (
    CrossArchAttnAligner,
    CrossArchMeanAligner,
    SemanticAligner,
)

__all__ = ["SemanticAligner", "CrossArchMeanAligner", "CrossArchAttnAligner"]
