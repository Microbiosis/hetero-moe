"""v19.0 — C-1 起源根因补全分析.

设计:
    v8.1 已有部分诊断, v9.0 修正 C 让 C-1 工作, 但根因分析不完整.
    v19 探索"是什么让 C-1 (单独中枢 token) 在 v8.0 失败":
        - H1: U 训练扰动破坏 v7 路由 (v8.1 已确认)
        - H2: EMA 太慢, c 没积累够信息 (v8.1 弱信号)
        - H3: 训练步数太少 (v19 新增: 用 1000 步重跑)
        - H4: 学生容量太小 (v19 新增: 用 BERT-base 替代 TinyBERT)
        - H5: 中枢广播位置不对 (v19 探索: 加到 attn 而不是 ffn)

可追溯性:
    DiagnosticRunner         -> 多假设对照实验
    CapacityComparison        -> 不同容量学生对照
    LongTrainingComparison    -> 100 步 vs 1000 步对照
"""
from .diagnostics import (
    DiagnosticRunner,
    CapacityComparison,
    LongTrainingComparison,
)

__all__ = ["DiagnosticRunner", "CapacityComparison", "LongTrainingComparison"]
__version__ = "19.0.0"