"""v13.0 — 混合对齐器 + 蒸馏到单底座。

两个目标:
    1. 更复杂的对齐器: MixtureAligner (跨底座共享专家池, 跨专家注意力)
       比 v10 的 CrossArchAttnAligner 表达力更强
    2. 嵌合 + 蒸馏: v10/v13 多底座协同 → 学生单底座 + LoRA,
       推理时只需学生, 不需要其他底座 → 真正的独立模型

与 v10 的关系:
    v10: 跨底座共享 K/V 注意力 (线性降维 + 跨底座 attn)
    v13: 跨底座 + 跨专家注意力 (每底座贡献 N 个专家, 跨专家池 attn)

与蒸馏的关系:
    教师 (v10/v13 多底座协同, 跨架构) → 蒸馏 KL loss → 学生 (单底座 + LoRA)
    学生独立推理, 不需要其他底座
"""
from .aligner import MixtureAligner
from .student import StudentModel
from .distill import distill_loss

__all__ = [
    "MixtureAligner",
    "StudentModel",
    "distill_loss",
]
__version__ = "13.0.0"