"""v17.0 — 多教师蒸馏到真实 BERT 学生。

设计:
    学生 = TinyBERT + LoRA (复用 v16)
    教师 = v10 attn / v13 mixture / v15 mom (任选组合)
    损失 = alpha · L_real + sum_k (beta_k · L_teacher_k)

    关键设计:
    - 每个教师提供不同视角的"概念方向"
    - 学生综合学多个教师的 hidden states
    - 与 v16 (单教师) 对比: 多教师是否更好?

相对 v16 的关系:
    v16: 单教师 (v13 mixture) 蒸馏 -> 学生 fuse 0.135
    v17: 多教师 (v10+v13+v15) 蒸馏 -> 期望更好
"""
from .distill import MultiTeacherDistiller

__all__ = ["MultiTeacherDistiller"]
__version__ = "17.0.0"