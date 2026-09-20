"""v16.0 — 真实 BERT 学生 + LoRA 蒸馏。

v13 学生是简化 MLP, 不能体现真实部署的能力.
v16 用真实 TinyBERT 当学生底座, 加 LoRA on hidden_states, 蒸馏 v13 教师.

设计:
    学生 = AutoModel.from_pretrained(TinyBERT)   # 冻结原有权重
         + LoRALinear(hidden_size, hidden_size)    # 加在 hidden_states 上
         + 输出投影到 D_shared

    教师 = v13 MixtureAligner (跨架构协同, fuse 0.26)
    蒸馏 = 学生 hidden_states 对齐教师 hidden_states

与 v13 的关系:
    v13: 简化 MLP 学生 + LoRA, 单 BERT 蒸馏
    v16: 真实 BERT 学生 + LoRA, 同样的蒸馏目标
"""
from .student import BERTStudent, LoRAAdapter

__all__ = ["BERTStudent", "LoRAAdapter"]
__version__ = "16.0.0"