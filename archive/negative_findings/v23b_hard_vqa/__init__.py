"""v23b — Hard VQA (8 类陷阱) 负结果归档。

原始位置: research/multimodal/v23_vqa/hard_dataset.py (与 v23 一起)
归档原因: 8 类陷阱设计下所有模型 (含 multimodal) 仍 50% = 随机水平,
         toy 数据集根本不可行 (见 PROJECT_REPORT §4 负面发现表).

可追溯性:
    HardVQADataset         → 8 类陷阱样本生成
    generate_hard_vqa_sample → 单样本
    HARD_ANSWER_LABELS / HARD_QUESTION_TEMPLATES → 标签与模板
"""
from .hard_dataset import (
    HardVQADataset,
    generate_hard_vqa_sample,
    HARD_ANSWER_LABELS,
    HARD_QUESTION_TEMPLATES,
)

__all__ = [
    "HardVQADataset",
    "generate_hard_vqa_sample",
    "HARD_ANSWER_LABELS",
    "HARD_QUESTION_TEMPLATES",
]
__version__ = "23b"
__archival_status__ = "negative-finding"
__archival_reason__ = "8-class trap toy data: all models reach 50% = random; toy datasets fundamentally infeasible"

