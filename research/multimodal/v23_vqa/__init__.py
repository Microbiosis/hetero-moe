"""v23.0 — Mini-VQA 真实任务验证多模态嵌合.

设计:
    内置 16 张合成图片 + 16 个问题 + 16 个答案 (4 类).
    图片: 224x224 纯色或几何图形 (CPU 可生成, 无需下载 VQAv2)
    问题: 4 种模板 (yes/no/color/count)
    答案: 4 类 one-hot

学生 = v21 MultiModalStudent (BERT + ViT + 模态路由器 + LoRA)
教师 = v13 mixture (跨模态融合)
任务 = 答案分类 (CE loss 替代 v21 的 MSE)

注意: 8 类陷阱版 HardVQA 已归档至 archive/negative-findings/v23b_hard_vqa/
      (8 类 toy 数据下所有模型 = 随机水平, toy 数据集根本不可行).
"""
from .dataset import (
    MiniVQADataset,
    generate_synthetic_vqa_sample,
    QUESTION_TEMPLATES,
    ANSWER_LABELS,
)

__all__ = [
    "MiniVQADataset",
    "generate_synthetic_vqa_sample",
    "QUESTION_TEMPLATES",
    "ANSWER_LABELS",
]
__version__ = "23.0.0"