"""v24.0 — CLIP 预训练教师 + 真实多模态数据.

设计:
    教师 = CLIP (frozen, 预训练 image-text 对齐)
    学生 = v21 MultiModalStudent (BERT + ViT + 模态路由器 + LoRA)
    任务 = 蒸馏学生 hidden → CLIP image+text 融合 embedding

相对 v23b toy 8 类:
    v23b 失败: 24 训练样本, 合成图像, 从零学习 image-text 对齐
    v24 解决: CLIP 已预训练 image-text 对齐, 学生只需学"匹配 CLIP 空间"
"""
from .teacher import CLIPTeacher
from .corpus import MiniImageTextCorpus, generate_color_image, COLOR_RGB

__all__ = ["CLIPTeacher", "MiniImageTextCorpus", "generate_color_image", "COLOR_RGB"]
__version__ = "24.0.0"