"""v21.0 — 嵌合 + 多模态学生 (BERT-text + ViT-image 双底座 + 模态路由器).

设计:
    学生 = MultiModalStudent = TinyBERT (text) + ViT-tiny (image) + LoRA + 模态路由器
    教师 = v13 mixture (跨 3 模态协同, fuse 0.26)
    蒸馏 = 学生模态路由器 (text vs image) 对齐教师的多模态融合

相对 v16 (单模态 BERT):
    v16: 学生只看 BERT 模态输入
    v21: 学生同时接收文本 + 图像, 模态路由器学会"何时调用哪个底座"

相对 v18 (真实语料):
    v18: 验证真实泛化 (单模态)
    v21: 验证多模态 (text + image)

与 v6-v10 跨架构的关系:
    v6: 跨架构 (BERT/Llama/ViT) 协同
    v21: 学生端实现跨架构 (BERT/ViT) - 但用 LoRA 蒸馏到单学生
"""
from .student import MultiModalStudent, LoRAHiddenAdapter

__all__ = ["MultiModalStudent", "LoRAHiddenAdapter"]
__version__ = "21.0.0"