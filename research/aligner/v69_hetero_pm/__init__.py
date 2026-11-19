"""69.0 — 异构 P_m — Qwen 降维(2048→1024) + BERT/ViT DimReconcileBlock 稠密升维门控

替代「统一 256 维冻结随机正交」：各模态按自身维度走不同路径

实现文件:
  - research/aligner/v62_dimension_projection_space/train_v2.py

本代修掉一个真实缺陷：调解/降维输出未接去相关正交（ac2c951）。异构 P_m 的动机是 v63 换底座后模态维度不再同构（Qwen3.5-2B 2048 / ModernBERT 768 / SigLIP2 1152）。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "69.0"
