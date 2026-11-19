"""70.0 — VL 跨模态融合旁路 — CrossModalVisionLanguageFusionSubOp 接入

四槽把三模态预混合进 canonical，视觉文本交互被抹平；VL 旁路直接吃分离的文本流与视觉流

实现文件:
  - research/aligner/v62_dimension_projection_space/vl_fusion.py
  - research/aligner/v62_dimension_projection_space/train_v2.py

算子 203K 参数。修掉三个真实 bug：① vl_out 是 256 维而 canonical 是 1024 维，需过 Linear 投影再残差；② patch 维硬编码 128（实际 ViT 是 197）；③ ViT 197 patch 未池化到文本 128 token 就融合。另修 VSR 图像改从 HF 预打包 zip 拉取（绕开 Kaggle COCO 联网限制）。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "70.0"
