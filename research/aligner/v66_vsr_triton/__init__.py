"""66.0 — VSR 真视觉任务 + MoA Triton 融合内核库

VSR 真视觉空间关系任务提取 + 四个 Triton 融合内核（附 numpy 位级验证器）

实现文件:
  - research/aligner/v62_dimension_projection_space/extract_vsr_v66.py
  - research/aligner/v62_dimension_projection_space/moa_triton_kernels.py

TritonTileSimulator 是 numpy 逐 program 模拟器，可在**无 GPU 的本地 CPU** 上复算内核位级正确性 —— 本代是 v63–v74 中唯一能本地复算硬指标的代。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "66.0"
