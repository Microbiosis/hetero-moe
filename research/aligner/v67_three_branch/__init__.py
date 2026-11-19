"""67.0 — MoA 去掉 SSM 分支 — 3 异构（Attn/Conv/SwiGLU）

从四分支降为三分支：SSM 的状态维度与显存/并行扫描构成三角冲突

实现文件:
  - research/aligner/v62_dimension_projection_space/moa_operators.py

本代是一次**能力削减决策**，依据是 OOM 硬约束 + 实测 held-out 对比（0.64 vs 0.53），不是审美偏好。四分支在 v68 以 zero-OOM 版短暂恢复，v71 又退回三分支。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "67.0"
