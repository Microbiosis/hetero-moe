"""68.0 — 四分支 zero-OOM SSM 恢复 + 架构图 + 单模态消融

autograd checkpointing + 状态 tile 实现 zero-OOM 版 SSM 恢复四分支；产架构图；加单模态消融

实现文件:
  - research/aligner/v62_dimension_projection_space/moa_operators_zerooom.py
  - research/aligner/v62_dimension_projection_space/moa_triton_zerooom.py
  - research/aligner/v62_dimension_projection_space/hetero_arch_v68.png
  - research/aligner/v62_dimension_projection_space/train_v2.py

zero-OOM 版四分支 held-out 仅 0.53，仍输给三分支 0.64 ⇒ 恢复四分支**未能反超**，v71 因此又退回三分支。这是本代最重要的一条结论：显存问题解决了，但收益没有。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "68.0"
