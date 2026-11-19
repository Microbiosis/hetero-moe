"""73.0 — Dim III 子空间夹角 — 断言权重完整加载 + P_m 投影 + eigh 提速

审计脚本重写：断言审计的是训练好的模型而非随机空间；梯度路径过 P_m 投影；Gram eigh 替代全 SVD

实现文件:
  - research/aligner/v62_dimension_projection_space/slot_independence_audit.py

连续修掉 6 个真实 bug（见 falsified/fixed 列表）。其中「审计随机空间」是最严重的一个——它让前几代审计结论全部无效。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "73.0"
