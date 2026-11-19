"""72.0 — 审计脚本 projspace 定位修复 — 绝对路径 + 包目录 fallback

Kaggle notebook 的 cwd 是 /kaggle/working，原按相对路径找 projspace.py 会 import 失败

实现文件:
  - research/aligner/v62_dimension_projection_space/slot_independence_audit.py

纯路径修复代，无新机制。但它修掉的是**审计脚本跑不起来**——一个跑不起来的审计等于没有审计。

本代性质: 代码就位，机制语义可本地证伪；端到端数字待 Kaggle 批量验证。
"""
__version__ = "72.0"
