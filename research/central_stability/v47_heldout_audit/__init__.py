"""v47_heldout_audit — 「in-sample 优势能否活到样本外」的批次审计.

16 个版本包此前只有 in-sample 读数 (runner 里 grep ``held_out|heldout`` 全部 0 命中),
其 ``leakage_audit.status`` 只能是 ``pending``。本包补齐三个口径:

- ``in_sample``        —— 与各包原口径一致, 用于对齐既有数字
- ``heldout_noise``    —— 同分布不同抽样 (seed+1000): 换噪声, 干净信号不变
- ``heldout_stimulus`` —— 同分布**不同刺激**: 换输入文本, 噪声亦换

本包只提供判定算子; 每个包的实验由 ``examples/run_v47_*_heldout.py`` 驱动,
并复用被审计 runner 的全部构件以保证逐行同口径。
"""
from .heldout import audit_table, leakage_verdict, paired_gain, retention

__all__ = ["paired_gain", "leakage_verdict", "retention", "audit_table"]
__version__ = "47.0.0"
