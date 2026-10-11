"""v46_attn_path_audit — 「增益集中在 attention 通路」主张的样本外审计.

microfusion.tex 第 77/151 行主张跨架构互补增益集中在 attention 通路。
v7 的四个通路分解量全部在训练同源数据上评估, 且该 runner 没有任何 held-out 构造。
本包补上 held-out 口径并判定该主张是否成立。
"""
from .path_audit import decomposition_table, path_claim_verdict, path_gap

__all__ = ["path_gap", "path_claim_verdict", "decomposition_table"]
__version__ = "46.0.0"
