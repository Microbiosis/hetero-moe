"""v15.0 — MoM (Mixture of Mixture) 嵌套对齐器。

设计:
    两级嵌套专家池:
        底层: 每底座贡献 n_inner 个原始专家, 跨底座 attn -> 共享表示 1
        上层: 共享表示 1 通过 n_outer 个高层专家, 跨专家 attn -> 最终表示

    与 v13 MixtureAligner 的区别:
        v13: 单层 M × N 专家池, 跨底座 + 跨专家 attn
        v15: 两层嵌套, 第一层是底座级, 第二层是池级 (更抽象)

可追溯性:
    MoMAligner  ->  v15_nested.aligner

相对 v14 的关系:
    v14 验证 mixture < attn < linear (单层对齐器)
    v15 验证 mixture-of-mixture < mixture < attn (嵌套对齐器)
"""
from .aligner import MoMAligner

__all__ = ["MoMAligner"]
__version__ = "15.0.0"