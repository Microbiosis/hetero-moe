"""v46.0 — 通路分解主张的判定算子.

审计对象
--------
``microfusion.tex`` 的主张(第 77 / 151 行)::

    "a path decomposition shows the gain concentrates in the attention path:
     cross-architecture differences surface mainly in attention patterns"

对应 v7 的四个评估量: ``fuse`` / ``attn_only`` / ``ffn_only`` / ``avg``。
"增益集中在 attention 通路" 的可判定形式是 **attn_only 明显优于 ffn_only**。

问题
----
``examples/run_v7_cross_arch.py`` 的四个量**全部在训练同源数据上评估**
(第 100-122 行, 与训练共用 ``x_shared_list`` / ``targets``),
且该文件**没有任何 held-out 构造**(grep 计数 0)。

本模块提供:
- :func:`path_gap` —— 通路差(attn 相对 ffn 的优势)的稳健读数
- :func:`path_claim_verdict` —— 判定"增益集中"是否在给定口径下成立

设计约束(L1 自包含): 只依赖标准库, 不 import torch 也不 import 任何
``research.*`` 子包。实验由 ``examples/run_v46_attn_path_audit.py`` 驱动。
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

__all__ = ["path_gap", "path_claim_verdict", "decomposition_table"]


def path_gap(attn: Sequence[float], ffn: Sequence[float]) -> Dict[str, float]:
    """attn 通路相对 ffn 通路的优势。

    误差**越小越好**, 故 attn 更优 ⟺ ``attn < ffn``, 优势以正百分比表示。
    读数一律用**配对逐样本中位数**, 不用均值 —— 本项目自 v40 起的纪律:
    均值会被单种子主导。
    """
    if len(attn) != len(ffn):
        raise ValueError(f"长度不符: attn {len(attn)} vs ffn {len(ffn)}")
    per = [(f - a) / f * 100 for a, f in zip(attn, ffn)]   # f=ffn 误差作分母
    return {
        "n": len(per),
        "median_pct": statistics.median(per),
        "mean_pct": statistics.mean(per),
        "attn_wins": sum(1 for x in per if x > 0),
        "per_sample": [round(x, 1) for x in per],
        "attn_median": statistics.median(attn),
        "ffn_median": statistics.median(ffn),
    }


def path_claim_verdict(gap: Dict[str, float], fuse: Dict[str, float] | None = None) -> str:
    """判定"增益集中在 attention 通路"。

    - ``holds``    attn 配对中位优势 > 0 且逐样本多数胜出
    - ``reversed`` attn 配对中位优势 < 0 (即 ffn 通路反而更好)
    - ``inconclusive`` 优势方向不稳定 (胜出数过半但中位为负, 或反之)
    """
    if gap["median_pct"] > 0 and gap["attn_wins"] > gap["n"] / 2:
        return "holds"
    if gap["median_pct"] < 0 and gap["attn_wins"] < gap["n"] / 2:
        return "reversed"
    return "inconclusive"


def decomposition_table(seeds: Sequence[Dict[str, float]]) -> str:
    """把 ``[{"seed":…, "fuse":…, "attn_only":…, "ffn_only":…, "avg":…}, …]``
    渲染成 markdown 表, 并附 in-sample / held-out 两列的汇总。
    """
    keys = ["fuse", "attn_only", "ffn_only", "avg"]
    lines = ["| 口径 | " + " | ".join(keys) + " | 通路差(attn−ffn) |",
             "|---|" + "---:|" * (len(keys) + 1)]
    for scope in ("in_sample", "heldout"):
        rows = [r[scope] for r in seeds if scope in r]
        if not rows:
            continue
        med = {k: statistics.median([r[k] for r in rows]) for k in keys}
        g = path_gap([r["attn_only"] for r in rows], [r["ffn_only"] for r in rows])
        lines.append(f"| {scope} | " + " | ".join(f"{med[k]:.4f}" for k in keys) +
                     f" | {g['median_pct']:+.1f}% ({g['attn_wins']}/{g['n']}) |")
    return "\n".join(lines)
