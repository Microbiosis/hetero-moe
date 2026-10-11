"""v45.0 — 语料规模边界的方向判定算子.

审计对象
--------
``paper.tex`` 的**理论主张**::

    "establish a corpus-scale boundary showing that the optimal hub mode
     switches from gate-style to adaptive-EMA as data scale grows"

主表(同一篇论文)给的数据却是:

===========  ==========  =============
规模          gate        adaptive-ema
===========  ==========  =============
tiny          0.7097      **0.7035**
small         0.7075      **0.6984**
medium        **0.6971**  0.7136
large         **0.7029**  0.7136
===========  ==========  =============

即 **摘要的方向与自己的主表相反**(摘要说小→gate、大→adaptive-ema;
表里是小→adaptive-ema、大→gate)。论文的贡献点、讨论、结论三处都与主表一致,
**只有摘要写反了**。

本模块提供把这件事**判定清楚**的算子: 给定"规模 × 模式"的
(in-sample, held-out) 表, 判定 (1) 边界的方向, (2) 边界在 held-out 上是否仍在。

设计约束(L1 自包含): 只依赖标准库, 不 import 任何 ``research.*`` 子包,
也不 import torch。实验由 ``examples/run_v45_scale_boundary_audit.py`` 驱动。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

__all__ = [
    "better_mode",
    "boundary_direction",
    "boundary_survives_heldout",
    "scale_table",
]

LOW, HIGH = "gate", "adaptive-ema"      # 主张涉及的两个模式


def better_mode(scores: Dict[str, float]) -> str:
    """给定 ``{模式: 误差}``, 返回更优者(误差小者)。分数必须不等。"""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    if len(items) >= 2 and items[0][1] == items[1][1]:
        raise ValueError(f"两种模式分数相同, 无法判定: {scores}")
    return items[0][0]


def boundary_direction(table: Sequence[Tuple[str, Dict[str, float]]]
                       ) -> Dict[str, object]:
    """判定规模边界的方向。

    ``table`` 为 ``[(规模标签, {模式: 误差}), ...]``, 按规模升序。

    返回:
        ``per_scale``   每档的最优模式
        ``low_winner``  最小规模档的最优模式
        ``high_winner`` 最大规模档的最优模式
        ``direction``   ``"low=high"`` (无边界) / ``"switch"`` (有边界) +
                        ``f"{low_winner} -> {high_winner}"``
        ``switch_at``   最优模式发生变化的第一档标签(无则 None)
    """
    if not table:
        raise ValueError("table 不能为空")
    per_scale = {label: better_mode(scores) for label, scores in table}
    low, high = per_scale[table[0][0]], per_scale[table[-1][0]]
    switch_at = None
    prev = per_scale[table[0][0]]
    for label, _ in table[1:]:
        if per_scale[label] != prev:
            switch_at = label
            break
        prev = per_scale[label]
    return {
        "per_scale": per_scale,
        "low_winner": low,
        "high_winner": high,
        "direction": f"{low} -> {high}",
        "has_boundary": low != high,
        "switch_at": switch_at,
    }


def boundary_survives_heldout(in_sample_table, heldout_table) -> Dict[str, object]:
    """比较"训练集口径"与"held-out 口径"下的边界方向是否一致。

    方向一致 ⇒ 边界结论不是 in-sample 产物;
    不一致 ⇒ 边界必须按 held-out 口径重述。
    """
    a = boundary_direction(in_sample_table)
    b = boundary_direction(heldout_table)
    return {
        "in_sample": a,
        "heldout": b,
        "direction_agrees": a["direction"] == b["direction"],
        "winners_agree": a["per_scale"] == b["per_scale"],
        "verdict": ("边界在 held-out 上**方向不变**"
                    if a["direction"] == b["direction"]
                    else "边界方向在 held-out 上**发生改变** —— 必须按样本外口径重述"),
    }


def scale_table(rows: Sequence[Tuple[str, str, float, float]]) -> str:
    """把 ``[(规模标签, 模式, in-sample, held-out)]`` 渲染成 markdown 表。"""
    lines = ["| 规模 | 模式 | in-sample | held-out | 落差 |", "|---|---|---:|---:|---:|"]
    for label, mode, ins, outs in rows:
        lines.append(f"| {label} | {mode} | {ins:.4f} | {outs:.4f} | "
                     f"{(outs - ins) / max(ins, 1e-9) * 100:+.1f}% |")
    return "\n".join(lines)
