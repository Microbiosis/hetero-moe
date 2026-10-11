"""v47.0 — 「in-sample 优势能否活到样本外」的判定算子.

审计对象
--------
一批**只有 in-sample 读数**的版本包: v33/v34/v35/v36/v37/v38/v39 (规模异构)、
v19/v20 (中枢诊断)、v30/v31 (中枢稳定性)、v23/v24 (多模态)、v12/v23b (负面发现)、
v8 (路由演化)。这些包的 runner 在训练同源数据上评估增益, 且**没有任何 held-out
构造** (grep ``held_out|heldout`` 全部 0 命中)。

问题
----
一个主张 (「机制 X 带来 +a% 增益」) 只有在样本外仍同向才算成立。本模块给出:

- :func:`paired_gain` —— 配对逐样本的优势读数 (中位数 + 胜出数, 不用均值:
  本项目自 v40 起的纪律, 均值会被单种子主导)
- :func:`leakage_verdict` —— 按 in-sample / held-out 两口径的同向性判定
  ``clean`` / ``invalidated`` / ``inconclusive``
- :func:`retention` —— held-out 保留下来的优势比例 (幅度塌缩的可见读数)
- :func:`audit_table` —— 渲染成 markdown 表

设计约束 (L1 自包含): 只依赖标准库, 不 import torch 也不 import 任何
``research.*`` 子包。实验由 ``examples/run_v47_*_heldout.py`` 驱动。
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

__all__ = ["paired_gain", "leakage_verdict", "retention", "audit_table"]

_VERDICTS = ("clean", "invalidated", "inconclusive")


def paired_gain(treatment: Sequence[float], baseline: Sequence[float],
                lower_is_better: bool = True) -> Dict[str, float]:
    """treatment 相对 baseline 的配对优势。

    ``lower_is_better=True`` (误差类指标) 时, ``gain = (base - treat) / base * 100``;
    ``lower_is_better=False`` (准确率类指标) 时取 ``(treat - base) / base * 100``。
    两种情况都使「更好」为**正**。

    读数一律用**配对逐样本中位数**, 同时给出胜出数 —— 只看中位数会被逐样本
    分裂的情况骗过去 (一半样本大幅获益、一半大幅退化, 中位数仍可能为正)。
    """
    if len(treatment) != len(baseline):
        raise ValueError(f"配对长度不符: treatment {len(treatment)} vs baseline {len(baseline)}")
    if not treatment:
        raise ValueError("空序列没有中位数")
    per: List[float] = []
    for t, b in zip(treatment, baseline):
        if b == 0:
            raise ValueError(f"baseline 为 0, 相对优势无定义 (treatment={t})")
        per.append((b - t) / b * 100 if lower_is_better else (t - b) / b * 100)
    return {
        "n": len(per),
        "median_pct": statistics.median(per),
        "mean_pct": statistics.mean(per),
        "wins": sum(1 for x in per if x > 0),
        "treat_median": statistics.median(treatment),
        "base_median": statistics.median(baseline),
        "per_sample": [round(x, 1) for x in per],
    }


def leakage_verdict(in_sample: Dict[str, float], heldout: Dict[str, float]) -> str:
    """按两口径判定 in-sample 优势是否成立。

    - ``clean``         held-out 中位优势 > 0 **且** 逐样本多数胜出
    - ``invalidated``   held-out 中位优势 ≤ 0 (方向翻转 / 塌成零)
    - ``inconclusive``  方向不稳定 (中位为正但胜出没过半, 或反之)

    幅度塌缩 (中位仍为正但只剩零头) 由 :func:`retention` 单独报出, **不**在这里
    硬编码阈值 —— 塌缩到什么程度算失效, 取决于主张本身的强弱, 应由 SPEC 决定。
    """
    for g in (in_sample, heldout):
        if g["median_pct"] is None or g["wins"] is None:
            return "inconclusive"
    majority = heldout["wins"] > heldout["n"] / 2
    if heldout["median_pct"] > 0 and majority:
        return "clean"
    if heldout["median_pct"] <= 0 and heldout["wins"] < heldout["n"] / 2:
        return "invalidated"
    return "inconclusive"


def retention(in_sample: Dict[str, float], heldout: Dict[str, float]) -> float | None:
    """held-out 保留了多少 in-sample 的优势 (0~1)。

    in-sample 中位优势 ≤ 0 时无定义 (没有可保留的东西), 返回 ``None``。
    """
    a = in_sample["median_pct"]
    if a is None or a <= 0:
        return None
    return heldout["median_pct"] / a


def audit_table(records: Sequence[Dict]) -> str:
    """把审计记录渲染成 markdown 表。

    ``records`` 的每一项形如::

        {"name": "v34 vs v33", "lower_is_better": True,
         "in_sample": {"fuse": …, "base": …},   # 每个 seed 一行
         "heldout_noise": {…}, "heldout_stimulus": {…}}

    每个 scope 的每个读数都是长度 = seed 数的序列; 本函数按 seed 做配对。
    """
    lines = ["| 版本 | 口径 | 改进(%) | 胜出 | 判定 | 保留率 |",
             "|---|---|---:|---:|:---:|---:|"]
    for r in records:
        lb = r.get("lower_is_better", True)
        scopes = [s for s in ("in_sample", "heldout_noise", "heldout_stimulus") if s in r]
        verdicts = {}
        for s in scopes:
            g = paired_gain(r[s]["treat"], r[s]["base"], lb)
            verdicts[s] = g
        base_g = verdicts.get("in_sample")
        for s in scopes:
            g = verdicts[s]
            if s == "in_sample":
                v, keep = "—", "—"
            else:
                v = leakage_verdict(base_g, g)
                keep = retention(base_g, g)
                keep = "—" if keep is None else f"{keep * 100:.0f}%"
            lines.append(f"| {r['name']} | {s} | {g['median_pct']:+.1f} | "
                         f"{g['wins']}/{g['n']} | {v} | {keep} |")
    return "\n".join(lines)
