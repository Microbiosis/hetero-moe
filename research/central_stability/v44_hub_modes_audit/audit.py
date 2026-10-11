"""v44.0 — 四种中枢模式的容量-泄漏审计算子.

审计对象
--------
``v22_central_theory`` 的四模式比较, 是 ``paper.tex`` 的主表
(``Adaptive-EMA achieves the lowest MSE on v22 (0.2972)``)。

四种模式与其中枢参数量
----------------------
==========  ==========================  ================  ==============
模式         机制                        中枢参数           总数(非 trainable)
==========  ==========================  ================  ==============
broadcast    ``z + U @ c``              ``c[d] + U[M,d]``  1,024
gate         ``z / (1+α tanh(W_t·c))``  ``c[d] + W_t[d]``  **512**
router-norm  ``LayerNorm(z + U @ c)``   ``c[d] + U[M,d]``  1,024
adaptive-ema broadcast + 衰减门          ``c[d] + U[M,d]``  1,024 (+ 衰减标量)
==========  ==========================  ================  ==============

**gate 的中枢参数是其余三种的一半**, 而主表里赢的却是 adaptive-ema(1,024)。
容量混杂的方向需要实测确认, 不能靠猜。

泄漏
----
``examples/run_v22_theory_compare.py:357-365`` 在**训练同源数据**上评估
(``h_list = h_by_modal``, 与训练同一批)。这是 v43 在 v9 上发现的问题在中枢线上的重复。

设计约束(L1 自包含): 只依赖 torch, 不 import 任何 ``research.*`` 子包。
模式对象由 ``examples/run_v44_hub_modes_audit.py`` 从
``research/_primitives/central_theory`` 构造。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

__all__ = [
    "MODES",
    "mode_central_params",
    "capacity_table",
    "gate_is_smallest",
    "paired_gain",
    "generalization_gap",
]


MODES = ["broadcast", "gate", "router-norm", "adaptive-ema"]


def mode_central_params(mode: str, d_shared: int = 256, num_experts: int = 3) -> Dict[str, int]:
    """某模式的中枢参数量。

    ``gate`` 用 ``W_t[d]``(标量温度, 与专家数无关), 其余三种用 ``U[M,d]``。
    ``adaptive-ema`` 额外有一个衰减标量 α, 记在 ``decay`` 字段。
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    c = d_shared
    if mode == "gate":
        return {"c": c, "W_t": c, "U": 0, "decay": 0, "total": 2 * c}
    u = num_experts * d_shared
    decay = 1 if mode == "adaptive-ema" else 0
    return {"c": c, "W_t": 0, "U": u, "decay": decay, "total": c + u + decay}


def capacity_table(d_shared: int = 256, num_experts: int = 3) -> List[Dict]:
    """四种模式的容量表, 按参数量升序。"""
    rows = []
    for m in MODES:
        p = mode_central_params(m, d_shared, num_experts)
        rows.append({"mode": m, **p,
                     "ratio_vs_gate": p["total"] / mode_central_params("gate", d_shared, num_experts)["total"]})
    return sorted(rows, key=lambda r: r["total"])


def gate_is_smallest(d_shared: int = 256, num_experts: int = 3) -> bool:
    """gate 是否是四模式里中枢参数最少的。"""
    g = mode_central_params("gate", d_shared, num_experts)["total"]
    return all(mode_central_params(m, d_shared, num_experts)["total"] >= g for m in MODES)


def paired_gain(baseline: Sequence[float], arm: Sequence[float]) -> Dict[str, float]:
    """同 seed 内 (基线 − 被审计臂) / 基线 的配对增益, 给出中位数与胜出数。

    全部读数一律用中位数与配对口径 —— v40/v43 两次教训: 均值会被单种子主导。
    """
    import statistics as st

    if len(baseline) != len(arm):
        raise ValueError(f"长度不符: 基线 {len(baseline)} vs 臂 {len(arm)}")
    gains = [(b - a) / b * 100 for b, a in zip(baseline, arm)]
    return {
        "n": len(gains),
        "median_gain_pct": st.median(gains) if gains else float("nan"),
        "mean_gain_pct": st.mean(gains) if gains else float("nan"),
        "seeds_won": sum(1 for g in gains if g > 0),
        "per_seed": [round(g, 1) for g in gains],
    }


def generalization_gap(in_sample: Sequence[float], heldout: Sequence[float]) -> Dict[str, float]:
    """同一个臂的 in-sample → held-out 落差。"""
    import statistics as st

    mi, mo = st.median(in_sample), st.median(heldout)
    return {"in_sample_median": mi, "heldout_median": mo,
            "absolute_gap": mo - mi, "relative_gap_pct": (mo - mi) / max(mi, 1e-9) * 100}
