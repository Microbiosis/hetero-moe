"""v43.0 — 中枢机制的结构-容量-泄漏审计算子.

审计对象
--------
``v9_gate_central`` 的核心主张: **加性广播(v8)摧毁路由, 温度调制(v9)修复它**,
即 "中枢应调节温度而非扰动 logits"。论文 `neurips_2025.tex` 整篇建立在此。

三类待审
--------
1. **容量混杂**: gate 的 +44.5% 会不会只是参数更多?
   本模块直接数两种中枢的参数量。**注意方向**: v8 broadcast 用 ``U[M,d]``,
   v9 gate 用 ``W_t[d]``,后者参数是前者的**一半** —— 混杂的方向与担心的相反。
2. **容量与机制是否可分离**: 若把 gate 展宽到 broadcast 的参数量
   (``W_t: d → [M,d]``, 每个专家一个温度), 保序性是否还在?
   本模块用随机检验回答。**若不保序, 则容量与机制在中枢这里不可分离**,
   与 v41 对齐器那里"可以构造等容量对"的情形完全不同。
3. **泄漏**: v9 的 ``run_v9_gate_layer.py:121-124`` 在**训练同源数据**上评估。

设计约束(L1 自包含): 只依赖 torch, 不 import 任何 ``research.*`` 子包。
中枢模块本身由 ``examples/run_v43_central_audit.py`` 从
``research/_primitives/central_mechanism`` 构造。
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

__all__ = [
    "broadcast_params",
    "gate_params",
    "capacity_ratio",
    "is_identity_at_init",
    "order_preserved",
    "order_break_rate",
    "widened_break_rate",
]


# ── 1. 容量 ───────────────────────────────────────────────────────────────

def broadcast_params(d_shared: int, num_experts: int) -> Dict[str, int]:
    """v8 ``BroadcastWorkspace`` 的参数量: ``c[d] + U[M,d]``。"""
    return {"c": d_shared, "U": num_experts * d_shared,
            "total": d_shared + num_experts * d_shared}


def gate_params(d_shared: int, num_experts: int) -> Dict[str, int]:
    """v9 ``GateStyleWorkspace`` 的参数量: ``c[d] + W_t[d]``。

    ``num_experts`` 不参与计算 —— 这正是要害: 标量温度与专家数无关,
    而 broadcast 的 ``U`` 是 ``[M,d]``, 随专家数线性增长。
    """
    return {"c": d_shared, "W_t": d_shared, "total": 2 * d_shared}


def capacity_ratio(d_shared: int, num_experts: int) -> float:
    """gate / broadcast 的参数比。小于 1 ⇒ gate 更省参数。"""
    return gate_params(d_shared, num_experts)["total"] / \
        broadcast_params(d_shared, num_experts)["total"]


# ── 2. 结构性质 ───────────────────────────────────────────────────────────

def is_identity_at_init(workspace, z: torch.Tensor) -> Tuple[bool, float]:
    """完美退化: ``W_t=0, c=0 ⇒ temperature=1 ⇒ 输出恒等于输入``。

    返回 ``(是否恒等, 最大绝对偏差)``。这是 v9 立论的第一块基石 ——
    禁用中枢时系统与无中枢基线严格等价。
    """
    with torch.no_grad():
        out = workspace.augment_router_logits(z.clone())
    dev = float((out - z).abs().max())
    return bool(torch.allclose(out, z, atol=1e-6)), dev


def order_preserved(z: torch.Tensor, temperature) -> bool:
    """温度调制后, M 个 logit 的排序是否与调制前一致。"""
    scaled = z / temperature
    return bool(torch.equal(torch.argsort(z), torch.argsort(scaled)))


def order_break_rate(n_trials: int = 20000, num_experts: int = 3,
                     d_shared: int = 256, alpha: float = 0.1,
                     generator: torch.Generator | None = None) -> Dict[str, float]:
    """随机检验保序性: 标量温度 vs 展宽后的逐专家温度。

    返回两种形态各自的**保序破坏率**。

    - 标量温度(gate-style 实际形态): 破坏率应为 0, 因为正标量缩放保序
    - 逐专家温度(展宽到 broadcast 容量后): 破坏率通常显著大于 0
    """
    g = generator or torch.Generator().manual_seed(0)
    break_scalar = break_wide = 0
    for _ in range(n_trials):
        z = torch.randn(num_experts, generator=g)
        c_vec = torch.randn(d_shared, generator=g)

        t_scalar = 1.0 + alpha * torch.tanh(torch.randn(d_shared, generator=g) @ c_vec)
        if not order_preserved(z, t_scalar):
            break_scalar += 1

        W_wide = torch.randn(num_experts, d_shared, generator=g)
        t_vec = 1.0 + alpha * torch.tanh(W_wide @ c_vec)
        if not order_preserved(z, t_vec):
            break_wide += 1

    return {"scalar_break_rate": break_scalar / n_trials,
            "wide_break_rate": break_wide / n_trials,
            "n_trials": n_trials}


def widened_break_rate(n_trials: int = 20000, **kw) -> float:
    """:func:`order_break_rate` 的便捷包装, 只取展宽形态的破坏率。"""
    return order_break_rate(n_trials=n_trials, **kw)["wide_break_rate"]
