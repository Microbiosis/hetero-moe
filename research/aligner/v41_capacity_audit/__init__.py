"""v41_capacity_audit — 对齐器阶梯的容量-泄漏双重审计.

针对 E2(容量混杂,自 v14 起挂账)与 v13 的 in-sample 评估两个未闭项:
``MixtureAligner(n_shared_experts=1)`` 的参数量恰好等于 ``CrossArchAttnAligner``,
构成构造出来的**精确等容量对**,可在容量严格相同的前提下比较两种机制。
"""
from .audit import (
    attn_param_budget,
    count_trainable,
    fixed_mixture,
    heldout_seed,
    ladder_table,
    mixture_capacity_ladder,
    mixture_experts_for_budget,
    mixture_params,
    param_report,
)

__all__ = [
    "count_trainable",
    "attn_param_budget",
    "mixture_params",
    "fixed_mixture",
    "mixture_experts_for_budget",
    "mixture_capacity_ladder",
    "param_report",
    "heldout_seed",
    "ladder_table",
]
__version__ = "41.0.0"
