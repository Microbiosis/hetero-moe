"""v41.0 — 对齐器阶梯的容量-泄漏双重审计的算子集.

针对两件事:

1. **E2(容量混杂, 自 v14 起挂账未闭)**:
   docs 记录的对齐器阶梯是 linear 1.0436 → attn 0.3448 → mixture 0.2575 → mom 0.2645。
   但参数量是 linear 325,632 → attn 720,640 → mixture 2,291,968 → mom 1,704,448,
   **性能序与容量序完全一致**, 阶梯因此与容量混淆, 一直无法判定"设计"还是"规模"。

   本模块提供**容量对齐对照**: ``MixtureAligner(n_shared_experts=1)`` 的参数量
   恰好等于 ``CrossArchAttnAligner`` 的 720,640 —— 这是构造出来的**精确等容量对**,
   因此可以在容量严格相同的前提下比较两种机制。

2. **泄漏(in-sample 评估)**: v13 的 ``eval_teacher`` 在**训练同源数据**上评估。
   本模块提供 held-out 划分的构造, 用于分离"机制增益"与"记住训练样本"。

设计约束(L1 自包含): 本模块只依赖 torch, 不 import 任何 ``research.*`` 子包
(``research/aligner`` 在 ``check_self_containment.py`` 的 ``ALLOWED_INTRA_LINE``
中没有条目)。对齐器本身由 ``examples/run_v41_capacity_audit.py`` 从各包构造。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

__all__ = [
    "count_trainable",
    "param_report",
    "mixture_params",
    "mixture_capacity_ladder",
    "attn_param_budget",
    "mixture_experts_for_budget",
    "heldout_seed",
    "fixed_mixture",
    "ladder_table",
]


def count_trainable(module) -> int:
    """可训练参数量 (只数 requires_grad 的)。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def attn_param_budget(modal_dims: List[int], d_shared: int, num_heads: int = 4) -> int:
    """CrossArchAttnAligner 的参数量解析式 (与实例化结果逐位一致, 可用于免实例化搜索)。

    组成:
        down_projections : Σ_m (D_m·d + d)
        W_k, W_v         : 2·d²            (bias=False)
        q_projections    : M·(d² + d)
        out_projection   : d² + d
    """
    M = len(modal_dims)
    d = d_shared
    down = sum(D * d + d for D in modal_dims)
    wk = wv = d * d
    q = M * (d * d + d)
    out = d * d + d
    return down + wk + wv + q + out


def mixture_experts_for_budget(
    modal_dims: List[int], d_shared: int, budget: int, num_heads: int = 4,
    max_experts: int = 64,
) -> Tuple[int, int]:
    """找 ``n_shared_experts`` 使 MixtureAligner 的参数量最接近 ``budget``。

    返回 ``(n_shared_experts, 实际参数量)``。纯算术, 不实例化 —— 用于快速定位
    容量对齐点。

    组成 (每个专家 = Linear(D_m,d)+Linear(d,d), 全 M 个底座各一份):
        experts   : n · Σ_m (D_m·d + d + d² + d)
        W_k,W_v   : 2·d²
        out_proj  : d² + d
    """
    d = d_shared
    per_expert = sum(D * d + d + d * d + d for D in modal_dims)
    fixed = 2 * d * d + (d * d + d)
    best = (1, float("inf"))
    for n in range(1, max_experts + 1):
        total = n * per_expert + fixed
        if abs(total - budget) < abs(best[1] - budget):
            best = (n, total)
        if total > budget * 1.5:
            break
    return best[0], best[1]


def mixture_params(modal_dims: List[int], d_shared: int, n_experts: int) -> int:
    """MixtureAligner 在给定 n_shared_experts 下的参数量 (解析式)。"""
    d = d_shared
    per_expert = sum(D * d + d + d * d + d for D in modal_dims)
    return n_experts * per_expert + fixed_mixture(d)


def param_report(modal_dims: List[int], d_shared: int, num_heads: int = 4) -> Dict[str, int]:
    """四种对齐器在给定配置下的可训练参数量。"""
    d = d_shared
    lin = sum(D * d for D in modal_dims)          # v6/v7 的矩形 P_m (无 bias)
    return {
        "linear_P_m": lin,
        "attn": attn_param_budget(modal_dims, d_shared, num_heads),
        "mixture_n1": mixture_params(modal_dims, d_shared, 1),
        "mixture_n4": mixture_params(modal_dims, d_shared, 4),
        "mom": mixture_params(modal_dims, d_shared, 2),   # MoM 内层为 2×2, 规模同阶
    }


def fixed_mixture(d: int) -> int:
    """MixtureAligner 的非专家部分 (W_k + W_v + out_projection)。"""
    return 2 * d * d + (d * d + d)


def mixture_capacity_ladder(
    modal_dims: List[int], d_shared: int, experts: Iterable[int],
) -> List[Dict[str, int]]:
    """给出 MixtureAligner 在若干 n_shared_experts 下的参数量阶梯。"""
    return [{"n_shared_experts": n, "params": mixture_params(modal_dims, d_shared, n)}
            for n in experts]


def heldout_seed(base_seed: int, offset: int = 1000) -> int:
    """held-out 划分的种子偏移。

    ``prepare_data`` 用 ``seed*10 + cls*6 + i`` 驱动噪声 generator, 故换一个足够大的
    base seed 即得到**同分布、不同抽样**的评估集。这是本模块对"held-out"的全部含义:
    不换分布, 只换抽样。
    """
    return base_seed + offset


def ladder_table(
    rows: List[Dict], base_arm: str = "attn",
) -> str:
    """把 (arm, params, mse_in_sample, mse_heldout) 渲染成 markdown 表。

    ``delta_at_matched_capacity`` 一列是 E2 的直接读数: 它只比较与 ``base_arm``
    容量相同的臂, 用来把"设计"从"规模"里剥出来。
    """
    base = next((r for r in rows if r["arm"] == base_arm), None)
    bpar = base["params"] if base else None
    lines = ["| arm | 参数量 | 相对 attn | in-sample MSE | held-out MSE | 泛化落差 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: x["params"]):
        rel = f"{r['params']/bpar:.2f}×" if bpar else "-"
        gap = r["mse_heldout"] - r["mse_in_sample"]
        lines.append(
            f"| {r['arm']} | {r['params']:,} | {rel} | {r['mse_in_sample']:.4f} | "
            f"{r['mse_heldout']:.4f} | {gap:+.4f} |")
    return "\n".join(lines)
