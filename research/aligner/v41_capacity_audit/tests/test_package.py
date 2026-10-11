"""v41.0 — 对齐器容量审计算子单元测试 (8 项).

本文件是权威副本。根目录 ``tests/`` 下的转发层若存在也只是薄转发层 —— 本代刻意**不**
添加, 理由见 MANIFEST._note(新增转发层会改变 run.sh test 的套件计数)。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_fusion")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V41.0 对齐器容量审计算子 — 单元测试 (8 项)"

import torch
import torch.nn as nn

from research.aligner.v41_capacity_audit import (
    attn_param_budget,
    count_trainable,
    heldout_seed,
    ladder_table,
    mixture_capacity_ladder,
    mixture_experts_for_budget,
    mixture_params,
    param_report,
)

MD = [312, 768, 192]
DS = 256


def test_count_trainable_counts_only_trainable():
    """1. 只数 requires_grad 的参数"""
    m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8))
    assert count_trainable(m) == 4 * 8 + 8 + 8 * 8 + 8
    for p in m[0].parameters():
        p.requires_grad_(False)
    assert count_trainable(m) == 8 * 8 + 8, "冻结参数不应计入"
    print("[PASS] test_count_trainable_counts_only_trainable")
    return True


def test_attn_budget_matches_instantiation():
    """2. attn 解析式与真实实例化逐位一致"""
    from research.aligner.v10_embedding import CrossArchAttnAligner
    real = count_trainable(CrossArchAttnAligner(modal_dims=MD, d_shared=DS, num_heads=4))
    formula = attn_param_budget(MD, DS, 4)
    assert real == formula, f"解析 {formula} != 实测 {real}"
    print(f"[PASS] test_attn_budget_matches_instantiation ({real:,})")
    return True


def test_mixture_formula_matches_instantiation():
    """3. mixture 解析式与真实实例化逐位一致"""
    from research.aligner.v13_mixture import MixtureAligner
    for n in (1, 4):
        real = count_trainable(
            MixtureAligner(modal_dims=MD, d_shared=DS, n_shared_experts=n, num_heads=4))
        formula = mixture_params(MD, DS, n)
        assert real == formula, f"n={n}: 解析 {formula} != 实测 {real}"
    print("[PASS] test_mixture_formula_matches_instantiation")
    return True


def test_capacity_matched_pair_is_exact():
    """4. 核心性质: mixture(n=1) 与 attn 参数量**完全相等** —— E2 的等容量对"""
    from research.aligner.v13_mixture import MixtureAligner
    from research.aligner.v10_embedding import CrossArchAttnAligner
    a = count_trainable(CrossArchAttnAligner(modal_dims=MD, d_shared=DS, num_heads=4))
    m = count_trainable(
        MixtureAligner(modal_dims=MD, d_shared=DS, n_shared_experts=1, num_heads=4))
    assert a == m, f"等容量对失效: attn {a} vs mixture-n1 {m}"
    print(f"[PASS] test_capacity_matched_pair_is_exact ({a:,} == {m:,})")
    return True


def test_mixture_capacity_is_monotone():
    """5. 参数量随 n_shared_experts 单调递增, 且 n=4 > n=1"""
    ladder = mixture_capacity_ladder(MD, DS, [1, 2, 4, 8])
    ps = [r["params"] for r in ladder]
    assert ps == sorted(ps) and len(set(ps)) == len(ps), ps
    assert ps[-1] > ps[0], "n=8 应大于 n=1"
    print("[PASS] test_mixture_capacity_is_monotone")
    return True


def test_experts_for_budget_finds_closest():
    """6. 按预算反查 n_shared_experts, 命中等容量点"""
    target = attn_param_budget(MD, DS, 4)
    n, got = mixture_experts_for_budget(MD, DS, target)
    assert n == 1 and got == target, f"应命中 n=1, 实得 n={n} params={got}"
    print("[PASS] test_experts_for_budget_finds_closest")
    return True


def test_heldout_seed_changes_draw_not_distribution():
    """7. held-out 只换抽样不换分布: 偏移恒定且可逆, 不与原种子碰撞"""
    s = heldout_seed(3, offset=1000)
    assert s == 1003 and s != 3
    assert heldout_seed(3, 1000) == heldout_seed(3, 1000), "必须确定性"
    assert heldout_seed(0) - 0 == heldout_seed(9) - 9, "偏移对所有种子一致"
    print("[PASS] test_heldout_seed_changes_draw_not_distribution")
    return True


def test_ladder_table_sorts_by_capacity():
    """8. 渲染按参数量升序, 且泛化落差列存在"""
    rows = [
        {"arm": "mixture-n4", "params": 2291968, "mse_in_sample": 0.25, "mse_heldout": 0.30},
        {"arm": "attn", "params": 720640, "mse_in_sample": 0.34, "mse_heldout": 0.40},
        {"arm": "mixture-n1", "params": 720640, "mse_in_sample": 0.30, "mse_heldout": 0.36},
    ]
    tbl = ladder_table(rows, base_arm="attn")
    assert tbl.count("\n") == 4, tbl
    order = [ln.split("|")[1].strip() for ln in tbl.splitlines()[2:]]
    assert order == ["attn", "mixture-n1", "mixture-n4"], order
    assert "1.00×" in tbl and "+0.0600" in tbl
    print("[PASS] test_ladder_table_sorts_by_capacity")
    return True


TESTS = [
    test_count_trainable_counts_only_trainable,
    test_attn_budget_matches_instantiation,
    test_mixture_formula_matches_instantiation,
    test_capacity_matched_pair_is_exact,
    test_mixture_capacity_is_monotone,
    test_experts_for_budget_finds_closest,
    test_heldout_seed_changes_draw_not_distribution,
    test_ladder_table_sorts_by_capacity,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
