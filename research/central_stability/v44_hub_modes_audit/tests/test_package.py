"""v44.0 — 四中枢模式审计算子单元测试 (8 项)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V44.0 四中枢模式审计算子 — 单元测试 (8 项)"

from research.central_stability.v44_hub_modes_audit import (
    MODES,
    capacity_table,
    gate_is_smallest,
    generalization_gap,
    mode_central_params,
    paired_gain,
)

D, M = 256, 3


def test_known_modes_only():
    """1. 只接受已知的四种模式"""
    assert MODES == ["broadcast", "gate", "router-norm", "adaptive-ema"]
    for m in MODES:
        mode_central_params(m, D, M)
    try:
        mode_central_params("nope", D, M)
    except ValueError:
        print("[PASS] test_known_modes_only")
        return True
    raise AssertionError("未知模式应当抛 ValueError")


def test_gate_params_formula():
    """2. gate = c[d] + W_t[d], 与专家数无关"""
    p = mode_central_params("gate", D, M)
    assert p["total"] == 2 * D and p["U"] == 0, p
    for m in (2, 3, 8, 32):
        assert mode_central_params("gate", D, m)["total"] == 2 * D, "gate 不应随 M 增长"
    print("[PASS] test_gate_params_formula")
    return True


def test_u_modes_params_formula():
    """3. broadcast / router-norm = c[d] + U[M,d]"""
    for m in ("broadcast", "router-norm"):
        p = mode_central_params(m, D, M)
        assert p["total"] == D + M * D, (m, p)
        assert p["W_t"] == 0, m
    assert mode_central_params("adaptive-ema", D, M)["total"] == D + M * D + 1
    print("[PASS] test_u_modes_params_formula")
    return True


def test_gate_is_smallest():
    """4. 核心事实: gate 的中枢参数是最少的 —— 混杂方向需实测确认"""
    assert gate_is_smallest(D, M), "gate 应当是最省的"
    t = capacity_table(D, M)
    assert t[0]["mode"] == "gate", t
    print(f"[PASS] test_gate_is_smallest (gate={t[0]['total']}, 其余={t[-1]['total']})")
    return True


def test_capacity_table_sorted_and_ratios():
    """5. 容量表按参数量升序, 比值相对 gate"""
    t = capacity_table(D, M)
    totals = [r["total"] for r in t]
    assert totals == sorted(totals), totals
    for r in t:
        assert abs(r["ratio_vs_gate"] - r["total"] / t[0]["total"]) < 1e-9
    print("[PASS] test_capacity_table_sorted_and_ratios")
    return True


def test_paired_gain_basic():
    """6. 配对增益: 逐 seed 比较, 中位数与胜出数"""
    base_in = [1.0, 1.0, 1.0, 1.0, 1.0]
    arm_in = [0.5, 0.6, 1.0, 1.2, 0.8]
    g = paired_gain(base_in, arm_in)
    assert g["n"] == 5
    assert g["seeds_won"] == 3, g           # 0.5/0.6/0.8 三档优于 1.0; 1.0 打平不计
    assert abs(g["median_gain_pct"] - 20.0) < 1e-6, g
    assert len(g["per_seed"]) == 5
    assert g["per_seed"] == [50.0, 40.0, 0.0, -20.0, 20.0], g["per_seed"]
    print(f"[PASS] test_paired_gain_basic (median={g['median_gain_pct']:.1f}%, 胜 {g['seeds_won']}/5)")
    return True


def test_paired_gain_rejects_length_mismatch():
    """7. 配对要求等长 —— 长度不符必须报错而不是静默截断"""
    try:
        paired_gain([1.0, 1.0], [1.0])
    except ValueError:
        print("[PASS] test_paired_gain_rejects_length_mismatch")
        return True
    raise AssertionError("长度不符时应当抛 ValueError")


def test_generalization_gap_direction():
    """8. 泛化落差: held-out 相对 in-sample 的方向与量级"""
    g = generalization_gap([0.5] * 5, [1.0] * 5)
    assert abs(g["in_sample_median"] - 0.5) < 1e-9
    assert abs(g["heldout_median"] - 1.0) < 1e-9
    assert abs(g["absolute_gap"] - 0.5) < 1e-9
    assert abs(g["relative_gap_pct"] - 100.0) < 1e-6
    print("[PASS] test_generalization_gap_direction")
    return True


TESTS = [
    test_known_modes_only,
    test_gate_params_formula,
    test_u_modes_params_formula,
    test_gate_is_smallest,
    test_capacity_table_sorted_and_ratios,
    test_paired_gain_basic,
    test_paired_gain_rejects_length_mismatch,
    test_generalization_gap_direction,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
