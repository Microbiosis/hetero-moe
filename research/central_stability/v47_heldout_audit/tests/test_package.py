"""v47.0 — 样本外判定算子单元测试 (9 项)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V47.0 样本外判定算子 — 单元测试 (9 项)"

from research.central_stability.v47_heldout_audit import (
    audit_table, leakage_verdict, paired_gain, retention,
)


def test_gain_positive_for_error_metric():
    """1. 误差类: 越小越好, treatment 更小 ⇒ 优势为正"""
    g = paired_gain([0.20, 0.22, 0.18, 0.21], [0.30, 0.32, 0.28, 0.31])
    assert g["median_pct"] > 0 and g["wins"] == 4 and g["n"] == 4
    print(f"[PASS] test_gain_positive_for_error_metric ({g['median_pct']:+.1f}%)")
    return True


def test_gain_positive_for_accuracy_metric():
    """2. 准确率类: 越大越好, treatment 更大 ⇒ 优势同为正"""
    g = paired_gain([0.80, 0.82, 0.79], [0.70, 0.72, 0.71], lower_is_better=False)
    assert g["median_pct"] > 0 and g["wins"] == 3, g
    print(f"[PASS] test_gain_positive_for_accuracy_metric ({g['median_pct']:+.1f}%)")
    return True


def test_gain_uses_median_not_mean():
    """3. 读数用中位数: 单样本极端值不得主导 (v40 起的纪律)"""
    treat = [1.0] * 9 + [100.0]          # 一个极端离群
    base = [2.0] * 10
    g = paired_gain(treat, base)
    assert abs(g["median_pct"] - 50.0) < 1e-9, g
    assert g["mean_pct"] < 0, "均值应被离群值拖成负 —— 这正是不能用均值的原因"
    print(f"[PASS] test_gain_uses_median_not_mean (中位 {g['median_pct']:+.1f}% vs 均值 {g['mean_pct']:+.1f}%)")
    return True


def test_gain_rejects_bad_input():
    """4. 配对长度不符 / 空序列 / baseline 为 0 都必须报错"""
    for bad in (lambda: paired_gain([1.0, 2.0], [1.0]),
                lambda: paired_gain([], []),
                lambda: paired_gain([1.0], [0.0])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("应当抛 ValueError")
    print("[PASS] test_gain_rejects_bad_input")
    return True


def test_verdict_clean():
    """5. 两口径同向 + held-out 多数胜出 ⇒ clean"""
    a = {"median_pct": 25.6, "wins": 5, "n": 5}
    b = {"median_pct": 11.0, "wins": 5, "n": 5}
    assert leakage_verdict(a, b) == "clean"
    print("[PASS] test_verdict_clean")
    return True


def test_verdict_invalidated_on_reversal():
    """6. held-out 中位优势 ≤ 0 ⇒ invalidated (方向翻转)"""
    a = {"median_pct": 25.6, "wins": 5, "n": 5}
    b = {"median_pct": -5.1, "wins": 0, "n": 5}
    assert leakage_verdict(a, b) == "invalidated"
    print("[PASS] test_verdict_invalidated_on_reversal")
    return True


def test_verdict_inconclusive_on_split():
    """7. 中位为正但逐样本没过半 ⇒ inconclusive (不能只看中位数)"""
    a = {"median_pct": 25.6, "wins": 5, "n": 5}
    b = {"median_pct": 3.0, "wins": 2, "n": 5}
    assert leakage_verdict(a, b) == "inconclusive"
    print("[PASS] test_verdict_inconclusive_on_split")
    return True


def test_retention_reports_magnitude_collapse():
    """8. 幅度塌缩可读: 中位仍为正但只剩零头时保留率 < 1 且非 None"""
    a = {"median_pct": 25.6, "wins": 5, "n": 5}
    b = {"median_pct": 2.6, "wins": 4, "n": 5}
    assert abs(retention(a, b) - 2.6 / 25.6) < 1e-12
    assert retention(a, {"median_pct": -1.0, "wins": 1, "n": 5}) < 0
    assert retention({"median_pct": -2.0, "wins": 1, "n": 5}, b) is None
    print(f"[PASS] test_retention_reports_magnitude_collapse ({retention(a, b) * 100:.0f}%)")
    return True


def test_table_renders_all_three_scopes():
    """9. 渲染表同时给出三个口径行, 并把判定与保留率写进 held-out 行"""
    rec = {"name": "v34 vs v33", "lower_is_better": True,
           "in_sample":      {"treat": [0.90, 0.91, 0.89], "base": [1.00, 1.00, 1.00]},
           "heldout_noise":  {"treat": [0.80, 0.81, 0.79], "base": [1.00, 1.00, 1.00]},
           "heldout_stimulus": {"treat": [0.95, 0.96, 0.94], "base": [1.00, 1.00, 1.00]}}
    t = audit_table([rec])
    assert "heldout_noise" in t and "heldout_stimulus" in t and t.count("\n") == 4, t
    assert t.count("| clean |") == 2, t
    print("[PASS] test_table_renders_all_three_scopes")
    return True


TESTS = [
    test_gain_positive_for_error_metric,
    test_gain_positive_for_accuracy_metric,
    test_gain_uses_median_not_mean,
    test_gain_rejects_bad_input,
    test_verdict_clean,
    test_verdict_invalidated_on_reversal,
    test_verdict_inconclusive_on_split,
    test_retention_reports_magnitude_collapse,
    test_table_renders_all_three_scopes,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
