"""v46.0 — 通路分解审计算子单元测试 (7 项)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V46.0 通路分解审计算子 — 单元测试 (7 项)"

from research.routing_evolution.v46_attn_path_audit import (
    decomposition_table, path_claim_verdict, path_gap,
)


def test_gap_positive_when_attn_better():
    """1. attn 误差更小 ⇒ 优势为正"""
    g = path_gap([0.20, 0.22, 0.18, 0.21], [0.30, 0.32, 0.28, 0.31])
    assert g["median_pct"] > 0, g
    assert g["attn_wins"] == 4 and g["n"] == 4
    print(f"[PASS] test_gap_positive_when_attn_better ({g['median_pct']:+.1f}%)")
    return True


def test_gap_negative_when_ffn_better():
    """2. ffn 误差更小 ⇒ 优势为负"""
    g = path_gap([0.30] * 5, [0.20] * 5)
    assert g["median_pct"] < 0 and g["attn_wins"] == 0
    print(f"[PASS] test_gap_negative_when_ffn_better ({g['median_pct']:+.1f}%)")
    return True


def test_gap_rejects_length_mismatch():
    """3. 配对长度不符必须报错"""
    try:
        path_gap([1.0], [1.0, 2.0])
    except ValueError:
        print("[PASS] test_gap_rejects_length_mismatch")
        return True
    raise AssertionError("长度不符应当抛 ValueError")


def test_verdict_holds():
    """4. attn 稳定更优 ⇒ holds"""
    g = path_gap([0.20] * 5, [0.30] * 5)
    assert path_claim_verdict(g) == "holds", g
    print("[PASS] test_verdict_holds")
    return True


def test_verdict_reversed():
    """5. ffn 稳定更优 ⇒ reversed(主张方向反了)"""
    g = path_gap([0.30] * 5, [0.20] * 5)
    assert path_claim_verdict(g) == "reversed", g
    print("[PASS] test_verdict_reversed")
    return True


def test_verdict_inconclusive_on_split():
    """6. 单点反转导致中位与胜出矛盾 ⇒ inconclusive"""
    attn = [0.10, 0.10, 0.10, 0.10, 0.90]
    ffn = [0.20, 0.20, 0.20, 0.20, 0.20]
    g = path_gap(attn, ffn)
    v = path_claim_verdict(g)
    assert v in ("holds", "inconclusive"), (v, g)
    print(f"[PASS] test_verdict_inconclusive_on_split (裁决 {v})")
    return True


def test_table_renders_both_scopes():
    """7. 渲染表同时给出 in-sample 与 held-out 两行"""
    seeds = [{"seed": 0,
              "in_sample": {"fuse": .3, "attn_only": .2, "ffn_only": .3, "avg": .35},
              "heldout": {"fuse": .5, "attn_only": .4, "ffn_only": .5, "avg": .6}}]
    t = decomposition_table(seeds)
    assert "in_sample" in t and "heldout" in t
    assert t.count("\n") == 3, t
    print("[PASS] test_table_renders_both_scopes")
    return True


TESTS = [
    test_gap_positive_when_attn_better,
    test_gap_negative_when_ffn_better,
    test_gap_rejects_length_mismatch,
    test_verdict_holds,
    test_verdict_reversed,
    test_verdict_inconclusive_on_split,
    test_table_renders_both_scopes,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
