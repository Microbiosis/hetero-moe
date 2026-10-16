"""v45.0 — 规模边界算子单元测试 (7 项)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V45.0 语料规模边界算子 — 单元测试 (7 项)"

from research.corpus_scale.v45_scale_boundary_audit import (
    boundary_direction, boundary_survives_heldout, better_mode, scale_table,
)

PAPER_TABLE = [("tiny", {"gate": 0.7097, "adaptive-ema": 0.7035}),
               ("small", {"gate": 0.7075, "adaptive-ema": 0.6984}),
               ("medium", {"gate": 0.6971, "adaptive-ema": 0.7136}),
               ("large", {"gate": 0.7029, "adaptive-ema": 0.7136})]


def test_better_mode_picks_lower():
    """1. 更优 = 误差更小"""
    assert better_mode({"gate": 0.70, "adaptive-ema": 0.69}) == "adaptive-ema"
    assert better_mode({"gate": 0.69, "adaptive-ema": 0.70}) == "gate"
    print("[PASS] test_better_mode_picks_lower")
    return True


def test_better_mode_rejects_tie():
    """2. 分数相同必须拒绝判定 —— 不能靠顺序取胜出"""
    try:
        better_mode({"gate": 0.70, "adaptive-ema": 0.70})
    except ValueError:
        print("[PASS] test_better_mode_rejects_tie")
        return True
    raise AssertionError("同分时应当抛 ValueError")


def test_paper_table_direction_is_ada_to_gate():
    """3. 论文主表自身的方向是 adaptive-ema -> gate, 与摘要主张相反"""
    r = boundary_direction(PAPER_TABLE)
    assert r["direction"] == "adaptive-ema -> gate", r
    assert r["has_boundary"] is True
    assert r["switch_at"] == "medium", r
    print(f"[PASS] test_paper_table_direction_is_ada_to_gate ({r['direction']}, 翻转于 {r['switch_at']})")
    return True


def test_abstract_claim_is_contradicted_by_its_own_table():
    """4. 摘要主张 'gate -> adaptive-ema' 被自己的主表证伪"""
    r = boundary_direction(PAPER_TABLE)
    assert r["low_winner"] == "adaptive-ema", "小语料应是 adaptive-ema 优"
    assert r["high_winner"] == "gate", "大语料应是 gate 优"
    assert not (r["low_winner"] == "gate" and r["high_winner"] == "adaptive-ema")
    print("[PASS] test_abstract_claim_is_contradicted_by_its_own_table")
    return True


def test_no_boundary_case():
    """5. 全档同一模式 ⇒ 无边界"""
    tbl = [(l, {"gate": 0.70, "adaptive-ema": 0.71}) for l in ("a", "b", "c")]
    r = boundary_direction(tbl)
    assert r["has_boundary"] is False and r["switch_at"] is None
    assert r["direction"] == "gate -> gate"
    print("[PASS] test_no_boundary_case")
    return True


def test_direction_disagreement_is_detected():
    """6. held-out 上边界方向改变时必须报出来, 而不是沿用训练集口径"""
    flipped = [("small", {"gate": 0.70, "adaptive-ema": 0.71}),
               ("large", {"gate": 0.69, "adaptive-ema": 0.72})]   # large 变成 adaptive-ema 优
    r = boundary_survives_heldout(PAPER_TABLE, flipped)
    assert r["direction_agrees"] is False, r
    assert "改变" in r["verdict"], r["verdict"]
    print("[PASS] test_direction_disagreement_is_detected")
    return True


def test_scale_table_rendering():
    """7. 渲染表含规模/模式/两个口径/落差"""
    md = scale_table([("large", "gate", 0.70, 0.77)])
    assert "large" in md and "gate" in md and "0.7000" in md and "0.7700" in md
    assert md.count("|") > 10
    print("[PASS] test_scale_table_rendering")
    return True


TESTS = [
    test_better_mode_picks_lower,
    test_better_mode_rejects_tie,
    test_paper_table_direction_is_ada_to_gate,
    test_abstract_claim_is_contradicted_by_its_own_table,
    test_no_boundary_case,
    test_direction_disagreement_is_detected,
    test_scale_table_rendering,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
