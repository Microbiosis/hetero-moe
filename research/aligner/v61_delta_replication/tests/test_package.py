"""V61 单测（远端跑）。核心是确认判据实现与 SPEC 一致。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != "/" and not (os.path.isdir(os.path.join(_ROOT, "research"))
                            and os.path.isdir(os.path.join(_ROOT, "examples"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from research.aligner.v60_token_locality.locality import cos_matrix
from examples.run_v61_delta_replication import (
    delta_d1, parity_gap, _stable_seed, analyse, PREDICTIONS,
)


def test_delta_definition():
    """Δ = M̄(1) − mean(M̄(2..S−1))。用手算可验证的矩阵。"""
    # M[i,j] = |i-j| 的递减函数: 相邻最相似
    S = 6
    M = torch.zeros(S, S)
    for i in range(S):
        for j in range(S):
            M[i, j] = 1.0 / (1 + abs(i - j))
    d = delta_d1(M)
    d1 = sum(float(M[i, i + 1]) for i in range(S - 1)) / (S - 1)
    rest = [float(M[i, i + dd]) for dd in range(2, S) for i in range(S - dd)]
    expect = d1 - sum(rest) / len(rest)
    assert abs(d - expect) < 1e-9 and d > 0
    print(f"[PASS] Δ 定义正确: Δ={d:.4f} (相邻更相似 → 正)")
    return True


def test_parity_gap_zero_for_smooth():
    """平滑剖面（只依赖 |i−j|）的奇偶差必须为 0 —— 它只测周期 2。"""
    S = 10
    M = torch.zeros(S, S)
    for i in range(S):
        for j in range(S):
            M[i, j] = 1.0 / (1 + abs(i - j))
    pg = parity_gap(M)
    assert abs(pg) < 1e-9, pg
    print(f"[PASS] 平滑剖面奇偶差 = {pg:.2e}（R4 只对周期 2 敏感）")
    return True


def test_parity_gap_detects_alternating():
    """周期 2 的交替结构必须被 R4 抓住。"""
    S = 10
    M = torch.zeros(S, S)
    for i in range(S):
        for j in range(S):
            M[i, j] = 1.0 if (i - j) % 2 == 0 else -1.0
    pg = parity_gap(M)
    assert abs(pg) > 1.5, pg
    print(f"[PASS] 交替结构奇偶差 = {pg:.2f}（大幅偏离 0）")
    return True


def test_planted_local_delta_positive():
    """植入相邻相似的结构 ⇒ Δ 应显著为正。"""
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    direction = torch.randn(1, 1, D)
    pos = torch.arange(S).view(1, S, 1).float()
    X = base + pos * direction * 0.05 + torch.randn(n, S, D) * 0.01
    r = analyse(X, 30, 7)
    assert r["delta"] > 0.05 and r["outside"], (r["delta"], r["outside"])
    print(f"[PASS] 植入相邻相似 → Δ={r['delta']:+.4f}, 显著={r['outside']}")
    return True


def test_planted_alternating_delta_negative_and_r4_flags():
    """植入周期 2 结构 ⇒ Δ 为负，**且 R4 必须报警**（这正是 R4 的用途）。"""
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    alt = torch.randn(n, 1, D)
    sign = torch.tensor([1.0 if t % 2 == 0 else -1.0 for t in range(S)]).view(1, S, 1)
    X = alt * sign * 2.0 + torch.randn(n, S, D) * 0.05
    r = analyse(X, 30, 11)
    assert r["delta"] < -0.05, ("交替结构的 Δ 应为负", r["delta"])
    assert abs(r["parity_gap"]) >= 0.05, ("R4 应报警", r["parity_gap"])
    print(f"[PASS] 周期 2 结构: Δ={r['delta']:+.4f}, 奇偶差={r['parity_gap']:+.4f}（R4 会拦下）")
    return True


def test_pure_noise_null():
    """纯噪声上 Δ 不应显著。"""
    torch.manual_seed(0)
    X = torch.randn(128, 12, 32)
    r = analyse(X, 30, 3)
    assert not r["outside"], (r["delta"], r["null_lo"], r["null_hi"])
    print(f"[PASS] 纯噪声 Δ={r['delta']:+.4f} 不显著")
    return True


def test_stable_seed_deterministic():
    """seed 必须可复现（不能用内置 hash）。"""
    a = _stable_seed("big", 16, "raw")
    b = _stable_seed("big", 16, "raw")
    c = _stable_seed("big", 32, "raw")
    assert a == b and a != c
    print(f"[PASS] _stable_seed 可复现: {a} == {b}, 且与不同参数区分 ({c})")
    return True


def test_predictions_match_spec():
    assert set(PREDICTIONS) == {"R0", "R1", "R2", "R3", "R4"}
    print(f"[PASS] 五条预登记判据齐全: {list(PREDICTIONS)}")
    return True


TESTS = [
    test_delta_definition,
    test_parity_gap_zero_for_smooth,
    test_parity_gap_detects_alternating,
    test_planted_local_delta_positive,
    test_planted_alternating_delta_negative_and_r4_flags,
    test_pure_noise_null,
    test_stable_seed_deterministic,
    test_predictions_match_spec,
]

TITLE = "V61.0 Δ 跨模型复制 — 单元测试 (8 项)"

if __name__ == "__main__":
    print(f"=== {TITLE} ===")
    ok = 0
    for fn in TESTS:
        try:
            if fn():
                ok += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
    print("---")
    print(f"合计: {ok}/{len(TESTS)} {'✓ ALL PASS' if ok == len(TESTS) else '✗ HAS FAIL'}")
    sys.exit(0 if ok == len(TESTS) else 1)
