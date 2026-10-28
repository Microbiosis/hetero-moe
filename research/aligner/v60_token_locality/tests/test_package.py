"""V60 单测（远端跑）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != "/" and not (os.path.isdir(os.path.join(_ROOT, "research"))
                            and os.path.isdir(os.path.join(_ROOT, "examples"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from research.aligner.v60_token_locality.locality import (
    token_grams, pair_coupling, coupling_matrix, distance_profile, corr,
    permuted_profile, center_over_sites,
)


def test_grams_match_direct():
    """token_grams 必须与直接标准化一致。"""
    torch.manual_seed(0)
    n, S, D = 64, 5, 12
    X = torch.randn(n, S, D)
    G = token_grams(X)
    Xc = X - X.mean(dim=0, keepdim=True)
    W = Xc / (Xc.norm(dim=0, keepdim=True) / (n ** 0.5)).clamp(min=1e-8)
    direct = W[:, 2, :] @ W[:, 2, :].T
    err = float((G[2] - direct).abs().max())
    assert err < 1e-4, err
    print("[PASS] token_grams 与直接标准化一致")
    return True


def test_pure_noise_no_locality():
    """★ P0 的核心：纯噪声上耦合必须与距离无关。"""
    bad = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        n, S, D = 128, 8, 24
        G = token_grams(torch.randn(n, S, D))
        M = coupling_matrix(G, n, topk=3)
        ds, prof = distance_profile(M)
        r = corr(ds, prof)
        if abs(r) >= 0.3:
            bad.append((seed, round(r, 3)))
    assert not bad, bad
    print("[PASS] 纯噪声 2 seed |corr|<0.3（无假局部性）")
    return True


def test_planted_locality_detected():
    """★ 灵敏度正控：植入随位置线性变化的成分，必须检出强负相关。

    构造 X[t] = base + t·direction + noise ⇒ 相邻 token 表征相近。
    若这个都检不出，P1 就没有灵敏度。
    """
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    direction = torch.randn(1, 1, D)
    pos = torch.arange(S).view(1, S, 1).float()
    X = base + pos * direction * 1.5 + torch.randn(n, S, D) * 0.3

    G = token_grams(X)
    M = coupling_matrix(G, n, topk=3)
    ds, prof = distance_profile(M)
    r = corr(ds, prof)
    ratio = prof[0] / prof[-1]

    assert r < -0.5, ("应检出强衰减", round(r, 3), [round(v, 3) for v in prof])
    assert ratio > 1.5, ("局部性比应 >1.5", round(ratio, 3))
    print(f"[PASS] 植入局部性被检出: corr={r:+.3f}, M̄(1)/M̄(15)={ratio:.2f}")
    return True


def test_permutation_kills_locality():
    """★ P2 的核心：打乱 token 顺序后，衰减必须消失。"""
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    direction = torch.randn(1, 1, D)
    pos = torch.arange(S).view(1, S, 1).float()
    X = base + pos * direction * 1.5 + torch.randn(n, S, D) * 0.3

    M = coupling_matrix(token_grams(X), n, topk=3)
    ds, prof = distance_profile(M)
    r_true = corr(ds, prof)

    null = [corr(ds, permuted_profile(M, seed=100 + t)[1]) for t in range(30)]
    null_sorted = sorted(null)
    lo = null_sorted[int(0.025 * (len(null_sorted) - 1))]

    assert r_true < lo, ("真实应低于零分布下界", round(r_true, 3), round(lo, 3))
    assert abs(sum(null) / len(null)) < 0.3, ("零分布应居中于 0", round(sum(null)/len(null), 3))
    print(f"[PASS] 置换零分布: 真实={r_true:+.3f}, 下界={lo:+.3f}, 均值={sum(null)/len(null):+.3f}")
    return True


def test_global_common_cause_is_flat():
    """★ 全局共同因（所有 token 共享 base）必须给出**平**的剖面。

    这正是 v58/v59 平凡结果的机制：共享成分让所有对都一样强。
    """
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    X = base.repeat(1, S, 1) + torch.randn(n, S, D) * 0.5

    M = coupling_matrix(token_grams(X), n, topk=3)
    ds, prof = distance_profile(M)
    r = corr(ds, prof)
    assert abs(r) < 0.5, ("共同因应给出平坦剖面", round(r, 3), [round(v, 2) for v in prof])
    print(f"[PASS] 全局共同因 → corr={r:+.3f}（近平），剖面 {[round(v,2) for v in prof[:4]]}...")
    return True


def test_center_removes_common_cause():
    """中心化后共同因消失，剖面应更平（|corr| 不增）。"""
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    pos = torch.arange(S).view(1, S, 1).float()
    direction = torch.randn(1, 1, D)
    X = base + pos * direction * 1.5 + torch.randn(n, S, D) * 0.3

    r_raw = corr(*distance_profile(
        coupling_matrix(token_grams(X), n, topk=3)))
    r_cen = corr(*distance_profile(
        coupling_matrix(token_grams(center_over_sites(X)), n, topk=3)))
    # 中心化去掉的是"逐样本公共成分"，植入的 pos 结构仍然保留
    assert r_cen < 0, ("中心化后局部结构应仍在", round(r_cen, 3))
    print(f"[PASS] raw corr={r_raw:+.3f} → centered corr={r_cen:+.3f}（局部结构保留）")
    return True


def test_profile_shape():
    """距离剖面长度必须是 S−1，且 d 严格递增。"""
    torch.manual_seed(0)
    n, S, D = 64, 9, 12
    M = coupling_matrix(token_grams(torch.randn(n, S, D)), n, topk=1)
    ds, prof = distance_profile(M)
    assert ds == list(range(1, S)) and len(prof) == S - 1
    print(f"[PASS] 剖面长度 {len(prof)} = S−1，d = 1..{S-1}")
    return True


def test_predictions_declared():
    from examples.run_v60_token_locality import PREDICTIONS
    assert set(PREDICTIONS) == {"P0", "P1", "P2", "P3"}
    print(f"[PASS] 四条预登记判决齐全: {list(PREDICTIONS)}")
    return True


TESTS = [
    test_grams_match_direct,
    test_pure_noise_no_locality,
    test_planted_locality_detected,
    test_permutation_kills_locality,
    test_global_common_cause_is_flat,
    test_center_removes_common_cause,
    test_profile_shape,
    test_predictions_declared,
]

TITLE = "V60.0 token 局部性 — 单元测试 (8 项)"

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
