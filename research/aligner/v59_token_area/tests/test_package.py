"""V59 单测。全部在远端跑（tools/run_v59_tests_remote.py）。"""
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

from research.aligner.v59_token_area.token_area import (
    cross_spectrum, noise_ceiling, chi_at, cut_tokens,
    center_over_sites, shuffle_rows, matched_ceiling, chi_matched, chi_iid,
    TokenGrams, spec_from_grams,
)


def _corr(a, b):
    """相关系数。任一侧方差为 0 时相关系数无定义 —— 按惯例记 0
    （np.corrcoef 会返回 nan，直接比较会静默失败）。"""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def test_gram_equals_direct():
    """★ Gram 技巧必须与直接 SVD 数值一致（否则大 d 下的读数不可信）。"""
    torch.manual_seed(0)
    for (n, da, db) in ((64, 32, 48), (96, 128, 96), (128, 300, 200), (200, 20, 400)):
        A, B = torch.randn(n, da), torch.randn(n, db)
        g = cross_spectrum(A, B, gram_threshold=0.0)
        b = cross_spectrum(A, B, gram_threshold=1e30)
        m = min(len(g), len(b))
        err = float((g[:m] - b[:m]).abs().max())
        scale = float(b[0].abs().clamp(min=1e-12))
        assert err / scale < 5e-3, (n, da, db, err, scale)
    print("[PASS] Gram 与直接 SVD 一致（4 组形状，含 d<n）")
    return True


def test_token_grams_equals_direct():
    """★ 加速路径必须与朴素路径给同一谱（否则所有读数作废）。"""
    torch.manual_seed(0)
    for (n, S, D) in ((64, 5, 12), (96, 8, 20), (128, 6, 40)):
        X = torch.randn(n, S, D)
        tg = TokenGrams(X)
        for k in (1, S // 2, S - 1):
            idx = list(range(k))
            A, B = cut_tokens(X, k)
            direct = cross_spectrum(A, B)
            fast = spec_from_grams(tg.G(idx), tg.G_complement(idx), n)
            m = min(len(direct), len(fast))
            err = float((direct[:m] - fast[:m]).abs().max())
            scale = float(direct[0].abs().clamp(min=1e-12))
            assert err / scale < 5e-3, (n, S, D, k, err, scale)
    print("[PASS] TokenGrams 与朴素路径一致（3 形状 × 3 切口）")
    return True


def test_pure_noise_zero_all_cuts():
    """★ 纯噪声在 token 轴全部切口上 χ 必须 = 0。"""
    bad = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        n, S, D = 96, 8, 32
        X = torch.randn(n, S, D)
        for k in range(1, S):
            tau = noise_ceiling(n, k * D, (S - k) * D, reps=14, seed=100 + k)
            A, B = cut_tokens(X, k)
            c = chi_at(A, B, tau)
            if c != 0:
                bad.append((seed, k, c))
    assert not bad, bad
    print("[PASS] 纯噪声 3 seed × 7 切口 全部 χ=0")
    return True


def test_shared_dims_recovered():
    """共享 k 维必须被数出来（就地零分布不能太钝）。"""
    torch.manual_seed(0)
    n, S, D = 512, 8, 32
    k = 5
    Z = torch.randn(n, k)
    X = torch.randn(n, S, D) + (Z @ torch.randn(k, S * D)).reshape(n, S, D)
    tau = noise_ceiling(n, 4 * D, 4 * D, reps=14, seed=99)
    A, B = cut_tokens(X, 4)
    got = chi_at(A, B, tau)
    assert got >= k, (k, got)
    print(f"[PASS] 打入 {k} 个共享方向 → 数出 {got}（≥{k}）")
    return True


def test_global_common_cause_detected_then_removed():
    """★ 核心机制检验：全局共同因必须被 raw 检出，且被中心化去掉。

    构造: 所有 token 共享 base，再加各自独立噪声。
      raw      → 任一切口两侧都含 base ⇒ χ ≥ 1
      centered → base 被逐样本均值吸收 ⇒ χ = 0
    这正是"为什么面积律会变成平凡性质"的直接演示。
    """
    torch.manual_seed(0)
    n, S, D = 192, 8, 32
    base = torch.randn(n, 1, D)
    X = base + torch.randn(n, S, D) * 0.5

    tau = noise_ceiling(n, 4 * D, 4 * D, reps=14, seed=7)
    A, B = cut_tokens(X, 4)
    raw = chi_at(A, B, tau)

    A2, B2 = cut_tokens(center_over_sites(X), 4)
    cen = chi_at(A2, B2, tau)

    assert raw >= 1, ("raw 应检出共同因", raw)
    assert cen == 0, ("中心化后应干净", cen)
    print(f"[PASS] 全局共同因: raw χ={raw} ≥1 → centered χ={cen} =0")
    return True


def test_iid_null_underestimates_with_correlated_columns():
    """★ 实测发现: iid 零分布会低估谱尾部，必须换成结构匹配零分布。

    构造: 每个 token = base(逐样本) + 独立噪声 ⇒ A 的列高度相关。
      · iid 天花板下，**样本已打乱**的数据仍读出 χ≈4（假阳性）
      · 匹配零分布下，样本打乱的数据 χ=0
      · 匹配零分布下，原始数据 χ≥1（真同源被检出）
    这条测试把"iid 零分布不可用"固定成回归项。
    """
    torch.manual_seed(0)
    n, S, D = 192, 8, 32
    base = torch.randn(n, 1, D)
    X = base + torch.randn(n, S, D) * 0.5

    tau_iid = noise_ceiling(n, 4 * D, 4 * D, reps=14, seed=8)
    A, B = cut_tokens(X, 4)
    B_shuf = shuffle_rows(B, seed=3)

    raw_iid = chi_iid(A, B, tau_iid)
    shuf_iid = chi_iid(A, B_shuf, tau_iid)          # ← 假阳性计数
    raw_mat, _ = chi_matched(A, B, reps=12, seed=40)
    shuf_mat, _ = chi_matched(A, B_shuf, reps=12, seed=50)

    assert raw_iid >= 1, ("iid 下原始应显著", raw_iid)
    assert shuf_iid >= 1, ("iid 下打乱数据应出现假阳性——若为 0 说明该现象本轮未复现", shuf_iid)
    assert raw_mat >= 1, ("匹配零分布下原始应显著", raw_mat)
    assert shuf_mat == 0, ("匹配零分布下打乱数据应干净", shuf_mat)
    print(f"[PASS] iid 零分布假阳性={shuf_iid} 个通道；匹配零分布: raw={raw_mat}, shuf={shuf_mat}")
    print(f"       ⇒ iid 天花板整体偏低约 {shuf_iid} 个通道，v58 的公式阈值同病")
    return True


def test_matched_null_stable_across_seeds():
    """匹配零分布在多个 seed 下都应给 0（不是碰巧）。"""
    torch.manual_seed(1)
    n, S, D = 192, 8, 16
    X = torch.randn(n, S, D)          # 纯噪声，无任何共同因
    A, B = cut_tokens(X, 4)
    bad = []
    for s in (0, 1, 2):
        c, tau = chi_matched(A, B, reps=10, seed=s)
        if c != 0:
            bad.append((s, c, round(tau, 3)))
    assert not bad, bad
    print("[PASS] 纯噪声上匹配零分布 3 seed 全 0")
    return True


def test_area_law_constant_profile():
    """χ 恒定 ⇒ corr≈0 且取值数=1（面积律的极限情形）。"""
    prof = [6] * 6
    sizes = [k * 16 for k in range(1, 7)]
    assert abs(_corr(prof, sizes)) < 1e-9 and len(set(prof)) == 1
    print("[PASS] 恒定剖面 corr=0, 取值数=1")
    return True


def test_volume_law_detected():
    """体积律（χ ∝ |A|）必须给出正相关，区别于面积律。"""
    c = _corr([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert c > 0.9, c
    print(f"[PASS] 体积律 corr={c:.2f} > 0.9")
    return True


def test_predictions_declared():
    from examples.run_v59_token_area import PREDICTIONS
    assert set(PREDICTIONS) == {"N0", "N1", "N2", "N3", "N4"}
    print(f"[PASS] 五条预登记判决齐全: {list(PREDICTIONS)}")
    return True


TESTS = [
    test_gram_equals_direct,
    test_token_grams_equals_direct,
    test_pure_noise_zero_all_cuts,
    test_shared_dims_recovered,
    test_global_common_cause_detected_then_removed,
    test_iid_null_underestimates_with_correlated_columns,
    test_matched_null_stable_across_seeds,
    test_area_law_constant_profile,
    test_volume_law_detected,
    test_predictions_declared,
]

TITLE = "V59.0 token 轴面积律 — 单元测试 (10 项)"

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
