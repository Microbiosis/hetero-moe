"""V60 单测（远端跑）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != "/" and not (os.path.isdir(os.path.join(_ROOT, "research"))
                            and os.path.isdir(os.path.join(_ROOT, "examples"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from research.aligner.v60_token_locality.locality import (
    cos_matrix, distance_profile, corr, permuted_profile, cos_profile,
    token_grams, coupling_matrix,
)


def _planted(n=192, S=16, D=32, amp=1.5, noise=0.3, seed=0):
    """X[t] = base + t·direction + noise ⇒ 相邻位置表征相近。"""
    torch.manual_seed(seed)
    base = torch.randn(n, 1, D)
    direction = torch.randn(1, 1, D)
    pos = torch.arange(S).view(1, S, 1).float()
    return base + pos * direction * amp + torch.randn(n, S, D) * noise


def test_v1_spectral_coupling_is_blind():
    """★ 回归: v1（谱范数型耦合）对位置结构**完全失明**，必须复现该盲区。

    原因: 逐列标准化把每列缩放到 ‖col‖=√n ⇒ 幅度信息被抹掉，
    植入的 t·direction 标准化后只剩 sign(t−7.5)，与距离无关。
    这条测试固化"为什么 v1 不能用"。
    """
    X = _planted()
    n, S, D = X.shape
    M = coupling_matrix(token_grams(X), n, topk=3)
    ds, prof = distance_profile(M)
    r = corr(ds, prof)
    spread = (max(prof) - min(prof)) / (sum(prof) / len(prof))
    assert abs(r) < 0.5, ("v1 应当失明", round(r, 3))
    assert spread < 0.02, ("v1 剖面应当近平", round(spread, 4))
    print(f"[PASS] v1 对植入结构失明: corr={r:+.3f}, 相对起伏={spread:.2%}（<2% 即近平）")
    return True


def test_v2_planted_locality_detected():
    """★ 灵敏度正控: v2 必须检出植入的位置结构。"""
    X = _planted()
    r = cos_profile(X, n_perm=20)
    assert r["verdict_P1"], ("应检出强衰减", r["corr"], r["profile"])
    assert r["verdict_P3"], ("局部性差应 >0.01", r["gap"])
    print(f"[PASS] v2 检出植入结构: corr={r['corr']:+.3f}, "
          f"M̄(1)−M̄(15)={r['gap']:.4f}")
    print(f"       M̄(d) = {r['profile']}")
    return True


def test_v2_pure_noise_no_locality():
    """★ P0: 纯噪声上无系统性衰减。

    ⚠️ 不能只看单个 seed: 15 个距离点上 corr 的抽样宽度约 ±0.3，
       实测 seed=0 单独就给出 0.67。故取 5 seed 均值，
       判据是**系统性偏移**而不是单次读数。
    """
    cs, ps = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        X = torch.randn(128, 12, 32)
        r = cos_profile(X, n_perm=20, seed=seed)
        cs.append(r["corr"])
        ps.append(int(r["verdict_P2"]))
    mean_c = sum(cs) / len(cs)
    assert abs(mean_c) < 0.3, ("纯噪声不应有系统性衰减", round(mean_c, 3), [round(c, 2) for c in cs])
    assert sum(ps) <= 1, ("纯噪声上 P2 假阳性过多", ps)
    print(f"[PASS] 纯噪声 5 seed: corr={[round(c,2) for c in cs]}, 均值={mean_c:+.3f}")
    print(f"       P2 假阳性 {sum(ps)}/5（≤1 可接受，阈值按 2.5% 分位本应约 0）")
    return True


def test_v2_permutation_separates():
    """★ P2: 打乱 token 顺序后衰减必须消失，真实值显著低于零分布。"""
    X = _planted()
    r = cos_profile(X, n_perm=40)
    assert r["verdict_P2"], ("真实应低于零分布下界", r["corr"], r["null_lo"])
    assert abs(r["null_mean"]) < 0.5, ("零分布应居中", r["null_mean"])
    print(f"[PASS] 置换可分: 真实={r['corr']:+.3f} < 下界={r['null_lo']:+.3f}, "
          f"零分布均值={r['null_mean']:+.3f}")
    return True


def test_common_cause_raises_intercept_and_keeps_sign():
    """★ 共同因的作用: 抬高**截距**；斜率**符号**保留。

    ⚠️ 关于斜率**幅度**，我先后猜错过两次，两次都被实测推翻:
       ① 猜"不改斜率"  → 实测 0.776→0.730（变了）
       ② 猜"压缩斜率"  → 实测 −0.933→−0.973（反而增强）
       推导后才明白: 共同因足够大时，分母被常数化，余弦变成
           cos ≈ 1 − c·(i−j)² / (2·scale²|base|²)
       即 **|i−j|² 的纯净抛物线**，衰减反而更规整。
       所以幅度效应**取决于位置项的形式**，不是普遍性质。
       ⇒ 本测试只断言两件确定的事: 截距抬高、符号保留。

    用同一组 base/direction/noise，只改 base 系数（否则两组不可比）。
    """
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    direction = torch.randn(1, 1, D)
    pos = torch.arange(S).view(1, S, 1).float()
    noise = torch.randn(n, S, D) * 0.02

    out = {}
    for scale, tag in ((0.1, "弱"), (5.0, "强")):
        X = base * scale + pos * direction * 0.05 + noise
        r = cos_profile(X, n_perm=10)
        out[tag] = r
        out[tag + "_mean"] = sum(r["profile"]) / len(r["profile"])

    weak, strong = out["弱"], out["强"]

    assert out["强_mean"] > out["弱_mean"], (
        "共同因应抬高截距", round(out["弱_mean"], 4), round(out["强_mean"], 4))
    assert weak["corr"] < 0 and strong["corr"] < 0, (
        "两档斜率符号都应为负", weak["corr"], strong["corr"])
    print(f"[PASS] 共同因: 截距 {out['弱_mean']:.4f}→{out['强_mean']:.4f}（抬高），"
          f"corr {weak['corr']:+.3f}→{strong['corr']:+.3f}（符号保留）")
    print("       幅度效应取决于位置项形式，不作断言（我猜错过两次）")
    return True


def test_no_common_cause_no_locality():
    """没有位置结构、只有共同因时，不应出现衰减（防假阳性）。"""
    torch.manual_seed(0)
    n, S, D = 192, 16, 32
    base = torch.randn(n, 1, D)
    X = base.repeat(1, S, 1) + torch.randn(n, S, D) * 0.5
    r = cos_profile(X, n_perm=20)
    assert abs(r["corr"]) < 0.5, ("不应出现衰减", r["corr"], r["profile"])
    print(f"[PASS] 只有共同因、无位置结构 → corr={r['corr']:+.3f}（近 0）")
    return True


def test_profile_shape_and_symmetry():
    """M 对称、对角为 1、剖面长度 S−1。"""
    torch.manual_seed(0)
    X = torch.randn(64, 9, 12)
    M = cos_matrix(X)
    assert M.shape == (9, 9)
    assert float((M - M.T).abs().max()) < 1e-6
    assert float((M.diagonal() - 1.0).abs().max()) < 1e-5
    ds, prof = distance_profile(M)
    assert ds == list(range(1, 9)) and len(prof) == 8
    print("[PASS] M 对称、对角=1、剖面长度 S−1=8")
    return True


def test_predictions_declared():
    from examples.run_v60_token_locality import PREDICTIONS
    assert set(PREDICTIONS) == {"P0", "P1", "P2", "P3"}
    print(f"[PASS] 四条预登记判决齐全: {list(PREDICTIONS)}")
    return True


TESTS = [
    test_v1_spectral_coupling_is_blind,
    test_v2_planted_locality_detected,
    test_v2_pure_noise_no_locality,
    test_v2_permutation_separates,
    test_common_cause_raises_intercept_and_keeps_sign,
    test_no_common_cause_no_locality,
    test_profile_shape_and_symmetry,
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
