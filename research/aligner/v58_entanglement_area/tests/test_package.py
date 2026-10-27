"""V58.0 膜 / 面积律算子 — 单元测试 (11 项)。不加载模型。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
from research._primitives.testing import run_suite

TITLE = "V58.0 膜 / 面积律算子 — 单元测试 (11 项)"

import torch
from research.aligner.v58_entanglement_area.brane import (
    cross_spectrum, noise_tau, entanglement_entropy, capacity_needed,
    make_lattice, scan_cuts, BraneSpace, PREDICTIONS,
)

N, DW, L = 512, 8, 12


def _chain(chi_c, seed=0):
    torch.manual_seed(seed)
    lat = torch.zeros(N, L, DW); prev = None
    for i in range(L):
        z = torch.randn(N, DW)
        if prev is not None and chi_c:
            z[:, :chi_c] = prev[:, :chi_c] + 0.1 * torch.randn(N, chi_c)
        lat[:, i, :] = z; prev = z
    return lat


def test_independent_blocks_give_zero():
    """独立的两块: χ 必须 0, 熵必须 ≈0。

    ★ 回归: 朴素 CCA 在 n=512, d=32/48 上给出 29–31 个"显著"通道
      —— 白化在 n 与 d 同量级时把样本噪声白成了信号。必须用置换零分布扣掉。
    """
    torch.manual_seed(0)
    A, B = torch.randn(N, 32), torch.randn(N, 48)
    chi = capacity_needed(A, B)
    S = entanglement_entropy(A, B)
    assert chi == 0, chi
    assert S < 0.02, S
    print(f"[PASS] 独立块 χ={chi} S={S:.4f} (朴素 CCA 会给 29+)")
    return True


def test_shared_dims_recovered_exactly():
    """共享 k 个方向 → χ 必须正好是 k。"""
    torch.manual_seed(0)
    for k in (1, 2, 5, 12):
        Z = torch.randn(N, k)
        A = torch.randn(N, 32) + Z @ torch.randn(k, 32)
        B = torch.randn(N, 48) + Z @ torch.randn(k, 48)
        got = capacity_needed(A, B)
        assert got == k, (k, got)
    print("[PASS] 共享维 k∈{1,2,5,12} 全部精确还原")
    return True


def test_area_law_saturates():
    """面积律: 相邻 site 定宽耦合 → χ 剖面恒定, corr(|A|,χ)≈0。"""
    for c in (3, 6):
        bs = BraneSpace(); bs.measure(_chain(c), axis=1)
        chis = [p["chi"] for p in bs.profile]
        assert all(x == c for x in chis), chis
        assert abs(bs.area_law_ratio()) < 0.2
    print("[PASS] 面积律: χ 剖面恒定, corr≈0")
    return True


def test_volume_law_detected():
    """体积律: 跨切通道数随块大小增长 → corr(|A|, χ) 明显 >0。

    ★ 构造要点 (第一版写错过): "所有 site 共享同一组全局方向" 给出的是
      **恒定** χ —— 因为任何切口的跨协方差都只经过那组全局方向, 与块大小无关。
      真正的体积律要让**共享方向的个数随位置增长**: 本测试令 site i 共享前 i 个方向。
    """
    torch.manual_seed(0)
    Lv, dwt = 12, 8
    Z = torch.randn(N, Lv)                       # 全局方向池
    lat = torch.zeros(N, Lv, dwt)
    for i in range(Lv):
        k_i = i                              # site i 与之前共享前 i 个方向
        lat[:, i, :min(k_i, dwt)] = (Z[:, :min(k_i, dwt)] * 2.0
                                     + 0.2 * torch.randn(N, min(k_i, dwt)))
        if k_i < dwt:
            lat[:, i, k_i:] = torch.randn(N, dwt - k_i)
    bs = BraneSpace(); bs.measure(lat, axis=1)
    r = bs.area_law_ratio()
    assert r > 0.2, (r, [p["chi"] for p in bs.profile])
    print(f"[PASS] 体积律可分辨: corr={r:+.2f}, χ={[p['chi'] for p in bs.profile]}")
    return True


def test_nonuniform_detected():
    """★ B4 证伪点: 单热点结构必须被读成非均匀, 否则膜退化成 v57 的标量 r。"""
    torch.manual_seed(0)
    lat = torch.zeros(N, L, DW); prev = None
    for i in range(L):
        z = torch.randn(N, DW)
        if prev is not None:
            c = 7 if i == 5 else (2 if i in (4, 6) else 0)
            if c: z[:, :c] = prev[:, :c] + 0.1 * torch.randn(N, c)
        lat[:, i, :] = z; prev = z
    bs = BraneSpace(); bs.measure(lat, axis=1)
    chis = [p["chi"] for p in bs.profile]
    assert not bs.is_uniform, chis
    assert max(chis) >= 6 and min(chis) == 0, chis
    print(f"[PASS] 单热点被读出: {chis} (非均匀)")
    return True


def test_uniform_equals_v57():
    """退化检验: 剖面恒定时 BraneSpace ≡ v57 的标量 r。"""
    bs = BraneSpace(); bs.measure(_chain(4), axis=1)
    assert bs.is_uniform
    print(f"[PASS] 恒定剖面 is_uniform=True, 总容量={bs.total_capacity} = χ×(L-1)")
    return True


def test_make_lattice_order_matters():
    """natural 与 random 必须是不同的排列 (否则 B3 的对照无意义)。"""
    torch.manual_seed(0)
    X = torch.randn(64, 96)
    a = make_lattice(X, (4, 6), order="natural")
    b = make_lattice(X, (4, 6), order="random", seed=1)
    assert a.shape == b.shape == (64, 4, 6, 4)
    assert not torch.allclose(a, b)
    print(f"[PASS] natural vs random 不同排列, 形状 {tuple(a.shape)}")
    return True


def test_lattice_shape_validation():
    """不能整分时必须报错, 不静默错。"""
    X = torch.randn(8, 100)
    try:
        make_lattice(X, (7, 7))
    except AssertionError:
        print("[PASS] 100 不能被 7×7 整除 → 报错")
        return True
    raise AssertionError("应当报错")


def test_spectrum_and_tau():
    """互协方差谱降序非负; 阈值随 max(da,db) 单调增; 独立时 τ 高于谱最大值。"""
    torch.manual_seed(0)
    A, B = torch.randn(N, 32), torch.randn(N, 48)
    spec = cross_spectrum(A, B)
    assert float(spec.min()) >= 0
    assert torch.all(spec[:-1] >= spec[1:] - 1e-7)
    assert noise_tau(32, 48, N) < noise_tau(32, 96, N)
    assert noise_tau(32, 48, N) > float(spec.max())
    print(f"[PASS] 谱降序; τ(32x48)={noise_tau(32,48,N):.3f} > s_max={float(spec.max()):.3f}")
    return True


def test_entropy_monotone_in_coupling():
    """熵必须随耦合强度单调增 (RT 公式的基本性质)。"""
    torch.manual_seed(0)
    Z = torch.randn(N, 6); ents = []
    for eps in (0.05, 0.3, 0.9):
        A = torch.randn(N, 24) + Z @ torch.randn(6, 24) * eps
        B = torch.randn(N, 24) + Z @ torch.randn(6, 24) * eps
        ents.append(entanglement_entropy(A, B))
    assert ents[0] < ents[1] < ents[2], ents
    print(f"[PASS] 熵随耦合单调增: {ents[0]:.2f} < {ents[1]:.2f} < {ents[2]:.2f}")
    return True


def test_predictions_declared():
    assert set(PREDICTIONS) == {"B1", "B2", "B3", "B4"}
    print(f"[PASS] 四条预登记判决齐全: {list(PREDICTIONS)}")
    return True


TESTS = [
    test_independent_blocks_give_zero,
    test_shared_dims_recovered_exactly,
    test_area_law_saturates,
    test_volume_law_detected,
    test_nonuniform_detected,
    test_uniform_equals_v57,
    test_make_lattice_order_matters,
    test_lattice_shape_validation,
    test_spectrum_and_tau,
    test_entropy_monotone_in_coupling,
    test_predictions_declared,
]

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
