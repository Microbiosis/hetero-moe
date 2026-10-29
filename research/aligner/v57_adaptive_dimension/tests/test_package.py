"""V57.0 自适应维度算子 — 单元测试 (10 项)。不加载任何模型。"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V57.0 自适应维度算子 — 单元测试 (10 项)"

import math

import torch

from research.aligner.v57_adaptive_dimension.operators import (
    mp_sqrt_median, omega, noise_sigma, optimal_hard_threshold,
    SpectralProject, AdaptiveSubspace,
)


def test_mp_median_range():
    """c_β 必须单调递减且在 (0.7, 1.05) 内 —— 解析式在 β=1 会塌到 0, 这是回归点。"""
    vals = {b: mp_sqrt_median(b) for b in (0.1, 0.25, 0.5, 0.75, 1.0)}
    assert all(0.70 < v < 1.05 for v in vals.values()), vals
    seq = [vals[b] for b in (0.1, 0.25, 0.5, 0.75, 1.0)]
    assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), seq
    print(f"[PASS] c_β 单调递减且落在 (0.70,1.05): {vals[1.0]:.4f} … {vals[0.1]:.4f}")
    return True


def test_noise_sigma_recovers_true_sigma():
    """σ̂ 必须还原真实噪声水平 (纯噪声矩阵)。"""
    torch.manual_seed(0)
    for sig in (0.5, 1.0, 3.0):
        Z = torch.randn(512, 512) * sig
        s = torch.linalg.svdvals(Z)
        est = noise_sigma(s, 512, 512)
        assert abs(est / sig - 1.0) < 0.08, (sig, est)
    print("[PASS] σ̂ 在 σ∈{0.5,1,3} 上还原到 8% 内")
    return True


def test_threshold_above_noise_edge():
    """τ 必须高于噪声谱上边缘 —— 否则纯噪声会被误判为有信号。"""
    torch.manual_seed(0)
    for (n, d) in ((512, 512), (512, 2048), (256, 512)):
        Z = torch.randn(n, d)
        s = torch.linalg.svdvals(Z)
        tau = optimal_hard_threshold(s, n, d)
        assert tau > float(s.max()), (n, d, tau, float(s.max()))
    print("[PASS] τ 高于噪声谱上边缘 (三组形状)")
    return True


def test_rank_exact_on_known_signal():
    """已知秩 k 的信号 + 噪声 → 估出的秩必须正好是 k。"""
    torch.manual_seed(0)
    for k in (0, 3, 8, 20):
        U = torch.randn(512, max(k, 1)); V = torch.randn(max(k, 1), 512)
        X = (U @ V) * 4.0 if k else torch.zeros(512, 512)
        Z = X + torch.randn(512, 512)
        s = torch.linalg.svdvals(Z)
        tau = optimal_hard_threshold(s, 512, 512)
        got = int((s > tau).sum())
        assert got == k, (k, got)
    print("[PASS] 真实秩 {0,3,8,20} 全部精确估出")
    return True


def test_omega_formula():
    """ω(β) 的端点值 (论文给的 1.43 / 2.86)。"""
    assert abs(omega(0.0) - 1.43) < 1e-9
    assert abs(omega(1.0) - 2.86) < 1e-9
    print(f"[PASS] ω(0)={omega(0):.2f}  ω(1)={omega(1):.2f}")
    return True


def test_projection_is_idempotent_and_grad_flows():
    """投影算子: P²=P, 且反向梯度走子空间内 (直通)。"""
    torch.manual_seed(0)
    d = 64
    V = torch.linalg.qr(torch.randn(d, 8))[0]
    Z = torch.randn(16, d, requires_grad=True)
    PZ = SpectralProject(Z, V)
    PPZ = SpectralProject(PZ.detach(), V)
    assert torch.allclose(PZ, PPZ, atol=1e-6)          # 幂等
    # 子空间外的分量被抹掉
    resid = SpectralProject(Z.detach() - PZ.detach(), V)
    assert float(resid.abs().max()) < 1e-5
    PZ.sum().backward()
    assert Z.grad is not None and float(Z.grad.abs().sum()) > 0
    print(f"[PASS] 投影幂等 + 子空间外残差 {float(resid.abs().max()):.1e} + 梯度非零")
    return True


def test_growth_on_fixed_basis():
    """固定基: 秩 3 数据 → r=3 (不多不少)。"""
    torch.manual_seed(0)
    d = 512
    B = torch.linalg.qr(torch.randn(d, 32))[0]
    sp = AdaptiveSubspace(d)
    for _ in range(8):
        coef = torch.randn(128, 3) * 6.0
        sp(coef @ B[:, :3].T + torch.randn(128, d), update=True)
    assert sp.r == 3, sp.r
    print(f"[PASS] 固定秩-3 数据流 → r={sp.r}")
    return True


def test_expansion_when_new_direction_appears():
    """出现新方向必须扩张。"""
    torch.manual_seed(0)
    d = 512
    B = torch.linalg.qr(torch.randn(d, 32))[0]
    sp = AdaptiveSubspace(d)
    for _ in range(6):
        sp(torch.randn(128, 3) * 6.0 @ B[:, :3].T + torch.randn(128, d), update=True)
    r0 = sp.r
    for _ in range(6):
        sp(torch.randn(128, 8) * 6.0 @ B[:, :8].T + torch.randn(128, d), update=True)
    assert sp.r > r0, (r0, sp.r)
    assert sp.log.born[-1] >= 0 and sum(sp.log.born) > 3
    print(f"[PASS] 新方向出现 → r {r0} → {sp.r} (born 合计 {sum(sp.log.born)})")
    return True


def test_collapse_when_signal_disappears():
    """信号消失后方向必须死亡 —— 只会扩张不会坍缩的空间等于固定维度。"""
    torch.manual_seed(0)
    d = 512
    B = torch.linalg.qr(torch.randn(d, 32))[0]
    sp = AdaptiveSubspace(d, death_window=2)
    for _ in range(6):
        sp(torch.randn(128, 8) * 6.0 @ B[:, :8].T + torch.randn(128, d), update=True)
    r_peak = sp.r
    for _ in range(10):
        sp(torch.randn(128, d), update=True)          # 纯噪声
    assert sp.r < r_peak, (r_peak, sp.r)
    assert sum(sp.log.died) > 0
    print(f"[PASS] 信号消失 → r {r_peak} → {sp.r} (died 合计 {sum(sp.log.died)})")
    return True


def test_coords_dimension_matches_r():
    """前向返回的坐标宽度必须等于 r —— 下游永远不设定维度。"""
    torch.manual_seed(0)
    d = 256
    sp = AdaptiveSubspace(d)
    for _ in range(5):
        Z = torch.randn(64, d)
        out = sp(Z, update=True)
        assert out.shape == (64, sp.r), (out.shape, sp.r)
    print(f"[PASS] 坐标宽度恒等于 r (末态 r={sp.r})")
    return True


TESTS = [
    test_mp_median_range,
    test_noise_sigma_recovers_true_sigma,
    test_threshold_above_noise_edge,
    test_rank_exact_on_known_signal,
    test_omega_formula,
    test_projection_is_idempotent_and_grad_flows,
    test_growth_on_fixed_basis,
    test_expansion_when_new_direction_appears,
    test_collapse_when_signal_disappears,
    test_coords_dimension_matches_r,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
