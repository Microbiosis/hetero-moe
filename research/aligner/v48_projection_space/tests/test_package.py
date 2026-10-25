"""V48.0 投影空间 — 单元测试 (10 项)。

不依赖 transformers / 不加载真实模型: 用随机张量与一个"形状与 v33 融合层
一致"的替身对象, 验证三件事:
  1) 空间诊断的数学量算得对 (谱/有效维数/典型相关/足迹重叠)
  2) 诊断函数不会在 p >> n 时被伪零特征值骗 (v48 修过的 bug)
  3) 三个臂的**参数量完全相同**, 只差"是否被训练" (v41 容量混杂的排除)
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V48.0 投影空间 — 单元测试 (11 项)"

import numpy as np
import torch
import torch.nn as nn

from research.aligner.v48_projection_space.space import (
    spectrum_report, union_report, k_grid_report, pairwise_canonical_correlations,
    top_canonical_correlation, center, CONCAT_REFERENCE,
)
from research.aligner.v48_projection_space.mechanism import ARMS, configure_arm


def _experts(n_expert=3, n_tok=24, dims=(64, 32, 32), seed=0):
    """构造 n_expert 个 (n_tok, D_m) 表征, 彼此几乎不相关 (随机独立基)。"""
    rng = np.random.default_rng(seed)
    out = []
    for d in dims:
        X = rng.standard_normal((n_tok, d))
        out.append(X + 0.05 * rng.standard_normal((n_tok, d)))   # 轻微共享结构
    return out


def test_spectrum_report_has_required_keys():
    r = spectrum_report(np.random.default_rng(0).standard_normal((40, 200)))
    for k in ("eigvals", "noise_floor", "k_aff", "k_1pct", "k_5pct", "elbow", "eff_rank"):
        assert k in r, k
    print("[PASS] spectrum_report 返回全部必需字段")
    return True


def test_spectrum_no_pseudo_zero_bug():
    """p >> n 时 K_aff 不得被数值零特征值撑大 (v48 修过的 bug)。"""
    r = spectrum_report(np.random.default_rng(1).standard_normal((20, 500)))
    assert r["k_aff"] <= 20, f"K_aff={r['k_aff']} 超过 token 行数 → 伪零特征值 bug 未修"
    assert r["k_1pct"] <= 20
    print(f"[PASS] p>>n 时维数估计不越界 (K_aff={r['k_aff']}, 99%维数={r['k_1pct']})")
    return True


def test_spectrum_rank_equals_true_rank():
    """低秩数据 (秩 3) 的 99% 维数应接近 3。"""
    rng = np.random.default_rng(2)
    B = rng.standard_normal((50, 3))
    X = B @ rng.standard_normal((3, 120))
    r = spectrum_report(X)
    assert r["k_1pct"] <= 5, f"99% 维数={r['k_1pct']} 明显高于真秩 3"
    print(f"[PASS] 低秩数据的有效维数估计正确 (99%维数={r['k_1pct']}, 期望≈3)")
    return True


def test_canonical_correlation_identical_is_one():
    A = np.random.default_rng(3).standard_normal((30, 10))
    cc = top_canonical_correlation(A, A)
    assert cc > 0.999, f"同一矩阵的 CC1 应为 1, 实得 {cc}"
    print("[PASS] 相同输入的典型相关为 1")
    return True


def test_cka_null_control():
    """置换零分布必须能区分"真共享"与"只是噪声" (CKA 为主指标)。"""
    A = np.random.default_rng(4).standard_normal((40, 20))
    B = A + 0.3 * np.random.default_rng(5).standard_normal((40, 20))   # 真共享
    C = np.random.default_rng(6).standard_normal((40, 20))             # 纯噪声
    res = pairwise_canonical_correlations([A, B], n_null=30)
    res2 = pairwise_canonical_correlations([A, C], n_null=30)
    assert res[0]["cka"] > res[0]["cka_null_max"], "真共享对未超过零分布上限"
    assert res2[0]["cka"] <= res2[0]["cka_null_max"], "纯噪声对不应超过零分布上限"
    print(f"[PASS] CKA 置换零分布能区分真共享 ({res[0]['cka']:.3f} > {res[0]['cka_null_max']:.3f}) "
          f"与噪声 ({res2[0]['cka']:.3f} ≤ {res2[0]['cka_null_max']:.3f})")
    return True


def test_cka_is_basis_invariant():
    """CKA 对正交变换不变 —— 这正是它比典型相关更适合的原因。"""
    from research.aligner.v48_projection_space.space import linear_cka
    rng = np.random.default_rng(7)
    A = rng.standard_normal((30, 12))
    Q = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    c1 = linear_cka(A, A)
    c2 = linear_cka(A, A @ Q)
    assert abs(c1 - c2) < 1e-8, (c1, c2)
    assert abs(c1 - 1.0) < 1e-8, c1
    print("[PASS] CKA 对正交变换不变, 且自身为 1")
    return True


def test_k_grid_fidelity_monotonic_in_K():
    H = _experts()
    kg = k_grid_report(H, [4, 8, 16, 32])
    fids = [r["fidelity_min"] for r in kg]
    assert all(fids[i] <= fids[i + 1] + 1e-9 for i in range(len(fids) - 1)), fids
    assert 0.0 <= fids[0] <= 1.0 and 0.0 <= fids[-1] <= 1.0
    print(f"[PASS] K 网格的保真度单调不减且落在 [0,1]: "
          + " ".join(f"{f:.3f}" for f in fids))
    return True


def test_k_grid_overlap_in_unit_range():
    H = _experts()
    kg = k_grid_report(H, [8, 16])
    for r in kg:
        assert 0.0 <= r["overlap"] <= 1.0, r
        assert 0.0 <= r["redundancy"] <= 1.0 + 1e-9, r
    print("[PASS] 足迹重叠度与冗余度都落在 [0,1]")
    return True


def test_union_report_block_dims():
    H = _experts()
    rep = union_report(H, ["a", "b", "c"])
    assert rep["block_dims"] == [64, 32, 32], rep["block_dims"]
    assert set(rep["per_expert"]) == {"a", "b", "c"}
    print("[PASS] 并集报告给出正确的分块维度")
    return True


class _FakeLayer(nn.Module):
    """与 v33 CrossModalFusionLayer 同构的替身 (只测 requires_grad 语义)。"""

    def __init__(self, d_shared=16, d_ff=32, modal_dims=(64, 32, 32)):
        super().__init__()
        self.P_m = nn.ParameterList([nn.Parameter(torch.empty(d_shared, D)) for D in modal_dims])
        for p in self.P_m:
            nn.init.orthogonal_(p)
        self.w_gates = nn.ParameterList([nn.Parameter(torch.zeros(d_ff, d_shared)) for _ in modal_dims])
        self.w_ups = nn.ParameterList([nn.Parameter(torch.zeros(d_ff, d_shared)) for _ in modal_dims])
        self.w_downs = nn.ParameterList([nn.Parameter(torch.zeros(d_shared, d_ff)) for _ in modal_dims])
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(d_shared)) for _ in modal_dims])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(d_shared)) for _ in modal_dims])
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in modal_dims])
        self.W_router = nn.Parameter(torch.zeros(len(modal_dims), d_shared))


def test_arms_share_total_params():
    """三臂参数量必须完全相同 —— 差异只能来自"是否被训练" (排除容量混杂)。"""
    totals, trains = {}, {}
    for arm in ARMS:
        layer = _FakeLayer()
        params, info = configure_arm(layer, arm)
        totals[arm] = sum(p.numel() for p in layer.parameters())
        trains[arm] = sum(p.numel() for p in params)
    assert len(set(totals.values())) == 1, totals
    assert trains["A_frozen"] < trains["B_train_proj"] < trains["C_train_all"], trains
    print(f"[PASS] 三臂总参数相同 ({list(totals.values())[0]}), "
          f"可训练参数 {trains['A_frozen']} < {trains['B_train_proj']} < {trains['C_train_all']}")
    return True


def test_frozen_arm_leaves_projection_untrained():
    layer = _FakeLayer()
    params, _ = configure_arm(layer, "A_frozen")
    assert all(not p.requires_grad for p in layer.P_m)
    assert all(not p.requires_grad for p in layer.w_gates)
    layer2 = _FakeLayer()
    params2, _ = configure_arm(layer2, "B_train_proj")
    assert all(p.requires_grad for p in layer2.P_m)
    assert all(not p.requires_grad for p in layer2.w_gates)   # B 不动专家
    print("[PASS] A 臂冻结投影/专家, B 臂只解冻投影")
    return True


TESTS = [
    test_spectrum_report_has_required_keys,
    test_spectrum_no_pseudo_zero_bug,
    test_spectrum_rank_equals_true_rank,
    test_canonical_correlation_identical_is_one,
    test_cka_null_control,
    test_cka_is_basis_invariant,
    test_k_grid_fidelity_monotonic_in_K,
    test_k_grid_overlap_in_unit_range,
    test_union_report_block_dims,
    test_arms_share_total_params,
    test_frozen_arm_leaves_projection_untrained,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
