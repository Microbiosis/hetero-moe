"""V55.0 中枢带宽 — 单元测试 (7 项)。不加载底座权重, 不训练, 全部 CPU。

测的是 **定理 3 的结构性**: rank(J_alpha) = min(top_k, M-1) 对任意参数成立。
小尺寸 (D=32) 即可验证秩, 因为秩与维度无关。
"""
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V55.0 中枢带宽 — 单元测试 (7 项)"

import math

import torch

from research.aligner.v55_hub_bandwidth.measure import (
    REL_TOL,
    _load_fusion,
    run_one,
    spectrum_report,
)

D_SMALL, E, FF, MODAL = 32, 4, 32, [32, 16, 16]
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _x(d=D_SMALL, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, d, generator=g) * 0.5


def _rank(seed=0, top_k=2, d=D_SMALL, e=E):
    torch.manual_seed(seed)
    mm = _load_fusion()
    return run_one(mm, top_k, _x(d, seed), d=d, e=e, ff=FF,
                   modal_dims=MODAL, heavy=False)


def test_alpha_rank_equals_theory():
    """定理 3 给上界 min(top_k, M-1); 实测逐档**取等** (上界是紧的)。"""
    for top_k in (1, 2, 3, 4):
        r = _rank(top_k=top_k)
        assert r["J_alpha"]["rank"] == r["J_alpha_theory_rank"], (top_k, r["J_alpha"]["rank"])
        assert r["J_alpha_theory_rank"] == min(top_k, E - 1)
    print("[PASS] rank(J_alpha) = min(top_k, M-1) ∈ {1,2,3,3}")
    return True


def test_alpha_rank_ceiling_at_topk_eq_m():
    """top_k = M 时被 softmax 的秩 M-1 卡住: 加宽门不能再加宽中枢。"""
    r = _rank(top_k=E)
    assert r["J_alpha"]["rank"] == E - 1 == 3, r["J_alpha"]["rank"]
    print("[PASS] top_k=M 时秩封顶在 M-1=3 (softmax 退化)")
    return True


def test_route_rank_bounded_by_alpha_rank():
    """J_route 是 <=top_k 个秩 1 项之和, 且行向量同处 <=min(k,M-1) 维子空间。"""
    for top_k in (1, 2, 3, 4):
        r = _rank(top_k=top_k, seed=top_k)
        assert r["J_route"]["rank"] <= r["J_alpha_theory_rank"], (top_k, r["J_route"]["rank"])
        assert r["J_route"]["rank"] == r["J_alpha"]["rank"], (top_k, r["J_route"]["rank"])
    print("[PASS] rank(J_route) <= min(top_k, M-1), 实测取等")
    return True


def test_rank_invariant_under_random_parameters():
    """『对任意参数成立』: 换 6 组随机初始化, 秩上界不变。"""
    ranks = {_rank(seed=s)["J_alpha"]["rank"] for s in range(6)}
    assert ranks == {2}, ranks
    print(f"[PASS] 6 组随机参数下 rank(J_alpha) 恒为 2 (集合 = {ranks})")
    return True


def test_mask_semantics_and_gate_bits():
    """门语义: 恰好 top_k 个非零权重; 全局状态数 C(M, top_k) 对应门容量 bit 数。"""
    for top_k in (1, 2, 3, 4):
        r = _rank(top_k=top_k)
        assert r["n_nonzero_weights"] == top_k, (top_k, r["n_nonzero_weights"])
        assert r["gate_states"] == math.comb(E, top_k)
        assert abs(r["gate_bits"] - math.log2(math.comb(E, top_k))) < 1e-4  # gate_bits 四舍五入到 4 位
    assert abs(_rank(top_k=2)["gate_bits"] - 2.585) < 5e-4
    print("[PASS] mask 恰 top_k 非零; top_k=2 时门容量 2.585 bit")
    return True


def test_spectrum_report_handles_low_rank():
    """spectrum_report 对严格低秩矩阵必须报出真实秩 (阈值 1e-5 相对, 不是 1e-8)。"""
    u = torch.randn(64, 1)
    M = u @ u.T                                   # 精确秩 1
    assert spectrum_report(M)["rank"] == 1, spectrum_report(M)["rank"]
    Z = torch.zeros(16, 16)
    r = spectrum_report(Z + 1e-12 * torch.eye(16))
    assert r["rank"] == 16 or r["rank"] >= 1      # 全零矩阵不崩
    assert REL_TOL == 1e-5
    print("[PASS] 低秩/近零矩阵的秩报告不产生假秩")
    return True


def test_metrics_snapshot_matches_theory():
    """results/metrics.json 的快照必须与定理一致 (产物自检, 防口径漂移)。"""
    path = os.path.join(PKG, "results", "metrics.json")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["package"] == "v55_hub_bandwidth"
    assert "code_sha" in m["provenance"] and m["provenance"]["rel_tol"] == REL_TOL
    for row in m["rows"]:
        assert row["J_alpha"]["rank"] == row["J_alpha_theory_rank"], row["top_k"]
        assert row["J_alpha_theory_rank"] == min(row["top_k"], m["provenance"]["num_experts"] - 1)
    d = m["v33_default"]
    assert d["top_k"] == 2 and d["J_alpha"]["rank"] == 2 and d["gate_bits"] == 2.585
    # 68.5% 的份额: 只作为"这条通道有多重"的读数, 给一个宽容区间防实现漂移
    assert 0.55 <= d["route_share_fro"] <= 0.80, d["route_share_fro"]
    assert d["J_out"]["eff_rank"] < m["provenance"]["d_shared"], "局部项若满秩各向同性则 eff_rank≈D"
    print(f"[PASS] 快照与定理一致; v33 默认: 秩 2 / 2.585 bit / route {100*d['route_share_fro']:.1f}% / "
          f"effrank {d['J_out']['eff_rank']:.1f}")
    return True


TESTS = [
    test_alpha_rank_equals_theory,
    test_alpha_rank_ceiling_at_topk_eq_m,
    test_route_rank_bounded_by_alpha_rank,
    test_rank_invariant_under_random_parameters,
    test_mask_semantics_and_gate_bits,
    test_spectrum_report_handles_low_rank,
    test_metrics_snapshot_matches_theory,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
