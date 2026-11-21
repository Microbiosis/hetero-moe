"""V83.0 - S4-AP: Q-K 接收场正交对齐 + 源侧同层正交解耦管道 权威测试.

背景: v82 (S4-P: 裁决 S4PIPE_FAIL) 查出三处机制卡点:
  (1) P5a 管道连通未过 (arm1−arm3 = −0.000983, p = 0.9053; 逐样本 p = 1.0):
      原因在于 v82 的载体 Δ = (a_s @ P_Aᵀ) @ P_B 使用未对齐的独立随机正交矩阵, 注入方向与目标
      模型第 0 层注意力 Query-Key 接收锥正交;
  (2) P5d 选择性未过 (r2_o 增量 +0.001921, p = 0.0) 且 Qwen->BERT 槽特异未过 (p = 0.1324):
      原因在于 v77-v82 对 O 槽取 o_proj 输入 ("in") 而非输出 ("out"), 且样本级 mean-pool 使
      同层 Q/O 共享句子主成分;
  (3) P2′ 深度语义未过 (同层 +0.000399 < 跨层 +0.004495, 倒挂 11 倍):
      单点第 0 层注入下, Qwen 第 0 层 Value-Output 算子 M_vo 秩仅 128 且 Frobenius 范数仅 0.2862
      (比 BERT L0 小 43 倍、比 Qwen L1-L23 小 11-65 倍)。

本代针对 (1)(2) 做纯前向、训练集内闭式修复 (零测试集泄漏, 不训练任何底座权重), 在相同 64 文本块 ×
4 臂 × 5 种子网格与同一套六项判据上重测, 并锁定 (3) 的 M_vo 谱瓶颈诊断。

每个 test_* 必须在末尾 return True。
"""
import hashlib
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)
_root = _pkg
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

TITLE = "V83.0 S4-AP (Q-K 接收场正交对齐 + 源侧同层正交解耦管道)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
CALIB = os.path.join(_pkg, "results", "calib.json")
DIAG = os.path.join(_pkg, "results", "diag_mvo_readout.json")
SRC = os.path.join(_pkg, "s4_aligned_pipe.py")
V82_MANIFEST = os.path.join(_pkg, "..", "v82_s4_pipe", "MANIFEST.json")

L_Q = (0, 1, 2, 3, 6, 12, 18, 23)
L_B = (0, 1, 2, 3)
ARMS = ("arm0_pipe_zero", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
SEEDS = (0, 1, 2, 3, 4)
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0)


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _m():
    return _load(METRICS)


def _manifest():
    return _load(MANIFEST)


def _src():
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read()


def _sha256(path):
    with open(path, "rb") as f:
        b = f.read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        return hashlib.sha256(b).hexdigest()


def _expected_blocks():
    out = set()
    for l in L_Q:
        for j in L_B:
            out.add(("Qwen->BERT", l, j))
    for l in L_B:
        for j in L_Q:
            out.add(("BERT->Qwen", l, j))
    return out


# ---------------------------------------------------------------- 1
def test_block_grid_matches_protocol():
    """1. 块网格与协议一致: 64 块 (全文本) x 4 臂 x 5 种子 = 1280 行"""
    m = _m()
    rows = m["blocks"]
    got = set((r["direction"], r["source_layer"], r["target_layer"]) for r in rows)
    assert got == _expected_blocks(), "块集合与预期不符 (差 %d 个)" % len(got ^ _expected_blocks())
    assert len(rows) == 64 * 4 * 5, "行数不对: %s" % len(rows)
    assert set(r["arm"] for r in rows) == set(ARMS), "臂不对"
    assert set(r["seed"] for r in rows) == set(SEEDS), "种子不对"
    print("[PASS] test_block_grid_matches_protocol (64 块 x 4 臂 x 5 种子 = %d 行)" % len(rows))
    return True


# ---------------------------------------------------------------- 2
def test_thresholds_were_fixed_before_the_run():
    """2. 阈值与口径全部跑前写死 (静态扫源码)"""
    src = _src()
    for frag in ("SEEDS = (0, 1, 2, 3, 4)",
                 "N_PERM = 10000",
                 "ALPHA = 0.01",
                 "LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0)",
                 "CALIB_MAX_FINAL = 0.3",
                 "ORTHO_RIDGE_LAM = 10.0",
                 'SLOT_SOURCE_V83 = {"Q": "out", "O": "out"}',
                 "INJECT_LAYER = 0",
                 "ATT_LAYER = 0",
                 "K_PIPE = 8",
                 "TEXT_SPLIT_TRAIN = 64",
                 "DROP_FIRST_TOKEN = True",
                 '"p3_min_rel_perturbation": 1e-3',
                 '"p3_max_rel_perturbation": 0.5',
                 '"p5d_selectivity_p_min": 0.05'):
        assert frag in src, "源码里找不到预登记声明: %s" % frag
    th = _m()["protocol"]["thresholds"]
    assert th["alpha"] == 0.01 and th["n_perm"] == 10000 and th["n_seeds"] == 5
    assert th["slot_source_o"] == "out", "O 槽必须取模块真输出 out"
    print("[PASS] test_thresholds_were_fixed_before_the_run (预登记常量 + O 槽真输出声明齐备)")
    return True


# ---------------------------------------------------------------- 3
def test_procrustes_is_orthogonal_isometry_and_train_only():
    """3. 正交 Procrustes 矩阵严格正交 (等距: 保持范数与夹角), 且只在 train_idx 上求解"""
    from research.weight_fusion.v83_s4_aligned_pipe.s4_aligned_pipe import (
        fit_orthogonal_procrustes, orthogonalize_slot_pair,
    )
    torch.manual_seed(42)
    Z = torch.randn(128, 256)
    Y = torch.randn(128, 256)
    tr = list(range(64))
    omega = fit_orthogonal_procrustes(Z[tr], Y[tr])
    err = float((omega.t() @ omega - torch.eye(256)).norm())
    assert err < 1e-4, "Omega 不是正交矩阵: ||Omega^T Omega - I||_F = %g" % err
    # 改动测试集 (64..127) 不影响 Omega 与训练集正交残差化算子
    Z2 = Z.clone()
    Y2 = Y.clone()
    Z2[64:] += 100.0
    Y2[64:] -= 100.0
    omega2 = fit_orthogonal_procrustes(Z2[tr], Y2[tr])
    assert torch.allclose(omega, omega2, atol=1e-6), "Omega 泄漏了测试集信息"
    q1, o1 = orthogonalize_slot_pair(Z, Y, tr, lam=10.0)
    q2, o2 = orthogonalize_slot_pair(Z2, Y2, tr, lam=10.0)
    assert torch.allclose(q1[:64], q2[:64], atol=1e-6), "orthogonalize_slot_pair 泄漏了测试集信息"
    print("[PASS] test_procrustes_is_orthogonal_isometry_and_train_only (正交误差 %.2e, 零测试集泄漏)" % err)
    return True


# ---------------------------------------------------------------- 4
def test_p5a_and_p5b_pipe_connectivity_and_addressing_pass():
    """4. P5a (管道连通: arm1-arm0 且 arm1-arm3) 与 P5b (通道寻址 flip) 双双通过 (p < 0.01)"""
    m = _m()
    p5a = m["criteria"]["P5a"]
    p5b = m["criteria"]["P5b"]
    assert p5a["passed"] is True, "P5a 应当通过: %s" % p5a
    assert p5a["median_delta"] > 0 and p5a["permutation_p"] < 0.01, p5a
    cs = p5a["content_specific_vs_random"]
    assert cs["median_delta"] > 0 and cs["permutation_p"] < 0.01, cs
    assert p5b["passed"] is True, "P5b 应当通过: %s" % p5b
    assert p5b["median_delta"] > 0 and p5b["permutation_p"] < 0.01, p5b
    print("[PASS] test_p5a_and_p5b_pipe_connectivity_and_addressing_pass "
          "(P5a Δ=+%.4f/+%.4f p=%.4f; P5b Δ=+%.4f p=%.4f)"
          % (p5a["median_delta"], cs["median_delta"], p5a["permutation_p"],
             p5b["median_delta"], p5b["permutation_p"]))
    return True


# ---------------------------------------------------------------- 5
def test_p5c_penetration_and_p5d_selectivity_pass():
    """5. P5c (正文贯通 r2_q) 与 P5d (非目标槽 r2_o 选择性不普涨) 双双通过"""
    m = _m()
    p5c = m["criteria"]["P5c"]
    p5d = m["criteria"]["P5d"]
    assert p5c["passed"] is True, "P5c 应当通过: %s" % p5c
    assert p5c["median_delta"] > 0.02 and p5c["permutation_p"] < 0.01, p5c
    assert p5d["passed"] is True, "P5d 选择性应当通过 (非目标槽不再普涨): %s" % p5d
    assert p5d["median_delta"] <= 0 or p5d["permutation_p"] >= 0.05, p5d
    print("[PASS] test_p5c_penetration_and_p5d_selectivity_pass "
          "(P5c Δ=+%.5f p=%s; P5d Δ=%+.5f p=%.4f)"
          % (p5c["median_delta"], p5c["permutation_p"],
             p5d["median_delta"], p5d["permutation_p"]))
    return True


# ---------------------------------------------------------------- 6
def test_p3_1_guard_passes_and_five_of_six_criteria_pass():
    """6. P3① 守门通过 (960 行 0 违例), 六项主判据 5/6 通过, 仅剩 P2_prime (且同层中位 > 跨层中位)"""
    m = _m()
    c = m["criteria"]
    p3 = c["P3_1"]
    assert p3["passed"] is True and p3["n_violations"] == 0 and p3["n_rows"] == 960, p3
    p2 = c["P2_prime"]
    assert p2["median_same_layer"] > p2["median_cross_layer"], (
        "v82 的同层/跨层 11 倍倒挂应被纠正为同层中位 > 跨层中位: %s" % p2)
    assert m["verdict"] == "S4AP_FAIL", m["verdict"]
    assert m["verdict_detail"]["failed_criteria"] == ["P2_prime"], m["verdict_detail"]
    print("[PASS] test_p3_1_guard_passes_and_five_of_six_criteria_pass "
          "(5/6 通过, 同层 %.5f > 跨层 %.5f, 仅 P2_prime p=%.4f 未达 0.01)"
          % (p2["median_same_layer"], p2["median_cross_layer"], p2["permutation_p"]))
    return True


# ---------------------------------------------------------------- 7
def test_lambda_calibrated_with_12_of_12_groups_passing():
    """7. λ 标定选出 λ*=0.5, 且在该 λ 下全部 12/12 文本组同时满足连通与通道寻址"""
    c = _load(CALIB)
    assert c["verdict"] == "S4AP_LAMBDA_SELECTED", c["verdict"]
    assert c["selected_lambda"] == 0.5, c["selected_lambda"]
    cand05 = c["candidates"]["0.5"]
    assert cand05["feasible"] is True, cand05
    assert cand05["n_cond2_pass"] == 12 and cand05["n_cond3_pass"] == 12, cand05
    assert cand05["max_final"] <= 0.3, cand05
    m = _m()
    assert abs(m["protocol"]["constants"]["LAMBDA_INJECT"] - c["selected_lambda"]) < 1e-9
    print("[PASS] test_lambda_calibrated_with_12_of_12_groups_passing "
          "(λ*=0.5, 12/12 连通 + 12/12 寻址, max_final=%.4f)" % cand05["max_final"])
    return True


# ---------------------------------------------------------------- 8
def test_no_training_no_ability_metric():
    """8. 本代不训练任何底座权重 (无 backward / 无 optimizer), 不产生任何能力指标"""
    src = _src()
    for bad in (".backward(", "torch.optim", "requires_grad_(True)", "loss.backward"):
        assert bad not in src, "源码里出现了训练痕迹: %s" % bad
    m = _m()
    assert "不产生任何能力指标" in m["protocol"]["scope"]
    man = _manifest()
    assert man["leakage_audit"]["status"] == "clean"
    print("[PASS] test_no_training_no_ability_metric (纯前向闭式算子, 无 backward / optimizer)")
    return True


# ---------------------------------------------------------------- 9
def test_posthoc_mvo_and_bidirectional_readout_diagnostics():
    """9. post_hoc 诊断验证: 正文侧槽特异双向 p<1e-3, 留出集逐样本寻址正号率 >75%, Qwen L0 M_vo 秩=128 瓶颈在案"""
    assert os.path.exists(DIAG), "缺 post_hoc 诊断产物: %s" % DIAG
    d = _load(DIAG)
    assert "post_hoc_only" in d.get("role", "")
    slot_spec = d["readout_specificity"]["arm1_minus_arm2_slot_specificity"]
    assert slot_spec["permutation_p"] <= 1e-3, slot_spec
    for dir_name in ("Qwen->BERT", "BERT->Qwen"):
        bd = slot_spec["by_direction"][dir_name]
        assert bd["median_block_delta"] > 0 and bd["permutation_p"] <= 1e-3, (dir_name, bd)
    ps = d["per_sample_pipe_attention"]["flip_arm1_minus_arm2_per_sample"]
    p_sig = ps.get("sign_test_p_naive_iid", ps.get("sign_test_p", 1.0))
    assert ps["positive_fraction"] > 0.75 and p_sig < 1e-10, ps
    mvo = d["mvo_spectral_bottleneck"]
    assert mvo["Qwen"]["0"]["rank_gt_1e5"] == 128, mvo["Qwen"]["0"]
    assert mvo["Qwen"]["0"]["frobenius_norm"] < 0.35, mvo["Qwen"]["0"]
    assert mvo["BERT"]["0"]["frobenius_norm"] > 5.0, mvo["BERT"]["0"]
    print("[PASS] test_posthoc_mvo_and_bidirectional_readout_diagnostics "
          "(双向槽特异 p=%.4f, 留出集寻址正号率 %.1f%%, Qwen L0 M_vo rank=128 fro=%.4f)"
          % (slot_spec["permutation_p"], ps["positive_fraction"] * 100, mvo["Qwen"]["0"]["frobenius_norm"]))
    return True


# ---------------------------------------------------------------- 10
def test_provenance_sha256_and_lineage_consistent():
    """10. 复用件 sha256 与快照一致, 且谱系 82.0 -> 83.0 双向自洽"""
    m = _m()
    p = m["protocol"]["provenance_sha256"]
    assert _sha256(SRC) == p["v83_s4_aligned_pipe"]
    man = _manifest()
    assert man["version"] == "83.0" and man["generation"]["parent"] == "82.0"
    v82 = _load(V82_MANIFEST)
    assert "83.0" in v82["generation"]["children"]
    print("[PASS] test_provenance_sha256_and_lineage_consistent (82.0 -> 83.0 双向自洽)")
    return True


TESTS = [
    test_block_grid_matches_protocol,
    test_thresholds_were_fixed_before_the_run,
    test_procrustes_is_orthogonal_isometry_and_train_only,
    test_p5a_and_p5b_pipe_connectivity_and_addressing_pass,
    test_p5c_penetration_and_p5d_selectivity_pass,
    test_p3_1_guard_passes_and_five_of_six_criteria_pass,
    test_lambda_calibrated_with_12_of_12_groups_passing,
    test_no_training_no_ability_metric,
    test_posthoc_mvo_and_bidirectional_readout_diagnostics,
    test_provenance_sha256_and_lineage_consistent,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
