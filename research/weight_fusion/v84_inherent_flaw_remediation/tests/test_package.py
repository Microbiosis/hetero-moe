"""V84.0 — 全历史固有缺陷审计与 hetero_moe v5.0 修复 权威测试 (10 项)。"""
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

TITLE = "V84.0 全历史固有缺陷审计与 hetero_moe v5.0 系统修复"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "audit_remediation.py")
V83_MANIFEST = os.path.join(_pkg, "..", "v83_s4_aligned_pipe", "MANIFEST.json")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_overall_verdict_is_remediation_pass():
    """1. 总裁决必须为 REMEDIATION_PASS"""
    m = _load(METRICS)
    assert m["verdict"] == "REMEDIATION_PASS", m["verdict"]
    print("[PASS] test_overall_verdict_is_remediation_pass (REMEDIATION_PASS)")
    return True


def test_exp1_v62_random_matrix_artifact_reproduced_and_fixed():
    """2. Exp1: 纯高斯随机矩阵复现 v62 CKA/fidelity 假象，正交 Procrustes 保满秩且无虚高"""
    e1 = _load(METRICS)["exp1_v62_artifact_and_fix"]
    assert e1["artifact_reproduced"] is True, e1
    assert e1["orthogonal_fix_verified"] is True, e1
    assert e1["unconstrained_2_random_W"]["rank64"]["cka"] > 0.95
    assert e1["unconstrained_3_random_W"]["rank64"]["cka"] < 0.80
    assert e1["unconstrained_2_random_W"]["rank64"]["fidelity"] > 0.99
    assert e1["unconstrained_2_random_W"]["rank64"]["eff_rank_A"] < 12.0
    assert e1["cka_inflation_by_rank_collapse"] > 0.35
    assert abs(e1["orthogonal_procrustes_2_random_W"]["eff_rank_Q"] - e1["ds"]) < 1e-2
    print("[PASS] test_exp1_v62_random_matrix_artifact_reproduced_and_fixed (2-rand=%.4f, inflation=+%.4f)"
          % (e1["unconstrained_2_random_W"]["rank64"]["cka"], e1["cka_inflation_by_rank_collapse"]))
    return True


def test_exp2_v83_circular_selectivity_and_identifiability_exposed():
    """3. Exp2: 揭露 v83 正交覆盖评估标签人为压低 >80% 串扰，并验证安全子空间秩"""
    e2 = _load(METRICS)["exp2_v83_selectivity_and_identifiability_fix"]
    assert e2["circular_false_pass_exposed"] is True, e2
    assert e2["artificial_suppression_ratio"] > 0.80
    assert e2["v83_procrustes_identifiability"]["ok"] is False
    assert e2["v83_procrustes_identifiability"]["dof"] == 32640
    assert e2["v84_procrustes_identifiability"]["ok"] is True
    assert e2["v84_subspace_rank_safe"] <= 12
    print("[PASS] test_exp2_v83_circular_selectivity_and_identifiability_exposed (压低 %.1f%%, r_safe=%d)"
          % (e2["artificial_suppression_ratio"] * 100, e2["v84_subspace_rank_safe"]))
    return True


def test_exp3_v5_breaks_v55_router_bandwidth_theorem():
    """4. Exp3a: 低秩向量门控打破 v55 Theorem 3 路由带宽上界 min(K, M-1)"""
    e3 = _load(METRICS)["exp3_v5_core_engineering_and_bandwidth_fix"]
    assert e3["bandwidth_ceiling_broken"] is True, e3
    assert e3["router_jacobian_rank_scalar"] <= e3["v55_theorem3_bound"]
    assert e3["router_jacobian_rank_vecgate"] > e3["v55_theorem3_bound"]
    print("[PASS] test_exp3_v5_breaks_v55_router_bandwidth_theorem (scalar=%d -> vecgate=%d > bound=%d)"
          % (e3["router_jacobian_rank_scalar"], e3["router_jacobian_rank_vecgate"], e3["v55_theorem3_bound"]))
    return True


def test_exp3_v5_sparse_checkpoint_phase1_and_skip_rate():
    """5. Exp3b: 真稀疏零误差 + Phase 1 零 z_loss 污染 + grad_checkpoint 等梯度 + 4L skip_rate<=1"""
    e3 = _load(METRICS)["exp3_v5_core_engineering_and_bandwidth_fix"]
    assert e3["sparse_vs_dense_max_diff"] < 1e-5
    assert abs(e3["sparse_compute_ratio"] - 0.25) < 1e-6
    assert e3["phase1_z_loss_grad_diff"] == 0.0
    assert e3["grad_checkpoint_x_grad_diff"] < 1e-5
    assert e3["grad_checkpoint_param_grad_diff"] < 1e-5
    assert 0.0 < e3["multilayer_4L_skip_rate"] <= 1.0
    print("[PASS] test_exp3_v5_sparse_checkpoint_phase1_and_skip_rate (sparse_ratio=%.2f, skip_4L=%.3f)"
          % (e3["sparse_compute_ratio"], e3["multilayer_4L_skip_rate"]))
    return True


def test_exp4_downstream_classification_passes_and_breaks_beta_wall():
    """6. Exp4: 严格三集下游分类门禁 PASS 且显著击败纯冻结 v4 β 墙"""
    e4 = _load(METRICS)["exp4_downstream_classification_beta_wall"]
    rep = e4["v5_gate_report"]
    assert rep["passed"] is True and rep["verdict"] == "PASS", rep
    assert e4["beta_wall_broken"] is True, e4
    assert e4["v5_vs_v4_beta_wall_break"]["ci_low"] > 0.0
    f_acc = rep["metrics"]["fusion"]["accuracy"]
    s_acc = rep["metrics"]["best_single_acc"]["accuracy"]
    assert f_acc > e4["v4_legacy_frozen_test_acc"] > s_acc
    print("[PASS] test_exp4_downstream_classification_passes_and_breaks_beta_wall "
          "(best_single=%.4f -> v4=%.4f -> v5=%.4f)"
          % (s_acc, e4["v4_legacy_frozen_test_acc"], f_acc))
    return True


def test_exp5_downstream_lm_perplexity_gate_passes():
    """7. Exp5: 严格三集序列 LM PPL 下游门禁 PASS（显著低于最强单底座与均匀基线）"""
    e5 = _load(METRICS)["exp5_downstream_lm_perplexity_gate"]
    assert e5["passed"] is True and e5["verdict"] == "PASS", e5
    assert e5["metrics"]["nll_reduction_vs_best_single"]["ci_low"] > 0.0
    assert e5["metrics"]["ppl_reduction_pct"] > 5.0
    print("[PASS] test_exp5_downstream_lm_perplexity_gate_passes (PPL %.3f -> %.3f, -%.1f%%)"
          % (e5["metrics"]["best_single"]["ppl"], e5["metrics"]["fusion"]["ppl"],
             e5["metrics"]["ppl_reduction_pct"]))
    return True


def test_zero_cross_line_or_dynamic_loads_in_v84():
    """8. v84 实现脚本完全自包含（零动态加载、零跨研究线 import）"""
    with open(SRC, "r", encoding="utf-8") as f:
        src = f.read()
    for bad in ("spec_from_file_location", "_load_by_path", "v62_dimension_projection_space", "v83_s4_aligned_pipe"):
        assert bad not in src, "v84 实现脚本不应含动态/跨包依赖: %s" % bad
    print("[PASS] test_zero_cross_line_or_dynamic_loads_in_v84 (零跨线与动态加载)")
    return True


def test_provenance_sha256_matches_source():
    """9. results/metrics.json 记录的 audit_remediation.py SHA256 与源码逐字节一致"""
    m = _load(METRICS)
    assert _sha256(SRC) == m["provenance_sha256"]["audit_remediation.py"]
    print("[PASS] test_provenance_sha256_matches_source (SHA256 校验通过)")
    return True


def test_lineage_83_to_84_bidirectional():
    """10. 世代谱系 83.0 -> 84.0 双向自洽且 leakage_audit=clean"""
    man = _load(MANIFEST)
    v83 = _load(V83_MANIFEST)
    assert man["version"] == "84.0" and man["generation"]["parent"] == "83.0"
    assert "84.0" in v83["generation"]["children"]
    assert man["leakage_audit"]["status"] == "clean"
    print("[PASS] test_lineage_83_to_84_bidirectional (83.0 -> 84.0 双向自洽)")
    return True


TESTS = [
    test_overall_verdict_is_remediation_pass,
    test_exp1_v62_random_matrix_artifact_reproduced_and_fixed,
    test_exp2_v83_circular_selectivity_and_identifiability_exposed,
    test_exp3_v5_breaks_v55_router_bandwidth_theorem,
    test_exp3_v5_sparse_checkpoint_phase1_and_skip_rate,
    test_exp4_downstream_classification_passes_and_breaks_beta_wall,
    test_exp5_downstream_lm_perplexity_gate_passes,
    test_zero_cross_line_or_dynamic_loads_in_v84,
    test_provenance_sha256_matches_source,
    test_lineage_83_to_84_bidirectional,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
