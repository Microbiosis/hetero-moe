"""V85.0 — 真预训练异构底座 (Qwen2.5-0.5B + BERT-L4) 与真实 AG_News 语料下游防泄漏门禁合龙 权威测试 (10 项)。"""
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

TITLE = "V85.0 真预训练异构底座 (Qwen2.5-0.5B + BERT-L4) 与真实 AG_News 语料下游门禁合龙 (v5.1)"

METRICS = os.path.join(_pkg, "results", "v85_true_model_downstream_gate.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "true_downstream_gate.py")
V84_MANIFEST = os.path.join(_pkg, "..", "v84_inherent_flaw_remediation", "MANIFEST.json")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_lf(path):
    with open(path, "rb") as f:
        b = f.read().replace(b"\r\n", b"\n")
        return hashlib.sha256(b).hexdigest()


def test_overall_verdict_is_true_model_downstream_gate_pass():
    """1. 总裁决必须为 TRUE_MODEL_DOWNSTREAM_GATE_PASS"""
    m = _load(METRICS)
    assert m["verdict"] == "TRUE_MODEL_DOWNSTREAM_GATE_PASS", m["verdict"]
    print("[PASS] test_overall_verdict_is_true_model_downstream_gate_pass (TRUE_MODEL_DOWNSTREAM_GATE_PASS)")
    return True


def test_real_pretrained_models_and_massive_activation_audit():
    """2. 确认使用真 Qwen2.5-0.5B (24L) 与 BERT-L4 底座且实测 Qwen Massive Activation (>50x BERT)"""
    m = _load(METRICS)
    pm = m["pretrained_models"]
    assert "Qwen/Qwen2.5-0.5B-Instruct" in pm["causal_gqa_decoder"]
    assert "google/bert_uncased_L-4_H-256_A-4" in pm["bidir_mha_encoder"]
    assert "AG_News" in pm["dataset"]
    ma = m["exp1_real_agnews_classification"]["real_model_massive_activation_audit"]
    assert ma["qwen_l3_max"] > 500.0
    assert ma["qwen_l3_to_bert_l3_max_ratio"] > 50.0
    assert ma["qwen_massive_constant_consistency"] > 0.85
    print(
        "[PASS] test_real_pretrained_models_and_massive_activation_audit (Qwen L3 max=%.1f vs BERT L3 max=%.2f, ratio=%.1fx)"
        % (ma["qwen_l3_max"], ma["bert_l3_max"], ma["qwen_l3_to_bert_l3_max_ratio"])
    )
    return True


def test_document_level_split_and_identifiability_guards():
    """3. 文档级与视角级三集零重叠划分 (SplitGuard) 及自由度守卫 (IdentifiabilityGuard) 全通过"""
    e1 = _load(METRICS)["exp1_real_agnews_classification"]
    assert e1["doc_split"]["ok"] is True and e1["doc_split"]["overlap_count"] == 0
    assert e1["view_split"]["ok"] is True and e1["view_split"]["overlap_count"] == 0
    assert e1["identifiability"]["ok"] is True
    assert e1["identifiability"]["dof_per_sample"] <= 1.0
    print(
        "[PASS] test_document_level_split_and_identifiability_guards (docs=%d/%d/%d, views=%d/%d/%d, dof/sample=%.3f)"
        % (
            e1["doc_split"]["n_train"],
            e1["doc_split"]["n_val"],
            e1["doc_split"]["n_test"],
            e1["view_split"]["n_train"],
            e1["view_split"]["n_val"],
            e1["view_split"]["n_test"],
            e1["identifiability"]["dof_per_sample"],
        )
    )
    return True


def test_exp1_real_agnews_classification_passes_downstream_eval_gate():
    """4. Exp1: 真实 AG_News 新闻分类通过 DownstreamEvalGate (95% CI 下界 > 0)"""
    rep = _load(METRICS)["exp1_real_agnews_classification"]["classification_gate_report"]
    assert rep["passed"] is True and rep["verdict"] == "PASS", rep
    cm = rep["metrics"]
    assert cm["margin_vs_best_single_acc"]["mean_delta"] > 0.0
    assert cm["margin_vs_best_single_acc"]["ci_low"] > 0.0
    assert cm["margin_vs_best_single_ce"]["ci_low"] > 0.0
    assert cm["fusion"]["accuracy"] > cm["zero_param"]["majority_accuracy"]
    assert cm["negative_control"]["fusion_minus_neg_acc"]["ci_low"] > 0.0
    print(
        "[PASS] test_exp1_real_agnews_classification_passes_downstream_eval_gate (v5.1=%.4f vs best_single=%.4f, CI=[+%.4f, +%.4f])"
        % (
            cm["fusion"]["accuracy"],
            cm["best_single_acc"]["accuracy"],
            cm["margin_vs_best_single_acc"]["ci_low"],
            cm["margin_vs_best_single_acc"]["ci_high"],
        )
    )
    return True


def test_exp1_v5_1_beats_all_6_single_static_and_unnormalized_baselines():
    """5. Exp1: hetero_moe v5.1 严格击败全部 6 个单底座冻结、单底座内部 LoRA、静态均值与未锁相基线"""
    cm = _load(METRICS)["exp1_real_agnews_classification"]["classification_gate_report"]["metrics"]
    f_acc = cm["fusion"]["accuracy"]
    singles = cm["singles"]
    required = (
        "bert_l4_frozen_probe",
        "qwen_05b_frozen_probe",
        "naive_unnormalized_massive_qwen_bert",
        "static_unrouted_equal_average",
        "bert_l4_single_internal_lora_nofusion",
        "qwen_05b_single_internal_lora_nofusion",
    )
    for arm in required:
        assert arm in singles, f"缺少关键对照基线: {arm}"
        assert f_acc > singles[arm]["accuracy"], (arm, singles[arm], f_acc)
    print(
        "[PASS] test_exp1_v5_1_beats_all_6_single_static_and_unnormalized_baselines (击败全部 6 个对照臂, naive_massive=%.4f -> v5.1=%.4f)"
        % (singles["naive_unnormalized_massive_qwen_bert"]["accuracy"], f_acc)
    )
    return True


def test_exp2_real_agnews_lm_perplexity_passes_downstream_eval_gate():
    """6. Exp2: 真实 AG_News 序列 LM 通过 DownstreamEvalGate 且击败全部 5 个单底座/静态平均/内部 LoRA 基线"""
    e2 = _load(METRICS)["exp2_real_agnews_lm_perplexity"]
    assert e2["passed"] is True and e2["verdict"] == "PASS", e2
    lm = e2["metrics"]
    assert lm["nll_reduction_vs_best_single"]["ci_low"] > 0.0
    assert lm["ppl_reduction_pct"] > 5.0
    f_nll = lm["fusion"]["nll"]
    singles = lm["singles"]
    required = (
        "bert_l4_causal_prefix_probe",
        "qwen_05b_causal_prefix_probe",
        "static_unrouted_equal_avg_lm",
        "bert_l4_single_internal_lora_lm",
        "qwen_05b_single_internal_lora_lm",
    )
    for arm in required:
        assert arm in singles, f"缺少 LM 对照基线: {arm}"
        assert f_nll < singles[arm]["nll"], (arm, singles[arm], f_nll)
    print(
        "[PASS] test_exp2_real_agnews_lm_perplexity_passes_downstream_eval_gate (PPL %.2f -> %.2f, -%.1f%%, CI=[+%.4f, +%.4f])"
        % (
            lm["best_single"]["ppl"],
            lm["fusion"]["ppl"],
            lm["ppl_reduction_pct"],
            lm["nll_reduction_vs_best_single"]["ci_low"],
            lm["nll_reduction_vs_best_single"]["ci_high"],
        )
    )
    return True


def test_exp3_v5_1_engineering_and_brain_isomorphism_passed():
    """7. Exp3: hetero_moe v5.1 内部子层 LoRA、训练容量 STE、TP SwiGLU、动量保留与 DeCollapsedLocalInterface 全部通过"""
    e3 = _load(METRICS)["exp3_v5_1_engineering_and_brain_isomorphism"]
    assert e3["all_v5_1_checks_passed"] is True, e3
    assert e3["internal_sublayer_lora_count"] == 5
    assert e3["internal_sublayer_min_grad_norm"] > 0.0
    assert e3["train_capacity_max_tokens_observed"] <= e3["train_capacity_bound"]
    assert e3["tp_swiglu_vs_single_max_diff"] < 1e-5
    assert e3["phase1_to_phase2_adam_momentum_preserved"] is True
    bd = e3["brain_interface_on_qwen_massive_activations"]
    assert bd["consistency_after"] < 0.05
    assert abs(bd["amplitude_ratio"] - 1.0) < 0.05
    print(
        "[PASS] test_exp3_v5_1_engineering_and_brain_isomorphism_passed (sublayers=%d, tp_diff=%.2e, brain_cons=%.4f, amp_ratio=%.4f)"
        % (
            e3["internal_sublayer_lora_count"],
            e3["tp_swiglu_vs_single_max_diff"],
            bd["consistency_after"],
            bd["amplitude_ratio"],
        )
    )
    return True


def test_zero_cross_line_or_dynamic_loads_in_v85():
    """8. v85 实现脚本完全自包含（零动态加载、零跨研究线 import）"""
    with open(SRC, "r", encoding="utf-8") as f:
        src = f.read()
    for bad in ("spec_from_file_location", "_load_by_path", "research.brain_domains", "v84_inherent_flaw_remediation"):
        assert bad not in src, "v85 实现脚本不应含动态/跨包依赖: %s" % bad
    print("[PASS] test_zero_cross_line_or_dynamic_loads_in_v85 (零跨线与动态加载)")
    return True


def test_provenance_sha256_matches_source():
    """9. results/v85_true_model_downstream_gate.json 记录的 script_sha256 与源码一致"""
    m = _load(METRICS)
    assert _sha256_lf(SRC) == m["script_sha256"]
    print("[PASS] test_provenance_sha256_matches_source (SHA256 校验通过)")
    return True


def test_lineage_84_to_85_bidirectional():
    """10. 世代谱系 84.0 -> 85.0 双向自洽且 leakage_audit=clean"""
    man = _load(MANIFEST)
    v84 = _load(V84_MANIFEST)
    assert man["version"] == "85.0" and man["generation"]["parent"] == "84.0"
    assert "85.0" in v84["generation"]["children"]
    assert man["leakage_audit"]["status"] == "clean"
    print("[PASS] test_lineage_84_to_85_bidirectional (84.0 -> 85.0 双向自洽)")
    return True


TESTS = [
    test_overall_verdict_is_true_model_downstream_gate_pass,
    test_real_pretrained_models_and_massive_activation_audit,
    test_document_level_split_and_identifiability_guards,
    test_exp1_real_agnews_classification_passes_downstream_eval_gate,
    test_exp1_v5_1_beats_all_6_single_static_and_unnormalized_baselines,
    test_exp2_real_agnews_lm_perplexity_passes_downstream_eval_gate,
    test_exp3_v5_1_engineering_and_brain_isomorphism_passed,
    test_zero_cross_line_or_dynamic_loads_in_v85,
    test_provenance_sha256_matches_source,
    test_lineage_84_to_85_bidirectional,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite

    sys.exit(run_suite(TITLE, TESTS))
