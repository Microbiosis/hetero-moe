"""V77.0 — A1 路线 β: 真 HF 模型的逐层驱动与层边界注入 (S1β) 权威测试。

背景: 主线前置关卡 A1 的语义层判据 (P1a/P1b/P2/P3) 必须在路线 β (真模型逐层
驱动 + 真输入 + 标签) 上做 —— v76 在路线 α 上证伪了「随机输入这条路测不了
语义」。本代做路线 β 的第一步: 证明这条路通不通。

本代结论有两面, 测试把两面都锁住:
  - 机制层 (B1-B4) 全过, 且 B1/B2 是**位级**证据:
      切片权重与真模型参数 72/72 块逐位相同 (rel_err = 0.0);
      零注入的「打断->续跑」与原前向逐层位级相同 (rel_err = 0.0)。
  - 唯一未过的是 B5 (槽身份可分辨的**对照**测量): 方向为正但功效不足
      (占优 12/16, 中位差 +0.0738, 双侧符号检验 p=0.0768 > 0.05)
      ⇒ 读作「不可判定」, **不是**「槽身份在真模型上不可分辨」的否证。

本测试**不重跑实验** (需要真模型权重), 它锁住的是:
  1. 判据与裁决的自洽 (B1-B4 过 / B5 不过 => S1B_FAIL, 阈值未放宽)
  2. 口径同源凭证 (v62 measure / v75 g2 / v76 a1_forward 的 sha256)
  3. 阈值是跑前写死的 (静态扫源码)
  4. 两条证伪与「事后拆解不参与裁决」的标注都在案

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

TITLE = "V77.0 A1 路线 β (真模型逐层驱动, S1β)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "true_model.py")
V75_G2 = os.path.join(_pkg, "..", "v75_weight_conjugation", "g2_conjugation.py")
V76_A1 = os.path.join(_pkg, "..", "v76_cross_layer_flow", "a1_forward.py")
V62_MEASURE = os.path.join(_pkg, "..", "..", "aligner",
                            "v62_dimension_projection_space", "measure.py")


def _m():
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


def _manifest():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)


def _src():
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read()


def _sha256(path):
    with open(path, "rb") as f:
        b = f.read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- 1
def test_four_of_five_criteria_pass_and_verdict_is_fail():
    """1. 判据与裁决: B1-B4 过 / B5 不过 => 机器裁决 S1B_FAIL"""
    m = _m()
    crit = m["criteria"]
    for name in ("B1_weight_same_source", "B2_layerwise_drive_ok",
                 "B3_injection_path_ok", "B4_stack_health_ok"):
        assert crit[name] is True, "机制层判据未过: %s" % name
    assert crit["B5_slot_identity_resolvable"] is False, "B5 竟然过了 —— 裁决需重写"
    assert m["verdict"] == "S1B_FAIL", "裁决不是 S1B_FAIL: %s" % m["verdict"]
    assert _manifest()["acceptance"]["outcome"] == "fail", "MANIFEST 的验收结论与快照不一致"
    print("[PASS] test_four_of_five_criteria_pass_and_verdict_is_fail (4/5, verdict=%s)"
          % m["verdict"])
    return True


# ---------------------------------------------------------------- 2
def test_b1_weights_are_bitwise_same_source():
    """2. B1: 切片权重与真模型参数逐位相同 (72/72 块)"""
    b1 = _m()["B1_weight_provenance"]
    total = 0
    for name, v in b1.items():
        assert v["ok"] is True, "%s 的权重同源判据未过" % name
        assert v["n_different"] == 0, "%s 有 %d 块权重对不上" % (name, v["n_different"])
        assert v["n_exact"] == v["n_blocks"] == 24, "%s 的块数/同源数变了: %s" % (name, v)
        assert v["max_rel_err"] == 0.0, "%s 的最大相对误差不是 0: %s" % (name, v["max_rel_err"])
        total += v["n_blocks"]
    assert total == 72, "总块数变了: %d" % total
    print("[PASS] test_b1_weights_are_bitwise_same_source (72/72 块 rel_err=0.0)")
    return True


# ---------------------------------------------------------------- 3
def test_b2_zero_injection_is_bitwise_equivalent():
    """3. B2: 零注入的打断/续跑与原前向位级等价 => 机制本身没改语义"""
    b2 = _m()["B2_layerwise_drive"]
    for name, v in b2.items():
        assert v["ok"] is True, "%s 的逐层驱动判据未过" % name
        assert v["zero_inject_max_rel_err"] == 0.0, "%s 的零注入不是位级等价: %s" % (
            name, v["zero_inject_max_rel_err"])
        assert v["n_hidden_states"] == v["expected"], "%s 的逐层状态数不对: %s" % (
            name, v)
        assert v["finite"] is True, "%s 的逐层状态有非有限值" % name
    print("[PASS] test_b2_zero_injection_is_bitwise_equivalent (三底座 max_rel_err=0.0)")
    return True


# ---------------------------------------------------------------- 4
def test_b3_injection_really_changes_output():
    """4. B3: 注入真的改变了后续层输出, 且没把模型搞坏"""
    b3 = _m()["B3_injection"]
    assert set(b3) == {"Qwen", "BERT"}, "注入检查的底座集合变了: %s" % list(b3)
    for name, v in b3.items():
        assert v["ok"] is True, "%s 的注入通路判据未过" % name
        assert v["finite"] is True, "%s 的输出有非有限值" % name
        rel = v["final"]["rel_perturbation"]
        assert 1e-3 < rel < 0.5, "%s 的相对扰动越界: %g" % (name, rel)
    print("[PASS] test_b3_injection_really_changes_output (BERT %.4f / Qwen %.4f)"
          % (b3["BERT"]["final"]["rel_perturbation"],
             b3["Qwen"]["final"]["rel_perturbation"]))
    return True


# ---------------------------------------------------------------- 5
def test_b4_scope_is_first_generation_and_vit_defect_is_gone():
    """5. B4: 判据范围 = 首代覆盖的 Qwen/BERT; ViT 在**真模型**上没有 v76 的缺陷"""
    b4 = _m()["B4_stack_health"]
    assert b4["scope"] == ["Qwen", "BERT"], "B4 判据范围变了: %s" % b4["scope"]
    assert b4["in_scope_ok"] is True
    assert "ViT" in b4["out_of_scope"], "ViT 不再登记为范围外"
    for name in b4["scope"]:
        v = b4["models"][name]
        assert v["ok"] is True, "%s 的层栈健康判据未过" % name
        lo, hi = v["bounds"]["ratio"]
        for r in v["rms_ratio"]:
            assert lo <= r <= hi, "%s 的范数比 %g 越界" % (name, r)
        assert v["min_eff_rank"] > v["bounds"]["eff_rank_min"], "%s 有效秩塌了" % name
    # v76 的 ViT 缺陷 (残差膨胀 20.11x) 是**路线 α 重建**的产物: 真模型上 ViT 健康
    vit = b4["out_of_scope"]["ViT"]
    assert vit["ok"] is True, "ViT 在真模型上仍不健康 ⇒ v76 的根因假设被证伪"
    assert max(vit["rms_ratio"]) < 10.0, "ViT 仍膨胀: %g" % max(vit["rms_ratio"])
    print("[PASS] test_b4_scope_is_first_generation_and_vit_defect_is_gone "
          "(Qwen/BERT 健康; 真模型 ViT 最大比 %.2f)"
          % max(vit["rms_ratio"]))
    return True


# ---------------------------------------------------------------- 6
def test_b5_failure_is_power_not_direction():
    """6. B5 未过的是**功效**不是方向: 必须读作不可判定, 不是否证"""
    m = _m()
    b5 = m["B5_slot_identity"]
    assert b5["ok"] is False, "B5 竟然过了"
    assert b5["frac_matched_better"] >= 0.75, "占优比例变了: %s" % b5["frac_matched_better"]
    assert b5["median_delta"] > 0.0, "中位差变成非正: %s" % b5["median_delta"]
    assert b5["sign_test_p"] > 0.05, "符号检验 p 变了: %s" % b5["sign_test_p"]
    note = " ".join(m["limits"]) + " " + _manifest()["acceptance"]["note"]
    for frag in ("功效", "不可判定"):
        assert frag in note, "缺少「%s」的读法限定, 裁决会被读歪" % frag
    assert "不是" in note, "缺少「不是否证」的限定"
    print("[PASS] test_b5_failure_is_power_not_direction (占优 %.2f, 中位差 %+.4f, p=%.4f)"
          % (b5["frac_matched_better"], b5["median_delta"], b5["sign_test_p"]))
    return True


# ---------------------------------------------------------------- 7
def test_post_hoc_breakdown_is_labelled_and_not_used_for_verdict():
    """7. 事后拆解必须带 post_hoc 标注, 且不得参与裁决"""
    m = _m()
    ph = m["post_hoc_per_slot"]
    assert ph["status"] == "post_hoc_only_not_used_for_verdict", "事后拆解的标注不对"
    assert "不参与" in ph["note"]
    q = ph["by_slot"]["Q"]
    o = ph["by_slot"]["O"]
    assert q["n_blocks"] == 8 and q["n_positive"] == 8, "Q 槽的拆解结果变了: %s" % q
    assert q["sign_test_p"] < 0.05, "Q 槽的符号检验 p 变了: %s" % q["sign_test_p"]
    assert o["n_blocks"] == 8 and o["n_positive"] == 4, "O 槽的拆解结果变了: %s" % o
    # 拆解不能反过来把 B5 变成通过
    assert m["criteria"]["B5_slot_identity_resolvable"] is False
    print("[PASS] test_post_hoc_breakdown_is_labelled_and_not_used_for_verdict "
          "(Q 8/8 p=%.4f; O 4/8; 未参与裁决)" % q["sign_test_p"])
    return True


# ---------------------------------------------------------------- 8
def test_thresholds_were_fixed_before_the_run():
    """8. 阈值是跑之前写死的 (静态扫源码), 防止事后放宽"""
    src = _src()
    for frag in ('"B1_rel_err_exact": 1e-5', '"B2_zero_inject_rel_err": 1e-6',
                 '"B3_min_rel_perturbation": 1e-3', '"B3_max_rel_perturbation": 0.5',
                 '"B4_norm_ratio": [0.1, 10.0]', '"B4_eff_rank_min": 1.1',
                 '"B5_matched_better_min": 0.75', '"B5_sign_test_p_max": 0.05',
                 'HEALTH_SCOPE = ("Qwen", "BERT")'):
        assert frag in src, "源码里找不到阈值声明: %s" % frag
    assert 'verdict = "S1B_PASS" if all(criteria.values()) else "S1B_FAIL"' in src, "裁决逻辑变了"
    assert 'return 0 if res["verdict"] == "S1B_PASS" else 1' in src, "退出码契约变了"
    print("[PASS] test_thresholds_were_fixed_before_the_run (八项阈值 + 判据范围在源码中写死)")
    return True


# ---------------------------------------------------------------- 9
def test_provenance_sha256_matches_live_files():
    """9. 口径同源凭证: v62 measure / v75 g2 / v76 a1_forward 的 sha256 与快照一致"""
    p = _m()["protocol"]["provenance_sha256"]
    assert os.path.exists(V62_MEASURE), "找不到 v62 measure.py: %s" % V62_MEASURE
    assert os.path.exists(V75_G2), "找不到 v75 g2_conjugation.py: %s" % V75_G2
    assert os.path.exists(V76_A1), "找不到 v76 a1_forward.py: %s" % V76_A1
    live = {"v62_measure": _sha256(V62_MEASURE),
            "v75_g2": _sha256(V75_G2),
            "v76_a1_forward": _sha256(V76_A1)}
    for key, actual in live.items():
        assert actual == p[key], "%s sha256 不一致: 实测 %s, 快照 %s" % (key, actual, p[key])
    print("[PASS] test_provenance_sha256_matches_live_files (三份复用件 sha256 全部一致)")
    return True


# ---------------------------------------------------------------- 10
def test_protocol_declares_route_beta_and_limits():
    """10. 路线 β 的身份与边界必须随数字一起登记"""
    m = _m()
    assert "beta" in m["protocol"]["route"], "route 不是路线 β"
    assert m["protocol"]["device"] == "cpu" and m["protocol"]["dtype"] == "float32"
    limits = " ".join(m["limits"])
    for frag in ("端到端", "P1b", "对照"):
        assert frag in limits, "limits 少了: %s" % frag
    choices = " ".join(m["protocol"]["declared_modeling_choices"])
    for frag in ("fp32", "hook", "mean-pool"):
        assert frag in choices, "declared_modeling_choices 少了: %s" % frag
    assert m["protocol"]["scope"].startswith("S1beta"), "scope 声明变了"
    print("[PASS] test_protocol_declares_route_beta_and_limits (%d 条局限 / %d 条建模选择已登记)"
          % (len(m["limits"]), len(m["protocol"]["declared_modeling_choices"])))
    return True


# ---------------------------------------------------------------- 11
def test_falsified_expectations_are_on_record():
    """11. 两条被数据推翻的预期必须在案 (注入方向被原样携带 / 探针可直接读 Qwen)"""
    falsified = _manifest()["falsified"]
    assert len(falsified) == 2, "证伪条目数变了: %d" % len(falsified)
    joined = " ".join(f["hypothesis"] + f["verdict"] for f in falsified)
    for frag in ("alignment", "有效秩"):
        assert frag in joined, "证伪里少了: %s" % frag
    for f in falsified:
        assert f["evidence"], "证伪条目缺证据位置"
    print("[PASS] test_falsified_expectations_are_on_record (2 条证伪在案)")
    return True


# ---------------------------------------------------------------- 12
def test_manifest_claim_is_honest_about_the_partial_outcome():
    """12. claim 的标签与 note 必须如实反映「机制过 + 对照不可判定」"""
    man = _manifest()
    assert man["claim"]["outcome_label"] == "partially_confirmed", man["claim"]["outcome_label"]
    assert man["claim"]["value"] == 0.8, "claim 的通过率变了: %s" % man["claim"]["value"]
    assert man["claim"]["metric"] == "s1b_criteria_pass_rate"
    note = man["claim"]["note"]
    assert "对照" in note and "不可判定" in note, "claim 的 note 缺少读法限定"
    assert man["generation"]["parent"] == "76.0" and man["version"] == "77.0"
    print("[PASS] test_manifest_claim_is_honest_about_the_partial_outcome "
          "(outcome_label=%s, pass_rate=%.1f)" % (man["claim"]["outcome_label"],
                                                  man["claim"]["value"]))
    return True


TESTS = [
    test_four_of_five_criteria_pass_and_verdict_is_fail,
    test_b1_weights_are_bitwise_same_source,
    test_b2_zero_injection_is_bitwise_equivalent,
    test_b3_injection_really_changes_output,
    test_b4_scope_is_first_generation_and_vit_defect_is_gone,
    test_b5_failure_is_power_not_direction,
    test_post_hoc_breakdown_is_labelled_and_not_used_for_verdict,
    test_thresholds_were_fixed_before_the_run,
    test_provenance_sha256_matches_live_files,
    test_protocol_declares_route_beta_and_limits,
    test_falsified_expectations_are_on_record,
    test_manifest_claim_is_honest_about_the_partial_outcome,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
