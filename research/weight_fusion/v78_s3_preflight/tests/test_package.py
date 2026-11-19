"""V78.0 — A1/S3 前置诊断: 把 v77 留下的缺陷归因清楚 权威测试。

背景: v77 (路线 β 的真模型可用性) 留下三类缺陷/未知:
  (1) Qwen 深层 hidden state 有效秩塌到个位数 —— 到底是「少数维主导」还是
      「少数 token 主导」? 两种归因对应两种完全不同的修法。
  (2) B5 (槽身份可分辨的对照测量) 功效不足 —— 假设是错槽基线被「共同成分」
      抬高, 那么修法应该是「去掉共同成分」。
  (3) O 槽跨模型对应不成立 (4/8) —— 是槽本身不行, 还是 O 槽取法与 Q 槽不同源?

本代做诊断 (D1-D4), 结论有三面, 测试把三面都锁住:
  - D1: 塌陷根因 = **少数维主导 + BOS token** (Qwen 特有; BERT 干净);
  - D2: 共同成分确实主导 (PC1 占 88.8% 方差), 但**去掉它不是解法** ——
        残差化后合并判据反而更差 (10/16, p=0.4545);
  - D3 + 订正: 样本级 mean-pool 探针**健康** (有效秩 47-62, 准确率 0.61-0.70)
        ⇒ v77 「探针会被绑架」的推论**过强**, 本代订正之。

本测试**不重跑实验** (需要真模型权重), 它锁住的是:
  1. 三项诊断的达成与裁决 (PREFLIGHT_DIAGNOSED);
  2. 上面三条结论的具体数字 (防止事后被软化);
  3. 「诊断性复测」不得被当成判据 / 不得改动 v77 的裁决;
  4. 阈值是跑前写死的 (静态扫源码)。

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

TITLE = "V78.0 A1/S3 前置诊断 (D1-D4)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "s3_preflight.py")
V77_METRICS = os.path.join(_pkg, "..", "v77_true_model_flow", "results", "metrics.json")
V77_SRC = os.path.join(_pkg, "..", "v77_true_model_flow", "true_model.py")


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
        return hashlib.sha256(f.read()).hexdigest()


def _row(d1, model, layer):
    for r in d1[model]["rows"]:
        if r["layer"] == layer:
            return r
    raise AssertionError("找不到 %s 的第 %d 层" % (model, layer))


# ---------------------------------------------------------------- 1
def test_three_diagnostics_achieved_and_verdict():
    """1. 三项诊断全部达成 => 机器裁决 PREFLIGHT_DIAGNOSED"""
    m = _m()
    crit = m["criteria"]
    assert crit["D1_eff_rank_cause_identified"] is True, "D1 未达成"
    assert crit["D2_common_component_measured"] is True, "D2 未达成"
    assert crit["D3_label_readout_measured"] is True, "D3 未达成"
    assert m["verdict"] == "PREFLIGHT_DIAGNOSED", "裁决不是 PREFLIGHT_DIAGNOSED: %s" % m["verdict"]
    assert _manifest()["acceptance"]["outcome"] == "pass", "MANIFEST 的验收结论与快照不一致"
    print("[PASS] test_three_diagnostics_achieved_and_verdict (3/3, verdict=%s)"
          % m["verdict"])
    return True


# ---------------------------------------------------------------- 2
def test_d1_collapse_is_dimension_dominated_and_bert_is_clean():
    """2. D1: Qwen 深层塌陷 = 少数维主导 (top1 维独占 >=90% 方差); BERT 干净"""
    d1 = _m()["D1_eff_rank_cause"]
    collapsed = (3, 4, 5, 12, 18)
    for li in collapsed:
        r = _row(d1, "Qwen", li)
        assert r["eff_rank_raw"] < 20.0, "Qwen L%d 的有效秩不是塌陷态: %s" % (li, r["eff_rank_raw"])
        assert r["dim_concentration"]["top1"] >= 0.90, (
            "Qwen L%d 的 top1 维方差占比没到 0.90: %s" % (li, r["dim_concentration"]["top1"]))
        assert r["token_concentration"]["top16"] >= 0.95, (
            "Qwen L%d 的 token 集中度对不上 (BOS 主导): %s"
            % (li, r["token_concentration"]["top16"]))
    # BERT 全部干净 => 塌陷是 Qwen 特有, 不是本仓库实现的产物
    for r in d1["BERT"]["rows"]:
        assert r["eff_rank_raw"] >= 120.0, "BERT 的有效秩塌了: %s" % r
        assert r["dim_concentration"]["top1"] <= 0.13, "BERT 的维度集中度过高: %s" % r
    print("[PASS] test_d1_collapse_is_dimension_dominated_and_bert_is_clean "
          "(Qwen L3-L18 塌到 <20 且 top1 维 >=0.90; BERT 全部 >=122)")
    return True


# ---------------------------------------------------------------- 3
def test_d1_rescue_variants_are_measured():
    """3. D1: 修法有效且可比较 —— 逐 token L2 / 去首 token 能救回, 剔 top4 维不够"""
    d1 = _m()["D1_eff_rank_cause"]
    for li in (3, 4, 5, 12, 18):
        v = _row(d1, "Qwen", li)["eff_rank_variants"]
        assert v["token_l2_normalized"] >= 200.0, "L%d 逐 token L2 救不回: %s" % (li, v)
        assert v["drop_first_token"] >= 200.0, "L%d 去首 token 救不回: %s" % (li, v)
    worst = _row(d1, "Qwen", 4)["eff_rank_variants"]
    assert worst["drop_top4_dims"] < 100.0, (
        "剔 top4 维居然够用, 与实测(62.49)不符: %s" % worst["drop_top4_dims"])
    print("[PASS] test_d1_rescue_variants_are_measured (L4: L2 %.1f / 去首 %.1f / 剔 top4 维 %.1f)"
          % (worst["token_l2_normalized"], worst["drop_first_token"], worst["drop_top4_dims"]))
    return True


# ---------------------------------------------------------------- 4
def test_d2_common_component_dominates():
    """4. D2: 共同成分确实主导 (PC1 占 88.8% 方差, 槽表示被 2 PC 解释到 R^2 0.63~1.00)"""
    d2 = _m()["D2_common_component"]
    cc = d2["common_component"]
    assert cc["pc_explained_variance"][0] >= 0.88, "PC1 的方差占比对不上: %s" % cc
    assert cc["pc_explained_variance"][1] >= 0.03, "PC2 的方差占比对不上: %s" % cc
    assert cc["mean_slot_explained_by_pcs"] >= 0.60, "槽被 2 PC 解释的平均 R^2 对不上: %s" % cc
    assert cc["max_slot_explained_by_pcs"] >= 0.99, "槽被 2 PC 解释的最大 R^2 对不上: %s" % cc
    assert cc["threshold_common_share_min"] == 0.5, "共同成分的阈值被改了"
    print("[PASS] test_d2_common_component_dominates (PC1 %.4f / PC2 %.4f, R^2 mean %.4f max %.4f)"
          % (cc["pc_explained_variance"][0], cc["pc_explained_variance"][1],
             cc["mean_slot_explained_by_pcs"], cc["max_slot_explained_by_pcs"]))
    return True


# ---------------------------------------------------------------- 5
def test_d2_residualization_is_not_the_fix():
    """5. D2 的**否定**结论: 去掉共同成分不是 B5 的解法 (合并判据更差)"""
    d2 = _m()["D2_common_component"]
    raw = d2["raw"]
    res = d2["residualized"]
    assert raw["n_blocks"] == 16 and raw["n_positive"] == 12, "原始口径变了: %s" % raw
    assert res["n_blocks"] == 16 and res["n_positive"] == 10, "残差化口径变了: %s" % res
    assert res["sign_test_p"] > raw["sign_test_p"], "残差化反而更显著了 (与实测相反)"
    assert res["median_delta"] < raw["median_delta"], "残差化反而更大 (与实测相反)"
    # 分槽: Q 槽的槽特异信号是真的; O 槽几乎全是共同成分
    by = d2["by_slot_residualized"]
    q, o = by["Q"], by["O"]
    assert q["n_blocks"] == 8 and q["n_positive"] == 8, "Q 槽的拆解变了: %s" % q
    assert q["sign_test_p_resid"] < 0.05, "Q 槽的符号检验 p 变了: %s" % q["sign_test_p_resid"]
    assert abs(q["median_delta_resid"] - q["median_delta_raw"]) < 0.005, (
        "Q 槽残差化后不该几乎不变: %s" % q)
    assert o["n_blocks"] == 8 and o["n_positive"] == 2, "O 槽的拆解变了: %s" % o
    assert o["sign_test_p_resid"] > 0.05, "O 槽的符号检验 p 变了: %s" % o["sign_test_p_resid"]
    print("[PASS] test_d2_residualization_is_not_the_fix (残差化 10/16 p=%.4f 更差; Q 8/8, O 2/8)"
          % res["sign_test_p"])
    return True


# ---------------------------------------------------------------- 6
def test_d3_sample_level_probe_is_healthy():
    """6. D3: 样本级 mean-pool 探针健康 —— token 级塌陷不影响它 (订正 v77 的推论)"""
    d3 = _m()["D3_label_readout"]
    q = [r for r in d3["Qwen"]["layers"] if r["layer"] >= 1]
    accs = sorted(r["raw"]["acc"] for r in q)
    med = accs[len(accs) // 2]
    assert med >= 0.62, "Qwen 样本级探针中位准确率掉下来了: %s" % med
    assert max(r["raw"]["acc"] for r in d3["Qwen"]["layers"]) >= 0.70, "Qwen 最高层读不出来了"
    assert d3["BERT"]["layers"][0]["raw"]["acc"] >= 0.68, "BERT L0 读不出来了"
    # 关键: v77 说「有效秩塌到个位数」的那几层, 样本级 pooled 有效秩是健康的
    for li in (3, 4, 5, 12, 18):
        r = _row(_m()["D1_eff_rank_cause"], "Qwen", li)
        pooled = [x for x in d3["Qwen"]["layers"] if x["layer"] == li][0]
        assert pooled["eff_rank_raw"] >= 45.0, (
            "Qwen L%d 的样本级 pooled 有效秩不健康: %s" % (li, pooled["eff_rank_raw"]))
        assert r["eff_rank_raw"] < 20.0
    print("[PASS] test_d3_sample_level_probe_is_healthy (Qwen 中位 %.4f / 最高 %.4f; "
          "L3-L18 pooled 有效秩 >=49)" % (med, max(r["raw"]["acc"] for r in d3["Qwen"]["layers"])))
    return True


# ---------------------------------------------------------------- 7
def test_d4_recommendation_is_computed_not_hardcoded():
    """7. D4: 推荐预处理**由实测算出** (与 D1/D3 的原始读数自洽)"""
    m = _m()
    d4 = m["D4_recommendation"]
    acc = d4["sample_level_acc_mean"]
    assert acc["raw_mean_pool"] is not None and acc["token_l2_pooled"] is not None, (
        "D4 缺样本级准确率读数: %s" % acc)
    assert acc["raw_mean_pool"] > acc["token_l2_pooled"], (
        "原始 mean-pool 竟然比 token-L2 后更差, 推荐结论要重写: %s" % acc)
    assert "mean-pool" in d4["sample_level_probe"], "D4 的推荐没提 mean-pool: %s" % d4
    # 推荐与 D3 的原始读数自洽 (不是写死的字符串)
    for mdl in ("Qwen", "BERT"):
        rows = m["D3_label_readout"][mdl]["layers"]
        expect = sorted(r["eff_rank_raw"] for r in rows)[len(rows) // 2]
        got = d4["sample_level_pooled_eff_rank_median"][mdl]
        assert abs(expect - got) < 1e-6, "%s 的 pooled 有效秩中位数与 D3 不自洽: %s vs %s" % (
            mdl, got, expect)
    assert d4["token_level_rescue_best"] in d4["token_level_rescue_median_eff_rank"], (
        "D4 的 token 级修法名不在读数表里: %s" % d4["token_level_rescue_best"])
    print("[PASS] test_d4_recommendation_is_computed_not_hardcoded "
          "(raw %.4f > l2 %.4f; 推荐=%s; token 级最优=%s)"
          % (acc["raw_mean_pool"], acc["token_l2_pooled"], d4["sample_level_probe"],
             d4["token_level_rescue_best"]))
    return True


# ---------------------------------------------------------------- 8
def test_diagnostic_readouts_are_not_judgement_criteria():
    """8. 诊断性复测必须带标注, 且不得被当成判据"""
    m = _m()
    note2 = m["D2_common_component"]["note"]
    for frag in ("诊断性复测", "不作为 P1b 判据"):
        assert frag in note2, "D2 的 note 缺少: %s" % frag
    assert "诊断" in m["D4_recommendation"]["note"], "D4 的 note 缺少诊断标注"
    man = _manifest()
    blob = man["claim"]["note"] + " " + man["protocol"]["caveat"]
    for frag in ("诊断", "不产生", "未判定"):
        assert frag in blob, "MANIFEST 缺少读法限定: %s" % frag
    limits = " ".join(m["limits"])
    for frag in ("诊断代", "PCA", "粗读"):
        assert frag in limits, "limits 少了: %s" % frag
    print("[PASS] test_diagnostic_readouts_are_not_judgement_criteria "
          "(D2/D4 标注 + %d 条局限在案)" % len(m["limits"]))
    return True


# ---------------------------------------------------------------- 9
def test_v77_verdict_is_untouched():
    """9. 本代**不改动** v77 的裁决 (S1B_FAIL 仍然成立)"""
    assert os.path.exists(V77_METRICS), "找不到 v77 快照: %s" % V77_METRICS
    with open(V77_METRICS, "r", encoding="utf-8") as f:
        v77 = json.load(f)
    assert v77["verdict"] == "S1B_FAIL", "v77 的裁决被改了: %s" % v77["verdict"]
    assert v77["criteria"]["B5_slot_identity_resolvable"] is False, "v77 的 B5 结论被翻了"
    falsified = _manifest()["falsified"]
    assert len(falsified) == 3, "证伪条目数变了: %d" % len(falsified)
    joined = " ".join(f["hypothesis"] + f["verdict"] for f in falsified)
    for frag in ("共同成分", "过强", "O 槽"):
        assert frag in joined, "证伪里少了: %s" % frag
    for f in falsified:
        assert f["evidence"], "证伪条目缺证据位置"
    print("[PASS] test_v77_verdict_is_untouched (v77 仍 S1B_FAIL; %d 条证伪在案)"
          % len(falsified))
    return True


# ---------------------------------------------------------------- 10
def test_thresholds_were_fixed_before_the_run():
    """10. 阈值是跑之前写死的 (静态扫源码), 防止事后放宽"""
    src = _src()
    for frag in ('"D1_dim_concentration_top4": 0.50', '"D1_token_concentration_top4": 0.50',
                 '"D1_rescue_eff_rank_min": 10.0', '"D2_common_share_min": 0.50',
                 '"D3_probe_acc_min": 0.40', '"D2_sign_test_p_max": 0.05',
                 '"D4_sign_test_p_max": 0.05'):
        assert frag in src, "源码里找不到阈值声明: %s" % frag
    assert 'verdict = "PREFLIGHT_DIAGNOSED" if all(criteria.values()) else "PREFLIGHT_PARTIAL"' in src, \
        "裁决逻辑变了"
    assert 'return 0 if res["verdict"] == "PREFLIGHT_DIAGNOSED" else 1' in src, "退出码契约变了"
    assert "post_hoc_only" not in src or "诊断性" in src, "诊断读数的标注被去掉了"
    print("[PASS] test_thresholds_were_fixed_before_the_run (七项阈值 + 裁决逻辑 + 退出码契约)")
    return True


# ---------------------------------------------------------------- 11
def test_provenance_sha256_matches_live_file():
    """11. 口径同源凭证: v77 true_model.py 的 sha256 与快照一致"""
    p = _m()["protocol"]["provenance_sha256"]
    assert os.path.exists(V77_SRC), "找不到 v77 true_model.py: %s" % V77_SRC
    live = _sha256(V77_SRC)
    assert live == p["v77_true_model"], (
        "v77 true_model.py 的 sha256 不一致: 实测 %s, 快照 %s" % (live, p["v77_true_model"]))
    print("[PASS] test_provenance_sha256_matches_live_file (复用件 sha256 一致)")
    return True


# ---------------------------------------------------------------- 12
def test_protocol_declares_scope_and_limits():
    """12. 范围与局限必须随数字一起登记"""
    m = _m()
    proto = m["protocol"]
    assert proto["route"] == "beta (真 HF 模型 + 真实语料)", "route 变了: %s" % proto["route"]
    assert proto["device"] == "cpu" and proto["dtype"] == "float32"
    assert set(proto["models"]) == {"Qwen", "BERT"}, "诊断范围变了: %s" % proto["models"]
    assert "ViT" not in proto["models"], "ViT 不该进本代诊断范围"
    assert "ag_news" in proto["corpus"], "语料登记变了: %s" % proto["corpus"]
    assert "同一" in proto["corpus_same_as_v77"], "与 v77 同源的声明丢了"
    assert "复用" in proto["tokenization_same_as_v77"], "与 v77 同 tokenization 的声明丢了"
    assert proto["thresholds"]["D3_probe_acc_min"] == 0.40, "阈值表与源码对不上"
    print("[PASS] test_protocol_declares_scope_and_limits (范围 Qwen/BERT, 语料与 tokenization 同源)")
    return True


# ---------------------------------------------------------------- 13
def test_manifest_claim_is_honest_about_diagnostic_scope():
    """13. claim 的标签与 note 必须如实反映「诊断达成, 但 A1 未判定」"""
    man = _manifest()
    assert man["claim"]["outcome_label"] == "confirmed", man["claim"]["outcome_label"]
    assert man["claim"]["metric"] == "preflight_diagnostics_pass_rate"
    assert man["claim"]["value"] == 1.0, "claim 的通过率变了: %s" % man["claim"]["value"]
    note = man["claim"]["note"]
    for frag in ("诊断", "未判定", "S1B_FAIL"):
        assert frag in note, "claim 的 note 少了: %s" % frag
    assert man["generation"]["parent"] == "77.0" and man["version"] == "78.0"
    assert man["leakage_audit"]["status"] == "not_applicable", "泄漏状态变了"
    assert man["superseded_by"] is None
    print("[PASS] test_manifest_claim_is_honest_about_diagnostic_scope "
          "(outcome_label=%s, pass_rate=%.1f)" % (man["claim"]["outcome_label"],
                                                  man["claim"]["value"]))
    return True


TESTS = [
    test_three_diagnostics_achieved_and_verdict,
    test_d1_collapse_is_dimension_dominated_and_bert_is_clean,
    test_d1_rescue_variants_are_measured,
    test_d2_common_component_dominates,
    test_d2_residualization_is_not_the_fix,
    test_d3_sample_level_probe_is_healthy,
    test_d4_recommendation_is_computed_not_hardcoded,
    test_diagnostic_readouts_are_not_judgement_criteria,
    test_v77_verdict_is_untouched,
    test_thresholds_were_fixed_before_the_run,
    test_provenance_sha256_matches_live_file,
    test_protocol_declares_scope_and_limits,
    test_manifest_claim_is_honest_about_diagnostic_scope,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
