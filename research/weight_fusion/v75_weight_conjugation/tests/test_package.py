"""V75.0 — 槽级共轭的权重级有效性检验 权威测试。

背景：主线生死线 G2 问的是「槽级搬运后, 同槽跨架构的权重是否比异槽更相似」。
本代把它跑成了可证伪实验, 结论是负的:
  - G1 口径回归 PASS (原生六槽逐位复现 v62 §2)
  - G2-a 保留检验 INCONCLUSIVE (原生基线在 JS 与 CKA 两种口径下都无功效)
  - G2-b 生成检验 FAIL (按槽搬运未在预登记 alpha=0.01 上击败同谱随机搬运/槽无关搬运)

本测试**不重跑实验** (需要权重切片缓存), 它锁住的是:
  1. 口径同源凭证 (v62 measure.py 的 sha256)
  2. 结论与判据的自洽 (重算一遍判据逻辑, 必须复现快照里的裁决)
  3. 阈值是跑前写死的 (静态扫源码)
  4. 本代推翻/限定的既有主张确实写在快照里

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

TITLE = "V75.0 槽级共轭的权重级有效性检验"

METRICS = os.path.join(_pkg, "results", "metrics.json")
SRC = os.path.join(_pkg, "g2_conjugation.py")
V62_MEASURE = os.path.join(_pkg, "..", "..", "aligner",
                            "v62_dimension_projection_space", "measure.py")


def _m():
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


def _src():
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- 1
def test_g1_reproduces_v62_within_tolerance():
    """1. G1 口径回归: 原生六槽谱统计逐位复现 v62 §2"""
    g1 = _m()["g1_native_spectral"]
    assert g1["passes"] is True, "G1 口径回归未通过"
    for key in ("same_function_js_mean", "diff_function_js_mean"):
        rel = g1["relative_error"][key]
        assert rel <= 1e-6, "%s 相对误差 %g 过大 (判据容差 5%%)" % (key, rel)
    assert g1["measured"]["n_pairs"] == g1["v62_published"]["n_pairs"] == 1728
    assert g1["measured"]["permutation_p"] == g1["v62_published"]["permutation_p"] == 0.0002
    print("[PASS] test_g1_reproduces_v62_within_tolerance (相对差 %.2e / %.2e)"
          % (g1["relative_error"]["same_function_js_mean"],
             g1["relative_error"]["diff_function_js_mean"]))
    return True


# ---------------------------------------------------------------- 2
def test_measure_sha256_is_the_provenance():
    """2. 口径同源凭证: v62 measure.py 的 sha256 与快照记录一致"""
    m = _m()
    recorded = m["protocol"]["v62_measure_sha256"]
    assert os.path.exists(V62_MEASURE), "找不到 v62 measure.py: %s" % V62_MEASURE
    with open(V62_MEASURE, "rb") as f:
        actual = hashlib.sha256(f.read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
    assert actual == recorded, "measure.py sha256 不一致: 实测 %s, 快照 %s" % (actual, recorded)
    print("[PASS] test_measure_sha256_is_the_provenance (measure.py sha256=%s...)" % actual[:16])
    return True


# ---------------------------------------------------------------- 3
def test_shape_audit_exposes_family_composition():
    """3. 前提审计: headline 的显著性主要来自族/形状构成, 不是槽对应"""
    aud = _m()["premise_shape_effect_audit"]
    cross = aud["C_cross_family"]
    assert cross["n_same"] == 0, "跨族桶里不该有同槽对"
    assert cross["n_diff"] == 768, "跨族桶对数变了: %s" % cross["n_diff"]
    bal = aud["F_family_balanced"]
    p_bal = bal["family_stratified_permutation_p"]
    assert p_bal > 0.05, "族平衡分离度显著 (p=%s) —— 与本代结论冲突" % p_bal
    within = aud["D_within_family_only"]["separation"]
    assert abs(bal["balanced_separation"]) < abs(within), "族平衡分离度没有小于族内分离度"
    print("[PASS] test_shape_audit_exposes_family_composition (族平衡 p=%.4f, 族内 p=%s)"
          % (p_bal, aud["D_within_family_only"]["permutation_p"]))
    return True


# ---------------------------------------------------------------- 4
def test_native_gate_has_no_power_in_either_metric():
    """4. G2-a 的 INCONCLUSIVE 不依赖口径: JS 原生闸门与 CKA 原生闸门都无功效"""
    m = _m()
    js_gate = m["premise_native_equishaped"]["slot"]["permutation_p"]
    assert js_gate > 0.05, "JS 口径原生闸门有功效 (p=%s) —— 与本代结论冲突" % js_gate
    cka_gate = m["arms"]["A0_none"]["granularities"]["slot"]["cka_strat_perm"]["permutation_p"]
    assert cka_gate > 0.05, "CKA 口径原生闸门有功效 (p=%s) —— 与本代结论冲突" % cka_gate
    for gran, crit in m["criteria_by_granularity"].items():
        assert crit["native_baseline_detectable_at_this_granularity"] is False, gran
        assert m["verdict_by_granularity"][gran]["G2a_preservation"] == \
            "G2a_INCONCLUSIVE_NO_BASELINE_POWER", gran
    print("[PASS] test_native_gate_has_no_power_in_either_metric (JS p=%.4f, CKA p=%.4f)"
          % (js_gate, cka_gate))
    return True


# ---------------------------------------------------------------- 5
def test_transport_path_works_in_sample():
    """5. 机械通路自检: 同层拟合的槽检索准确率必须接近 1, 否则搬运机制本身没通"""
    m = _m()
    acc = m["arms"]["B_slot"]["retrieval"]["in_sample_slot_accuracy"]
    assert acc >= 0.5, "同层自检检索准确率 %g 过低 ⇒ 搬运通路本身有问题" % acc
    for gran, crit in m["criteria_by_granularity"].items():
        assert crit["transport_path_works_in_sample"] is True, gran
    print("[PASS] test_transport_path_works_in_sample (同层自检检索 %.3f)" % acc)
    return True


# ---------------------------------------------------------------- 6
def test_cross_layer_slot_identity_does_not_travel():
    """6. 诊断: 跨层搬运后槽检索 ≈ 随机 ⇒ 槽级对齐不跨深度携带槽身份"""
    m = _m()
    r = m["arms"]["B_slot"]["retrieval"]
    assert r["slot_accuracy"] <= 0.35, "跨层槽检索 %.3f 明显高于随机, 与本代诊断冲突" % r["slot_accuracy"]
    assert r.get("slot_binom_p") is None or r["slot_binom_p"] > 0.05, "跨层槽检索与随机无异"
    assert r["in_sample_slot_accuracy"] >= 0.99, "同层自检应当接近 1.000"
    print("[PASS] test_cross_layer_slot_identity_does_not_travel (跨层 %.3f vs 同层 %.3f)"
          % (r["slot_accuracy"], r["in_sample_slot_accuracy"]))
    return True


# ---------------------------------------------------------------- 7
def test_g2b_fails_against_controls():
    """7. G2-b: 按槽搬运未在预登记 alpha 上击败同谱随机搬运与槽无关搬运"""
    m = _m()
    crit = m["criteria_by_granularity"]["slot"]
    assert crit["B_beats_C_random_paired"] is False, "B 竟击败了同谱随机搬运"
    assert crit["B_beats_Bplus_pooled_paired"] is False, "B 竟击败了槽无关搬运"
    assert crit["B_beats_C_wrongslot_paired"] is True, "B 应当显著优于错槽搬运"
    p = m["paired_contrasts"]["slot"]["B_minus_C_random"]["permutation_p"]
    # 预登记 alpha=0.01 ⇒ 必须没通过; 同时它落在 0.05 之下 (按 0.05 判就会翻案),
    # 这正是 SPEC.md §5 那条"alpha 纪律"要拦住的事 —— 数字变了就得重写裁决。
    assert 0.01 <= p < 0.05, "B-C_random 的 p 变了 (%s) —— 裁决需重写" % p
    assert m["verdict_by_granularity"]["slot"]["G2b_creation"] == "G2b_FAIL_RANDOM_AS_GOOD"
    print("[PASS] test_g2b_fails_against_controls (B-C_random p=%.4f, B-C_wrongslot p=%.4f)"
          % (p, m["paired_contrasts"]["slot"]["B_minus_C_wrongslot"]["permutation_p"]))
    return True


# ---------------------------------------------------------------- 8
def test_verdict_is_derivable_from_criteria():
    """8. 裁决必须能从判据矩阵重算出来 (防止结论与判据脱钩)"""
    m = _m()

    def derive(c):
        if not c["transport_path_works_in_sample"]:
            return ("G2a_INCONCLUSIVE_TRANSPORT_PATH_BROKEN",
                    "G2b_INCONCLUSIVE_TRANSPORT_PATH_BROKEN")
        if not c["native_baseline_detectable_at_this_granularity"]:
            va = "G2a_INCONCLUSIVE_NO_BASELINE_POWER"
        elif not c["B_separation_positive"] or not c["B_strat_perm_p_lt_0.01"]:
            va = "G2a_FAIL_NO_SEPARATION"
        elif not c["B_separation_not_smaller_than_A0"]:
            va = "G2a_FAIL_TRANSPORT_HURTS"
        else:
            va = "G2a_PASS"
        if not c["B_beats_C_random_paired"]:
            vb = "G2b_FAIL_RANDOM_AS_GOOD"
        elif not c["B_beats_C_wrongslot_paired"]:
            vb = "G2b_FAIL_WRONG_SLOT_AS_GOOD"
        elif not c["B_beats_Bplus_pooled_paired"]:
            vb = "G2b_FAIL_CIRCULAR"
        elif not c["B_separation_positive"]:
            vb = "G2b_FAIL_NO_SEPARATION"
        else:
            vb = "G2b_PASS"
        return (va, vb)

    for gran, c in m["criteria_by_granularity"].items():
        va, vb = derive(c)
        got = m["verdict_by_granularity"][gran]
        assert got["G2a_preservation"] == va, "%s: G2a 裁决与判据脱钩 (%s != %s)" % (gran, got["G2a_preservation"], va)
        assert got["G2b_creation"] == vb, "%s: G2b 裁决与判据脱钩 (%s != %s)" % (gran, got["G2b_creation"], vb)
    assert m["verdict"] == m["verdict_by_granularity"]["slot"]["G2a_preservation"]
    print("[PASS] test_verdict_is_derivable_from_criteria (三粒度裁决均可由判据重算)")
    return True


# ---------------------------------------------------------------- 9
def test_thresholds_were_fixed_before_the_run():
    """9. 阈值是跑之前写死的 (静态扫源码), 防止事后放宽 alpha"""
    src = _src()
    for frag in ("P_G2 = 0.01", "P_POWER = 0.05", "C_REL_TOL = 0.20",
                 "INSAMPLE_RETR_MIN = 0.50", "TOL_A0 = 0.05",
                 "判据阈值 (跑之前写死)"):
        assert frag in src, "源码里找不到阈值声明: %s" % frag
    assert 'return 0 if primary == "G2a_PASS" else 1' in src, "退出码契约变了"
    print("[PASS] test_thresholds_were_fixed_before_the_run (P_G2=0.01 等阈值在源码中写死)")
    return True


# ---------------------------------------------------------------- 10
def test_protocol_is_recorded_as_pre_registered():
    """10. 协议快照与跑前设计一致 (种子数/置换次数/主口径/拟合方案/统计单位)"""
    p = _m()["protocol"]
    assert p["seeds"] == [0, 1, 2, 3, 4], "种子数变了: %s" % p["seeds"]
    assert p["n_perm"] == 5000 and p["n_rep"] == 2048
    assert p["primary_metric"].startswith("cka"), "主口径必须是函数相似度"
    assert "leave-one-layer-out" in p["fit_scheme"], "跨层拟合方案没记上"
    assert "机械通路自检" in p["in_sample_scheme"], "同层拟合必须标为仅自检"
    assert "12" in p["experimental_unit"], "统计单位必须写明 12 个块"
    print("[PASS] test_protocol_is_recorded_as_pre_registered (5 种子/跨层拟合/CKA 主口径)")
    return True


# ---------------------------------------------------------------- 11
def test_known_caveats_are_declared():
    """11. 口径局限必须随数字一起登记 (JS 测不到共轭 / 压缩投影 / 缺 gate 路)"""
    caveats = " ".join(_m()["protocol"]["known_caveats"])
    for frag in ("奇异值", "压缩投影", "gate"):
        assert frag in caveats, "known_caveats 少了: %s" % frag
    print("[PASS] test_known_caveats_are_declared (%d 条局限已登记)" % len(_m()["protocol"]["known_caveats"]))
    return True


TESTS = [
    test_g1_reproduces_v62_within_tolerance,
    test_measure_sha256_is_the_provenance,
    test_shape_audit_exposes_family_composition,
    test_native_gate_has_no_power_in_either_metric,
    test_transport_path_works_in_sample,
    test_cross_layer_slot_identity_does_not_travel,
    test_g2b_fails_against_controls,
    test_verdict_is_derivable_from_criteria,
    test_thresholds_were_fixed_before_the_run,
    test_protocol_is_recorded_as_pre_registered,
    test_known_caveats_are_declared,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
