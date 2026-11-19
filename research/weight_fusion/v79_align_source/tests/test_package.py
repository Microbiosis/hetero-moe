"""V79.0 — S2-0: 定死 align[s|A→B] 的来源 (真模型激活上的成对对齐拟合与核验) 权威测试。

背景: S3 的注入算子需要一个跨模型对齐映射 align[s|A→B] (把源模型槽激活的共享空间
表示映到目标模型槽激活的共享空间)。S3_PROTOCOL §2.3 把「这个映射的来源」标成
**头号工程风险 (S2-0)**, 要求二选一并登记 sha256。

候选 (甲): v75 的成对对齐流程 (切片 + 随机输入 + 闭式 ridge + cyclic leave-one-layer-out)
候选 (乙): 在**真模型激活**上现拟合 (ridge / procrustes / low-rank)

本代结论是**负结果**, 测试把它的两面都锁住:
  - 判据面: A1 (槽特异性) 成立, 但 A2 (优于「不对齐」基线) 对**全部候选**不成立
            ⇒ 机器裁决 ALIGN_SOURCE_NOT_FOUND —— 不是崩溃, 是「没找到来源」。
  - 机理面 (post_hoc): Q 槽 mean-pool 后有效秩只有 4.8-38.7 (O 槽 58-69) ⇒ 跨模型的
            高 CKA 主要来自低秩共同成分; 去掉 top1 PC 后连块级映射的优势也消失。

本测试**不重跑实验** (需要真模型权重 + 切片), 它锁住的是:
  1. 负结果的判据与裁决 (A1 过 / A2 不过 ⇒ ALIGN_SOURCE_NOT_FOUND)
  2. 「A2 对全部候选不成立」这条核心结论 (防止日后被软化成「差一点」)
  3. post_hoc 只作机理, 不得回头翻案
  4. 阈值是跑前写死的 (静态扫源码) + 口径同源凭证

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

TITLE = "V79.0 S2-0 (对齐来源核验)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "s2_0_align.py")
MAPS = os.path.join(_pkg, "results", "align_maps.pt")
V78_MANIFEST = os.path.join(_pkg, "..", "v78_s3_preflight", "MANIFEST.json")
V78_SRC = os.path.join(_pkg, "..", "v78_s3_preflight", "s3_preflight.py")
V75_SRC = os.path.join(_pkg, "..", "v75_weight_conjugation", "g2_conjugation.py")
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
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- 1
def test_verdict_is_align_source_not_found():
    """1. 判据与裁决: A1 过 / A2 不过 => 机器裁决 ALIGN_SOURCE_NOT_FOUND"""
    m = _m()
    crit = m["criteria"]
    assert crit["power_gate_blocks_ok"] is True, "功效闸门没过 (块数不足)"
    assert crit["A1_some_candidate_has_slot_specificity"] is True, "A1 竟然没过"
    assert crit["A2_some_candidate_beats_noalign"] is False, "A2 竟然过了 ⇒ 裁决要重写"
    assert crit["A3_ridge_lambda_robust"] is False, "A3 竟然过了"
    assert crit["A4_source_decided_uniquely"] is False, "A4 竟然过了"
    assert m["verdict"] == "ALIGN_SOURCE_NOT_FOUND", "裁决不对: %s" % m["verdict"]
    assert m["chosen_source"] is None, "负结果却选出了来源: %s" % m["chosen_source"]
    assert m["eligible_candidates"] == [], "负结果却还有合格候选"
    assert _manifest()["acceptance"]["outcome"] == "fail", "MANIFEST 的验收结论与快照不一致"
    print("[PASS] test_verdict_is_align_source_not_found (A1 过 / A2 不过, verdict=%s)"
          % m["verdict"])
    return True


# ---------------------------------------------------------------- 2
def test_a2_fails_for_every_candidate():
    """2. 核心结论: **没有任何候选**能在留出集上超过「不对齐」基线"""
    by = _m()["by_candidate"]
    for cand, per in by.items():
        for slot, v in per.items():
            a2 = v["A2_beats_noalign"]
            assert a2["median"] <= 1e-6, (
                "%s/%s 的 A2 中位差变成正了: %s" % (cand, slot, a2["median"]))
            assert v["A2_ok"] is False, "%s/%s 的 A2 判据竟然判过" % (cand, slot)
    # 基线候选的 A2 中位差必须恰好是 0 (matched 与 noalign 是同一个量)
    none_q = by["none"]["Q"]
    assert abs(none_q["A2_beats_noalign"]["median"]) < 1e-12, (
        "基线的 A2 中位差不为 0: %s" % none_q["A2_beats_noalign"]["median"])
    # 而「错的」候选必须**更差** (否则是拟合没生效, 不是「学不到」)
    assert by["ridge"]["Q"]["A2_beats_noalign"]["median"] < 0.0
    assert by["lowrank16"]["Q"]["A2_beats_noalign"]["median"] < 0.0
    print("[PASS] test_a2_fails_for_every_candidate (五候选 × 两槽的 A2 中位差全 <= 0)")
    return True


# ---------------------------------------------------------------- 3
def test_a1_slot_specificity_is_visible():
    """3. A1 成立: 不对齐基线上同槽就比异槽像 (槽的差别跨模型确实可见)"""
    by = _m()["by_candidate"]
    q = by["none"]["Q"]
    assert q["mean_cka_matched"] > q["mean_cka_wrong_slot"], (
        "基线上同槽不比异槽像: %s" % q)
    assert q["A1_slot_specificity"]["n_positive"] >= 28, (
        "A1 的正号数变少了: %s" % q["A1_slot_specificity"])
    assert q["A1_slot_specificity"]["sign_p_two_sided"] < 0.001, (
        "A1 的 p 变了: %s" % q["A1_slot_specificity"])
    print("[PASS] test_a1_slot_specificity_is_visible (matched %.4f > wrong %.4f, p<1e-3)"
          % (q["mean_cka_matched"], q["mean_cka_wrong_slot"]))
    return True


# ---------------------------------------------------------------- 4
def test_post_hoc_is_labelled_and_not_used_for_verdict():
    """4. post_hoc 必须带标注, 且不得参与裁决"""
    m = _m()
    ph = m["post_hoc"]
    assert ph["status"] == "post_hoc_only_not_used_for_verdict", "post_hoc 的标注不对"
    assert "不参与" in ph["note"], "post_hoc 的 note 缺少限定"
    assert m["criteria"]["A2_some_candidate_beats_noalign"] is False, (
        "post_hoc 不得反过来把判据改成过")
    assert m["chosen_source"] is None
    print("[PASS] test_post_hoc_is_labelled_and_not_used_for_verdict (标注在案, 未参与裁决)")
    return True


# ---------------------------------------------------------------- 5
def test_post_hoc_mechanism_is_common_component():
    """5. post_hoc 机理: 去掉 top1 PC 后, 连块级映射也不再超过基线"""
    ph = _m()["post_hoc"]
    bs = ph["block_specific"]["ridge"]
    assert bs["A2_beats_noalign"]["median"] > 0.0, (
        "块级映射在去 PC 之前就该超过基线: %s" % bs["A2_beats_noalign"])
    res = ph["residualized_top1pc_blockspecific_ridge"]
    assert res["A2_beats_noalign"]["median"] < 0.0, (
        "去 top1 PC 后仍超过基线 ⇒ 机理判据变了: %s" % res["A2_beats_noalign"])
    assert res["A2_beats_noalign"]["sign_p_two_sided"] > 0.05, (
        "去 top1 PC 后的 A2 竟然显著: %s" % res["A2_beats_noalign"])
    print("[PASS] test_post_hoc_mechanism_is_common_component (去 PC 前 A2 %+.4f, 去 PC 后 %+.4f)"
          % (bs["A2_beats_noalign"]["median"], res["A2_beats_noalign"]["median"]))
    return True


# ---------------------------------------------------------------- 6
def test_pooled_eff_rank_explains_the_high_cka():
    """6. 机理的一半: Q 槽的池化表示有效秩远低于 O 槽 (低秩 ⇒ 天然易线性相关)"""
    er = _m()["post_hoc"]["pooled_eff_rank"]
    for l in (0, 1, 2, 3):
        q = er["Qwen|L%d|Q" % l]
        o = er["Qwen|L%d|O" % l]
        assert q < o, "L%d 的 Q 槽有效秩不该高于 O 槽: Q=%.2f O=%.2f" % (l, q, o)
    assert er["Qwen|L0|Q"] < 10.0, "Qwen L0 的 Q 槽有效秩不是个位数: %s" % er["Qwen|L0|Q"]
    assert er["BERT|L0|O"] > 50.0, "BERT O 槽的有效秩不该这么低: %s" % er["BERT|L0|O"]
    print("[PASS] test_pooled_eff_rank_explains_the_high_cka (Qwen L0: Q=%.1f vs O=%.1f)"
          % (er["Qwen|L0|Q"], er["Qwen|L0|O"]))
    return True


# ---------------------------------------------------------------- 7
def test_in_sample_vs_heldout_shows_no_overfitting():
    """7. 机理: 池化映射在训练集上也不高 ⇒ 不是过拟合, 是「没东西可学」"""
    io = _m()["post_hoc"]["in_sample_vs_heldout"]
    assert io["ridge_pooled_train_cka_median"] < io["noalign_test_cka_median"], (
        "池化映射在训练集上竟然超过基线 (那是过拟合, 不是本代结论): %s" % io)
    assert io["ridge_pooled_test_cka_median"] < io["noalign_test_cka_median"], (
        "池化映射在留出集上竟然超过基线 (与 A2 的结论矛盾): %s" % io)
    print("[PASS] test_in_sample_vs_heldout_shows_no_overfitting (train %.4f / test %.4f / base %.4f)"
          % (io["ridge_pooled_train_cka_median"], io["ridge_pooled_test_cka_median"],
             io["noalign_test_cka_median"]))
    return True


# ---------------------------------------------------------------- 8
def test_artifact_is_labelled_as_posthoc_product():
    """8. 产物: align_maps.pt 存在且**明确标注**是 post_hoc 依据的工程件, 不是裁决产物"""
    m = _m()
    assert os.path.exists(MAPS), "找不到产物: %s" % MAPS
    assert m["n_align_maps"] == 64, "映射张量数变了: %s" % m["n_align_maps"]
    live = _sha256(MAPS)
    assert live == m["align_maps_sha256"], (
        "align_maps.pt 的 sha256 不一致: 实测 %s, 快照 %s" % (live, m["align_maps_sha256"]))
    import torch
    pack = torch.load(MAPS, map_location="cpu", weights_only=False)
    assert pack["chosen_source"] is None, "产物里居然登记了被选中的来源"
    assert "post_hoc" in pack["artifact_kind"], (
        "产物的标注没写明是 post_hoc 依据: %s" % pack["artifact_kind"])
    print("[PASS] test_artifact_is_labelled_as_posthoc_product (64 张映射, 标注与 sha256 一致)")
    return True


# ---------------------------------------------------------------- 9
def test_thresholds_were_fixed_before_the_run():
    """9. 阈值是跑之前写死的 (静态扫源码), 防止事后放宽"""
    src = _src()
    for frag in ('"sign_p_max": 0.05', '"min_blocks": 32', '"ridge_lam": 1.0',
                 '"ridge_lam_robust": [0.1, 10.0]', '"lowrank_k": 16',
                 '"n_train": 64', '"n_test": 64'):
        assert frag in src, "源码里找不到阈值声明: %s" % frag
    assert 'verdict = "ALIGN_SOURCE_DECIDED" if all(criteria.values()) else "ALIGN_SOURCE_NOT_FOUND"' in src, \
        "裁决逻辑变了"
    assert 'return 0 if res["verdict"] == "ALIGN_SOURCE_DECIDED" else 1' in src, "退出码契约变了"
    print("[PASS] test_thresholds_were_fixed_before_the_run (七项阈值 + 裁决逻辑 + 退出码契约)")
    return True


# ---------------------------------------------------------------- 10
def test_provenance_sha256_matches_live_files():
    """10. 口径同源凭证: v78 / v75 / v62 的 sha256 与快照一致"""
    p = _m()["protocol"]["provenance_sha256"]
    for path, key in ((V78_SRC, "v78_preflight"), (V75_SRC, "v75_g2"), (V62_MEASURE, "v62_measure")):
        assert os.path.exists(path), "找不到复用件: %s" % path
        live = _sha256(path)
        assert live == p[key], "%s sha256 不一致: 实测 %s, 快照 %s" % (key, live, p[key])
    print("[PASS] test_provenance_sha256_matches_live_files (三份复用件 sha256 全部一致)")
    return True


# ---------------------------------------------------------------- 11
def test_protocol_declares_discipline_and_limits():
    """11. 三条纪律与局限必须随数字一起登记"""
    m = _m()
    disc = " ".join(m["protocol"]["discipline"])
    for frag in ("leave-block-out", "样本留出", "同口径"):
        assert frag in disc, "纪律少了: %s" % frag
    limits = " ".join(m["limits"])
    for frag in ("不产生", "前 4 层", "样本级"):
        assert frag in limits, "limits 少了: %s" % frag
    assert len(m["protocol"]["split"]["train_idx"]) == 63, "训练切分变了"
    assert len(m["protocol"]["split"]["test_idx"]) == 64, "留出切分变了"
    print("[PASS] test_protocol_declares_discipline_and_limits (%d 条纪律 / %d 条局限在案)"
          % (len(m["protocol"]["discipline"]), len(m["limits"])))
    return True


# ---------------------------------------------------------------- 12
def test_lineage_is_consistent():
    """12. 谱系自洽: v78 的 children 含 79.0, 本代 parent = 78.0"""
    with open(V78_MANIFEST, "r", encoding="utf-8") as f:
        v78 = json.load(f)
    assert "79.0" in (v78["generation"]["children"] or []), (
        "v78 的 children 没含 79.0: %s" % v78["generation"]["children"])
    man = _manifest()
    assert man["generation"]["parent"] == "78.0" and man["version"] == "79.0"
    assert "80.0" in man["generation"]["children"] or man["generation"]["children"] == []
    print("[PASS] test_lineage_is_consistent (78.0 -> 79.0)")
    return True


# ---------------------------------------------------------------- 13
def test_manifest_claim_is_honest_about_the_negative_result():
    """13. claim 的标签与 note 必须如实反映「负结果 + 机理」"""
    man = _manifest()
    assert man["claim"]["outcome_label"] == "falsified", man["claim"]["outcome_label"]
    assert man["acceptance"]["outcome"] == "fail"
    note = man["claim"]["note"]
    for frag in ("负结果", "A2", "共同成分"):
        assert frag in note, "claim 的 note 少了: %s" % frag
    assert man["leakage_audit"]["status"] == "not_applicable"
    assert man["superseded_by"] is None
    fals = man["falsified"]
    assert len(fals) >= 1, "证伪条目为空"
    for f in fals:
        assert f["evidence"], "证伪条目缺证据位置"
    print("[PASS] test_manifest_claim_is_honest_about_the_negative_result (%d 条证伪在案)"
          % len(fals))
    return True


TESTS = [
    test_verdict_is_align_source_not_found,
    test_a2_fails_for_every_candidate,
    test_a1_slot_specificity_is_visible,
    test_post_hoc_is_labelled_and_not_used_for_verdict,
    test_post_hoc_mechanism_is_common_component,
    test_pooled_eff_rank_explains_the_high_cka,
    test_in_sample_vs_heldout_shows_no_overfitting,
    test_artifact_is_labelled_as_posthoc_product,
    test_thresholds_were_fixed_before_the_run,
    test_provenance_sha256_matches_live_files,
    test_protocol_declares_discipline_and_limits,
    test_lineage_is_consistent,
    test_manifest_claim_is_honest_about_the_negative_result,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
