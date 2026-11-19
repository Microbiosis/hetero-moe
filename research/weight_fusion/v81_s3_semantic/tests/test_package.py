"""V81.0 - S3: 语义判据 (token 级探针 R2 + 置换检验) 权威测试.

背景: docs/S3_PROTOCOL.md (S3 的协议正本) + docs/A1_DESIGN.md §12。
v77 (真模型可用性) / v79 (对齐来源: 负结果) / v80 (注入算子 + λ 标定, 裁决
S2_OPERATOR_READY) 已经把机械前提全部打通; 本代第一次回答「注进去的信息,
目标模型读不读得出来」。

本代走的是一条链:
  (1) **协议预登记**: 判据 P1a/P1b/P2/P3①/P3② + P1c + P4, alpha=0.01, 置换 10^4,
      5 个 nuisance 种子, 块网格 84 块 (文本 64 + ViT 20) —— 全部写进源码常量
      (THRESHOLDS / SEEDS / N_PERM / ALPHA / LAMBDA_INJECT), 跑前写死;
  (2) **跑批**: 四臂 x 5 种子 x 84 块 x 128 样本, 逐块 JSONL checkpoint + 断点续跑;
  (3) **机器裁决**: S3_PASS (P1a ∧ P1b ∧ P2 ∧ P3① ∧ P3② 全过) 否则 S3_FAIL。

测试把三件事都锁住:
  - 机制面: 块网格/臂/种子与协议一致; arm0 是零注入基线且逐位等价; 注入范数比 == λ;
             探针的 R² 在**留出集**上算; λ 由**训练集内** CV 选出 (测试样本不参与)
  - 纪律面: 阈值跑前写死; P1c (ViT 平行臂) **不参与**主裁决; 样本级辅助探针
             **不参与**任何裁决; 置换次数与 p 一起报告; 不引入任何学出来的跨模型映射
  - 谱系面: v80.children 含 81.0, 本代 parent = 80.0

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

TITLE = "V81.0 S3 (语义判据: 探针 R2 + 置换检验)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "s3_flow.py")
DIAG_SRC = os.path.join(_pkg, "s3_ceiling_diag.py")
V80_MANIFEST = os.path.join(_pkg, "..", "v80_inject_operator", "MANIFEST.json")
V80_SRC = os.path.join(_pkg, "..", "v80_inject_operator", "s2_operator.py")
V77_SRC = os.path.join(_pkg, "..", "v77_true_model_flow", "true_model.py")
V78_SRC = os.path.join(_pkg, "..", "v78_s3_preflight", "s3_preflight.py")
V75_SRC = os.path.join(_pkg, "..", "v75_weight_conjugation", "g2_conjugation.py")
V62_MEASURE = os.path.join(_pkg, "..", "..", "aligner",
                            "v62_dimension_projection_space", "measure.py")

L_Q = (0, 1, 2, 3, 6, 12, 18, 23)
L_B = (0, 1, 2, 3)
L_V = (0, 3, 6, 9, 11)
ARMS = ("arm0_no_inject", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
SEEDS = (0, 1, 2, 3, 4)


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
        return hashlib.sha256(f.read()).hexdigest()


def _expected_blocks():
    out = set()
    for l in L_Q:
        for j in L_B:
            out.add(("Qwen->BERT", l, j))
    for l in L_B:
        for j in L_Q:
            out.add(("BERT->Qwen", l, j))
    for l in L_V:
        for j in L_V:
            if l != j:
                out.add(("ViT->ViT", l, j))
    return out


# ---------------------------------------------------------------- 1
def test_block_grid_matches_protocol():
    """1. 块网格与协议 §2.1 一致: 84 块 x 4 臂 x 5 种子 = 1680 行"""
    m = _m()
    rows = m["blocks"]
    got = set((r["direction"], r["source_layer"], r["target_layer"]) for r in rows)
    assert got == _expected_blocks(), "块集合与预期不符 (差 %d 个)" % len(got ^ _expected_blocks())
    assert len(rows) == 84 * 4 * 5, "行数不对: %s" % len(rows)
    assert set(r["arm"] for r in rows) == set(ARMS), "臂不对"
    assert set(r["seed"] for r in rows) == set(SEEDS), "种子不对"
    n_text = len(set((r["direction"], r["source_layer"], r["target_layer"])
                     for r in rows if r["direction"] != "ViT->ViT"))
    n_vit = len(set((r["direction"], r["source_layer"], r["target_layer"])
                    for r in rows if r["direction"] == "ViT->ViT"))
    assert n_text == 64, "主判据集 (文本块) 数不对: %s" % n_text
    assert n_vit == 20, "ViT 平行臂块数不对: %s" % n_vit
    print("[PASS] test_block_grid_matches_protocol (84 块 x 4 臂 x 5 种子 = %d 行; 文本 %d + ViT %d)"
          % (len(rows), n_text, n_vit))
    return True


# ---------------------------------------------------------------- 2
def test_thresholds_were_fixed_before_the_run():
    """2. 阈值与口径全部**跑前写死** (静态扫源码), 防止事后放宽"""
    src = _src()
    for frag in ("SEEDS = (0, 1, 2, 3, 4)",
                 "N_PERM = 10000",
                 "ALPHA = 0.01",
                 "P1C_ALPHA = 0.05",
                 "LAMBDA_INJECT = 0.03",
                 "PROBE_LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)",
                 "PROBE_CV_FOLDS = 5",
                 "TEXT_SPLIT_TRAIN = 64",
                 "VIT_SPLIT_TRAIN = 32",
                 "DROP_FIRST_TOKEN = True",
                 '"min_text_blocks": 32',
                 '"n_text_blocks": 64',
                 '"n_vit_blocks": 20',
                 '"p3_min_rel_perturbation": 1e-3',
                 '"p3_max_rel_perturbation": 0.5',
                 '"p3_selectivity_p_min": 0.05',
                 '"p2_require_same_ge_cross": True',
                 '"verdict": "S3_PASS" if passed else "S3_FAIL"',
                 'return 0 if res["verdict"] == "S3_PASS" else 1'):
        assert frag in src, "源码里找不到预登记声明: %s" % frag
    th = _m()["protocol"]["thresholds"]
    assert th["alpha"] == 0.01 and th["n_perm"] == 10000 and th["n_seeds"] == 5
    assert th["inject_lambda"] == 0.03, "注入 λ 不是 v80 标定出的 0.03"
    print("[PASS] test_thresholds_were_fixed_before_the_run (十七项预登记声明 + 裁决逻辑 + 退出码契约)")
    return True


# ---------------------------------------------------------------- 3
def test_arm0_is_zero_injection_baseline():
    """3. arm0 是零注入基线: 逐种子值相同, 且零注入与参考前向**逐位等价**"""
    m = _m()
    rows = [r for r in m["blocks"] if r["arm"] == "arm0_no_inject"]
    assert len(rows) == 84 * 5, "arm0 行数不对: %s" % len(rows)
    by_block = {}
    for r in rows:
        key = (r["direction"], r["source_layer"], r["target_layer"])
        by_block.setdefault(key, []).append(r)
    for key, rs in by_block.items():
        assert len(rs) == 5, "%s 的 arm0 种子数不对" % (key,)
        vals = set(round(r["r2_q"], 12) for r in rs)
        assert len(vals) == 1, "arm0 与种子无关, %s 的取值却随种子变" % (key,)
    worst = max(r["g1_max_rel_err"] for r in rows)
    assert worst == 0.0, "零注入与原前向不是逐位等价: max rel_err = %s" % worst
    print("[PASS] test_arm0_is_zero_injection_baseline (arm0 逐种子恒定, 84 块 g1_max_rel_err 恰为 0)")
    return True


# ---------------------------------------------------------------- 4
def test_injection_norm_ratio_is_exact():
    """4. 注入强度: 逐样本 ||delta||/||h_B^j|| == λ = 0.03 (v80 S2-1 标定值)"""
    m = _m()
    lam = m["protocol"]["thresholds"]["inject_lambda"]
    inj = [r for r in m["blocks"] if r["arm"] != "arm0_no_inject"]
    worst = max(r["norm_ratio_max_abs_dev"] for r in inj)
    assert worst <= 1e-6, "注入臂的范数比偏差超容差: %s" % worst
    # arm0 是零注入 => 范数比恒为 0 => 与 lambda 的偏差恰为 lambda (设计如此, 不是缺陷)
    a0 = max(abs(r["norm_ratio_max_abs_dev"] - lam) for r in m["blocks"]
             if r["arm"] == "arm0_no_inject")
    assert a0 <= 1e-6, "arm0 的零注入范数比不是 0: %s" % a0
    print("[PASS] test_injection_norm_ratio_is_exact (%d 行注入臂范数比偏差 max %.2e <= 1e-6; "
          "arm0 零注入比值恒 0)" % (len(inj), worst))
    return True


# ---------------------------------------------------------------- 5
def test_probe_lambda_selected_on_train_only():
    """5. 探针 λ 由**训练集内** 5 折 CV 在写死网格上选出; 测试样本全程不参与"""
    m = _m()
    lam_info = m["protocol"]["probe_lambda_by_direction"]
    assert set(lam_info) == set(("Qwen->BERT", "BERT->Qwen", "ViT->ViT")), (
        "三个方向都该有 λ 标定记录: %s" % list(lam_info))
    grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    for dname, info in lam_info.items():
        assert tuple(info["grid"]) == grid, "%s 的 λ 候选网格变了: %s" % (dname, info["grid"])
        assert info["n_folds"] == 5, "%s 的 CV 折数变了" % dname
        assert info["probe_lambda"] in grid, "%s 选出的 λ 不在网格里" % dname
        assert info["probe_lambda_pooled_aux"] in grid, "%s 的辅助探针 λ 不在网格里" % dname
        assert info["label_lambda"] in grid, "%s 的标签探针 λ 不在网格里" % dname
        assert len(info["pilot_blocks"]) == 3, "%s 的 pilot 块数不对" % dname
    for r in m["blocks"]:
        assert r["n_train"] == r["n_test"], "训练/测试样本数应当相等: %s" % r
    print("[PASS] test_probe_lambda_selected_on_train_only (三方向 λ 均出自写死网格 + 5 折训练内 CV)")
    return True


# ---------------------------------------------------------------- 6
def test_permutation_reports_p_and_n_perm():
    """6. 置换检验必须同时报告 p 与置换次数 (10^4), 且 alpha = 0.01"""
    m = _m()
    c = m["criteria"]
    assert c["P1a"]["n_perm"] == 10000, "P1a 的置换次数不对"
    assert c["P1a"]["alpha"] == 0.01, "P1a 的 alpha 不对"
    assert c["P2"]["n_perm"] == 10000, "P2 的置换次数不对"
    for key, v in c["P1b"]["comparisons"].items():
        assert v["n_perm"] == 10000, "P1b(%s) 的置换次数不对" % key
        assert v["permutation_p"] is not None, "P1b(%s) 缺 p" % key
    assert c["P3_2"]["n_perm"] == 10000, "P3② 的置换次数不对"
    assert c["P1a"]["permutation_p"] is not None, "P1a 缺 p"
    assert c["P2"]["permutation_p"] is not None, "P2 缺 p"
    print("[PASS] test_permutation_reports_p_and_n_perm (P1a/P1b/P2/P3② 均报 p 与 n_perm=10000)")
    return True


# ---------------------------------------------------------------- 7
def test_p1c_is_reported_but_not_in_verdict():
    """7. P1c (ViT 平行臂) 单独报告, **不参与**主裁决 (协议 §2.2)"""
    m = _m()
    c = m["criteria"]
    assert "P1c" in c, "P1c 必须单独报告"
    assert "P1c" not in m["verdict_detail"]["failed_criteria"], "P1c 混进了失败判据"
    assert "不参与" in c["P1c"]["role"], "P1c 的角色说明缺「不参与」"
    assert c["P1a"]["n_blocks"] == 64, "主判据集必须是 64 个文本块"
    assert c["P2"]["n_same_layer"] + c["P2"]["n_cross_layer"] == 64, (
        "P2 的同层 + 跨层块数必须等于 64")
    print("[PASS] test_p1c_is_reported_but_not_in_verdict (P1c 单列; 主判据集 = 64 个文本块)")
    return True


# ---------------------------------------------------------------- 8
def test_pooled_aux_probe_does_not_affect_verdict():
    """8. 样本级辅助探针是**预登记的兜底读数**, 不参与任何裁决"""
    m = _m()
    src = _src()
    rows = m["blocks"]
    assert all("r2_q_pooled_aux" in r for r in rows), "辅助读数缺失"
    assert "pooled_aux" in src, "源码里没有辅助读数"
    body = src[src.index("def evaluate_criteria"):src.index("# ================================================================ 主流程")]
    for bad in ("r2_q_pooled_aux", "r2_o_pooled_aux"):
        assert bad not in body, "辅助读数 %s 混进了判据聚合: 不得参与裁决" % bad
    print("[PASS] test_pooled_aux_probe_does_not_affect_verdict (辅助读数在案, 但未进入判据聚合)")
    return True


# ---------------------------------------------------------------- 9
def test_no_learned_cross_model_mapping():
    """9. 注入算子里**没有任何学出来的跨模型映射** (协议 §2.3 修订记录)"""
    src = _src()
    for bad in ("procrustes(", "fit_transport(", "scipy.optimize", "align["):
        assert bad not in src, "源码里出现了学出来的跨模型映射步骤: %s" % bad
    assert "build_delta" in src, "注入走的是 v80 的 build_delta"
    print("[PASS] test_no_learned_cross_model_mapping (无 align/procrustes/transport 拟合步骤)")
    return True


# ---------------------------------------------------------------- 10
def test_provenance_sha256_matches_live_files():
    """10. 口径同源凭证: 六份复用件/本代脚本的 sha256 与快照一致"""
    p = _m()["protocol"]["provenance_sha256"]
    for path, key in ((V62_MEASURE, "v62_measure"),
                      (V75_SRC, "v75_g2_conjugation"),
                      (V77_SRC, "v77_true_model"),
                      (V78_SRC, "v78_s3_preflight"),
                      (V80_SRC, "v80_s2_operator"),
                      (SRC, "v81_s3_flow")):
        assert os.path.exists(path), "找不到复用件: %s" % path
        live = _sha256(path)
        assert live == p[key], "%s sha256 不一致: 实测 %s, 快照 %s" % (key, live, p[key])
    print("[PASS] test_provenance_sha256_matches_live_files (六份复用件 sha256 全部一致)")
    return True


# ---------------------------------------------------------------- 11
def test_lineage_is_consistent():
    """11. 谱系自洽: v80 的 children 含 81.0, 本代 parent = 80.0"""
    v80 = _load(V80_MANIFEST)
    assert "81.0" in (v80["generation"]["children"] or []), (
        "v80 的 children 没含 81.0: %s" % v80["generation"]["children"])
    man = _manifest()
    assert man["generation"]["parent"] == "80.0" and man["version"] == "81.0"
    assert "82.0" in man["generation"]["children"] or man["generation"]["children"] == []
    print("[PASS] test_lineage_is_consistent (80.0 -> 81.0)")
    return True


# ---------------------------------------------------------------- 12
def test_protocol_declares_discipline_and_limits():
    """12. 纪律与局限必须随数字一起登记"""
    m = _m()
    disc = " ".join(m["protocol"]["discipline"])
    for frag in ("跑前写死", "置换", "留出集"):
        assert frag in disc, "纪律少了: %s" % frag
    lims = m["protocol"]["limits"]
    assert len(lims) >= 5, "limits 条目不足"
    lim = " ".join(lims)
    for frag in ("256", "ViT", "层号"):
        assert frag in lim, "limits 少了: %s" % frag
    print("[PASS] test_protocol_declares_discipline_and_limits (%d 条纪律 / %d 条局限在案)"
          % (len(m["protocol"]["discipline"]), len(lims)))
    return True


# ---------------------------------------------------------------- 13
def test_verdict_matches_criteria():
    """13. 机器裁决与判据表必须**自洽** (verdict 只能由 P1a/P1b/P2/P3①/P3② 推出)"""
    m = _m()
    c = m["criteria"]
    flags = {"P1a": c["P1a"]["passed"],
             "P1b": c["P1b"]["passed"],
             "P2": c["P2"]["passed"],
             "P3_1": c["P3_1"]["passed"],
             "P3_2": c["P3_2"]["passed"]}
    expect = "S3_PASS" if all(flags.values()) else "S3_FAIL"
    assert m["verdict"] == expect, (
        "裁决与判据不自洽: verdict=%s, 期望 %s" % (m["verdict"], expect))
    failed = sorted(n for n, ok in flags.items() if not ok)
    assert sorted(m["verdict_detail"]["failed_criteria"]) == failed, (
        "失败判据清单不对: %s vs %s" % (m["verdict_detail"]["failed_criteria"], failed))
    assert "P1c" not in failed, "P1c 不得进入失败清单"
    print("[PASS] test_verdict_matches_criteria (verdict=%s, 失败判据 %s)"
          % (m["verdict"], failed or "无"))
    return True


# ---------------------------------------------------------------- 14
def test_manifest_claim_is_honest():
    """14. claim / acceptance 必须如实反映实际裁决"""
    m = _m()
    man = _manifest()
    assert man["claim"]["outcome_label"] in ("supported", "refuted", "inconclusive"), (
        "claim 的 outcome_label 不合法: %s" % man["claim"]["outcome_label"])
    assert man["acceptance"]["outcome"] in ("pass", "fail"), (
        "acceptance 的 outcome 不合法: %s" % man["acceptance"]["outcome"])
    expect = "pass" if m["verdict"] == "S3_PASS" else "fail"
    assert man["acceptance"]["outcome"] == expect, (
        "MANIFEST 的验收结论与机器裁决不一致: %s vs %s" % (
            man["acceptance"]["outcome"], expect))
    assert man["leakage_audit"]["status"] in ("clean", "not_applicable"), (
        "leakage_audit 状态不合法: %s" % man["leakage_audit"]["status"])
    assert man["superseded_by"] is None
    print("[PASS] test_manifest_claim_is_honest (acceptance=%s 与机器裁决一致)"
          % man["acceptance"]["outcome"])
    return True


# ---------------------------------------------------------------- 15
def test_p1a_penetration_is_locked():
    """15. P1a 穿透**通过**: 注入确实把源内容搬进了目标模型 (数字锁定)"""
    m = _m()
    p1a = m["criteria"]["P1a"]
    assert p1a["passed"] is True, "P1a 竟然没过"
    assert p1a["n_blocks"] == 64 and p1a["n_obs"] == 320, "P1a 的块数/观测数变了"
    assert p1a["n_positive"] == 256, "P1a 的正号数变了: %s" % p1a["n_positive"]
    assert abs(p1a["median_delta"] - 0.003232) < 1e-5, "P1a 的中位增量变了: %s" % p1a["median_delta"]
    assert abs(p1a["mean_delta"] - 0.055932) < 1e-5, "P1a 的均值增量变了: %s" % p1a["mean_delta"]
    assert p1a["permutation_p"] == 0.0, "P1a 的 p 变了: %s" % p1a["permutation_p"]
    # 与随机方向臂的对照 (P1b 的另一半) 才是「内容有效」的证据
    m3 = m["criteria"]["P1b"]["comparisons"]["arm3_random_dir"]
    assert m3["permutation_p"] == 0.0, "arm1 vs arm3 的 p 变了: %s" % m3["permutation_p"]
    print("[PASS] test_p1a_penetration_is_locked (P1a 中位 Δ=%+.6f, p=0.0, 256/320 正; "
          "arm1>>arm3 p=0.0)" % p1a["median_delta"])
    return True


# ---------------------------------------------------------------- 16
def test_p1b_slot_specificity_failed():
    """16. P1b 槽特异**失败**: Q 槽与 O 槽的注入效果不可区分 (数字锁定)"""
    m = _m()
    p1b = m["criteria"]["P1b"]
    assert p1b["passed"] is False, "P1b 竟然过了"
    c2 = p1b["comparisons"]["arm2_wrong_slot_O"]
    assert c2["n_obs"] == 320 and c2["n_positive"] == 112, "arm1-arm2 的观测数/正号数变了"
    assert abs(c2["median_delta"] - (-0.002455)) < 1e-5, "arm1-arm2 的中位差变了: %s" % c2["median_delta"]
    assert abs(c2["permutation_p"] - 0.9521) < 1e-3, "arm1-arm2 的 p 变了: %s" % c2["permutation_p"]
    c3 = p1b["comparisons"]["arm3_random_dir"]
    assert c3["n_positive"] == 299, "arm1-arm3 的正号数变了"
    print("[PASS] test_p1b_slot_specificity_failed (arm1-arm2 中位 %+.6f, p=%.4f => 不可区分)"
          % (c2["median_delta"], c2["permutation_p"]))
    return True


# ---------------------------------------------------------------- 17
def test_p2_depth_semantics_failed():
    """17. P2 深度语义**失败**: 同层与跨层的注入增量不可区分 (数字锁定)"""
    m = _m()
    p2 = m["criteria"]["P2"]
    assert p2["passed"] is False, "P2 竟然过了"
    assert p2["n_same_layer"] == 8 and p2["n_cross_layer"] == 56, "P2 的分组块数变了"
    assert abs(p2["median_same_layer"] - 0.002586) < 1e-5, "P2 同层中位变了"
    assert abs(p2["median_cross_layer"] - 0.002295) < 1e-5, "P2 跨层中位变了"
    assert abs(p2["permutation_p"] - 0.8714) < 1e-3, "P2 的 p 变了: %s" % p2["permutation_p"]
    print("[PASS] test_p2_depth_semantics_failed (同层 %+.6f vs 跨层 %+.6f, p=%.4f)"
          % (p2["median_same_layer"], p2["median_cross_layer"], p2["permutation_p"]))
    return True


# ---------------------------------------------------------------- 18
def test_p3_1_guard_passes():
    """18. P3① 守门**通过**: 全部注入臂块的输出扰动都在区间内, 没把 B 搞坏"""
    m = _m()
    p31 = m["criteria"]["P3_1"]
    assert p31["passed"] is True, "P3① 竟然没过"
    assert p31["n_rows"] == 1260, "P3① 的行数变了: %s" % p31["n_rows"]
    assert p31["n_violations"] == 0, "P3① 有违例: %s" % p31["n_violations"]
    assert p31["min_rel_perturbation"] > 1e-3, "P3① 下界被破"
    assert p31["max_rel_perturbation"] <= 0.5, "P3① 上界被破"
    print("[PASS] test_p3_1_guard_passes (1260 行, 违例 0, 扰动 %.5f-%.5f)"
          % (p31["min_rel_perturbation"], p31["max_rel_perturbation"]))
    return True


# ---------------------------------------------------------------- 19
def test_p3_2_selectivity_failed():
    """19. P3② 选择性**失败**: 注入 Q 槽也把非目标槽 (O) 抬可读了 (数字锁定)"""
    m = _m()
    p32 = m["criteria"]["P3_2"]
    assert p32["passed"] is False, "P3② 竟然过了"
    assert p32["n_obs"] == 320 and p32["n_positive"] == 248, "P3② 的观测数/正号数变了"
    assert abs(p32["median_delta_non_target_slot"] - 0.000836) < 1e-5, "P3② 的中位增量变了"
    assert p32["permutation_p"] == 0.0, "P3② 的 p 变了: %s" % p32["permutation_p"]
    print("[PASS] test_p3_2_selectivity_failed (非目标槽 Δ=%+.6f, p=%.4f => 无选择性)"
          % (p32["median_delta_non_target_slot"], p32["permutation_p"]))
    return True


# ---------------------------------------------------------------- 20
def test_vit_label_probe_defect_is_locked():
    """20. **已知缺陷留档**: ViT 臂的位置切分使标签探针恒为 0 ⇒ P1c 不可判定"""
    m = _m()
    rows = m["blocks"]
    vit = [r for r in rows if r["direction"] == "ViT->ViT"]
    assert vit, "找不到 ViT 臂的行"
    assert all(r["label_acc"] == 0.0 for r in vit), "ViT 的标签可读性不是恒 0 (缺陷变了)"
    # 文本臂的标签探针是好的 => 缺陷是 ViT 切分特有, 不是探针实现坏了
    txt = [r["label_acc"] for r in rows if r["direction"] != "ViT->ViT"]
    assert min(txt) > 0.3, "文本臂的标签可读性也坏了: %s" % min(txt)
    p1c = m["criteria"]["P1c"]
    assert "P1c" not in m["verdict_detail"]["failed_criteria"], "P1c 混进了失败清单"
    man = _manifest()
    joined = " ".join([json.dumps(man.get("falsified", []), ensure_ascii=False),
                       json.dumps(man.get("acceptance", {}), ensure_ascii=False)])
    assert "ViT" in joined and "P1c" in joined, "MANIFEST 没登记 ViT 标签探针缺陷"
    print("[PASS] test_vit_label_probe_defect_is_locked (ViT label_acc 恒 0; 文本臂 >= %.3f; "
          "P1c 不参与主裁决; 缺陷已在 MANIFEST 登记)" % min(txt))
    return True


# ---------------------------------------------------------------- 21
def test_post_hoc_diagnostics_are_not_criteria():
    """21. 两个 post_hoc_only 诊断**不参与**任何裁决, 但必须留档"""
    pkg = os.path.dirname(_pkg)
    for rel, key in (("results/ceiling_diag.json", "ceiling_diag"),
                     ("results/qo_similarity.json", "qo_similarity")):
        path = os.path.join(_pkg, rel)
        assert os.path.exists(path), "缺诊断快照: %s" % rel
        d = _load(path)
        assert "post_hoc_only" in d.get("role", ""), "%s 没标 post_hoc_only" % rel
        man = _manifest()
        assert key in man["artifacts"], "MANIFEST 没登记 %s" % key
    # 判据聚合里不得出现这两个诊断的任何字段
    src = _src()
    body = src[src.index("def evaluate_criteria"):src.index("# ================================================================ 主流程")]
    for bad in ("r2_project", "delta_over_project", "qo_similarity", "ceiling_diag"):
        assert bad not in body, "post_hoc 诊断的量 %s 混进了判据聚合" % bad
    print("[PASS] test_post_hoc_diagnostics_are_not_criteria (两份诊断留档且未进入判据聚合)")
    return True


TESTS = [
    test_block_grid_matches_protocol,
    test_thresholds_were_fixed_before_the_run,
    test_arm0_is_zero_injection_baseline,
    test_injection_norm_ratio_is_exact,
    test_probe_lambda_selected_on_train_only,
    test_permutation_reports_p_and_n_perm,
    test_p1c_is_reported_but_not_in_verdict,
    test_pooled_aux_probe_does_not_affect_verdict,
    test_no_learned_cross_model_mapping,
    test_provenance_sha256_matches_live_files,
    test_lineage_is_consistent,
    test_protocol_declares_discipline_and_limits,
    test_verdict_matches_criteria,
    test_manifest_claim_is_honest,
    test_p1a_penetration_is_locked,
    test_p1b_slot_specificity_failed,
    test_p2_depth_semantics_failed,
    test_p3_1_guard_passes,
    test_p3_2_selectivity_failed,
    test_vit_label_probe_defect_is_locked,
    test_post_hoc_diagnostics_are_not_criteria,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
