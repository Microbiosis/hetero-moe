"""V80.0 - S2: 注入算子下沉到真模型层边界 (四臂跑通 + 机械闸门) 权威测试.

背景: docs/A1_DESIGN.md §8 的 S2 + docs/S3_PROTOCOL.md §2.3 / §2.5 / §4.
v79 (S2-0) 的负结果已把注入算子口径定死: 保留共享空间载体, **不学任何学出来的映射**。

本代走了一条完整的链:
  (1) **先导运行** (λ=0.1, 块 Frobenius 口径): 机制层五项全过, 但 C4「没把 B 搞坏」
      被证伪 (36/252 臂块 > 0.5, 全部在 BERT->Qwen 的 j∈{3,6,12}) ⇒ 归档在
      results/run_lambda0p1.json, **裁决 S2_OPERATOR_FAIL**;
  (2) **S2-1 λ 标定** (写死的候选集 + 写死的选择规则): 裁决 CALIB_LAMBDA_SELECTED,
      λ* = 0.03 (快照 results/lambda_calibration.json);
  (3) **闸门重跑** (λ=0.03): 六项机械闸门全过 ⇒ 裁决 S2_OPERATOR_READY。

测试把三件事都锁住:
  - 机制面: 零注入的「打断 -> 续跑」与原前向**逐位等价**; 注入量逐样本精确归一化;
             四臂跑得通; 读出层扰动全部在范围内
  - 纪律面: **先导运行的失败必须留档且不得被抹掉**; 标定规则是写死的规则而非看结果挑;
             阈值跑前写死; 算子里没有任何学出来的映射; 不得被当成语义判据的证据

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

TITLE = "V80.0 S2 (注入算子下沉 + λ 标定)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
BASE_RUN = os.path.join(_pkg, "results", "run_lambda0p1.json")
BASE_SRC = os.path.join(_pkg, "results", "run_lambda0p1_source.py")
CALIB = os.path.join(_pkg, "results", "lambda_calibration.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "s2_operator.py")
CALIB_SRC = os.path.join(_pkg, "s2_lambda_calib.py")
V79_MANIFEST = os.path.join(_pkg, "..", "v79_align_source", "MANIFEST.json")
V77_SRC = os.path.join(_pkg, "..", "v77_true_model_flow", "true_model.py")
V77_METRICS = os.path.join(_pkg, "..", "v77_true_model_flow", "results", "metrics.json")
V62_MEASURE = os.path.join(_pkg, "..", "..", "aligner",
                            "v62_dimension_projection_space", "measure.py")

L_Q = (0, 1, 2, 3, 6, 12, 18, 23)
L_B = (0, 1, 2, 3)
L_V = (0, 3, 6, 9, 11)
ARMS = ("arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _m():
    return _load(METRICS)


def _base():
    return _load(BASE_RUN)


def _calib():
    return _load(CALIB)


def _manifest():
    return _load(MANIFEST)


def _src():
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read()


def _calib_src():
    with open(CALIB_SRC, "r", encoding="utf-8") as f:
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
    for l in L_V:
        for j in L_V:
            if l != j:
                out.add(("ViT->ViT", l, j))
    return out


# ---------------------------------------------------------------- 1
def test_verdict_is_s2_operator_ready():
    """1. 标定后重跑: 六项机械闸门全过 => 机器裁决 S2_OPERATOR_READY"""
    m = _m()
    crit = m["criteria"]
    for key in ("G1_arm0_bit_equivalent", "C2_norm_ratio_exact",
                "C3_readout_perturbation_visible", "C4_no_model_break",
                "C5_arm3_decorrelated", "C6_all_finite"):
        assert key in crit, "判据少了: %s" % key
        assert crit[key] is True, "判据没过: %s" % key
    assert m["verdict"] == "S2_OPERATOR_READY", "裁决不对: %s" % m["verdict"]
    assert _manifest()["acceptance"]["outcome"] == "pass", "MANIFEST 的验收结论与快照不一致"
    print("[PASS] test_verdict_is_s2_operator_ready (六项闸门全过, verdict=%s)" % m["verdict"])
    return True


# ---------------------------------------------------------------- 2
def test_arm0_is_bit_equivalent():
    """2. 机械前提: 零注入的「打断 -> 续跑」与原前向**逐位等价** (rel_err == 0)"""
    m = _m()
    g1 = m["G1_arm0"]
    assert g1["n_checks"] == 17, "注入点核验数不对 (应 4 BERT + 8 Qwen + 5 ViT): %s" % g1["n_checks"]
    assert g1["max_rel_err"] == 0.0, "arm0 的最大相对误差不是 0: %s" % g1["max_rel_err"]
    for r in g1["rows"]:
        assert r["max_rel_err"] == 0.0, "arm0 在 %s 目标层 %d 上不是逐位等价: %s" % (
            r["direction"], r["target_layer"], r["max_rel_err"])
        for v in r["per_layer_rel_err"]:
            assert v == 0.0, "逐层相对误差不为 0: %s" % v
    print("[PASS] test_arm0_is_bit_equivalent (%d 个注入点, 全部 rel_err == 0)" % g1["n_checks"])
    return True


# ---------------------------------------------------------------- 3
def test_norm_ratio_is_exact():
    """3. 注入强度: 逐样本 ||delta||_F / ||h_B^j||_F == λ (λ = 0.03, 由 S2-1 标定)"""
    m = _m()
    lam = m["protocol"]["thresholds"]["C2_norm_ratio_target"]
    assert lam == 0.03, "λ 不是标定选出的 0.03: %s" % lam
    tol = m["protocol"]["thresholds"]["C2_norm_ratio_atol"]
    for arm, v in m["summary_by_arm"].items():
        assert v["norm_ratio_max_abs_dev"] <= tol, (
            "%s 的范数比偏差超容差: %s > %s" % (arm, v["norm_ratio_max_abs_dev"], tol))
    for r in m["blocks"]:
        assert abs(r["norm_ratio_median"] - lam) <= tol, (
            "%s/%d->%d/%s 的范数比不是 λ: %s" % (
                r["direction"], r["source_layer"], r["target_layer"], r["arm"],
                r["norm_ratio_median"]))
    print("[PASS] test_norm_ratio_is_exact (252 行注入的范数比 == λ=%.3g, 偏差 <= %.g)"
          % (lam, tol))
    return True


# ---------------------------------------------------------------- 4
def test_readout_perturbation_is_visible():
    """4. 注入在读出层确实可见: 每臂落进 (1e-3, 0.5) 的文本块数 >= 32"""
    m = _m()
    need = m["protocol"]["thresholds"]["C3_min_blocks_in_range"]
    for arm, v in m["summary_by_arm"].items():
        assert v["n_blocks_text"] == 64, "%s 的文本块数不对: %s" % (arm, v["n_blocks_text"])
        assert v["n_in_range_readout_text"] >= need, (
            "%s 落进范围的文本块数不足: %s < %s" % (arm, v["n_in_range_readout_text"], need))
    worst = max(r["rel_perturb_readout"] for r in m["blocks"])
    assert worst < 0.5, "读出层扰动不该超上限: %s" % worst
    print("[PASS] test_readout_perturbation_is_visible (三臂各 >= %d 块落进范围, 最大 %.4f)"
          % (need, worst))
    return True


# ---------------------------------------------------------------- 5
def test_c4_now_holds_at_calibrated_lambda():
    """5. 标定后的 λ: 全部块的 final 输出相对扰动都在 P3① 的 0.5 上限内"""
    m = _m()
    cap = m["protocol"]["thresholds"]["C4_max_rel_perturbation"]
    worst = max(r["rel_perturb_final"] for r in m["blocks"])
    assert worst <= cap, "final 输出扰动仍超上限: %s > %s" % (worst, cap)
    assert m["criteria"]["C4_no_model_break"] is True, "C4 竟然没过"
    print("[PASS] test_c4_now_holds_at_calibrated_lambda (252 块最大 final 扰动 %.4f <= %.2f)"
          % (worst, cap))
    return True


# ---------------------------------------------------------------- 6
def test_leading_failure_is_preserved_on_record():
    """6. 纪律锁: λ=0.1 的**先导失败**必须留档, 不得被标定后的成功抹掉"""
    b = _base()
    assert b["verdict"] == "S2_OPERATOR_FAIL", "先导运行的裁决变了: %s" % b["verdict"]
    assert b["criteria"]["C4_no_model_break"] is False, "先导运行的 C4 竟然过了"
    cap = b["protocol"]["thresholds"]["C4_max_rel_perturbation"]
    viol = [r for r in b["blocks"] if r["rel_perturb_final"] > cap]
    assert len(viol) == 36, "先导运行的超限臂块数变了: %s" % len(viol)
    assert set(r["direction"] for r in viol) == set(["BERT->Qwen"]), "先导运行的失效方向变了"
    assert set(r["target_layer"] for r in viol) == set([3, 6, 12]), "先导运行的失效目标层变了"
    man = _manifest()
    assert man["artifacts"].get("lambda01_leading_run") == "results/run_lambda0p1.json", (
        "MANIFEST 没登记先导运行: %s" % man["artifacts"])
    fals = man["falsified"]
    assert len(fals) >= 1, "证伪条目为空"
    assert any("0.1" in (f.get("hypothesis", "") + f.get("verdict", "")) for f in fals), (
        "证伪条目里没有 λ=0.1 那条")
    print("[PASS] test_leading_failure_is_preserved_on_record (先导运行仍判 FAIL, 36/252 超限在案)")
    return True


# ---------------------------------------------------------------- 7
def test_leading_failure_was_content_independent():
    """7. 机理锁: 先导运行的失效**内容无关** (三臂同样被打坏) ⇒ 不是槽内容的问题"""
    b = _base()
    cap = b["protocol"]["thresholds"]["C4_max_rel_perturbation"]
    by_block = {}
    for r in b["blocks"]:
        by_block.setdefault((r["direction"], r["source_layer"], r["target_layer"]), {})[r["arm"]] = r
    checked = 0
    for key, arms in by_block.items():
        if not any(a["rel_perturb_final"] > cap for a in arms.values()):
            continue
        checked += 1
        vals = sorted(a["rel_perturb_final"] for a in arms.values())
        assert vals[-1] / max(vals[0], 1e-9) < 1.6, "%s 三臂差异过大: %s" % (key, vals)
    assert checked == 12, "受检的违规块数不对: %s" % checked
    arm3_bad = sum(1 for r in b["blocks"]
                   if r["arm"] == "arm3_random_dir" and r["rel_perturb_final"] > cap)
    assert arm3_bad == 12, "随机方向臂的违规数不对: %s" % arm3_bad
    print("[PASS] test_leading_failure_was_content_independent (12 个违规块三臂比值 < 1.6)")
    return True


# ---------------------------------------------------------------- 8
def test_arm3_is_decorrelated():
    """8. arm3 与真实内容不相关: |cos| >= 0.35 的样本数为 0"""
    m = _m()
    c5 = m["C5_cos"]
    thr = m["protocol"]["thresholds"]["C5_arm3_cos_threshold"]
    assert c5["total_violations"] == 0, "arm3 出现了高相关样本: %s" % c5["total_violations"]
    assert c5["max_abs_cos_arm3_vs_arm1"] < thr, "arm3 vs arm1 的 |cos| 超阈: %s" % c5["max_abs_cos_arm3_vs_arm1"]
    assert c5["max_abs_cos_arm3_vs_arm2"] < thr, "arm3 vs arm2 的 |cos| 超阈: %s" % c5["max_abs_cos_arm3_vs_arm2"]
    print("[PASS] test_arm3_is_decorrelated (最大 |cos| %.4f / %.4f < %.2f, 违例 %d)"
          % (c5["max_abs_cos_arm3_vs_arm1"], c5["max_abs_cos_arm3_vs_arm2"],
             thr, c5["total_violations"]))
    return True


# ---------------------------------------------------------------- 9
def test_block_grid_matches_protocol():
    """9. 块网格与协议 §2.1 一致: 文本 64 块 + ViT 20 块, 每块三臂"""
    m = _m()
    grid = m["protocol"]["grid"]
    assert grid["L_Q"] == list(L_Q) and grid["L_B"] == list(L_B) and grid["L_V"] == list(L_V), (
        "层阶梯与协议不一致: %s" % grid)
    assert grid["n_text_blocks"] == 64 and grid["n_vit_blocks"] == 20, "块数不对: %s" % grid
    assert grid["arms"] == list(ARMS), "臂不对: %s" % grid["arms"]
    got = set((r["direction"], r["source_layer"], r["target_layer"]) for r in m["blocks"])
    assert got == _expected_blocks(), "块集合与预期不符 (差 %d 个)" % len(got ^ _expected_blocks())
    assert len(m["blocks"]) == 84 * 3, "行数不对: %s" % len(m["blocks"])
    print("[PASS] test_block_grid_matches_protocol (84 块 x 3 臂 = %d 行, 网格与协议一致)"
          % len(m["blocks"]))
    return True


# ---------------------------------------------------------------- 10
def test_operator_has_no_learned_mapping():
    """10. 注入算子里**没有任何学出来的映射** (协议 §2.3 修订记录: 保留载体、不学映射)"""
    src = _src()
    for bad in ("procrustes(", "ridge_fit(", "fit_transport(", "lstsq(",
                "scipy.optimize", "align["):
        assert bad not in src, "源码里出现了学出来的映射步骤: %s" % bad
    op = _m()["protocol"]["operator"]
    assert op["no_learned_mapping"] is True, "协议快照没声明 no_learned_mapping"
    assert "align" not in op["formula"], "算子公式里出现了 align: %s" % op["formula"]
    for frag in ("z = a_s @ P_A", "P_B", "||h_B^{j,i}||_F"):
        assert frag in op["formula"], "算子公式缺了: %s" % frag
    print("[PASS] test_operator_has_no_learned_mapping (源码无拟合步骤, 快照声明 no_learned_mapping)")
    return True


# ---------------------------------------------------------------- 11
def test_thresholds_were_fixed_before_the_run():
    """11. 阈值与 λ 是跑之前写死的 (静态扫源码), 防止事后放宽"""
    src = _src()
    lam = _m()["protocol"]["thresholds"]["C2_norm_ratio_target"]
    for frag in ('"G1_arm0_rel_err_max": 0.0',
                 '"C2_norm_ratio_target": %s' % repr(float(lam)),
                 '"C2_norm_ratio_atol": 1e-6',
                 '"C3_min_rel_perturbation": 1e-3',
                 '"C3_max_rel_perturbation": 0.5',
                 '"C3_min_blocks_in_range": 32',
                 '"C4_max_rel_perturbation": 0.5',
                 '"C5_arm3_cos_threshold": 0.35',
                 '"C5_arm3_cos_violations_max": 0'):
        assert frag in src, "源码里找不到阈值声明: %s" % frag
    assert 'verdict = "S2_OPERATOR_READY" if all(criteria.values()) else "S2_OPERATOR_FAIL"' in src, \
        "裁决逻辑变了"
    assert 'return 0 if res["verdict"] == "S2_OPERATOR_READY" else 1' in src, "退出码契约变了"
    print("[PASS] test_thresholds_were_fixed_before_the_run (九项阈值含 λ=%.3g + 裁决逻辑 + 退出码契约)"
          % lam)
    return True


# ---------------------------------------------------------------- 12
def test_lambda_calibration_is_rule_based_and_registered():
    """12. λ 的标定是**写死的规则**跑出来的, 且快照与源码常量一致"""
    c = _calib()
    src = _calib_src()
    assert c["verdict"] == "CALIB_LAMBDA_SELECTED", "标定裁决不对: %s" % c["verdict"]
    assert c["chosen_lambda"] == 0.03, "选出的 λ 不对: %s" % c["chosen_lambda"]
    cands = tuple(c["protocol"]["candidates"])
    assert cands == (0.01, 0.02, 0.03, 0.04, 0.05), "候选集变了: %s" % (cands,)
    assert "LAMBDA_CANDIDATES = (0.01, 0.02, 0.03, 0.04, 0.05)" in src, "源码候选集变了"
    assert '"CALIB_R1_MAX_FINAL": 0.3' in src, "源码 R1 变了"
    assert '"CALIB_R2_MIN_READOUT": 1e-3' in src, "源码 R2 变了"
    th = c["protocol"]["thresholds"]
    assert th["CALIB_R1_MAX_FINAL"] == 0.3 and th["CALIB_R2_MIN_READOUT"] == 1e-3, (
        "快照阈值与规则不符: %s" % th)
    # 选择规则: 取满足 (R1)(R2) 的最大候选
    feasible = [v["lambda"] for v in c["per_lambda"].values() if v["feasible"]]
    assert feasible == sorted(feasible), "可行集不对: %s" % feasible
    assert c["chosen_lambda"] == max(feasible), "选的 λ 不是可行集里的最大者"
    assert c["chosen_lambda"] in c["feasible_lambdas"], "feasible_lambdas 与选中值不符"
    # 选中者的 R1/R2 余量
    sel = c["per_lambda"]["%.4g" % c["chosen_lambda"]]
    assert sel["R1_final_within_cap"] and sel["R2_readout_above_floor"], "选中者不满足规则"
    man = _manifest()
    assert man["artifacts"].get("lambda_calibration") == "results/lambda_calibration.json", (
        "MANIFEST 没登记标定快照: %s" % man["artifacts"])
    print("[PASS] test_lambda_calibration_is_rule_based_and_registered (λ*=%.3g, 候选 %s, R1/R2 规则在案)"
          % (c["chosen_lambda"], list(cands)))
    return True


# ---------------------------------------------------------------- 13
def test_no_semantic_criteria_are_claimed():
    """13. 本代是机械闸门: 不得产生 / 不得被引用成 P1a/P1b/P2/P3 的证据"""
    m = _m()
    for key in m["criteria"]:
        assert not (key.startswith("P1") or key.startswith("P3") or key.startswith("P2")), (
            "机械代里不该出现语义判据: %s" % key)
    scope = m["protocol"]["scope"]
    assert "不产生" in scope, "protocol.scope 缺少「不产生语义判据」的限定"
    limits = " ".join(m["limits"])
    for frag in ("不产生", "机械", "16 条"):
        assert frag in limits, "limits 少了: %s" % frag
    assert "P1a/P1b/P2/P3" in limits, "limits 没写明不得被引用成语义层证据"
    print("[PASS] test_no_semantic_criteria_are_claimed (无 P* 判据, scope/limits 限定在案)")
    return True


# ---------------------------------------------------------------- 14
def test_provenance_sha256_matches_live_files():
    """14. 口径同源凭证: v62 measure / v77 true_model / 本代脚本的 sha256 与快照一致"""
    p = _m()["protocol"]["provenance_sha256"]
    for path, key in ((V62_MEASURE, "v62_measure"),
                      (V77_SRC, "v77_true_model"),
                      (SRC, "v80_s2_operator")):
        assert os.path.exists(path), "找不到复用件: %s" % path
        live = _sha256(path)
        assert live == p[key], "%s sha256 不一致: 实测 %s, 快照 %s" % (key, live, p[key])
    cp = _calib()["protocol"]["provenance_sha256"]
    # 标定快照记的 v80_s2_operator 是**标定当时**的 s2_operator.py (即 lambda=0.1 那一版),
    # 后来活文件被改成 lambda=0.03 => 必须与**先导运行的源码归档**逐位吻合, 而不是与活文件比.
    assert os.path.exists(BASE_SRC), "找不到先导运行的源码归档: %s" % BASE_SRC
    archived = _sha256(BASE_SRC)
    assert archived == cp["v80_s2_operator"], (
        "标定快照的 v80_s2_operator 与先导源码归档不一致: 归档 %s, 快照 %s"
        % (archived, cp["v80_s2_operator"]))
    live = _sha256(CALIB_SRC)
    assert live == cp["v80_s2_lambda_calib"], (
        "标定快照的 v80_s2_lambda_calib sha256 不一致: 实测 %s, 快照 %s"
        % (live, cp["v80_s2_lambda_calib"]))
    print("[PASS] test_provenance_sha256_matches_live_files (五份复用件 sha256 全部一致; "
          "标定当时的 s2_operator 与先导源码归档逐位吻合)")
    return True


# ---------------------------------------------------------------- 15
def test_protocol_declares_discipline_and_limits():
    """15. 纪律与局限必须随数字一起登记"""
    m = _m()
    disc = " ".join(m["protocol"]["discipline"])
    for frag in ("跑前写死", "逐位等价", "不产生语义判据", "不含任何学出来的映射"):
        assert frag in disc, "纪律少了: %s" % frag
    assert m["protocol"]["sample_size"]["per_block"] == 16, "样本量口径变了"
    assert len(m["limits"]) >= 6, "limits 条目不足"
    c = _calib()
    cdisc = " ".join(c["protocol"]["discipline"])
    for frag in ("跑前写死", "最大"):
        assert frag in cdisc, "标定纪律少了: %s" % frag
    print("[PASS] test_protocol_declares_discipline_and_limits (%d 条纪律 / %d 条局限在案)"
          % (len(m["protocol"]["discipline"]), len(m["limits"])))
    return True


# ---------------------------------------------------------------- 16
def test_injection_scales_with_lambda():
    """16. 注入强度 -> 扰动的响应近似线性 (标定与重跑互为交叉验证)"""
    m = _m()
    b = _base()
    key = ("BERT->Qwen", 1, 1, "arm1_slot_Q")   # (direction, source_layer, target_layer, arm)
    lam_new = m["protocol"]["thresholds"]["C2_norm_ratio_target"]
    lam_old = b["protocol"]["thresholds"]["C2_norm_ratio_target"]
    new = [r["rel_perturb_final"] for r in m["blocks"]
           if (r["direction"], r["source_layer"], r["target_layer"], r["arm"]) == key]
    old = [r["rel_perturb_final"] for r in b["blocks"]
           if (r["direction"], r["source_layer"], r["target_layer"], r["arm"]) == key]
    assert new and old, "找不到对照行"
    ratio = (sum(new) / len(new)) / (sum(old) / len(old))
    expect = lam_new / lam_old
    assert 0.7 * expect <= ratio <= 1.4 * expect, (
        "扰动没随 λ 近似线性缩放: 实测 %.3f, 期望 %.3f" % (ratio, expect))
    print("[PASS] test_injection_scales_with_lambda (λ %.3g->%.3g, 扰动比 %.3f ≈ 期望 %.3f)"
          % (lam_old, lam_new, ratio, expect))
    return True


# ---------------------------------------------------------------- 17
def test_lineage_is_consistent():
    """17. 谱系自洽: v79 的 children 含 80.0, 本代 parent = 79.0"""
    v79 = _load(V79_MANIFEST)
    assert "80.0" in (v79["generation"]["children"] or []), (
        "v79 的 children 没含 80.0: %s" % v79["generation"]["children"])
    man = _manifest()
    assert man["generation"]["parent"] == "79.0" and man["version"] == "80.0"
    assert "81.0" in man["generation"]["children"] or man["generation"]["children"] == []
    print("[PASS] test_lineage_is_consistent (79.0 -> 80.0)")
    return True


# ---------------------------------------------------------------- 18
def test_manifest_claim_is_honest():
    """18. claim / acceptance 必须如实反映「标定后全过 + 先导失败留档」"""
    man = _manifest()
    assert man["claim"]["outcome_label"] == "supported", man["claim"]["outcome_label"]
    assert man["acceptance"]["outcome"] == "pass"
    note = man["claim"]["note"]
    for frag in ("λ", "0.03"):
        assert frag in note, "claim 的 note 少了: %s" % frag
    assert "逐位等价" in man["claim"]["asserts"], "claim 的 asserts 少了「逐位等价」"
    assert man["leakage_audit"]["status"] in ("clean", "not_applicable"), \
        "leakage_audit 状态不合法: %s" % man["leakage_audit"]["status"]
    assert man["superseded_by"] is None
    fals = man["falsified"]
    assert len(fals) >= 1, "证伪条目为空"
    for f in fals:
        assert f["evidence"], "证伪条目缺证据位置"
    print("[PASS] test_manifest_claim_is_honest (%d 条证伪在案)" % len(fals))
    return True


TESTS = [
    test_verdict_is_s2_operator_ready,
    test_arm0_is_bit_equivalent,
    test_norm_ratio_is_exact,
    test_readout_perturbation_is_visible,
    test_c4_now_holds_at_calibrated_lambda,
    test_leading_failure_is_preserved_on_record,
    test_leading_failure_was_content_independent,
    test_arm3_is_decorrelated,
    test_block_grid_matches_protocol,
    test_operator_has_no_learned_mapping,
    test_thresholds_were_fixed_before_the_run,
    test_lambda_calibration_is_rule_based_and_registered,
    test_no_semantic_criteria_are_claimed,
    test_provenance_sha256_matches_live_files,
    test_protocol_declares_discipline_and_limits,
    test_injection_scales_with_lambda,
    test_lineage_is_consistent,
    test_manifest_claim_is_honest,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
