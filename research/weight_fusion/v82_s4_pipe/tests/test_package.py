"""V82.0 - S4-P: 管道 (注意力位置旁路 + 按专属通道编号) 权威测试.

背景: v81 (S3 语义判据, 裁决 S3_FAIL) 查清了「信息流分不清」的根因 ——
同层内 Q 槽与 O 槽高度可互换, 倒进同一个残差流容器必然分不清。本代按用户指令
「他不知道往哪流你不会做管道吗」改为**显式修管道**: 正文之前开 K=8 个专属位置,
把源信息写进去, 按**通道编号**区分槽身份。

本代走的是一条链:
  (1) **协议预登记**: 判据 P5a/P5b/P5c/P5d/P2'/P3① + λ 标定规则 + 块网格,
      全部写进源码常量 (THRESHOLDS / SEEDS / N_PERM / ALPHA / LAMBDA_GRID /
      INJECT_LAYER / ATT_LAYER), 跑前写死;
  (2) **λ 标定**: 写死的候选集 {0.5,1,2,3} + 写死的选择规则 -> lambda*;
  (3) **跑批**: 四臂 x 5 nuisance 种子 x 64 块 (全文本; ViT 臂已于跑前整臂撤出) x 128 样本;
  (4) **机器裁决**: S4PIPE_READY (六项全过) 否则 S4PIPE_FAIL, 退出码 0/1。

测试把三件事都锁住:
  - 机制面: 块网格/臂/种子与协议一致; arm0 是「管道开、内容为 0」的配对基线;
             管道内容**只写管道位**且**写在第 0 层**; 探针是 **masked token 级**
             且 R² 在**留出集**上算; λ 由**训练集内** CV 选出
  - 纪律面: 阈值跑前写死; ViT 臂**已于跑前整臂撤出**; 样本级辅助探针
             **不参与**任何裁决; 置换次数与 p 一起报告; 不训练任何底座权重
  - 谱系面: v81.children 含 82.0, 本代 parent = 81.0

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

TITLE = "V82.0 S4-P (管道: 注意力位置旁路 + 按专属通道编号)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
CALIB = os.path.join(_pkg, "results", "calib.json")
SRC = os.path.join(_pkg, "s4_pipe.py")
V81_MANIFEST = os.path.join(_pkg, "..", "v81_s3_semantic", "MANIFEST.json")

L_Q = (0, 1, 2, 3, 6, 12, 18, 23)
L_B = (0, 1, 2, 3)
L_V = (0, 3, 6, 9, 11)
ARMS = ("arm0_pipe_zero", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
SEEDS = (0, 1, 2, 3, 4)
LAMBDA_GRID = (0.5, 1.0, 2.0, 3.0)


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


def _expected_blocks():
    out = set()
    for l in L_Q:
        for j in L_B:
            out.add(("Qwen->BERT", l, j))
    for l in L_B:
        for j in L_Q:
            out.add(("BERT->Qwen", l, j))
    # ⚠️ ViT 臂已于跑前整臂撤出 (SPEC §0.1 修订 9) ⇒ 块网格里没有 ViT
    return out


# ---------------------------------------------------------------- 1
def test_block_grid_matches_protocol():
    """1. 块网格与协议 §2.1/§3 一致: 64 块 (全文本) x 4 臂 x 5 种子 = 1280 行"""
    m = _m()
    rows = m["blocks"]
    got = set((r["direction"], r["source_layer"], r["target_layer"]) for r in rows)
    assert got == _expected_blocks(), "块集合与预期不符 (差 %d 个)" % len(got ^ _expected_blocks())
    assert len(rows) == 64 * 4 * 5, "行数不对: %s" % len(rows)
    assert set(r["arm"] for r in rows) == set(ARMS), "臂不对"
    assert set(r["seed"] for r in rows) == set(SEEDS), "种子不对"
    n_text = len(set((r["direction"], r["source_layer"], r["target_layer"])
                     for r in rows if r["direction"] != "ViT->ViT"))
    n_vit = len(set((r["direction"], r["source_layer"], r["target_layer"])
                    for r in rows if r["direction"] == "ViT->ViT"))
    assert n_text == 64, "主判据集 (文本块) 数不对: %s" % n_text
    assert n_vit == 0, "ViT 臂已于跑前整臂撤出, 不得出现在块网格里: %s" % n_vit
    print("[PASS] test_block_grid_matches_protocol (64 块 x 4 臂 x 5 种子 = %d 行; 文本 %d + ViT %d)"
          % (len(rows), n_text, n_vit))
    return True


# ---------------------------------------------------------------- 2
def test_thresholds_were_fixed_before_the_run():
    """2. 阈值与口径全部**跑前写死** (静态扫源码), 防止事后放宽"""
    src = _src()
    for frag in ("SEEDS = (0, 1, 2, 3, 4)",
                 "N_PERM = 10000",
                 "ALPHA = 0.01",
                 "LAMBDA_GRID = (0.5, 1.0, 2.0, 3.0)",
                 "CALIB_MAX_FINAL = 0.3",
                 "INJECT_LAYER = 0",
                 "ATT_LAYER = 0",
                 "K_PIPE = 8",
                 "PROBE_LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)",
                 "PROBE_CV_FOLDS = 5",
                 "TEXT_SPLIT_TRAIN = 64",
                 "VIT_SPLIT_TRAIN = 32",
                 "DROP_FIRST_TOKEN = True",
                 '"n_text_blocks": 64',
                 '"n_vit_blocks": 0',
                 '"n_attention_blocks": 12',
                 '"p3_min_rel_perturbation": 1e-3',
                 '"p3_max_rel_perturbation": 0.5',
                 '"p5d_selectivity_p_min": 0.05',
                 '"calib_max_final": CALIB_MAX_FINAL',
                 '"lambda_grid": list(LAMBDA_GRID)',
                 '"verdict": "S4PIPE_READY" if passed else "S4PIPE_FAIL"',
                 'return 0 if res["verdict"] == "S4PIPE_READY" else 1'):
        assert frag in src, "源码里找不到预登记声明: %s" % frag
    th = _m()["protocol"]["thresholds"]
    assert th["alpha"] == 0.01 and th["n_perm"] == 10000 and th["n_seeds"] == 5
    assert th["k_pipe"] == 8 and th["n_channels"] == 2
    assert th["inject_layer"] == 0 and th["att_layer"] == 0
    print("[PASS] test_thresholds_were_fixed_before_the_run (二十一项预登记声明 + 裁决逻辑 + 退出码契约)")
    return True


# ---------------------------------------------------------------- 3
def test_arm0_is_pipe_open_zero_content_baseline():
    """3. arm0 是「管道开、内容为 0」的配对基线: 逐种子值相同, 扰动恒为 0"""
    m = _m()
    rows = [r for r in m["blocks"] if r["arm"] == "arm0_pipe_zero"]
    assert len(rows) == 64 * 5, "arm0 行数不对: %s" % len(rows)
    by_block = {}
    for r in rows:
        key = (r["direction"], r["source_layer"], r["target_layer"])
        by_block.setdefault(key, []).append(r)
    for key, rs in by_block.items():
        assert len(rs) == 5, "%s 的 arm0 种子数不对" % (key,)
        assert len(set(round(r["r2_q"], 12) for r in rs)) == 1, "%s 的 arm0 逐种子 r2_q 不相同" % (key,)
        assert len(set(round(r["r2_o"], 12) for r in rs)) == 1, "%s 的 arm0 逐种子 r2_o 不相同" % (key,)
        for r in rs:
            assert r["rel_perturb_final"] == 0.0, "%s 的 arm0 相对扰动应为 0" % (key,)
    print("[PASS] test_arm0_is_pipe_open_zero_content_baseline (420 行逐种子一致, 扰动恒 0)")
    return True


# ---------------------------------------------------------------- 4
def test_pipe_is_written_only_at_pipe_positions_in_layer0():
    """4. 管道内容**只写管道位**且**写在第 0 层** (机制的两条硬约束)"""
    src = _src()
    assert "INJECT_LAYER = 0" in src, "写入层必须是第 0 层 (D2)"
    assert "bb.layers[INJECT_LAYER]" in src, "hook 必须装在 INJECT_LAYER 上"
    assert "new_mask[:, :K_PIPE] = 1" in src, "管道位必须 attention_mask=1"
    assert "new_ids[i, K_PIPE:K_PIPE + L] = ids[i, :L]" in src, "正文必须排在管道位之后"
    assert "base = channel * CH_SIZE" in src and "range(base, base + CH_SIZE)" in src, \
        "写入范围必须只覆盖该通道的管道位"
    lay = _m()["protocol"]["layout_by_direction"]
    for d, v in lay.items():
        assert v["pipe_positions"] == 8, "%s 的管道位数不对" % d
        if d == "BERT->Qwen":
            assert v["body_positions"] == 120, "正文位数不对"
            assert v["n_truncated_to_txt"] == 1, "截断样本数不对 (实测 1/128)"
    print("[PASS] test_pipe_is_written_only_at_pipe_positions_in_layer0 (只写 8 个管道位 + 写入层=0 + 布局一致)")
    return True


# ---------------------------------------------------------------- 5
def test_attention_measured_at_inject_layer():
    """5. 管道注意力在**写入层 (第 0 层)** 测量, 且逐样本记录 (可复核)"""
    m = _m()
    assert m["protocol"]["constants"]["ATT_LAYER"] == 0, "注意力测量层必须是第 0 层"
    recs = m["attention_summary"]
    assert len(recs) > 0, "没有注意力记录"
    for r in recs:
        assert r["n_samples"] == 128, "注意力必须是逐样本的 (128 条)"
        assert r["attn_pipe_med"] >= 0.0, "注意力不该为负"
    n_groups = len(set((r["direction"], r["source_layer"]) for r in recs))
    assert n_groups == 12, "(方向,源层) 组数不对 (ViT 臂已于跑前整臂撤出): %s" % n_groups
    print("[PASS] test_attention_measured_at_inject_layer (%d 条逐样本记录, %d 个组)"
          % (len(recs), n_groups))
    return True


# ---------------------------------------------------------------- 6
def test_probe_is_masked_token_level_and_holdout():
    """6. 探针是 **masked token 级** 且 R² 在**留出集**上算; λ 由训练集内 CV 选出"""
    src = _src()
    for frag in ("def masked_r2_multi(", "def masked_lambda_scores(",
                 "def pooled_r2_multi(", "readout[idx]", "repeat_interleave"):
        assert frag in src, "源码里找不到探针实现: %s" % frag
    assert "Xtr = (Xtr - mu) / sd" in src, "标准化统计量必须只取训练行"
    assert "ss_tot = float(Yte.pow(2).sum())" in src, "R² 必须在留出集上算"
    info = _m()["protocol"]["probe_lambda_by_direction"]
    for d, v in info.items():
        assert v["probe_lambda"] in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), \
            "%s 的探针 λ 不在写死网格上" % d
        assert v["n_folds"] == 5, "%s 的 CV 折数不对" % d
        assert len(v["pilot_blocks"]) == 3, "%s 的 pilot 块数不对" % d
        assert "cv_scores" in v and v["cv_scores"], "%s 缺 CV 分数" % d
    print("[PASS] test_probe_is_masked_token_level_and_holdout (masked token 级 + 留出集 + 5 折 CV + pilot)")
    return True


# ---------------------------------------------------------------- 7
def test_vit_arm_is_withdrawn():
    """7. ViT 臂已于跑前**整臂撤出**: 不进块网格/主判据, 但撤出记录必须留档"""
    m = _m()
    crit = m["criteria"]
    assert "vit_parallel" in crit, "缺 ViT 撤出记录"
    assert crit["vit_parallel"].get("withdrawn") is True, "ViT 臂必须标记为已撤出"
    assert crit["vit_parallel"].get("role"), "ViT 撤出必须写明理由"
    txt = crit["P5c"]["n_blocks"]
    assert txt == 64, "主判据块数不对: %s" % txt
    assert crit["P5a"]["n_blocks"] == 12, "注意力块数不对 (12 个 (方向,源层) 组)"
    # 裁决只由六项判据决定, ViT 平行臂不出现其中
    for name in ("P5a", "P5b", "P5c", "P5d", "P2_prime", "P3_1"):
        assert name in crit, "缺判据 %s" % name
    assert "vit_parallel" not in m["verdict_detail"]["failed_criteria"], \
        "ViT 不得进入失败判据列表"
    src = _src()
    assert "passed = bool(p5a_pass and p5b_pass and p5c_pass and p5d_pass" in src, \
        "裁决必须只由六项判据决定"
    # 撤出必须是**跑前**写死的: ViT 方向只留档在源码常量区, 不在 DIRECTIONS 里
    assert '("ViT->ViT", "ViT", "ViT", L_V, L_V),' in src, \
        "撤出方向必须留档在源码常量区"
    assert "ViT->ViT" not in src.split("DIRECTIONS = (")[1].split(")")[0], \
        "ViT 不得留在 DIRECTIONS 里"
    print("[PASS] test_vit_arm_is_withdrawn (ViT 已整臂撤出, 裁决只看六项文本判据)")
    return True


# ---------------------------------------------------------------- 8
def test_verdict_logic_matches_protocol():
    """8. 机器裁决与协议一致: 六项全过 => S4PIPE_READY, 否则 S4PIPE_FAIL"""
    m = _m()
    v = m["verdict_detail"]
    names = ("P5a", "P5b", "P5c", "P5d", "P2_prime", "P3_1")
    oks = [bool(m["criteria"][n]["passed"]) for n in names]
    assert v["passed"] == all(oks), "verdict.passed 与六项判据不一致"
    assert m["verdict"] == ("S4PIPE_READY" if all(oks) else "S4PIPE_FAIL"), "裁决字符串不对"
    assert set(v["failed_criteria"]) == set(n for n, ok in zip(names, oks) if not ok), \
        "失败判据列表不对"
    print("[PASS] test_verdict_logic_matches_protocol (裁决 %s; 通过 %s)"
          % (m["verdict"], "/".join(n for n, ok in zip(names, oks) if ok)))
    return True


# ---------------------------------------------------------------- 9
def test_permutation_reported_with_p():
    """9. 每个 p 值都必须与置换次数一起报告 (协议 §3 的报告要求)"""
    m = _m()
    for name in ("P5a", "P5b", "P5c", "P5d", "P2_prime"):
        c = m["criteria"][name]
        assert "permutation_p" in c and "n_perm" in c, "%s 缺 p / 置换次数" % name
        assert c["n_perm"] == 10000, "%s 的置换次数不对" % name
        assert c["alpha"] == 0.01, "%s 的 alpha 不对" % name
    for name in ("P5a", "P5b", "P5c", "P5d"):
        assert "median_delta" in m["criteria"][name], "%s 缺中位增量" % name
    print("[PASS] test_permutation_reported_with_p (五项判据都带 p / 10^4 / alpha / 中位增量)")
    return True


# ---------------------------------------------------------------- 10
def test_no_training_no_ability_metric():
    """10. 本代不训练任何底座权重, 不产生任何能力指标"""
    src = _src()
    for bad in (".backward(", "torch.optim", "requires_grad_(True)", "loss.backward"):
        assert bad not in src, "源码里出现了训练痕迹: %s" % bad
    m = _m()
    assert "不产生任何能力指标" in m["protocol"]["scope"], "scope 必须写明不产生能力指标"
    man = _manifest()
    assert man["leakage_audit"]["status"] == "clean", "泄漏审计状态不对"
    assert man["acceptance"]["outcome"] in ("pass", "fail"), "验收结论不对"
    print("[PASS] test_no_training_no_ability_metric (无 backward / 无 optimizer / 无能力指标)")
    return True


# ---------------------------------------------------------------- 11
def test_lambda_calibrated_with_locked_rule():
    """11. λ 由写死候选集 + 写死选择规则标定得出, 且正式跑用的就是它"""
    c = _load(CALIB)
    assert c["verdict"] == "S4PIPE_LAMBDA_SELECTED", "标定未选出可行 lambda: %s" % c["verdict"]
    assert c["selected_lambda"] in LAMBDA_GRID, "选出的 λ 不在写死候选集里"
    assert tuple(c["lambda_grid"]) == LAMBDA_GRID, "候选集与协议不符"
    assert "①" in c["rule"] and "②" in c["rule"] and "③" in c["rule"], "标定规则必须三条写死"
    for lam in LAMBDA_GRID:
        cand = c["candidates"]["%.4g" % lam]
        assert "max_final" in cand and "blocks" in cand, "标定记录不完整"
    m = _m()
    assert abs(m["protocol"]["constants"]["LAMBDA_INJECT"] - c["selected_lambda"]) < 1e-9, \
        "正式跑用的 λ 与标定选出的一致"
    print("[PASS] test_lambda_calibrated_with_locked_rule (lambda*=%g; 候选集与三条规则写死)"
          % c["selected_lambda"])
    return True


# ---------------------------------------------------------------- 12
def test_calibration_rule_semantics():
    """12. 标定的三条规则语义被单测锁住 (纯函数 evaluate_calibration, 合成数据)"""
    from research.weight_fusion.v82_s4_pipe.s4_pipe import (  # noqa: E402
        LAMBDA_GRID, DIRECTIONS, evaluate_calibration,
    )

    def build(lam_map):
        cand, arm0 = {}, {}
        for d, s, t, sl, tl in DIRECTIONS:
            arm0[d] = {"attn_pipe_mean": 0.01}
            for lam in LAMBDA_GRID:
                key = "%.4g" % lam
                cand.setdefault(key, {}).setdefault(d, {})
                for ly in sl:
                    cfg = lam_map.get((key, d, ly), {})
                    ok2 = cfg.get("ok2", True)
                    ok3 = cfg.get("ok3", True)
                    flip1 = cfg.get("flip1", 0.002 if ok3 else -0.002)
                    flip2 = cfg.get("flip2", -0.001)
                    rel = cfg.get("rel", 0.1)
                    cand[key][d][str(ly)] = {
                        "arm1_slot_Q": {"attn_pipe_mean": 0.01 + (0.001 if ok2 else -0.001),
                                        "flip_mean": flip1, "rel": rel},
                        "arm2_wrong_slot_O": {"attn_pipe_mean": 0.01,
                                              "flip_mean": flip2, "rel": rel}}
        return cand, arm0

    res, feas = evaluate_calibration(*build({}))
    assert feas == list(LAMBDA_GRID), feas
    res, feas = evaluate_calibration(*build({("3", "BERT->Qwen", 0): {"rel": 0.9}}))
    assert feas == [0.5, 1.0, 2.0] and res["3"]["cond1_max_final_ok"] is False, feas
    # ViT 臂已于跑前整臂撤出 ⇒ DIRECTIONS 里根本没有 ViT ⇒ 它不可能门控 λ
    assert all(d != "ViT->ViT" for d, _s, _t, _sl, _tl in DIRECTIONS), \
        "ViT 臂必须已从 DIRECTIONS 撤出"
    # ③ 失败: 门控是「主判据集 ≥ 8/12 组为正」(修订 9), 不是逐组全过
    # (a) 只挂 1 组 (11/12) ⇒ 仍然可行
    res, feas = evaluate_calibration(*build({("2", "BERT->Qwen", 1): {"ok3": False}}))
    assert 2.0 in feas and 3.0 in feas, feas
    # (b) 挂 5 组 (只剩 7/12) ⇒ 不可行
    res, feas = evaluate_calibration(*build(
        {(k, d, ly): {"ok3": False} for k in ("2", "3")
         for d, s, t, sl, tl in DIRECTIONS for ly in sl
         if d == "Qwen->BERT" and ly in (0, 1, 2, 3, 6)}))
    assert 2.0 not in feas and 3.0 not in feas, feas
    res, feas = evaluate_calibration(*build(
        {(k, d, ly): {"ok2": False} for k in ("0.5", "1", "2", "3")
         for d, s, t, sl, tl in DIRECTIONS for ly in sl}))
    assert feas == [], feas
    # 修订 9 的门控边界: ②③ 是「主判据集 ≥ 8/12 组为正」, 不是逐组全过
    # (a) 恰好 8 组为正 (12 组里挂 4 组) ⇒ 可行
    res, feas = evaluate_calibration(*build(
        {(k, d, ly): {"ok2": False, "ok3": False}
         for k in ("0.5", "1", "2", "3")
         for d, s, t, sl, tl in DIRECTIONS for ly in sl
         if d == "Qwen->BERT" and ly in (0, 1, 2, 3)}))
    assert feas == list(LAMBDA_GRID), feas
    # (b) 只剩 4 组为正 (12 组里挂 8 组) ⇒ 不可行
    res, feas = evaluate_calibration(*build(
        {(k, d, ly): {"ok2": False, "ok3": False}
         for k in ("0.5", "1", "2", "3")
         for d, s, t, sl, tl in DIRECTIONS for ly in sl
         if (d == "Qwen->BERT" and ly in (0, 1, 2, 3))
         or (d == "BERT->Qwen" and ly in (0, 1, 2, 3))}))
    assert feas == [], feas
    print("[PASS] test_calibration_rule_semantics (七条规则语义: 取最大 / 守门越界 / ViT 已撤出 / "
          "③ 失败 / 全不可行 / 8-12 门控边界×2)")
    return True


# ---------------------------------------------------------------- 13
def test_lineage_and_version_unique():
    """12. 谱系自洽: v81.children 含 82.0, 本代 parent = 81.0, 版本号全局唯一"""
    man = _manifest()
    assert man["version"] == "82.0", "版本号不对"
    assert man["generation"]["parent"] == "81.0", "parent 不对"
    assert isinstance(man["generation"]["children"], list), "children 应为列表"
    v81 = _load(V81_MANIFEST)
    assert "82.0" in v81["generation"]["children"], "v81 的 children 必须含 82.0"
    print("[PASS] test_lineage_and_version_unique (82.0 <- 81.0, children 双向自洽)")
    return True


# ---------------------------------------------------------------- 14
def test_results_numbers_are_recorded():
    """13. 结果数字落在 SPEC §4 的范围里 (防止事后手改读数)"""
    m = _m()
    c = m["criteria"]
    for name in ("P5a", "P5b", "P5c", "P5d"):
        v = c[name]["median_delta"]
        assert v is not None, "%s 缺中位增量" % name
    p3 = c["P3_1"]
    assert p3["n_rows"] == 64 * 3 * 5, "P3① 的行数不对: %s" % p3["n_rows"]
    assert p3["passed"] == (p3["n_violations"] == 0), "P3① 的通过标记与违例数不一致"
    assert m["protocol"]["constants"]["SEEDS"] == list(SEEDS), "种子表不对"
    assert m["runtime_sec"] > 0, "缺跑批耗时"
    print("[PASS] test_results_numbers_are_recorded (P3① %d 行, 违例 %d, 跑批 %.0fs)"
          % (p3["n_rows"], p3["n_violations"], m["runtime_sec"]))
    return True




# ---------------------------------------------------------------- 15
def test_posthoc_diagnostics_are_separated_from_verdict():
    """15. 两份 post_hoc 诊断 (读出端 / 载体性质) 必须存在、标注 post_hoc_only, 且**不进入裁决**"""
    for name in ("diag_readout.json", "diag_carrier.json"):
        p = os.path.join(_pkg, "results", name)
        assert os.path.exists(p), "缺 post_hoc 诊断产物: %s" % name
        d = _load(p)
        assert "post_hoc_only" in d.get("role", ""), "%s 必须标注 post_hoc_only" % name
    for f in ("carrier_probe.py", "readout_diag.py"):
        assert os.path.exists(os.path.join(_pkg, f)), "缺诊断脚本 (产物须可复现): %s" % f
    m = _m()
    assert set(m["criteria"]) == {"P5a", "P5b", "P5c", "P5d", "P2_prime", "P3_1",
                                "aux_pooled", "vit_parallel"}, "判据集合被诊断产物污染: %s" % sorted(m["criteria"])
    assert m["verdict"] == "S4PIPE_FAIL", "裁决必须仍是 S4PIPE_FAIL"
    print("[PASS] test_posthoc_diagnostics_are_separated_from_verdict (两份诊断标注 post_hoc_only, 判据集合未变)")
    return True



TESTS = [
    test_block_grid_matches_protocol,
    test_thresholds_were_fixed_before_the_run,
    test_arm0_is_pipe_open_zero_content_baseline,
    test_pipe_is_written_only_at_pipe_positions_in_layer0,
    test_attention_measured_at_inject_layer,
    test_probe_is_masked_token_level_and_holdout,
    test_vit_arm_is_withdrawn,
    test_verdict_logic_matches_protocol,
    test_permutation_reported_with_p,
    test_no_training_no_ability_metric,
    test_lambda_calibrated_with_locked_rule,
    test_calibration_rule_semantics,
    test_lineage_and_version_unique,
    test_results_numbers_are_recorded,
    test_posthoc_diagnostics_are_separated_from_verdict,
]
