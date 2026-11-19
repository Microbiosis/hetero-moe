"""V76.0 — A1 跨层信息流: 从权重切片重建逐层前向 (S1) 权威测试。

背景: 主线前置关卡 A1 问的是「信息流能不能在计算图内部从模型 A 的第 l 层
流到模型 B 的第 j 层」。本代只做 S1 (机械可行性): 从权重切片重建每个底座的
逐层前向, 实现「层边界打断 -> 注入 -> 续跑」, 跑完五项机械自检。

本代结论有两面, 测试把两面都锁住:
  - S1_PASS: 五项自检全过 ⇒ 计算图层面的跨模型跨层信息流**机制存在**。
  - 两条被证伪的设计前提 (比 PASS 更重要):
      (a) 路线 α (随机输入的权重级前向) 测不了语义层判据 ⇒ P1a/P1b/P2/P3
          必须上路线 β (真模型逐层驱动 + 真输入 + 标签);
      (b) ViT 进不了首代范围 (layer0 残差膨胀 20.11x, 有效秩 744->64)。

本测试**不重跑实验** (需要权重切片缓存), 它锁住的是:
  1. 五项判据与裁决的自洽 (重算一遍判据逻辑, 必须复现快照里的裁决)
  2. 口径同源凭证 (v75 g2_conjugation.py 与 v62 measure.py 的 sha256)
  3. 阈值是跑前写死的 (静态扫源码)
  4. 两条证伪仍然在案 (不得被后续话术淡化或抹掉)

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

TITLE = "V76.0 A1 跨层信息流 (S1 机械可行性)"

METRICS = os.path.join(_pkg, "results", "metrics.json")
MANIFEST = os.path.join(_pkg, "MANIFEST.json")
SRC = os.path.join(_pkg, "a1_forward.py")
V75_G2 = os.path.join(_pkg, "..", "v75_weight_conjugation", "g2_conjugation.py")
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
def test_s1_criteria_all_pass():
    """1. S1 五项机械自检全过, 且裁决是 S1_PASS"""
    m = _m()
    crit = m["criteria"]
    for name in ("C1_structure_ok", "C2_provenance_within_tol", "C3_stack_health_ok",
                 "C4_injection_path_ok", "C5_cross_model_path_ok"):
        assert crit[name] is True, "判据未过: %s" % name
    assert m["verdict"] == "S1_PASS", "裁决不是 S1_PASS: %s" % m["verdict"]
    assert _manifest()["acceptance"]["outcome"] == "pass", "MANIFEST 的验收结论与快照不一致"
    print("[PASS] test_s1_criteria_all_pass (5/5 判据, verdict=%s)" % m["verdict"])
    return True


# ---------------------------------------------------------------- 2
def test_c2_provenance_reproduces_v62():
    """2. C2 口径同源: 谱统计逐位复现 v62 §2 (不是"接近", 是逐位)"""
    c2 = _m()["C2_provenance"]
    assert c2["ok"] is True
    for key in ("same_function_js_mean", "diff_function_js_mean"):
        rel = c2["relative_error"][key]
        assert rel <= 1e-6, "%s 相对误差 %g 过大 (判据容差 5%%)" % (key, rel)
    assert c2["n_pairs"] == 1728, "配对数变了: %s" % c2["n_pairs"]
    assert c2["permutation_p"] == 0.0002, "置换 p 变了: %s" % c2["permutation_p"]
    # 绝对差与相对差同源: 相对差 ~5e-8 x 数值 ~0.03 => 绝对差 ~1.6e-9
    assert abs(c2["measured_same"] - c2["published_same"]) < 1e-7
    assert abs(c2["measured_diff"] - c2["published_diff"]) < 1e-7
    print("[PASS] test_c2_provenance_reproduces_v62 (相对差 %.2e / %.2e, n=1728, p=0.0002)"
          % (c2["relative_error"]["same_function_js_mean"],
             c2["relative_error"]["diff_function_js_mean"]))
    return True


# ---------------------------------------------------------------- 3
def test_live_source_sha256_matches_snapshot():
    """3. 口径同源凭证: v75 g2 与 v62 measure 的 sha256 与快照记录一致"""
    p = _m()["protocol"]
    assert os.path.exists(V75_G2), "找不到 v75 g2_conjugation.py: %s" % V75_G2
    assert os.path.exists(V62_MEASURE), "找不到 v62 measure.py: %s" % V62_MEASURE
    g2 = _sha256(V75_G2)
    ms = _sha256(V62_MEASURE)
    assert g2 == p["v75_g2_sha256"], "v75 g2 sha256 不一致: 实测 %s, 快照 %s" % (g2, p["v75_g2_sha256"])
    assert ms == p["v62_measure_sha256"], "v62 measure sha256 不一致: 实测 %s, 快照 %s" % (ms, p["v62_measure_sha256"])
    assert _m()["C2_provenance"]["v62_measure_sha256"] == ms
    print("[PASS] test_live_source_sha256_matches_snapshot (g2=%s..., measure=%s...)"
          % (g2[:16], ms[:16]))
    return True


# ---------------------------------------------------------------- 4
def test_c1_structure_and_gqa_heads():
    """4. C1 结构合法: 三底座每层六槽齐全; Qwen 的 GQA 头数必须被校验过"""
    c1 = _m()["C1_structure"]
    assert c1["all_ok"] is True
    for name, model in c1["models"].items():
        assert model["ok"] is True, "%s 结构未过" % name
        assert len(model["layers"]) == 4, "%s 层数变了: %d" % (name, len(model["layers"]))
        for layer in model["layers"]:
            sh = layer["shapes"]
            assert set(sh) == {"Q", "K", "V", "O", "expand", "contract"}, "%s 层 %s 槽不全" % (name, layer["layer"])
            assert sh["Q"] == sh["O"], "%s 层 %s 的 Q/O 不是方阵" % (name, layer["layer"])
            assert sh["expand"][1] == sh["Q"][1], "%s 层 %s 的 expand 输入维不对" % (name, layer["layer"])
            assert sh["contract"][1] == sh["expand"][0], "%s 层 %s 的 contract 输入维不对" % (name, layer["layer"])
    q = c1["models"]["Qwen"]["heads"]
    assert (q["n_heads"], q["n_kv"], q["head_dim"]) == (14, 2, 64), "Qwen 头数变了: %s" % q
    assert c1["models"]["Qwen"]["gqa"] is True, "Qwen 的 GQA 标记没了"
    assert c1["models"]["Qwen"]["layers"][0]["shapes"]["K"] == [128, 896], "Qwen K 形状变了"
    assert c1["models"]["Qwen"]["layers"][0]["shapes"]["V"] == [128, 896], "Qwen V 形状变了"
    print("[PASS] test_c1_structure_and_gqa_heads (Qwen 14 头 / 2 KV 头 x 64, K=[128,896])")
    return True


# ---------------------------------------------------------------- 5
def test_c3_health_scope_and_bounds():
    """5. C3 层栈健康: 判据范围 = 首代覆盖的 Qwen/BERT, 范数比与有效秩都在界内"""
    c3 = _m()["C3_health"]
    assert c3["all_ok"] is True
    assert c3["scope"] == ["Qwen", "BERT"], "C3 判据范围变了: %s" % c3["scope"]
    lo, hi = c3["bounds"]["ratio"]
    for name in c3["scope"]:
        m = c3["models"][name]
        assert m["ok"] is True and m["in_scope"] is True, "%s 不在范围内却判了健康" % name
        for r in m["rms_ratio"]:
            assert lo <= r <= hi, "%s 的相邻层范数比 %g 越界 [%g,%g]" % (name, r, lo, hi)
        assert m["min_eff_rank"] > c3["bounds"]["eff_rank_min"], "%s 有效秩塌了: %g" % (name, m["min_eff_rank"])
    print("[PASS] test_c3_health_scope_and_bounds (Qwen 比 %.2f~%.2f 秩 %g, BERT 比 %.2f~%.2f 秩 %g)"
          % (c3["models"]["Qwen"]["min_ratio"], c3["models"]["Qwen"]["max_ratio"],
             c3["models"]["Qwen"]["min_eff_rank"],
             c3["models"]["BERT"]["min_ratio"], c3["models"]["BERT"]["max_ratio"],
             c3["models"]["BERT"]["min_eff_rank"]))
    return True


# ---------------------------------------------------------------- 6
def test_vit_defect_is_still_on_record():
    """6. ViT 的范围外缺陷必须仍在案: 不得被抹掉, 也不得混进判据"""
    c3 = _m()["C3_health"]
    assert "ViT" in c3["out_of_scope"], "ViT 不再登记为范围外 ⇒ 缺陷被抹掉了"
    vit = c3["out_of_scope"]["ViT"]
    assert vit["ok"] is False, "ViT 被判成健康了"
    assert vit["in_scope"] is False
    assert c3["models"]["ViT"]["ok"] is False, "ViT 在主表里被判成健康"
    assert max(vit["rms_ratio"]) > 10.0, "ViT 残差膨胀倍数变了: %g" % max(vit["rms_ratio"])
    defect = c3["out_of_scope_defects"]["ViT"]
    assert "20.1" in defect, "缺陷描述里的 20.1x 倍数被改掉了"
    assert "ViT" not in c3["scope"], "ViT 不该在判据范围内"
    print("[PASS] test_vit_defect_is_still_on_record (ViT layer0 膨胀 %.2fx, 有效秩 %g, 仍登记为范围外)"
          % (max(vit["rms_ratio"]), vit["min_eff_rank"]))
    return True


# ---------------------------------------------------------------- 7
def test_c4_injection_path_is_live():
    """7. C4 注入通路: 打断->注入->续跑真的改变了后续层输出, 且方向没被原样携带"""
    c4 = _m()["C4_injection"]
    assert c4["ok"] is True
    curve = c4["curve"]
    assert len(curve) == 3, "注入曲线层数变了: %d" % len(curve)
    for row in curve:
        assert row["rel_perturbation"] > 1e-3, "层 %s 的相对扰动太小" % row["layer"]
        assert row["alignment_mean"] > 0.0, "层 %s 的注入方向被洗掉了" % row["layer"]
        assert row["alignment_frac_pos"] == 1.0, "层 %s 的方向一致性不是 100%%" % row["layer"]
    final = c4["final"]
    assert final["alignment_mean"] < 1.0, "末层 alignment 变成 1.0 ⇒ 只是残差原样携带"
    assert final["alignment_mean"] > 0.5, "末层 alignment 低到无法区分于噪声"
    assert c4["min_rel_perturbation"] == 1e-3
    print("[PASS] test_c4_injection_path_is_live (扰动 %.3f~%.3f, alignment %.3f~%.3f, 方向 100%%)"
          % (curve[0]["rel_perturbation"], curve[-1]["rel_perturbation"],
             min(r["alignment_mean"] for r in curve), final["alignment_mean"]))
    return True


# ---------------------------------------------------------------- 8
def test_route_alpha_cannot_measure_semantics():
    """8. 关键证伪: 随机输入下匹配槽并不比错槽好 ⇒ 路线 α 测不了语义层"""
    c5 = _m()["C5_cross_model"]
    rows = c5["rows"]
    assert c5["n_rows"] == 24, "槽级 ridge 行数变了: %d" % c5["n_rows"]
    assert len(rows) == 24
    for r in rows:
        assert r["r2_matched"] > 0.90, "%s 层 %s 槽 %s 的匹配槽 R2 掉到 %g" % (
            r["pair"], r["layer"], r["slot"], r["r2_matched"])
    assert abs(c5["median_delta"]) < 0.01, "匹配槽与错槽的中位差变大: %g" % c5["median_delta"]
    assert 0.30 < c5["frac_matched_better"] < 0.70, "匹配槽明显占优: %g" % c5["frac_matched_better"]
    for name, v in c5["orthogonality_max_abs"].items():
        assert v < 1e-5, "%s 的正交残差 %g 过大" % (name, v)
    print("[PASS] test_route_alpha_cannot_measure_semantics (匹配槽 R2>0.90, 中位差 %.2e, 占优仅 %.1f%%)"
          % (c5["median_delta"], 100 * c5["frac_matched_better"]))
    return True


# ---------------------------------------------------------------- 9
def test_thresholds_were_fixed_before_the_run():
    """9. 阈值是跑之前写死的 (静态扫源码), 防止事后放宽"""
    src = _src()
    for frag in ("NORM_RATIO_MIN = 0.10", "NORM_RATIO_MAX = 10.0",
                 "MIN_EFF_RANK = 1.10", "MIN_REL_PERTURB = 1e-3",
                 "TOL_A0 = 0.05", 'HEALTH_SCOPE = ("Qwen", "BERT")'):
        assert frag in src, "源码里找不到阈值声明: %s" % frag
    assert 'verdict = "S1_PASS" if all(criteria.values()) else "S1_FAIL"' in src, "裁决逻辑变了"
    assert 'return 0 if res["verdict"] == "S1_PASS" else 1' in src, "退出码契约变了"
    print("[PASS] test_thresholds_were_fixed_before_the_run (C3/C4/C2 阈值在源码中写死)")
    return True


# ---------------------------------------------------------------- 10
def test_limits_and_modeling_choices_declared():
    """10. 口径局限与建模选择必须随数字一起登记"""
    m = _m()
    choices = " ".join(m["protocol"]["declared_modeling_choices"])
    for frag in ("norm", "bias", "embedding", "position"):
        assert frag in choices, "declared_modeling_choices 少了: %s" % frag
    limits = " ".join(m["limits"])
    for frag in ("路线 β", "语义", "ViT"):
        assert frag in limits, "limits 少了: %s" % frag
    assert m["protocol"]["scope"].startswith("S1"), "scope 声明变了"
    print("[PASS] test_limits_and_modeling_choices_declared (%d 条建模选择 / %d 条局限已登记)"
          % (len(m["protocol"]["declared_modeling_choices"]), len(m["limits"])))
    return True


# ---------------------------------------------------------------- 11
def test_verdict_derivable_and_falsified_on_record():
    """11. 裁决可由判据重算; 两条证伪必须在 MANIFEST 里在案"""
    m = _m()
    derived = "S1_PASS" if all(m["criteria"].values()) else "S1_FAIL"
    assert derived == m["verdict"], "裁决与判据脱钩 (%s != %s)" % (m["verdict"], derived)
    man = _manifest()
    falsified = man["falsified"]
    assert len(falsified) == 2, "证伪条目数变了: %d" % len(falsified)
    joined = " ".join(f["verdict"] for f in falsified)
    assert "路线 β" in joined, "证伪里没有路线 β 的边界"
    assert "ViT" in joined, "证伪里没有 ViT 的缺陷"
    assert man["claim"]["outcome_label"] == "confirmed"
    assert "不确认语义" in man["claim"]["note"], "claim 少了只证机制不证语义的限定"
    print("[PASS] test_verdict_derivable_and_falsified_on_record (裁决可重算, 2 条证伪在案)")
    return True


TESTS = [
    test_s1_criteria_all_pass,
    test_c2_provenance_reproduces_v62,
    test_live_source_sha256_matches_snapshot,
    test_c1_structure_and_gqa_heads,
    test_c3_health_scope_and_bounds,
    test_vit_defect_is_still_on_record,
    test_c4_injection_path_is_live,
    test_route_alpha_cannot_measure_semantics,
    test_thresholds_were_fixed_before_the_run,
    test_limits_and_modeling_choices_declared,
    test_verdict_derivable_and_falsified_on_record,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
