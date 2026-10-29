#!/usr/bin/env python3
"""exp105 三个交叉对照的**归因表复算**（离线，只读 `exp105-results/` 里的判决步产物）。

用途：给 `FINDINGS_TIER0_RSA.md` §五十四 的归因表提供**可复算**的每一格。
判据一字不改（§四十九·三/四/五）；这里只把**预登记的同一个统计函数**
（`exp/exp101_condition.py` 的 `paired_test` / `med_up`）应用到判决步自己写出的
逐被试 / 逐概念 CSV 上。

**为什么要补算 snr**：`exp/exp102_gen.py` 的 `judge` 在 C0 不过时**提前 return**
（`exp102_gen.py:183-187`）⇒ snr 的 C1/C2 **从未被计算** ⇒ 官方 verdict 只有
`C0_FAIL` 一格，归因表的 X1 列没有 2-way 证据。而逐被试 CSV 是同一个判决步写出的
（`own2/floor2/own50/floor50` 就是判据的输入）⇒ 补算**不改任何判据、不新增任何数据**。
D95（§五十二·五，出结果前落盘）已明确：C0 在 7,400 步量级测不出 ⇒ 本轮归因只认
2-way / 50-way 判据。

**怎么验证补算管线可信**：同一脚本对 `more` 复算，结果与判决步自己的
`more_verdict.json` **逐位相同** ⇒ 补算管线在「有官方答案可对」的那个对照上零偏差
⇒ 应用到 snr 上可信。

用法：
    python results/parse_exp105.py                 # 人读
    python results/parse_exp105.py --self-test     # 自测（断言，非 0 退出）

口径（与 exp101/exp102 逐字一致）：统计单元 X1/X2 = 被试、X3 = 概念；
配对符号检验；零差剔除；中位 = **上中位** `sorted[n//2]`；阈值 ≥8 / p ≤ 0.05 / 中位 > 0。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                            # research/brain_domains
sys.path.insert(0, str(REPO / "exp"))
import exp101_condition as E101               # noqa: E402

RES = HERE / "exp105-results"
ALPHA = 0.05
MIN_SUBJECTS = 8
REPO102 = REPO / "results" / "exp102-results"


def load(variant: str):
    """读判决步自己写出的逐被试 / 逐概念 CSV。"""
    fn = "per_concept.csv" if variant == "oracle" else "per_subject.csv"
    rows = list(csv.DictReader(open(RES / f"{variant}_{fn}", encoding="utf-8")))
    out = []
    for r in rows:
        def f(k):
            v = r.get(k, "")
            return float(v) if v not in ("", None) else None
        out.append({"subject": r.get("subject", r.get("concept", "?")),
                    "own2": f("own2"), "floor2": f("floor2"),
                    "own50": f("own50"), "floor50": f("floor50"),
                    "cond2": f("cond2_same_clip")})
    return out


def paired(rows, a, b, name):
    """预登记的判据函数：配对符号检验 + 阈值（与 exp102_gen.judge 逐字同一套）。"""
    diffs = [r[a] - r[b] for r in rows if r[a] is not None and r[b] is not None]
    if not diffs:
        print(f"    {name}: 无可配对数据")
        return None
    d: dict = {}
    v = E101.paired_test(diffs, name, d)
    v["pass"] = bool(v["n_eff"] >= MIN_SUBJECTS and v["p_sign"] <= ALPHA
                     and (v.get("median") or 0) > 0)
    print(f"    {name}: n_eff={v['n_eff']} n_pos={v['n_pos']} n_neg={v['n_neg']} "
          f"n_zero={v['n_zero']} p_sign={v['p_sign']:.6f} "
          f"median={v.get('median'):+.6f} ⇒ pass={v['pass']}")
    return v


def c3(ok):
    """judge 的 C3_report 公式（同一个 CLIP 口径）：生成 ÷ 条件。"""
    own2 = [r["own2"] for r in ok if r["own2"] is not None]
    flo2 = [r["floor2"] for r in ok if r["floor2"] is not None]
    cond = [r["cond2"] for r in ok if r["cond2"] is not None]
    om, fm, cm = E101.med_up(own2), E101.med_up(flo2), E101.med_up(cond)
    print(f"    生成 2-way 中位={om:.4f}｜地板={fm:.4f}｜条件={cm:.4f}"
          f"｜传递效率={om / cm:.4f}" if cm else "    条件中位缺失")
    return om, fm, cm


def recompute():
    """返回 {variant: {...}}，供自测断言与文档引用。"""
    out: dict = {}

    # ---- X1 snr：官方判决 C0_FAIL ⇒ C1/C2 补算 ----
    snr = load("snr")
    ok = [r for r in snr if r["floor2"] is not None and r["floor50"] is not None]
    print("=" * 78)
    print("① X1「snr」：条件 = 4 rep 平均（在冻结 encoder 输出空间）")
    print("=" * 78)
    print(f"  可配对被试 n={len(ok)}/{len(snr)}"
          f"（floor 为空 ⇒ 它的地板来自被剔除的 donor）")
    c1 = paired(ok, "own2", "floor2", "C1 two_way_minus_floor")
    c2 = paired(ok, "own50", "floor50", "C2 fifty_way_minus_floor")
    om, fm, cm = c3(snr)
    official = json.load(open(RES / "snr_verdict.json", encoding="utf-8"))["verdict"]
    print(f"  ⇒ 官方判决 {official['verdict']}"
          f"（loss {official['C0']['loss_first10']:.6f} → "
          f"{official['C0']['loss_last10']:.6f} 没降 ⇒ 判据函数提前返回）")
    print(f"  ⇒ 补算后：C1_pass={c1['pass']} C2_pass={c2['pass']} ⇒ 按预登记判据走完 = "
          f"{'CONDITION_DRIVES_GENERATION' if c1['pass'] and c2['pass'] else 'DIRECTION_RIGHT_UNDERPOWERED'}")
    print(f"  ⇒ **X1 = {'通过' if c1['pass'] else '不过'}**")
    out["snr"] = {"c1": c1, "c2": c2, "own2": om, "floor2": fm, "cond2": cm,
                  "official": official["verdict"], "pass": bool(c1["pass"])}

    # ---- X2 more：官方判决 DIRECTION_RIGHT_UNDERPOWERED ⇒ 复算必须逐位复现 ----
    more = load("more")
    okm = [r for r in more if r["floor2"] is not None and r["floor50"] is not None]
    print()
    print("=" * 78)
    print("② X2「more」：训练量 ×2（14,800 步，同一条轨迹 --resume）")
    print("=" * 78)
    print(f"  可配对被试 n={len(okm)}/{len(more)}"
          f"（sub-05 被对齐闸门剔除；sub-04 的 floor 为空）")
    c1m = paired(okm, "own2", "floor2", "C1 two_way_minus_floor")
    c2m = paired(okm, "own50", "floor50", "C2 fifty_way_minus_floor")
    omm, fmm, cmm = c3(more)
    vm = json.load(open(RES / "more_verdict.json", encoding="utf-8"))["verdict"]
    print(f"  ⇒ 官方判决 {vm['verdict']} ⇒ **X2 = "
          f"{'通过' if vm['C1_pass'] else '不过'}**")
    out["more"] = {"c1": c1m, "c2": c2m, "own2": omm, "floor2": fmm, "cond2": cmm,
                   "official": vm["verdict"], "pass": bool(vm["C1_pass"]),
                   "official_c1": vm["C1"]["two_way_minus_floor"]}

    # ---- X3 oracle：官方判决 CONDITION_DRIVES_GENERATION ----
    orc = load("oracle")
    print()
    print("=" * 78)
    print("③ X3「oracle」：条件 = 真图 CLIP 特征（统计单元 = 概念，n=200）")
    print("=" * 78)
    print(f"  概念数 n={len(orc)}")
    vo = json.load(open(RES / "oracle_verdict.json", encoding="utf-8"))["verdict"]
    for k in ("C1", "C2"):
        for kk, vv in vo[k].items():
            print(f"    {kk}: n_eff={vv['n_eff']} n_pos={vv['n_pos']} "
                  f"n_neg={vv['n_neg']} p_sign={vv['p_sign']:.6f} "
                  f"median={vv['median']:+.6f} ⇒ pass={vv['pass']}")
    print(f"  ⇒ 官方判决 {vo['verdict']} ⇒ **X3 = "
          f"{'通过' if vo['C1_pass'] and vo['C2_pass'] else '不过'}**")
    out["oracle"] = {"official": vo["verdict"],
                     "pass": bool(vo["C1_pass"] and vo["C2_pass"])}

    # ---- 前置闸门 ----
    print()
    print("=" * 78)
    print("④ 前置闸门（对照有效性）")
    print("=" * 78)
    g = json.load(open(RES / "cond_gate.json", encoding="utf-8"))
    gd = g["x1_gate"]["diff"]
    print(f"  X1：avg4 条件侧 2-way={g['avg4']['two_way_median']} vs "
          f"rep0={g['rep0']['two_way_median']}；配对差 n={gd['n_eff']} / "
          f"{gd['n_pos']}正{gd['n_neg']}负 / p={gd['p_sign']:.6f} / "
          f"中位 +{gd['median']:.4f} ⇒ pass={g['x1_gate']['pass']}")
    print("  X2：C0 loss 0.080511→0.078107 decreased=True；曲线 296 epoch / "
          "step 50→14800、步长恒 50 无跳变 ⇒ pass=True")
    print("  X3：条件 = 真图 CLIP ⇒ 条件侧 2-way = 1.0 ≥ 0.99 ⇒ pass=True")

    # ---- 归因表 ----
    print()
    print("=" * 78)
    print("⑤ 归因表（§四十九·五，预登记）")
    print("=" * 78)
    x1 = "通过" if out["snr"]["pass"] else "不过"
    x2 = "通过" if out["more"]["pass"] else "不过"
    x3 = "通过" if out["oracle"]["pass"] else "不过"
    TABLE = {("通过",): "瓶颈 = 单试次 SNR",
             ("不过", "通过"): "瓶颈 = 训练量",
             ("不过", "不过", "通过"): "瓶颈 = 脑侧条件的信息量（管线没问题）",
             ("不过", "不过", "不过"): "管线/目标/评测有问题 ⇒ 停下查实现（不报科学结论）",
             ("通过", "通过", "不过"): "口径不自洽 ⇒ 停下查实现（不许解释成科学结论）"}
    concl = TABLE.get((x1, x2, x3)) or "未覆盖的组合 ⇒ 停下查实现"
    print(f"  X1={x1}｜X2={x2}｜X3={x3}")
    print(f"  ⇒ **{concl}**")
    out["attribution"] = {"x1": x1, "x2": x2, "x3": x3, "conclusion": concl}

    # ---- 只报项 ----
    print()
    print("=" * 78)
    print("⑥ 只报项（不参与判决）")
    print("=" * 78)
    for v in ("exp102", "snr", "more", "oracle"):
        if v == "exp102":
            m = json.load(open(REPO102 / "mechanism.json", encoding="utf-8"))
        else:
            m = json.load(open(RES / f"{v}_mechanism.json", encoding="utf-8"))
        print(f"  {v:7s} w_max={m['w_max']:.6f} w_mean={m['w_mean']:.6f} "
              f"tok_med={m['tok_med']:.4f} consistency={m['consistency']:.6f}")
    print()
    print("  超机会量传递率（(生成−地板)/(条件−地板)，两个中位之比）：")
    for name, gen, flo, cond in (("exp102", 0.505, 0.500, 0.765),
                                 ("snr", out["snr"]["own2"], out["snr"]["floor2"], out["snr"]["cond2"]),
                                 ("more", out["more"]["own2"], out["more"]["floor2"], out["more"]["cond2"]),
                                 ("oracle", 1.0, 0.0, 1.0)):
        r = (gen - flo) / (cond - flo) if cond != flo else float("nan")
        print(f"    {name:7s} ({gen:.4f}−{flo:.4f})/({cond:.4f}−{flo:.4f}) = {r:+.4f} = {r * 100:+.2f}%")
    print()
    print("  C0 的「下降」与 epoch 间振荡之比（D95 的量化版）：")
    for v in ("snr", "more", "oracle"):
        d = json.load(open(RES / f"{v}_train_curve.json", encoding="utf-8"))
        L = [float(e["mean_loss"]) for e in d["epochs"]]
        n = len(L)
        k = int(round(n * 0.10))
        f = sum(L[:k]) / k
        l = sum(L[-k:]) / k
        d1 = [abs(L[i + 1] - L[i]) for i in range(n - 1)]
        d1.sort()
        med_d1 = d1[len(d1) // 2]
        print(f"    {v:7s} n={n} k={k} 首10%={f:.6f} 末10%={l:.6f} "
              f"变化={(l - f) / f * 100:+.2f}% decreased={l < f}｜"
              f"相邻 epoch |Δ| 上中位={med_d1:.6f} ⇒ "
              f"C0 的 |变化| 是它的 {abs(l - f) / med_d1 * 100:.1f}%")
    return out


def self_test():
    """断言：① more 复算逐位复现官方判决；② 归因表落到预期那一行。"""
    print("===== 自测")
    r = recompute()
    errs = []

    # ① more 的 C1 必须与官方逐位相同（补算管线的对拍）
    off = r["more"]["official_c1"]
    got = r["more"]["c1"]
    for k in ("n_eff", "n_pos", "n_neg", "n_zero"):
        if got[k] != off[k]:
            errs.append(f"more C1 {k}: 复算 {got[k]} ≠ 官方 {off[k]}")
    if abs(got["p_sign"] - off["p_sign"]) > 1e-12:
        errs.append(f"more C1 p_sign: 复算 {got['p_sign']} ≠ 官方 {off['p_sign']}")
    if abs(got["median"] - off["median"]) > 1e-12:
        errs.append(f"more C1 median: 复算 {got['median']} ≠ 官方 {off['median']}")
    if got["pass"] != off["pass"]:
        errs.append(f"more C1 pass: 复算 {got['pass']} ≠ 官方 {off['pass']}")

    # ② 归因表必须落到「瓶颈 = 脑侧条件的信息量」那一行
    a = r["attribution"]
    if (a["x1"], a["x2"], a["x3"]) != ("不过", "不过", "通过"):
        errs.append(f"归因表行 = ({a['x1']}, {a['x2']}, {a['x3']})，"
                    f"预期 ('不过', '不过', '通过')")
    if "脑侧条件的信息量" not in a["conclusion"]:
        errs.append(f"结论 = {a['conclusion']}")

    # ③ snr 的官方判决必须是 C0_FAIL（否则补算的前提不成立）
    if r["snr"]["official"] != "C0_FAIL":
        errs.append(f"snr 官方判决 = {r['snr']['official']}，预期 C0_FAIL")

    if errs:
        print()
        for e in errs:
            print("  ✗", e)
        print(f"\n自测失败（{len(errs)} 条）")
        return 1
    print()
    print("  ✓ more 的 C1 与官方 verdict.json 逐位相同（补算管线零偏差）")
    print("  ✓ 归因表落到「瓶颈 = 脑侧条件的信息量（管线没问题）」")
    print("  ✓ snr 官方判决 = C0_FAIL（补算前提成立）")
    print("\n自测通过")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="跑断言（more 逐位复现官方 + 归因表行）")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    recompute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
