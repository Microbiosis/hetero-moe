"""lateral_sitepairs.py — **同位点左右配对**的偏侧检验（用 exp89b 已有的逐通道曲线，零成本）。

## 为什么要有这个

exp90-lateral 把 63 个电极按左右分成 28 + 28 + 7 两组来测左右轴，
但那样**每组都摊薄了位置**：左 28 个里有 3 个是左枕（真正带图像语义的只有它们），
⇒ 两组各自只剩极弱的信号，于是「跨侧不迁移」这件事**分不清**是
「左右不可分」还是「两边都太弱」。

⇒ 更干净的检验：**同一位点的左右两个电极配对比**（O1↔O2、PO3↔PO4、…，共 28 对）
⇒ 每个配对**位置相同**，唯一的差别就是左右 ⇒ 这才是「偏侧」该问的问题。
⇒ 数据不用重算：exp89b 的逐通道曲线（每被试 63 个 RSA）**已经有了**。

## 判据

| 结果 | 结论 |
|---|---|
| 28 对配对里「左 > 右」与「右 > 左」大致各半（配对符号检验不显著） | **EEG 尺度上检不出偏侧** ⇒ 左右轴在当前数据里**不构成可用的轴**（登记为「未测·当前数据测不了」） |
| 某一组位点（如枕叶）系统性地偏向一侧 | 该处**有**偏侧 ⇒ 左右轴在该处**可测**，应进骨架 |
| 普遍偏向一侧 | 全局偏侧（如语言左偏 ⇒ 视觉右侧强） |

## 输出

`results/lateral-results/lateral_sitepairs.{json,md}`
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CURVE = HERE / "posaxis-results" / "per_channel_curve.json"
OUT = HERE / "lateral-results"

# 哪些位点对特别值得单看（视觉语义最可能带偏侧的地方）
FOCUS = {"O": "枕（O1/O2）", "PO": "枕顶（PO3/PO4, PO7/PO8）",
         "P": "顶（P1/P2, P3/P4, P5/P6, P7/P8）", "T": "颞（T7/T8）",
         "C": "中央（C1/C2, C3/C4, C5/C6）", "F": "额（F1/F2, F3/F4, F5/F6, F7/F8）",
         "FC": "额中央（FC1/FC2…）", "CP": "中央顶（CP1/CP2…）",
         "TP": "顶颞（TP7/TP8, TP9/TP10）", "FT": "额颞（FT7/FT8, FT9/FT10）",
         "AF": "额极（AF3/AF4, AF7/AF8）", "Fp": "额极（Fp1/Fp2）"}


def site_of(name):
    """电极名 → (位点前缀, 对序号)；与 exp90_lateral.site_key 同规则。

    ⚠️ **不改变大小写**：通道名里 `Fp1/Fp2` 是混合大小写，
    大写后再拼前缀会得到 "FP1" ⇒ 找不到通道 ⇒ 配对被悄悄丢掉（实测少一对）。
    """
    import re
    n = str(name).strip().replace("-", "").replace("_", "")
    m = re.match(r"^(.*?)(\d+)$", n)
    if not m:
        return (n.upper(), 0)
    pre, d = m.group(1), int(m.group(2))
    return (pre.upper(), (d + 1) // 2 if d % 2 else d // 2)


def sign_test_p(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


def main():
    if not CURVE.exists():
        print(f"✗ 缺 {CURVE}（先跑 exp89b 的 Job 再解析）")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(CURVE.read_text(encoding="utf-8"))
    names = d.get("ch_names") or []
    raw = d.get("per_subject_curves") or []
    # 解析器里 `per_subject_curves` 是 {subject: [63 个数]} ⇒ 展成列表
    curves = [raw[k] for k in sorted(raw)] if isinstance(raw, dict) else list(raw)
    if not names or not curves:
        print("✗ 曲线 JSON 里没有 ch_names / per_subject_curves")
        return 1

    idx = {c: i for i, c in enumerate(names)}
    # 同位点的左右配对（按 site 键找伙伴 ⇒ 不依赖前缀拼写）
    key_of = {c: site_of(c) for c in names}
    by_key = {}
    for c in names:
        by_key.setdefault(key_of[c], []).append(c)
    pairs = []
    for key, group in by_key.items():
        if key[1] == 0:                 # 中线
            continue
        if len(group) != 2:
            continue
        a, b = group
        # 左 = 末位奇数
        la = a[-1].isdigit() and int(a[-1]) % 2 == 1
        lb = b[-1].isdigit() and int(b[-1]) % 2 == 1
        if la == lb:
            continue
        l, r = (a, b) if la else (b, a)
        pairs.append((l, r))
    pairs.sort(key=lambda p: idx[p[0]])
    log = []
    per_pair = {}
    for l, r in pairs:
        diffs = [cv[idx[l]] - cv[idx[r]] for cv in curves]
        k, m, p = sign_test_p(diffs)
        per_pair[f"{l}/{r}"] = {"median_diff": sorted(diffs)[len(diffs) // 2],
                               "n_pos": k, "n_neg": m, "p_sign": p,
                               "left_median": sorted(cv[idx[l]] for cv in curves)[len(curves) // 2],
                               "right_median": sorted(cv[idx[r]] for cv in curves)[len(curves) // 2]}
        log.append((l, r, per_pair[f"{l}/{r}"]))

    # 汇总：把每一对 × 每个被试展平（配对符号检验）
    all_diffs = []
    for l, r in pairs:
        all_diffs += [cv[idx[l]] - cv[idx[r]] for cv in curves]
    k_all, m_all, p_all = sign_test_p(all_diffs)
    # 按位点族汇总
    fam = {}
    for l, r in pairs:
        pre = site_of(l)[0]
        fam.setdefault(pre, []).append((l, r))
    fam_out = {}
    for pre, ps in fam.items():
        ds = []
        for l, r in ps:
            ds += [cv[idx[l]] - cv[idx[r]] for cv in curves]
        k, m, p = sign_test_p(ds)
        fam_out[pre] = {"label": FOCUS.get(pre, pre), "n_pairs": len(ps),
                        "n_pos": k, "n_neg": m, "p_sign": p,
                        "median_diff": sorted(ds)[len(ds) // 2]}

    out = {"n_pairs": len(pairs), "n_subjects": len(curves),
           "per_pair": per_pair, "per_family": fam_out,
           "overall": {"n": len(all_diffs), "n_pos": k_all, "n_neg": m_all,
                       "p_sign": p_all,
                       "median_diff": sorted(all_diffs)[len(all_diffs) // 2]}}

    # --- 逐被试聚合（**避免同族配对之间的伪重复**：同族的配对共享被试）---
    # exp89 已证：带图像语义的是 **枕/顶/颞**（O/PO/P/T）⇒ 这一族才可能出现偏侧
    POST_PRE = {"O", "PO", "P", "T"}
    groups_agg = {
        "posterior_O_PO_P_T": [(l, r) for l, r in pairs
                              if site_of(l)[0] in POST_PRE],
        "all_pairs": list(pairs),
        "anterior_F_FC_AF_FT_C_CP": [(l, r) for l, r in pairs
                                      if site_of(l)[0] in
                                      {"F", "FC", "AF", "FT", "C", "CP"}],
    }
    agg = {}
    for gname, ps in groups_agg.items():
        if not ps:
            continue
        per_subj = []
        for cv in curves:
            ds = [cv[idx[l]] - cv[idx[r]] for l, r in ps]
            per_subj.append(sum(ds) / len(ds))
        k, m, p = sign_test_p(per_subj)
        agg[gname] = {"n_pairs": len(ps), "n_subjects": len(per_subj),
                      "n_pos": k, "n_neg": m, "p_sign": p,
                      "median_of_subject_means": sorted(per_subj)[len(per_subj) // 2],
                      "per_subject": per_subj}
    out["per_subject_aggregate"] = agg
    (OUT / "lateral_sitepairs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 左右轴：同位点左右配对的偏侧检验（用 exp89b 已有的逐通道曲线）", ""]
    L.append(f"配对数 {len(pairs)}（28 个位点对）｜被试 {len(curves)}")
    L.append("")
    L.append("## 总体（把 28 对 × 每被试展平后的配对差）")
    L.append("")
    o = out["overall"]
    L.append(f"- 配对差中位：**{o['median_diff']:+.4f}**（左 − 右）")
    L.append(f"- 「左 > 右」**{o['n_pos']}/{o['n']}**，双侧符号检验 **p = {o['p_sign']:.4f}**")
    L.append("")
    L.append("## ⭐ 逐被试聚合（**同族配对共享被试 ⇒ 上面的 p 是伪重复**；"
             "这才是 10 个独立单位的检验）")
    L.append("")
    L.append("| 组 | 对数 | 左>右/右>左（被试） | 符号检验 p | 每被试均值的中位 |")
    L.append("|---|---|---|---|---|")
    for gname, e in out.get("per_subject_aggregate", {}).items():
        L.append(f"| {gname} | {e['n_pairs']} | {e['n_pos']}/{e['n_pos']+e['n_neg']} "
                 f"| {e['p_sign']:.4f} | {e['median_of_subject_means']:+.4f} |")
    L.append("")
    L.append("## 按位点族")
    L.append("")
    L.append("| 位点族 | 对数 | 左>右 / 右>左 | 符号检验 p | 配对差中位 |")
    L.append("|---|---|---|---|---|")
    for pre, e in sorted(fam_out.items()):
        L.append(f"| {e['label']} | {e['n_pairs']} | {e['n_pos']}/{e['n_pos']+e['n_neg']} "
                 f"| {e['p_sign']:.4f} | {e['median_diff']:+.4f} |")
    L.append("")
    L.append("## 逐对")
    L.append("")
    L.append("| 配对（左/右） | 左中位 r | 右中位 r | 差中位 | 左>右/右>左 | p |")
    L.append("|---|---|---|---|---|---|")
    for l, r, e in log:
        L.append(f"| {l}/{r} | {e['left_median']:+.4f} | {e['right_median']:+.4f} "
                 f"| {e['median_diff']:+.4f} | {e['n_pos']}/{e['n_pos']+e['n_neg']} "
                 f"| {e['p_sign']:.4f} |")
    L.append("")
    (OUT / "lateral_sitepairs.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"写入 {OUT}/lateral_sitepairs.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
