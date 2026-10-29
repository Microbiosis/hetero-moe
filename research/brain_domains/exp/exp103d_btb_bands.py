#!/usr/bin/env python3
"""exp103d_btb_bands.py — 「乙」判决性探针 v4：**用 Brain Treebank 自带的皮层图谱**
把「每个被试同时记录了几个带」算出来。

## 为什么有 v4（v1→v3 的教训链）

- v1：按 h5 读 `electrode_labels` ⇒ 实际是 **JSON** ⇒ 脑区分布空。
- v2：JSON 读到了，但 ① **坐标表路径写死**（漏了 zip 的顶层嵌套）⇒ 坐标步失败；
  ② 用「解剖关键词」正则去匹配电极标签 ⇒ **0 命中**（标签是 `F3aOFa1` 这种**唯一 ID**，不含解剖名）。
- v3（`exp103c_btb_labels.py`）：修好路径 + 全量 dump ⇒ **发现关键事实**：
  - `localization/localization/sub_N/depth-wm.csv` 的列 = `Electrode, L, I, P,
    **DesikanKilliany, Destrieux, DKT**, ShiftDist, **ConfType**`
    ⇒ ⇒ **Brain Treebank 自带三个皮层图谱的逐电极标签**（+ 一个「置信/类型」列）
  - `localization/localization/elec_coords_full.csv` 的列 = `'', ID, Z, X, Y,
    Hemisphere, Subject, Electrode, **Region**` ⇒ ⇒ 坐标表里**也有一列脑区**
  ⇒ ⇒ ⇒ **解剖根本不需要自己聚类** ⇒ ⇒ 「每个被试几个带」可以直接数。

## 本脚本做什么（三层，逐层可查）

1. **原始清单（不做任何映射）**：每个被试、每个图谱的**标签清单 + 计数**、`ConfType` 交叉表
   ⇒ 这是**事实层**，任何映射争议都改不动它。
2. **带映射（事先写死的关键词表）**：按**脑科学解剖**把图谱标签归到
   **POST（感觉/语义）· CENT（运动）· FRONT（执行）**⇒ 给出每个被试的**带覆盖**。
   ⇒ 映射表写在下方常量里，**先写死再跑**；并给出**三种映射变体**（见 VARIANTS）
   ⇒ ⇒ **结论不依赖单一映射**。
3. **五条规格的初判**：逐条给数字 + 明确标注哪些是**已判**、哪些是**仍 UNKNOWN**。

## 映射的依据（项目内正本）

- 带的定义见 `BRAIN_STRUCTURE_SPEC.md` §4.2 第 2 条 + `FINDINGS_TIER0_RSA.md` §三十五·五 第 1 条：
  **POST = 后部感觉/语义 · CENT = 中部运动 · FRONT = 最前执行**。
- C19（本项目已核）：「前部」**不是一块**（运动皮层也属前部；前额内部还有前后层级）
  ⇒ ⇒ ⇒ **CENT 与 FRONT 必须按解剖分开数**，不能把 precentral 算进 FRONT。
  ⇒ ⇒ ⇒ 本映射**刻意**把 precentral / paracentral 归 CENT、把 frontal pole / orbitofrontal
  / superior frontal / pars* 归 FRONT。

## 仍然 UNKNOWN 的（不猜）

- **第⑤条（每带显著可解码）**：需要**神经数据本体**（130 GB）⇒ 本脚本**不下神经数据**
  ⇒ ⇒ 只给**必要条件**（每带到底有几个电极；一个带 0 个电极 ⇒ 第⑤条当场死）。
- **每 session 的带覆盖**：Brain Treebank 的电极是**每个被试一次植入、跨 trial 复用**
  ⇒ ⇒ 同一被试的所有 trial 的带覆盖**相同** ⇒ 「同一 session 多带」**按被试判定**即可。

## 输出

`/domains/runs/illusionbrain-align/exp103/exp103d_btb_bands.json`
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

BASE = "https://braintreebank.dev/data/"
WORK = Path("/domains/btb")
OUT = Path("/domains/runs/illusionbrain-align/exp103")
LOC = WORK / "localization" / "localization"

# ---------------------------------------------------------------------------
# 带映射（**先写死**）：关键词 → 带。匹配用小写子串；命中多个时按 ORDER 取第一个。
# ---------------------------------------------------------------------------
POST_KW = ["occipital", "cuneus", "lingual", "fusiform", "temporal", "entorhinal",
           "parahippocampal", "hippocamp", "parietal", "supramarginal", "angular",
           "precuneus", "postcentral", "insula", "island", "visual", "auditory",
           "bank", "superior temporal", "middle temporal", "inferior temporal"]
CENT_KW = ["precentral", "paracentral", "premotor", "motor", "caudal middle frontal",
           "rolandic", "central"]
FRONT_KW = ["frontal", "orbitofrontal", "frontal pole", "parsopercularis",
            "parsorbitalis", "parstriangularis", "broca", "anterior cingulate",
            "rostral", "caudate", "putamen", "pallidum", "accumbens", "claustrum"]

BAND_ORDER = ["CENT", "FRONT", "POST"]   # 命中多个时的优先级（CENT/FRONT 优先于 POST）

# 变体：结论不依赖单一映射
VARIANTS = {
    # 主映射
    "v_main": {"post": POST_KW, "cent": CENT_KW, "front": FRONT_KW},
    # 变体 A：postcentral（初级体感）算 CENT（运动/体感都在中央）
    "v_postcentral_to_cent": {"post": [k for k in POST_KW if k != "postcentral"],
                               "cent": CENT_KW + ["postcentral"],
                               "front": FRONT_KW},
    # 变体 B：保守 —— 只认最硬的解剖词（不认 cingulate / basal ganglia / insula）
    "v_conservative": {"post": ["occipital", "cuneus", "lingual", "fusiform",
                                "temporal", "parietal", "supramarginal", "angular",
                                "precuneus", "postcentral"],
                       "cent": ["precentral", "paracentral"],
                       "front": ["frontal pole", "orbitofrontal", "superior frontal",
                                 "parsopercularis", "parsorbitalis",
                                 "parstriangularis", "rostral middle frontal"]},
}


def log(*a):
    print(*a, flush=True)


def ensure_localization():
    """保证 /domains/btb/localization/localization/ 在（v3 已经解压过；否则重下）。"""
    if (LOC / "elec_coords_full.csv").exists():
        return {"status": "already_extracted"}
    WORK.mkdir(parents=True, exist_ok=True)
    zp = WORK / "localization.zip"
    if not zp.exists():
        import urllib.request
        with urllib.request.urlopen(BASE + "localization.zip", timeout=300) as r:
            zp.write_bytes(r.read())
    with zipfile.ZipFile(zp) as z:
        z.extractall(WORK / "localization")
    return {"status": "extracted_now"}


def read_csv(path, max_rows=None):
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        return [], []
    head = rows[0]
    body = rows[1:] if max_rows is None else rows[1:1 + max_rows]
    return head, body


def band_of(label, mapping):
    """按事先写死的关键词表把图谱标签归到带（未命中 ⇒ None）。"""
    s = (label or "").strip().lower()
    if not s:
        return None
    hits = []
    for band in BAND_ORDER:
        kws = mapping[band.lower()]
        for kw in kws:
            if kw in s:
                hits.append((band, kw))
                break
    if not hits:
        return None
    for band, _ in hits:
        return band
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"ensure": ensure_localization()}

    # ---------------- ① 坐标表：Region 列 + 坐标 ----------------
    coord_path = LOC / "elec_coords_full.csv"
    head, body = read_csv(coord_path)
    res["coords_columns"] = head
    per_subject_coord = OrderedDict()
    idx = {h: i for i, h in enumerate(head)}
    for row in body:
        if len(row) < len(head):
            continue
        sid = row[idx.get("Subject", 6)] if "Subject" in idx else "?"
        per_subject_coord.setdefault(sid, []).append(row)
    coord_regions = {}
    for sid, rows in per_subject_coord.items():
        reg = Counter(r[idx["Region"]].strip() for r in rows) if "Region" in idx else Counter()
        hemi = Counter(r[idx["Hemisphere"]].strip() for r in rows) if "Hemisphere" in idx else Counter()
        coord_regions[sid] = {"n_rows": len(rows),
                              "n_distinct_region": len(reg),
                              "regions": reg.most_common(),
                              "hemisphere": hemi.most_common()}
    res["coords_per_subject"] = coord_regions
    log(f"[OK ] 坐标表 {len(body)} 行 / {len(per_subject_coord)} 被试 ⇒ "
        f"Region 列样例：{list(coord_regions.values())[0]['regions'][:6] if coord_regions else []}")

    # ---------------- ② 逐被试 depth-wm.csv：三个图谱 ----------------
    atlases = ["DesikanKilliany", "Destrieux", "DKT"]
    per_subject = OrderedDict()
    for d in sorted(LOC.glob("sub_*/depth-wm.csv")):
        sid = d.parent.name
        h, b = read_csv(d)
        ix = {c: i for i, c in enumerate(h)}
        rec = {"file": str(d.relative_to(WORK)), "n_rows": len(b),
               "columns": h, "atlases": {}}
        for a in atlases:
            if a not in ix:
                continue
            vals = [r[ix[a]].strip() for r in b if len(r) > ix[a]]
            cnt = Counter(v for v in vals if v)
            rec["atlases"][a] = {"n_labeled": sum(cnt.values()),
                                 "n_distinct": len(cnt),
                                 "labels": cnt.most_common()}
        if "ConfType" in ix:
            rec["conftype"] = Counter(r[ix["ConfType"]].strip() for r in b
                                       if len(r) > ix["ConfType"]).most_common()
        per_subject[sid] = rec
        log(f"[OK ] {sid}: {rec['n_rows']} 电极 ⇒ " +
            " | ".join(f"{a}={rec['atlases'][a]['n_distinct']}"
                       for a in atlases if a in rec["atlases"]))
    res["per_subject"] = per_subject

    # ---------------- ③ 带映射（三个变体） ----------------
    band_cov = OrderedDict()
    for vname, mapping in VARIANTS.items():
        cov = OrderedDict()
        for sid, rec in per_subject.items():
            vres = {}
            for a in atlases:
                if a not in rec["atlases"]:
                    continue
                band_cnt = Counter()
                band_elec = Counter()
                for lab, n in rec["atlases"][a]["labels"]:
                    bd = band_of(lab, mapping)
                    if bd:
                        band_cnt[bd] += n
                        band_elec[bd] += 1
                vres[a] = {"bands": {b: band_elec[b] for b in ("POST", "CENT", "FRONT")},
                           "n_bands": sum(1 for b in ("POST", "CENT", "FRONT")
                                          if band_elec[b] > 0),
                           "n_bands_ge2elec": sum(1 for b in ("POST", "CENT", "FRONT")
                                                  if band_elec[b] >= 2)}
            cov[sid] = vres
        band_cov[vname] = cov
    res["band_coverage"] = band_cov
    for vname in VARIANTS:
        row = []
        for sid, v in band_cov[vname].items():
            dk = v.get("DesikanKilliany", {})
            row.append(f"{sid}:{dk.get('n_bands', 0)}带/{dk.get('n_bands_ge2elec', 0)}带≥2极")
        log(f"[..] 变体 {vname}（DK）⇒ " + "  ".join(row))

    # ---------------- ④ 五条规格的初判 ----------------
    n_subjects = len(per_subject)
    trial_dir = WORK / "subject_metadata"
    trials = sorted(trial_dir.rglob("*_metadata.json")) if trial_dir.exists() else []
    per_sub_trials = Counter(p.name.split("_trial")[0] for p in trials)

    main_cov = band_cov["v_main"]
    n3 = [v["DesikanKilliany"]["n_bands"] for v in main_cov.values()
          if "DesikanKilliany" in v]
    n3_all_variants = {vn: [v["DesikanKilliany"]["n_bands"] for v in band_cov[vn].values()
                             if "DesikanKilliany" in v] for vn in VARIANTS}
    verdict = OrderedDict()
    verdict["criterion_1_same_session_three_bands"] = {
        "status": "DECIDED_BY_ATLAS",
        "per_subject_bands_DK_v_main": {s: v["DesikanKilliany"]["bands"]
                                        for s, v in main_cov.items() if "DesikanKilliany" in v},
        "n_bands_per_subject_v_main": n3,
        "n_subjects_with_all3": sum(1 for x in n3 if x == 3),
        "n_subjects_with_ge2": sum(1 for x in n3 if x >= 2),
        "n_bands_per_subject_all_variants": n3_all_variants,
        "note": "电极按被试一次植入、跨 trial 复用 ⇒ 同被试所有 session 的带覆盖相同",
    }
    verdict["criterion_2_behavior_label_per_band"] = {
        "status": "DECIDED_BY_DATASET_DESIGN",
        "brain_treebank": "被动看 Hollywood 电影 + 转写 + UD 句法树 ⇒ "
                          "有**语义/语言**行为标签；**无运动**（CENT）、**无规则学习/任务集切换/决策**（FRONT）行为标签",
        "conclusion": "第②条**不满足**（POST 有语义标签，CENT/FRONT 没有对应行为）",
    }
    verdict["criterion_3_sessions_and_subjects"] = {
        "status": "DECIDED",
        "trials_total": len(trials),
        "trials_per_subject": dict(sorted(per_sub_trials.items())),
        "max_trials_per_subject": max(per_sub_trials.values()) if per_sub_trials else 0,
        "n_subjects": n_subjects,
        "spec_min_5_sessions_x_2_subjects": {
            "n_subjects_with_ge5": sum(1 for v in per_sub_trials.values() if v >= 5),
            "pass": sum(1 for v in per_sub_trials.values() if v >= 5) >= 2},
        "exp97c_threshold_8": {
            "n_subjects_with_ge8": sum(1 for v in per_sub_trials.values() if v >= 8),
            "pass": sum(1 for v in per_sub_trials.values() if v >= 8) >= 2},
    }
    verdict["criterion_5_band_decodability"] = {
        "status": "UNKNOWN",
        "why": "需要神经数据本体（130 GB）+ 解码；本探针只下元数据",
        "necessary_condition": "每带的电极数（见 band_coverage）⇒ 0 电极的带当场出局",
    }
    verdict["criterion_4_licence"] = {
        "status": "DECIDED",
        "licence": "CC BY 4.0（官网页脚原文 'Data released under a CC BY 4.0 license'）",
        "access": "直链下载、无注册",
    }
    res["verdict"] = verdict

    log("\n=== 「乙」五条规格：Brain Treebank 初判 ===")
    for k, v in verdict.items():
        log(f"  {k}: {v.get('status')} | "
            f"{v.get('conclusion') or v.get('note') or ''}")

    (OUT / "exp103d_btb_bands.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {OUT}/exp103d_btb_bands.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
