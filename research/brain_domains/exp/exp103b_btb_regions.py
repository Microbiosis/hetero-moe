#!/usr/bin/env python3
"""exp103b_btb_regions.py — 「乙」判决性探针 v2：读 Brain Treebank 的 **JSON/CSV** 元数据
⇒ ⇒ 数清楚**每个被试的电极分布在几个脑区**（⇒ 「三带」能不能成立）。

## 为什么有 v2

v1（`exp103_btb_regions.py`）按 h5 读 `electrode_labels` ⇒ 但**实际格式是 JSON**
（`electrode_labels/sub_N/electrode_labels.json`）⇒ ⇒ v1 的「脑区分布」是空的。
⇒ v2 改成：① **先打印 schema**（键名 + 样例）⇒ ② 再按「解剖关键词」统计**每被试的脑区数**。

## 只用元数据（**不下 130 GB 神经数据**）⇒ 便宜且可审计

- `electrode_labels/sub_N/electrode_labels.json`（7.9 KB 压缩包）⇒ 每电极的语义标签
- `localization/elec_coords_full.csv` + `localization/sub_N/depth-wm.csv`（453 KB）⇒ 每电极坐标
- `subject_metadata/sub_N_trialXXX_metadata.json`（7.9 KB）⇒ 每 trial 的元数据

⇒ ⇒ 输出：`/domains/runs/illusionbrain-align/exp103/exp103b_btb_regions.json`
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

WORK = Path("/domains/btb")
OUT = Path("/domains/runs/illusionbrain-align/exp103")

# 解剖关键词（用于把电极标签归到「带」；**只作初步分组**，最终分组要人工核）
REGION_PAT = re.compile(
    r"(frontal|parietal|temporal|occipital|hippocamp|amygdala|insula|cingulate|"
    r"thalam|stem|cerebell|gyrus|STG|MTG|ITG|STG|bank|BA|entorhinal|"
    r"moto|premotor|somatosensory|auditory|visual|putamen|caudate|pallidum)",
    re.I)


def log(*a):
    print(*a, flush=True)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    # ① schema 自发现：电极标注
    lab_dir = WORK / "electrode_labels"
    # ⚠️ zip 自带顶层目录 ⇒ 解压后是 `electrode_labels/electrode_labels/sub_N/...` ⇒ 用 rglob
    subs = sorted(lab_dir.rglob("electrode_labels.json")) if lab_dir.exists() else []
    log(f"[..] 电极标注文件 {len(subs)} 个")
    per_subject = {}
    schema_sample = None
    for p in subs:
        sid = p.parent.name
        try:
            d = load_json(p)
        except Exception as e:
            log(f"[FAIL] {sid}: {type(e).__name__}: {e}")
            continue
        if schema_sample is None:
            schema_sample = {"file": str(p.relative_to(WORK)),
                             "type": type(d).__name__,
                             "top_keys": list(d.keys())[:12] if isinstance(d, dict) else None,
                             "sample": (json.dumps(d, ensure_ascii=False)[:600]
                                        if not isinstance(d, list) else
                                        json.dumps(d[:3], ensure_ascii=False)[:600])}
        # 摊平成「标签 → 个数」
        labels = []

        def walk(o):
            if isinstance(o, str):
                labels.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
        cnt = Counter(labels)
        regions = {k: v for k, v in cnt.items() if REGION_PAT.search(k)}
        per_subject[sid] = {"n_strings": len(labels), "n_unique": len(cnt),
                            "n_region_like": len(regions),
                            "region_like": sorted(regions.items(), key=lambda kv: -kv[1])[:25],
                            "top_all": cnt.most_common(12)}
        log(f"[OK ] {sid}: {len(labels)} 个标签串 / {len(cnt)} 唯一 / **脑区样 {len(regions)}** "
            f"⇒ {sorted(regions.items(), key=lambda kv: -kv[1])[:6]}")
    res["schema_sample"] = schema_sample
    res["per_subject"] = per_subject

    # ② 坐标表：列名 + 每个被试的电极数
    loc_csv = WORK / "localization" / "elec_coords_full.csv"
    if loc_csv.exists():
        try:
            with open(loc_csv, newline="", encoding="utf-8", errors="replace") as f:
                rdr = csv.reader(f)
                head = next(rdr)
                rows = list(rdr)
            res["coords"] = {"columns": head, "n_rows": len(rows),
                             "sample_row": rows[0] if rows else None}
            log(f"[OK ] 坐标表：{len(rows)} 行，列 = {head}")
        except Exception as e:
            res["coords"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        res["coords"] = {"error": "elec_coords_full.csv 不存在"}

    # ③ 每 trial 元数据：trial 清单
    md_dir = WORK / "subject_metadata"
    trials = sorted([p for p in md_dir.glob("*/sub_*_metadata.json")]) if md_dir.exists() else []
    per_sub_trials = {}
    for p in trials:
        sid = p.name.split("_trial")[0]
        per_sub_trials[sid] = per_sub_trials.get(sid, 0) + 1
    res["n_trials_per_subject"] = per_sub_trials
    res["n_trials_total"] = len(trials)
    log(f"[OK ] trial 元数据：{len(trials)} 个 ⇒ 每被试 {sorted(per_sub_trials.items())}")

    # ④ 判决：**每被试能不能分出 ≥3 个非重叠的带**
    verdict = {"criterion_1_same_session_multi_region": None,
               "criterion_3_sessions_ge_8": len(trials) >= 8,
               "criterion_4_subjects_ge_2": len(per_sub_trials) >= 2,
               "criterion_5_licence": "CC BY 4.0（官网页脚原文）",
               "criterion_2_per_region_decodable": "UNKNOWN（要等脑区分组后再解码）"}
    n_reg = [v["n_region_like"] for v in per_subject.values()]
    if n_reg:
        verdict["criterion_1_same_session_multi_region"] = bool(max(n_reg) >= 3)
        verdict["max_region_like_per_subject"] = max(n_reg)
        verdict["median_region_like_per_subject"] = sorted(n_reg)[len(n_reg) // 2]
    res["verdict"] = verdict
    log("\n=== 对「乙」五条规格的初判 ===")
    for k, v in verdict.items():
        log(f"  {k}: {v}")

    (OUT / "exp103b_btb_regions.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {OUT}/exp103b_btb_regions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
