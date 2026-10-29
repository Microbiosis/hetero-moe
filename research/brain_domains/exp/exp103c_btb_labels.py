#!/usr/bin/env python3
"""exp103c_btb_labels.py — 「乙」判决性探针 v3：**把 Brain Treebank 的解剖线索全 dump 出来**。

## 为什么有 v3

- v1（`exp103_btb_regions.py`）：按 h5 读 ⇒ 实际是 **JSON** ⇒ 脑区分布空。
- v2（`exp103b_btb_regions.py`）：JSON 读到了（26 trial / 每被试 1~7 个），
  但 ① **坐标表路径写死**（`localization/elec_coords_full.csv`）⇒ **实际多一层嵌套** ⇒ 坐标步失败；
  ② 电极标签是**每电极唯一 ID**（`F3aOFa1`）⇒ 用「解剖关键词」正则匹配 ⇒ **0 命中**（标签里没有解剖词）。

⇒ ⇒ **本版只做一件事：把原始线索原样 dump 出来 + 自动发现结构**（不预设任何解剖假设）：

1. **每个被试的全部电极标签**（完整列表，不截断）⇒ 人眼可读 ⇒ 能看出命名规则；
2. **标签结构自发现**：去尾号 ⇒ 分组键；按 `a` 切分 ⇒ 找出「哪一段在变」；
3. **坐标表**：用 `rglob` 找（不再写死路径）⇒ 列名 + 行数 + 前若干行；
4. **每个被试的 `depth-wm.csv`**（若存在）⇒ 头几行。

⇒ ⇒ 判决「三带」能不能成立，靠的是**这一步的原始线索**，不是任何预设规则。

## 只用元数据（**不下 130 GB 神经数据本体**）⇒ 便宜且可审计

## 输出

`/domains/runs/illusionbrain-align/exp103/exp103c_btb_labels.json`
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

BASE = "https://braintreebank.dev/data/"
WORK = Path("/domains/btb")
OUT = Path("/domains/runs/illusionbrain-align/exp103")
ZIPS = ("electrode_labels.zip", "localization.zip", "subject_metadata.zip")


def log(*a):
    print(*a, flush=True)


def fetch(url, dest):
    import urllib.request
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def ensure_zips():
    WORK.mkdir(parents=True, exist_ok=True)
    info = {}
    for name in ZIPS:
        p = WORK / name
        try:
            if not p.exists():
                n = fetch(BASE + name, p)
            else:
                n = p.stat().st_size
            target = WORK / name.replace(".zip", "")
            if not target.exists():
                with zipfile.ZipFile(p) as z:
                    z.extractall(target)
            info[name] = {"bytes": n,
                          "members": zipfile.ZipFile(p).namelist()[:40]}
            log(f"[OK ] {name}: {n} bytes")
        except Exception as e:
            info[name] = {"error": f"{type(e).__name__}: {e}"}
            log(f"[FAIL] {name}: {type(e).__name__}: {e}")
    return info


def strip_tail(s):
    """去尾号：'F3aOFa1' → 'F3aOFa'；'F3aOFa12' → 'F3aOFa'。"""
    return re.sub(r"\d+$", "", s)


def discover_structure(labels):
    """标签结构自发现：① 按 'a' 切分看各段取值数；② 找公共前缀。"""
    out = {}
    parts = [s.split("a") for s in labels]
    n_parts = Counter(len(p) for p in parts)
    out["split_by_a_part_count"] = dict(n_parts)
    if n_parts:
        k = n_parts.most_common(1)[0][0]
        segs = [p for p in parts if len(p) == k]
        if k >= 2:
            out["segments"] = [
                {"index": i,
                 "n_unique": len(set(p[i] for p in segs)),
                 "examples": sorted(set(p[i] for p in segs))[:40]}
                for i in range(k)]
    pref = Counter()
    for s in labels:
        m = re.match(r"^([A-Za-z]+)", s)
        pref[m.group(1) if m else ""] += 1
    out["leading_alpha_prefix"] = dict(pref.most_common(30))
    return out


def read_csv_head(path, max_rows=25):
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        rows = list(csv.reader(io.StringIO(raw)))
    except Exception as e:
        return {"error": f"csv parse: {type(e).__name__}: {e}"}
    return {"n_rows": len(rows),
            "columns": rows[0] if rows else None,
            "head": rows[1:1 + max_rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    res["zips"] = ensure_zips()

    # ---------- ① 电极标签：全量 dump ----------
    lab_files = sorted(WORK.rglob("electrode_labels.json"))
    per_subject = OrderedDict()
    for p in lab_files:
        sid = p.parent.name
        try:
            d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            per_subject[sid] = {"error": f"{type(e).__name__}: {e}"}
            continue
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
        groups = Counter(strip_tail(s) for s in labels)
        per_subject[sid] = {
            "file": str(p.relative_to(WORK)),
            "n_labels": len(labels),
            "n_unique": len(set(labels)),
            "labels": labels,                       # 全量
            "n_groups_by_stripped_tail": len(groups),
            "groups_by_stripped_tail": groups.most_common(),
            "structure": discover_structure(labels),
        }
        log(f"[OK ] {sid}: {len(labels)} 标签 / {len(groups)} 组 ⇒ "
            f"{groups.most_common()[:8]}")
    res["per_subject"] = per_subject
    res["n_subjects"] = len(per_subject)
    res["label_groups_per_subject"] = {
        sid: v.get("n_groups_by_stripped_tail") for sid, v in per_subject.items()}

    # ---------- ② 坐标：rglob（不再写死路径） ----------
    loc_dir = WORK / "localization"
    csv_files = sorted(loc_dir.rglob("*.csv")) if loc_dir.exists() else []
    res["localization_files"] = [str(p.relative_to(WORK)) for p in csv_files]
    log(f"[..] localization 下 {len(csv_files)} 个 csv")
    coords = {}
    for p in csv_files:
        key = str(p.relative_to(WORK))
        if "depth" in p.name.lower() or "wm" in p.name.lower():
            coords[key] = read_csv_head(p, max_rows=12)
        else:
            coords[key] = read_csv_head(p, max_rows=8)
        log(f"[OK ] {key}: {coords[key].get('n_rows')} 行 ⇒ "
            f"{coords[key].get('columns')}")
    res["coords"] = coords

    # ---------- ③ trial 元数据 ----------
    md = sorted(WORK.rglob("*_metadata.json"))
    per_sub_trials = Counter()
    titles = {}
    for p in md:
        sid = p.name.split("_trial")[0]
        per_sub_trials[sid] += 1
        try:
            m = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            titles[p.name] = {"title": m.get("title"), "filename": m.get("filename")}
        except Exception:
            pass
    res["n_trials_per_subject"] = dict(per_sub_trials)
    res["n_trials_total"] = len(md)
    res["trial_titles"] = titles
    log(f"[OK ] trial {len(md)} 个 ⇒ {sorted(per_sub_trials.items())}")

    (OUT / "exp103c_btb_labels.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {OUT}/exp103c_btb_labels.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
