#!/usr/bin/env python3
"""exp103_btb_regions.py — 「乙」可行性的**判决性探针**：**Brain Treebank** 的脑区分布。

## 为什么是这一个

`BRAIN_STRUCTURE_SPEC.md` §4.2 第 2 条要「三个原语还是两个」⇒ 需要**同一 session 里有 ≥3 个可分区的带**
⇒ ⇒ 「乙」原计划是**自己采**（§三十五·五 五条）⇒ ⇒ **但公开数据里可能已经有一个满足的** ⇒ ⇒
先核它 ⇒ ⇒ **Brain Treebank**（human iEEG，arXiv 2411.08343，NeurIPS D&B 2024）：

- 官网 `https://braintreebank.dev/`：**直链下载**（无注册）+ 页面底部写明 **"Data released under a CC BY 4.0 license"**
- **10 个被试**、每被试平均 168 个电极、电影自然刺激 + 转写（UD 句法）+ 电极标注 + 电极空间坐标
- 下载清单里有 **26 个 trial**（sub_1:3 / sub_2:7 / sub_3:3 / sub_4:3 / sub_5:1 / sub_6:3 / sub_7:2 / sub_8:1 / sub_9:1 / sub_10:2）

⇒ ⇒ 本脚本**只下三个小 zip**（`electrode_labels` / `localization` / `subject_metadata`）
⇒ ⇒ **数清楚：每个被试、每个 trial 里，电极落在几个脑区、能不能分出 3 个非重叠的带** ⇒ ⇒
⇒ ⇒ **这决定「乙」要不要自己采。**

⚠️ **本脚本不下载神经数据本体**（那 130 GB 在 `subject_data/`）⇒ ⇒ 只用元数据
⇒ ⇒ ⇒ 所以它**便宜且可审计**；真要跑分析时再按需取子集。

## 输出

`/domains/runs/illusionbrain-align/exp103/exp103_btb_regions.json`
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

BASE = "https://braintreebank.dev/data/"
OUT = Path("/domains/runs/illusionbrain-align/exp103")
WORK = Path("/domains/btb")


def log(*a):
    print(*a, flush=True)


def fetch(url, dest):
    import urllib.request
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    res = {"steps": []}

    # ① 三个小 zip
    for name in ("electrode_labels.zip", "localization.zip", "subject_metadata.zip"):
        p = WORK / name
        try:
            if not p.exists():
                n = fetch(BASE + name, p)
            else:
                n = p.stat().st_size
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                z.extractall(WORK / name.replace(".zip", ""))
            res["steps"].append({"file": name, "bytes": n, "members": names[:20],
                                "n_members": len(names)})
            log(f"[OK ] {name}: {n} bytes, {len(names)} 个成员 ⇒ {names[:8]}")
        except Exception as e:
            res["steps"].append({"file": name, "error": f"{type(e).__name__}: {e}"})
            log(f"[FAIL] {name}: {type(e).__name__}: {e}")

    # ② 电极标注：每个电极的语义 ID（脑区/皮层叶/回等）
    lab_files = sorted((WORK / "electrode_labels").rglob("*")) if (WORK / "electrode_labels").exists() else []
    res["label_files"] = [str(p.relative_to(WORK)) for p in lab_files if p.is_file()][:40]
    log(f"[..] 标注文件: {res['label_files'][:10]}")

    # ③ 坐标：每个电极的 x/y/z
    loc_files = sorted((WORK / "localization").rglob("*")) if (WORK / "localization").exists() else []
    res["loc_files"] = [str(p.relative_to(WORK)) for p in loc_files if p.is_file()][:40]
    log(f"[..] 坐标文件: {res['loc_files'][:10]}")

    # ④ 元数据：每被试的 trial 数
    md_files = sorted((WORK / "subject_metadata").rglob("*")) if (WORK / "subject_metadata").exists() else []
    res["meta_files"] = [str(p.relative_to(WORK)) for p in md_files if p.is_file()][:40]
    log(f"[..] 元数据文件: {res['meta_files'][:10]}")

    # ⑤ 读标注与坐标 ⇒ 每被试的脑区数（**判决性**）
    per_subject = {}
    try:
        import h5py
        for lf in [p for p in lab_files if p.suffix in (".h5", ".hdf5")]:
            with h5py.File(lf, "r") as h:
                def walk(g, prefix=""):
                    for k, v in g.items():
                        if isinstance(v, h5py.Group):
                            yield from walk(v, prefix + k + "/")
                        else:
                            yield prefix + k, v
                items = dict(walk(h))
                sid = lf.stem
                # 找语义 ID 与坐标
                sem = None
                for k, v in items.items():
                    if "label" in k.lower() and v.dtype.kind in "SU":
                        sem = v
                        break
                if sem is None:
                    continue
                sem = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in sem[:]])
                uniq = {}
                for s in sem:
                    uniq[s] = uniq.get(s, 0) + 1
                per_subject[sid] = {"n_electrodes": int(len(sem)),
                                    "n_unique_labels": len(uniq),
                                    "top_labels": sorted(uniq.items(), key=lambda kv: -kv[1])[:20]}
                log(f"[OK ] {sid}: {len(sem)} 个电极，{len(uniq)} 个不同标签")
    except Exception as e:
        res["h5_error"] = f"{type(e).__name__}: {e}"
        log(f"[FAIL] 读标注 h5: {type(e).__name__}: {e}")

    res["per_subject"] = per_subject
    (out / "exp103_btb_regions.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("\n=== 脑区分布 ===")
    for sid, v in sorted(per_subject.items()):
        log(f"  {sid}: {v['n_electrodes']} 电极 / {v['n_unique_labels']} 标签 ⇒ {v['top_labels'][:8]}")
    log(f"结果写入 {out}/exp103_btb_regions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
