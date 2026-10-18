"""探测 THINGS-EEG 的 .pt 结构，确认试次维是否保留。

关键问题：train.pt 是
  A) (n_trial, 63, T)        → 试次维完整保留，可做四维数据集
  B) (63, n_trial * T)       → 试次已拼接，不可恢复
  C) dict with keys          → 需看具体字段

只知道总大小 2.1 GB 无法判断，必须读出来看。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

BUCKET = "hf://buckets/illusion-brain/training-data/eeg/things-eeg"


def describe(obj, name="root", depth=0, max_depth=3):
    """递归描述嵌套结构（dict / list / tensor）。"""
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict, {len(obj)} keys → {list(obj.keys())[:10]}")
        if depth < max_depth:
            for k, v in list(obj.items())[:8]:
                describe(v, str(k), depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}[{len(obj)}]")
        if len(obj) and depth < max_depth:
            describe(obj[0], f"{name}[0]", depth + 1, max_depth)
    elif hasattr(obj, "shape"):
        try:
            shp = tuple(obj.shape)
        except Exception:
            shp = "?"
        print(f"{pad}{name}: {type(obj).__name__} shape={shp} dtype={getattr(obj, 'dtype', '?')}")
    else:
        s = str(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {s[:80]}")


def main():
    import torch

    print("=" * 84)
    print("探测 THINGS-EEG .pt 结构（从挂载的桶读）")
    print("=" * 84)

    # 桶挂载在 /td，不是 dataset repo
    p = "/td/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt"
    target = p
    print(f"  路径 {p}")
    import os as _os
    if not _os.path.exists(p):
        print(f"  ❌ 不存在。挂载点内容:")
        for root, dirs, files in _os.walk("/td"):
            depth = root.count("/")
            if depth > 3:
                dirs[:] = []
                continue
            print(f"    {root}: {dirs[:5]} {files[:3]}")
            if depth > 2:
                break
        return
    print(f"  大小 {_os.path.getsize(p)/1024**2:.0f} MB")

    obj = torch.load(p, map_location="cpu", weights_only=False)
    print()
    print("=" * 84)
    print("结构")
    print("=" * 84)
    describe(obj, "test.pt", 0, 4)

    print()
    print("=" * 84)
    print("维度判定")
    print("=" * 84)

    report = {"file": target}

    def find_arrays(obj, path="root", out=None, depth=0):
        if out is None:
            out = []
        if depth > 4:
            return out
        if isinstance(obj, dict):
            for k, v in obj.items():
                find_arrays(v, f"{path}.{k}", out, depth + 1)
        elif isinstance(obj, (list, tuple)):
            out.append((f"{path}", "list", len(obj)))
            if len(obj):
                find_arrays(obj[0], f"{path}[0]", out, depth + 1)
        elif hasattr(obj, "shape"):
            out.append((f"{path}", tuple(obj.shape), str(getattr(obj, "dtype", "?"))))
        return out

    arrays = find_arrays(obj)
    for a in arrays[:20]:
        print(f"  {a[0]}: shape={a[1]} dtype={a[2]}")
    report["arrays"] = [{"path": a[0], "shape": str(a[1]), "dtype": a[2]}
                        for a in arrays[:20]]

    # 判断试次维
    verdict = "未知"
    for path, shape, _ in arrays:
        if isinstance(shape, tuple) and len(shape) == 3:
            # 找 (n_trial, ch, T) 或 (n_trial, T, ch)
            if 63 in shape or 64 in shape:
                n_trial = shape[0]
                verdict = (f"✅ 试次维完整: {path} shape={shape}, "
                           f"n_trial={n_trial}")
                break
    if verdict == "未知":
        for path, shape, _ in arrays:
            if isinstance(shape, tuple) and len(shape) == 2 and 63 in shape:
                verdict = (f"⚠️ 只有 2 维: {path} shape={shape}, "
                           f"试次可能已拼接")
                break
    print(f"\n  {verdict}")
    report["trial_dim_verdict"] = verdict

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out/exp30")
    out.mkdir(parents=True, exist_ok=True)
    (out / "things_eeg_probe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  报告 → {out}/things_eeg_probe.json")


if __name__ == "__main__":
    main()
