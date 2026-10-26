#!/usr/bin/env python3
"""exp98_extract.py — 从 Allen VC-NP 的 **session 级 NWB** 抽 exp98 要的逐 trial 特征。

## 抽什么（**协议由 exp98_posneg_control.py 的预登记定死**，本脚本只负责取数）

- **窗口**：每次自然图像呈现后的 **[0, 200] ms**（呈现本身 250 ms ⇒ 这是能取到的最长窗；
  实测最短呈现 0.2155 s > 0.2 s ⇒ 窗口始终落在呈现内）。
- **特征**：每单元在该窗内的**放电率**（spikes/s）。
- **类别**：`intervals/natural_scenes_presentations/frame`（`stimulus_name` 全同、`stimulus_index`
  恒为块号 ⇒ **都不可用**）。`frame = −1` 是**空白屏** ⇒ **剔除** ⇒ K = **118 张图像**，
  重标为连续 0..K−1（机会水平 = 1/K）。
- **单元→脑区**：`units/peak_channel_id` → `general/extracellular_ephys/electrodes/location`。
- **只保留** `BANDS` 里的脑区（VISp/VISl/VISrl/VISam/VISal/VISpm/CA1）⇒ 其余（LP/APN/TH/DG/SUB…）不读。
- **不做质量过滤**（不按 Allen 的 `quality` 标签筛 ⇒ 避免事后挑数据）。

## 怎么读（**这是本脚本的工程要点**）

- `units/spike_times` 是**连续 1-D float64**（该 session 约 1.31 亿个 spike = 1.05 GB），
  `units/spike_times_index`（2779）是每个单元的**偏移** ⇒ ⇒ **按需切片**即可，不必整表载入。
- 用 h5py + fsspec(HTTP) 的**范围读**；**读取必须在 `with fs.open(...)` 内部**（否则 h5py 报
  `ValueError: I/O operation on closed file`）。
- 连续保留单元合并成 **run**（间隙 ≤ 16 个单元），每个 run 一次连续读 ⇒ 请求数少、顺序读。

## 输出

- `/domains/allen_trials/<session>.npz`：`feat`(float32, 试次×单元) / `unit_area` / `trial_label` /
  `trial_time` / `session` / 元数据（K、窗、每区单元数、空白剔除数…）。
- stdout：`===X===` 每 session 一行摘要（供日志核对）。

## 用法

```powershell
python -u exp98_extract.py --shard 0 --nshards 4 --out /domains/allen_trials
python -u exp98_extract.py --sessions sub-699733573,sub-703279277 --out /domains/allen_trials
```
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import urllib.request

DANDI = "000021"
VER = "0.251116.2246"
BASE = f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}"
BANDS = ("VISp", "VISl", "VISrl", "VISam", "VISal", "VISpm", "CA1")
WIN = 0.2          # 窗 = 呈现后 [0, 200] ms（预登记）
BLANK_FRAME = -1.0  # 空白屏（剔除）


def ensure_deps():
    for mod, pkg in (("h5py", "h5py"), ("fsspec", "fsspec"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except Exception:
            print(f"installing {pkg} ...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "--no-cache-dir", pkg], check=True)


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def all_assets():
    out, url = [], f"{BASE}/assets/?page_size=100&fields=asset_id,path,size"
    while url:
        d = get_json(url)
        out += d.get("results", [])
        url = d.get("next")
    return out


def dec(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def make_runs(kept, gap=16, cap=256):
    """把保留的单元下标切成连续 run（间隙 ≤ gap 的合并；每 run ≤ cap 个单元）。"""
    runs, cur = [], []
    for i in kept:
        if cur and (i - cur[-1] > gap or len(cur) >= cap):
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return runs


def extract_session(asset_id, session, out_dir):
    import fsspec
    import h5py
    import numpy as np
    url = f"{BASE}/assets/{asset_id}/download/"
    fs = fsspec.filesystem("http")
    out_dir.mkdir(parents=True, exist_ok=True)
    with fs.open(url, "rb", block_size=1 << 20, cache_type="readahead") as f:
        with h5py.File(f, "r") as h:
            el = h["general/extracellular_ephys/electrodes"]
            eid = el["id"][:]
            loc = [dec(x) for x in el["location"][:]]
            e2loc = {int(i): l for i, l in zip(eid, loc)}
            units = h["units"]
            peak = units["peak_channel_id"][:]
            sidx = units["spike_times_index"][:]
            st_ds = units["spike_times"]
            n_spikes = int(st_ds.shape[0])
            ns = h["intervals/natural_scenes_presentations"]
            t_start = ns["start_time"][:]
            t_stop = ns["stop_time"][:]
            t_frame = ns["frame"][:]
            # ---- trial 表：剔空白屏、按开始时间排序、frame 重标 0..K-1
            keep = np.array([i for i in range(len(t_start))
                             if float(t_frame[i]) != BLANK_FRAME], dtype=np.int64)
            order = keep[np.argsort(t_start[keep])]
            frames = t_frame[order].astype(np.int64)
            uniq = np.unique(frames)
            lab = np.searchsorted(uniq, frames).astype(np.int64)
            K = int(len(uniq))
            starts = t_start[order].astype(np.float64)
            stops = t_stop[order].astype(np.float64)
            # ---- 单元 → 脑区（只留 BANDS）
            areas, keep_u = [], []
            for ui, p in enumerate(peak):
                try:
                    a = e2loc.get(int(p))
                except Exception:
                    a = None
                if a in BANDS:
                    areas.append(a)
                    keep_u.append(ui)
            n_units = len(keep_u)
            feat = np.zeros((len(order), n_units), dtype=np.float32)
            bounds = list(int(x) for x in sidx) + [n_spikes]
            t_end = starts + WIN
            # ---- 分 run 顺序读 spike_times ⇒ 逐单元数窗内 spike
            col = 0
            for run in make_runs(keep_u):
                lo = bounds[run[0]]
                hi = bounds[run[-1] + 1]
                chunk = st_ds[lo:hi]
                for ui in run:
                    s = chunk[bounds[ui] - lo: bounds[ui + 1] - lo]
                    if len(s):
                        cnt = (np.searchsorted(s, t_end, side="left")
                               - np.searchsorted(s, starts, side="left"))
                    else:
                        cnt = np.zeros(len(starts), dtype=np.int64)
                    feat[:, col] = cnt / WIN
                    col += 1
                del chunk
            # ---- 落盘
            np.savez_compressed(
                out_dir / f"{session}.npz",
                feat=feat,
                unit_area=np.array(areas, dtype=object),
                trial_label=lab,
                trial_time=starts,
                session=np.array(session),
                meta=np.array(json.dumps({
                    "K": K, "win_s": WIN, "n_trials": int(len(order)),
                    "n_units": int(n_units),
                    "units_by_area": {a: int(areas.count(a)) for a in BANDS
                                      if areas.count(a)},
                    "blank_dropped": int(len(t_start) - len(order)),
                    "dur_min": float(np.min(stops - starts)),
                    "dur_max": float(np.max(stops - starts)),
                    "frame_min": float(np.min(frames)), "frame_max": float(np.max(frames)),
                    "n_spikes_total": n_spikes,
                }, ensure_ascii=False)))
    return {"session": session, "K": K, "n_trials": int(len(order)), "n_units": int(n_units),
            "units_by_area": {a: int(areas.count(a)) for a in BANDS if areas.count(a)},
            "blank_dropped": int(len(t_start) - len(order))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="/domains/allen_trials")
    args = ap.parse_args()
    ensure_deps()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = all_assets()
    sess_assets = {}
    for a in assets:
        if a["path"].endswith(".nwb") and "_probe-" not in a["path"]:
            sess_assets[a["path"].split("/")[0]] = a["asset_id"]
    names = sorted(sess_assets)
    if args.sessions:
        want = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        want = [n for i, n in enumerate(names) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards} ⇒ {len(want)} sessions", flush=True)
    for n in want:
        if n not in sess_assets:
            print(f"===X=== skip {n}（没这个 session）", flush=True)
            continue
        try:
            rec = extract_session(sess_assets[n], n, out_dir)
            err = None
        except Exception as e:
            rec, err = {"session": n}, f"{type(e).__name__}: {e}"
        rec["err"] = err
        print("===X===" + json.dumps(rec, ensure_ascii=False), flush=True)
    print("===DONE===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
