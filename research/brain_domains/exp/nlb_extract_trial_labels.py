#!/usr/bin/env python3
"""nlb_extract_trial_labels.py — 从 NWB 重建**带 trial 对齐**的 NLB 视图（云端跑）。

## 为什么必须重建（而不是"补标签"）

我们在桶里的 `macaque_nlb/spikes_*.npz` **每个只有一个键 `spikes`**（n_units × n_bins）
⇒ **段内没有任何时间/试次信息** ⇒ 行为标签**贴不到现有 67 段上**。
⇒ 所以这里**从 NWB 重新分箱**（spike_times → 20 ms bin，session 绝对时间轴），
并把 trial 表与**同轴**的计数一起写出去 ⇒ 对齐天然正确。

## 数据来源

| 数据集 | 来源 | 说明 |
|---|---|---|
| `churchland`（猕猴 M1/PMd）| DANDI `dandi:000070`（10 个 session，我们用其中 6 个）| 伸手方向标签：8 类（官方 NLB `trial_reachDir` 的口径）|
| `dmfc-rsg`（猕猴 DMFC）| **桶内已有** `/domains/primate/dmfc_rsg/*.nwb`（零下载）| 决策时程标签：`go_offset = go_time − start_time` |

## 伸手方向怎么算（官方 NLB 口径）

在每个 trial 的**运动窗** `[move_begins_time, move_ends_time]` 内取光标轨迹的**净位移**
⇒ `atan2(dy, dx)` ⇒ 量化为 **8 个方向**（每 45°）。
交叉核对：用 `hit_target_position − 运动起点` 的方向 ⇒ 两者应高度一致（**自检**）。
不可用（窗口内光标点 < 2）⇒ `reach_dir = -1`。

## 用法（云端）

    python -u /domains/scripts/nlb_extract_trial_labels.py --out /domains/primate/nlb_trials

逐个 session 处理：**下载 → 抽 → 写 → 删 NWB**（峰值磁盘 = 单个 session 大小）
⇒ 已完成的 session 会被跳过 ⇒ **可断点续跑**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

DANDI_ID = "000070"
API = f"https://api.dandiarchive.org/api/dandisets/{DANDI_ID}/versions/draft/assets/?page_size=100"
BUCKET_DMFC = Path("/domains/primate/dmfc_rsg")
BIN_MS = 20
N_DIR = 8                       # 伸手方向 8 类
EPS = 1e-9


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 伸手方向
# ---------------------------------------------------------------------------

def direction8(dx: float, dy: float) -> int:
    """角度 → 8 个方向（0 = +x，逆时针）。"""
    import math
    ang = math.atan2(float(dy), float(dx))
    return int(round((ang + math.pi) / (2 * math.pi) * N_DIR)) % N_DIR


def reach_dirs_from_cursor(cur_data, cur_ts, move_begins, move_ends):
    """在运动窗内取光标净位移 ⇒ 8 类方向；不可用 ⇒ −1。"""
    out = np.full(len(move_begins), -1, dtype=np.int8)
    t = np.asarray(cur_ts, dtype=np.float64)
    d = np.asarray(cur_data, dtype=np.float64)
    for i, (mb, me) in enumerate(zip(move_begins, move_ends)):
        if not np.isfinite(mb) or not np.isfinite(me) or me <= mb:
            continue
        sel = (t >= mb) & (t <= me)
        if sel.sum() < 2:
            continue
        p = d[sel]
        dx = p[-1, 0] - p[0, 0]
        dy = p[-1, 1] - p[0, 1]
        if (dx * dx + dy * dy) < EPS:
            continue
        out[i] = direction8(dx, dy)
    return out


# ---------------------------------------------------------------------------
# 单个 session
# ---------------------------------------------------------------------------

def bin_spikes(spike_times, spike_times_index, duration, bin_ms=BIN_MS):
    """spike_times（ragged）→ (n_units, n_bins) 的 20 ms 计数矩阵。"""
    n_units = len(spike_times_index) - 1
    n_bins = int(np.ceil(duration / (bin_ms / 1000.0)))
    mat = np.zeros((n_units, n_bins), dtype=np.float16)
    st = np.asarray(spike_times, dtype=np.float64)
    si = np.asarray(spike_times_index, dtype=np.int64)
    for u in range(n_units):
        s = st[si[u]:si[u + 1]]
        if s.size == 0:
            continue
        b = np.floor(s / (bin_ms / 1000.0)).astype(np.int64)
        b = b[(b >= 0) & (b < n_bins)]
        if b.size:
            mat[u] = np.bincount(b, minlength=n_bins).astype(np.float16)
    return mat


def process_nwb(path: Path, kind: str):
    """打开一个 NWB ⇒ 抽出 (counts, trial 表, 标签)。"""
    import h5py
    f = h5py.File(str(path), "r")
    out = {}
    try:
        # ---- 单元与 20 ms 计数 ----
        u = f["units"]
        st = u["spike_times"][:]
        si = u["spike_times_index"][:]
        tri = f["intervals"]["trials"]
        n_tri = len(tri["id"])
        start = np.asarray(tri["start_time"], dtype=np.float64)
        stop = np.asarray(tri["stop_time"], dtype=np.float64)
        dur = float(np.nanmax(stop)) if n_tri else 0.0
        out["counts"] = bin_spikes(st, si, dur)
        out["trial_start"] = start
        out["trial_stop"] = stop
        out["n_units"] = int(len(si) - 1)
        # 单元位置（若有）
        try:
            out["electrode_location"] = np.asarray(
                [x.decode() if isinstance(x, bytes) else str(x)
                 for x in u["electrodes"]["location"][:]])
        except Exception:
            out["electrode_location"] = np.array([], dtype=object)

        if kind == "churchland":
            mb = np.asarray(tri["move_begins_time"], dtype=np.float64)
            me = np.asarray(tri["move_ends_time"], dtype=np.float64)
            out["move_begins"] = mb
            out["move_ends"] = me
            out["task_success"] = np.asarray(tri["task_success"], dtype=np.int8)
            out["correct_reach"] = np.asarray(tri["correct_reach"], dtype=np.int8)
            out["discard_trial"] = np.asarray(tri["discard_trial"], dtype=np.int8)
            out["trial_type"] = np.asarray(tri["trial_type"], dtype=np.int16)
            out["maze_condition"] = np.asarray(tri["maze_condition"], dtype=np.int16)
            out["reaction_time"] = np.asarray(tri["reaction_time"], dtype=np.float64)
            # 目标位置（ragged → 命中目标用 hit_target_position）
            out["hit_target_position"] = np.asarray(
                tri["hit_target_position"], dtype=np.float64)
            # 伸手方向：光标净位移
            cg = f["processing"]["behavior"]["Position"]["Cursor"]
            cdata = cg["data"][:]
            cts = cg["timestamps"][:]
            rd = reach_dirs_from_cursor(cdata, cts, mb, me)
            out["reach_dir"] = rd
            # 交叉核对：命中目标方向 vs 光标方向
            start_pt = np.full((n_tri, 2), np.nan)
            t = np.asarray(cts, dtype=np.float64)
            d = np.asarray(cdata, dtype=np.float64)
            for i, mb_i in enumerate(mb):
                if not np.isfinite(mb_i):
                    continue
                sel = t >= mb_i
                if sel.any():
                    start_pt[i] = d[sel][0]
            tgt = out["hit_target_position"]
            agree = np.zeros(n_tri, dtype=np.int8)
            for i in range(n_tri):
                if rd[i] < 0 or not np.isfinite(start_pt[i]).all():
                    continue
                dx = tgt[i, 0] - start_pt[i, 0]
                dy = tgt[i, 1] - start_pt[i, 1]
                if (dx * dx + dy * dy) < EPS:
                    continue
                agree[i] = int(direction8(dx, dy) == rd[i])
            out["dir_agree"] = agree
        else:   # dmfc-rsg
            go = np.asarray(tri["go_time"], dtype=np.float64)
            out["go_time"] = go
            out["go_offset"] = go - start
            out["split"] = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                                      for x in tri["split"][:]], dtype=object)
            out["heldout"] = np.asarray(f["units"]["heldout"], dtype=bool) \
                if "heldout" in f["units"] else np.zeros(0, dtype=bool)
    finally:
        f.close()
    return out


def write_npz(path: Path, d: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in d.items()})


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/domains/primate/nlb_trials")
    ap.add_argument("--datasets", default="churchland,dmfc")
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/nwb_work")
    tmp.mkdir(parents=True, exist_ok=True)

    # 需要的 session（与我们在 segments.parquet 里用到的对齐）
    want = {
        "churchland": [
            "sub-Jenkins_ses-20090912_behavior+ecephys",
            "sub-Jenkins_ses-20090916_behavior+ecephys",
            "sub-Jenkins_ses-20090918_behavior+ecephys",
            "sub-Jenkins_ses-20090923_behavior+ecephys",
            "sub-Nitschke_ses-20090812_behavior+ecephys",
            "sub-Nitschke_ses-20090819_behavior+ecephys",
        ],
        "dmfc-rsg": ["sub-Haydn_desc-test_ecephys", "sub-Haydn_desc-train_ecephys"],
    }

    assets = []
    try:
        d = json.load(urllib.request.urlopen(API, timeout=90))
        assets = [(a["asset_id"], a["path"], int(a.get("size") or 0))
                  for a in d["results"]]
        log(f"DANDI {DANDI_ID}: {len(assets)} 个资产")
    except Exception as e:
        log(f"✗ DANDI 查询失败：{type(e).__name__} {e}")

    summary = []
    for kind in args.datasets.split(","):
        kind = kind.strip()
        for sess in want.get(kind, []):
            dst = out_dir / f"{kind}__{sess}.npz"
            if dst.exists():
                log(f"↷ 跳过（已存在）{dst.name}")
                continue
            nwb = None
            try:
                if kind == "dmfc-rsg":
                    nwb = BUCKET_DMFC / f"{sess}.nwb"
                    if not nwb.exists():
                        log(f"✗ 桶内没有 {nwb}")
                        continue
                    log(f"\n=== {sess}（桶内 NWB，零下载）===")
                else:
                    hit = [a for a in assets if sess in a[1]]
                    if not hit:
                        log(f"✗ DANDI 里找不到 {sess}")
                        continue
                    aid, path, size = hit[0]
                    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDI_ID}"
                           f"/versions/draft/assets/{aid}/download/")
                    nwb = tmp / f"{sess}.nwb"
                    t0 = time.time()
                    urllib.request.urlretrieve(url, str(nwb))
                    log(f"\n=== {sess}：下载 {size/1e9:.2f} GB 用 {time.time()-t0:.0f}s ===")
                t1 = time.time()
                d = process_nwb(Path(nwb), "churchland" if kind == "churchland" else "dmfc")
                d["session"] = sess
                d["dataset"] = kind
                write_npz(dst, d)
                extra = ""
                if "reach_dir" in d:
                    rd = d["reach_dir"]
                    ag = d["dir_agree"]
                    valid = rd >= 0
                    extra = (f" | 方向可用 {int(valid.sum())}/{len(rd)}"
                             f" | 目标方向一致率 {ag[valid].mean() if valid.any() else float('nan'):.3f}")
                log(f"  ✓ {dst.name}: units={d['n_units']} bins={d['counts'].shape[1]} "
                    f"trials={len(d['trial_start'])} 用时 {time.time()-t1:.0f}s{extra}")
                summary.append({"dataset": kind, "session": sess,
                                "file": dst.name,
                                "n_units": int(d["n_units"]),
                                "n_bins": int(d["counts"].shape[1]),
                                "n_trials": int(len(d["trial_start"]))})
            except Exception as e:
                import traceback
                log(f"  ✗ {sess}: {type(e).__name__} {e}")
                log(traceback.format_exc()[-600:])
            finally:
                if nwb is not None and kind != "dmfc-rsg":
                    try:
                        os.remove(nwb)
                    except Exception:
                        pass

    (out_dir / "extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n完成 {len(summary)} 个 session ⇒ {out_dir}")


if __name__ == "__main__":
    main()
