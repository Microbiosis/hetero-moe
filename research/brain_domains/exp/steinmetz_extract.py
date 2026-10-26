#!/usr/bin/env python3
"""steinmetz_extract.py — 从 DANDI:000017（Steinmetz 2019）逐 session 抽
**带 trial 对齐 + 逐单元脑区标签**的视图（云端跑）。

## 为什么必须抽

我们桶里 `/domains/steinmetz/`（Neuromatch 版）有 39 个 `spikes_session_*.npz`（10 ms 分箱）
+ `units.parquet`（逐单元 `region` / `functional_domain`）⇒ **但缺 trial 表**（行为标签）。
⇒ 而 DANDI 的 NWB 里有 `intervals/trials`：`response_choice` /
`visual_stimulus_left_contrast` / `visual_stimulus_right_contrast` / `included` /
`visual_stimulus_time` / `response_time` / `feedback_type`。
⇒ ⇒ 行为标签只能从 DANDI 来 ⇒ 那就**整个从 DANDI 抽**（脑区标签也能从 NWB 自己拿到，见下）。

## 脑区标签从哪来（**踩过的坑，写在这里**）

DANDI 的 NWB **没有** `units.brain_region` 列 ⇒ v1 试过用**时间序列对齐**把 DANDI 单元
对到桶里 Neuromatch 的单元（FFT 互相关 + 逐单元相关）⇒ **失败**：桶里那份是原始数据的
**子集**（总放电 255 万 vs NWB 1328 万、单元 771 vs 2072）⇒ 时间轴对齐根本不成立。
v2 又试过 `units.parquet` 的 **`orig_unit_idx` 直连** ⇒ **也失败**（匹配率 0.02）。

⇒ ⇒ **正解在 NWB 自己身上**：`units.electrodes`（逐 spike 的电极号）+ `units.electrodes_index`
（逐单元的偏移）⇒ **单元 → 电极**；再查
`general/extracellular_ephys/electrodes/location`（标准解剖标签，如 `MOp` / `VISp` / `PL`）
⇒ **单元 → 脑区**。⇒ 零猜测、零外部对齐。

## 输出（**只存 trial 窗口特征，不存整段分箱** ⇒ 体积小两个数量级）

`/domains/steinmetz_trials/<session>.npz`：
- `feat` (n_trials × n_units) float32 —— 每 trial 在**固定窗**内的每单元平均放电率；
- `trial_*`：`stim_time` / `resp_time` / `choice`（±1）/ `stim_side`（±1）/ `included` / `reward`；
- `unit_region`（逐单元，object）；`unit_total`（逐单元总放电数，诊断用）；
- `diag_*`：单元数、带标签单元数、各带单元数、trial 数。

## 用法（云端）

    python -u /domains/scripts/steinmetz_extract.py --out /domains/steinmetz_trials
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

DANDI = "000017"
VERSION = "0.240329.1926"
API = (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
       f"/assets/?page_size=100&fields=asset_id,path,size")
BIN_S = 0.010
WIN_START = 0.05            # trial 窗起点（刺激后 50 ms）
WIN_LEN = 0.40              # trial 窗长（400 ms）

# 位置轴三带（**按解剖区名**，Allen CCF 标签；不用 Neuromatch 的聚类 ⇒ 不自造分区标准）
BAND_RULES = {
    "visual": ("VIS",),                              # 感觉/视觉侧 ≈ POST
    "motor": ("MO",),                                # 运动侧 ≈ CENT
    "prefrontal": ("PL", "ACA", "ILA", "ORB"),        # 执行侧 ≈ FRONT
}


def log(*a):
    print(*a, flush=True)


def dandi_assets():
    with urllib.request.urlopen(API, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    return [{"asset_id": a["asset_id"], "path": a["path"], "size": a["size"]}
            for a in data.get("results", [])]


def asset_url(asset_id):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
            f"/assets/{asset_id}/download/")


def dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def band_of(region):
    for b, prefs in BAND_RULES.items():
        if any(region == p or region.startswith(p) for p in prefs):
            return b
    return ""


def bin_spikes(st, si, n_bins):
    n_units = len(si) - 1
    mat = np.zeros((n_units, n_bins), dtype=np.float32)
    for u in range(n_units):
        s = st[si[u]:si[u + 1]]
        if s.size == 0:
            continue
        b = np.floor(s / BIN_S).astype(np.int64)
        b = b[(b >= 0) & (b < n_bins)]
        if b.size:
            mat[u] = np.bincount(b, minlength=n_bins)[:n_bins]
    return mat


def extract_one(asset, work_dir, out_dir):
    import h5py
    t0 = time.time()
    nwb = work_dir / "s.nwb"
    log(f"  ↓ {asset['path']}（{asset['size']/1e6:.0f} MB）")
    urllib.request.urlretrieve(asset_url(asset["asset_id"]), nwb)
    dl = time.time() - t0
    try:
        with h5py.File(str(nwb), "r") as f:
            u = f["units"]
            st = np.asarray(u["spike_times"][:], dtype=np.float64)
            si = np.asarray(u["spike_times_index"][:], dtype=np.int64)
            # ---- 单元 → 电极 → 脑区 ----
            eph = f["general"]["extracellular_ephys"]["electrodes"]
            loc = np.array([dec(x) for x in eph["location"][:]], dtype=object)
            ei = np.asarray(u["electrodes"][:], dtype=np.int64)
            eidx = np.asarray(u["electrodes_index"][:], dtype=np.int64)
            n_units = len(eidx) - 1
            unit_el = np.full(n_units, -1, dtype=np.int64)
            for i in range(n_units):
                seg = ei[eidx[i]:eidx[i + 1]]
                if len(seg):
                    unit_el[i] = collections.Counter(seg.tolist()).most_common(1)[0][0]
            region = np.array([dec(loc[e]) if 0 <= e < len(loc) else "" for e in unit_el],
                              dtype=object)
            band = np.array([band_of(str(r)) for r in region], dtype=object)
            # ---- trial 表 ----
            t = f["intervals"]["trials"]
            stop = np.asarray(t["stop_time"][:], dtype=np.float64)
            stim = np.asarray(t["visual_stimulus_time"][:], dtype=np.float64)
            resp = np.asarray(t["response_time"][:], dtype=np.float64)
            choice = np.asarray(t["response_choice"][:], dtype=np.float64)
            lc = np.asarray(t["visual_stimulus_left_contrast"][:], dtype=np.float64)
            rc = np.asarray(t["visual_stimulus_right_contrast"][:], dtype=np.float64)
            inc = np.asarray(t["included"][:], dtype=bool)
            fb = np.asarray(t["feedback_type"][:], dtype=np.float64)
        dur = float(np.nanmax(stop))
        n_bins = int(np.ceil(dur / BIN_S))
        counts = bin_spikes(st, si, n_bins)
        side = np.where(np.abs(lc) > np.abs(rc), 1.0, -1.0)
        # ---- trial 窗口特征 ----
        a = np.floor((stim + WIN_START) / BIN_S).astype(np.int64)
        b = np.floor((stim + WIN_START + WIN_LEN) / BIN_S).astype(np.int64)
        feat = np.zeros((len(stim), n_units), dtype=np.float32)
        ok = np.isfinite(a) & np.isfinite(b) & (b > a) & (b <= n_bins)
        for i in np.flatnonzero(ok):
            feat[i] = counts[:, a[i]:b[i]].mean(axis=1)
        unit_total = np.asarray([si[i + 1] - si[i] for i in range(n_units)], dtype=np.float64)
        band_counts = {k: int((band == k).sum()) for k in BAND_RULES}
        log(f"  ✓ 单元 {n_units}（带标签 {int((region != '').sum())}）；"
            f"trial {len(stim)}（included {int(inc.sum())}）；三带 {band_counts}")
        if min(band_counts.values()) < 20:
            log("  ✗ 某个带 < 20 单元 ⇒ 该 session 不满足 exp91 的前提 ⇒ 仍写盘（由 exp91 决定跳过）")
        out = {
            "session": asset["path"],
            "feat": feat,
            "trial_stim_time": stim,
            "trial_resp_time": resp,
            "trial_choice": choice,
            "trial_stim_side": side,
            "trial_included": inc,
            "trial_reward": fb,
            "unit_region": region,
            "unit_band": band,
            "unit_total": unit_total,
            "diag_n_units": np.int32(n_units),
            "diag_n_labelled": np.int32(int((region != "").sum())),
            "diag_n_trials": np.int32(len(stim)),
            "diag_n_included": np.int32(int(inc.sum())),
            "win_start": np.float32(WIN_START),
            "win_len": np.float32(WIN_LEN),
            "bin_s": np.float32(BIN_S),
        }
        p = out_dir / (Path(asset["path"]).stem + ".npz")
        np.savez_compressed(p, **out)
        log(f"  → {p.name}（{p.stat().st_size/1e6:.1f} MB；下载 {dl:.0f}s，总 {time.time()-t0:.0f}s）")
        return {"path": asset["path"], "n_units": int(n_units), "n_trials": int(len(stim)),
                "n_included": int(inc.sum()), "bands": band_counts,
                "regions": {r: int((region == r).sum()) for r in sorted(set(region.tolist()))}}
    finally:
        try:
            nwb.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/domains/steinmetz_trials")
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--work-dir", default="/tmp")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    assets = dandi_assets()
    log(f"DANDI {DANDI}@{VERSION}：{len(assets)} 个 session")
    if args.limit_sessions:
        assets = assets[: args.limit_sessions]

    recs = []
    for a in assets:
        stem = Path(a["path"]).stem
        if (out_dir / (stem + ".npz")).exists():
            log(f"· {stem} 已存在 ⇒ 跳过")
            continue
        try:
            r = extract_one(a, work_dir, out_dir)
        except Exception as e:
            log(f"  ✗ {stem} 失败：{type(e).__name__}: {e}")
            r = None
        if r:
            recs.append(r)
    if recs:
        mp = out_dir / "_manifest.json"
        mp.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"清单写入 {mp}")
    log(f"完成：本次抽了 {len(recs)} 个 session")


if __name__ == "__main__":
    main()
