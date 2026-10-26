#!/usr/bin/env python3
"""steinmetz_align_probe.py — 诊断「DANDI 单元 ↔ 桶里 Neuromatch 单元」的对齐。

## 为什么需要这个（v1 → v2 的转折）

v1 假设「桶里 Neuromatch 的 `spikes` 是 10 ms 分箱、且与 NWB 同一起点」⇒ 实测 session 级
剖面相关只有 **0.01–0.02**。v1 的诊断（`steinmetz_align_probe.py` 旧版）给出了真正的原因：

| | DANDI NWB（Forssmann 20171101）| 桶里 Neuromatch（最接近的那个）|
|---|---|---|
| 总放电数 | **13,285,502** | **2,555,630**（≈1/5）|
| 单元数 | **2072** | 771 |
| 时长 | 1195.8 s | 67738 bin（若 10 ms ⇒ 677 s）|

⇒ **桶里的 npz 是原始数据的一个子集**（不是同一段记录的同一批单元）⇒
**时间序列对齐（FFT 互相关、逐单元相关）在这两者之间根本不成立** ⇒ v1 那条路是死的。

## v2 的思路：不碰时间轴，用**结构化直连**

桶里的 `units.parquet` 有 **`orig_unit_idx`**（= 该单元在**原始数据**里的序号）⇒
若它恰好对应 DANDI NWB 的单元下标 ⇒ **单元 → 脑区标签** 就是**查表**，零时间对齐、零猜测。

⇒ **判据（可验证）**：对每个 NM 单元 j，取 `i = orig_unit_idx[j]`，比较
**「NM 的该单元总放电数」与「DANDI 的第 i 个单元总放电数」** ⇒ 同一份数据 ⇒ 应当**几乎相等**。
⇒ 于是：对每个 NM session 算「匹配率」= 相对差 < 5% 的单元占比
⇒ **正确的那个 session 应当匹配率 ≈ 1.0**，其余接近 0 ⇒ 这就是**可证伪的 session 配对判据**。

## 用法（云端）

    python -u /domains/scripts/steinmetz_align_probe.py --limit-nm 39
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

DANDI = "000017"
VERSION = "0.240329.1926"
API = (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
       f"/assets/?page_size=100&fields=asset_id,path,size")
NM_DIR = Path("/domains/steinmetz")
EPS = 1e-9


def log(*a):
    print(*a, flush=True)


def dandi_assets():
    with urllib.request.urlopen(API, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    return [{"asset_id": a["asset_id"], "path": a["path"], "size": a["size"]}
            for a in data.get("results", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-nm", type=int, default=39)
    ap.add_argument("--session-index", type=int, default=0)
    ap.add_argument("--work-dir", default="/tmp")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    import h5py
    import pandas as pd

    assets = dandi_assets()
    a = assets[args.session_index]
    log(f"目标 session：{a['path']}（{a['size']/1e6:.0f} MB）")
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
           f"/assets/{a['asset_id']}/download/")
    nwb = Path(args.work_dir) / "s.nwb"
    t0 = time.time()
    urllib.request.urlretrieve(url, nwb)
    log(f"下载完 {nwb.stat().st_size/1e6:.0f} MB（{time.time()-t0:.0f}s）")

    with h5py.File(str(nwb), "r") as f:
        u = f["units"]
        st = np.asarray(u["spike_times"][:], dtype=np.float64)
        si = np.asarray(u["spike_times_index"][:], dtype=np.int64)
        uid = np.asarray(u["id"][:], dtype=np.int64) if "id" in u else None
        t = f["intervals"]["trials"]
        stop = np.asarray(t["stop_time"][:], dtype=np.float64)
    n_ours = len(si) - 1
    our_tot = np.zeros(n_ours, dtype=np.float64)
    for i in range(n_ours):
        our_tot[i] = si[i + 1] - si[i]
    log(f"我们这边：单元 {n_ours}；总放电 {our_tot.sum():.0f}；"
        f"时长 {np.nanmax(stop):.1f}s；units.id 头 {uid[:5] if uid is not None else None}")

    df = pd.read_parquet(NM_DIR / "units.parquet")
    log(f"\n桶里 units.parquet：{len(df)} 行；列 {list(df.columns)}")
    log(f"  orig_unit_idx 范围 [{int(df['orig_unit_idx'].min())}, {int(df['orig_unit_idx'].max())}]")

    rows = []
    for idx in sorted(df["session"].unique()):
        sub = df[df["session"] == idx].sort_values("unit_idx")
        fp = NM_DIR / f"spikes_session_{int(idx):03d}.npz"
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        sp = np.asarray(z["spikes"], dtype=np.float32)
        nm_tot = sp.sum(axis=1)
        oi = sub["orig_unit_idx"].to_numpy()
        uidx = sub["unit_idx"].to_numpy()
        if len(oi) != len(nm_tot):
            log(f"  NM{idx:>2}: 行数 {len(oi)} 与 npz 单元 {len(nm_tot)} 不一致 ⇒ 跳过")
            continue
        ok = (oi >= 0) & (oi < n_ours)
        if ok.sum() == 0:
            rows.append((int(idx), 0, 0.0, 0.0, len(nm_tot)))
            continue
        rel = np.abs(nm_tot[ok] - our_tot[oi[ok]]) / (our_tot[oi[ok]] + EPS)
        frac = float((rel < args.tol).mean())
        rows.append((int(idx), int(ok.sum()), frac, float(np.median(rel)), len(nm_tot)))
        if frac > 0.05:
            log(f"  NM{idx:>2}: 单元 {ok.sum()}/{len(nm_tot)}；orig_unit_idx 可用率 1.00；"
                f"**相对差 <{args.tol:.0%} 的占 {frac:.2f}**；中位相对差 {np.median(rel):.3f}"
                f"（NM 单元数 {len(nm_tot)}）")
    rows.sort(key=lambda r: -r[2])
    log("\n按匹配率排序（前 6）：")
    for r in rows[:6]:
        log(f"  NM session {r[0]:>2}：匹配率 {r[2]:.3f}（中位相对差 {r[3]:.3f}，"
            f"可比单元 {r[1]}，NM 单元 {r[4]}）")
    if rows and rows[0][2] > 0.8:
        log(f"\n★ 结论：orig_unit_idx **直连成立** ⇒ session {rows[0][0]} 匹配率 {rows[0][2]:.3f}"
            "⇒ 抽取器可以直接查表拿脑区标签（**零时间对齐**）")
    else:
        log("\n✗ orig_unit_idx 直连不成立 ⇒ 需要另找桥（见日志）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
