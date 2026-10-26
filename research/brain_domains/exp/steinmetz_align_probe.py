#!/usr/bin/env python3
"""steinmetz_align_probe.py — 诊断「DANDI 单元 ↔ 桶里 Neuromatch 单元」的对齐参数。

## 为什么需要这个

`steinmetz_extract.py` 假设「桶里 Neuromatch 的 `spikes` 是 **10 ms** 分箱、且与 NWB 同一起点」
⇒ 但实测 **session 级剖面相关只有 0.01–0.02** ⇒ 假设错了（bin 尺寸或时间原点）。
⇒ 而**脑区标签只在 Neuromatch 那边有**（DANDI 的 NWB 没有 `brain_region` 列）⇒ 对齐是必须解决的。

## 本脚本干什么

对**一个** DANDI session：

1. 打印我们这边的诊断：总放电数、时长、trial 数、单元数；
2. 对桶里**每个** Neuromatch session：总放电数 / bin 数 / 单元数；
3. 用**总放电数**做 session 级粗匹配（列出最像的几个）；
4. 对最像的几个，**逐个候选 bin 尺寸**（1/2/5/10/20/25/50 ms）做 **FFT 互相关**
   （可在 O(n log n) 里搜**全时间轴**的偏移，不受 ±5 s 限制）⇒ 给出 (bin 尺寸, 偏移, 相关系数)；
5. 用胜出的参数做**逐单元**余弦匹配 ⇒ 报告对齐质量（中位相关、>0.5 的比例）
   ⇒ 这就是抽取器要用的参数。

## 用法（云端）

    python -u /domains/scripts/steinmetz_align_probe.py --limit-nm 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

DANDI = "000017"
VERSION = "0.240329.1926"
API = (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
       f"/assets/?page_size=100&fields=asset_id,path,size")
NM_DIR = Path("/domains/steinmetz")
CAND_BIN_MS = (1, 2, 5, 10, 20, 25, 50)
PROFILE_MS = 100
EPS = 1e-9


def log(*a):
    print(*a, flush=True)


def dandi_assets():
    with urllib.request.urlopen(API, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    return [{"asset_id": a["asset_id"], "path": a["path"], "size": a["size"]}
            for a in data.get("results", [])]


def best_lag_corr(P, Q):
    """FFT 互相关 ⇒ (lag, corr)：lag = Q 相对 P 的偏移（Q 的索引 = P 的索引 + lag）。"""
    n = 1
    while n < len(P) + len(Q):
        n *= 2
    cc = np.fft.irfft(np.fft.rfft(P, n) * np.conj(np.fft.rfft(Q, n)), n)
    # 在 [-len(Q)+1, len(P)-1] 范围内找峰（环形折叠 ⇒ 分两段取最大）
    half = n // 2
    seg1 = cc[:half]          # 正偏移
    seg2 = cc[half:]          # 负偏移（环形）
    i1 = int(np.argmax(seg1))
    i2 = int(np.argmax(seg2))
    if seg1[i1] >= seg2[i2]:
        lag = i1
    else:
        lag = i2 - half
    # 在该 lag 上算归一化相关（直接算，别信 FFT 的幅值）
    a0, a1 = max(0, lag), min(len(P), len(Q) + lag)
    b0, b1 = a0 - lag, a1 - lag
    if a1 - a0 < 50:
        return lag, 0.0
    x, y = P[a0:a1], Q[b0:b1]
    nx = float(np.sqrt((x * x).sum())) + EPS
    ny = float(np.sqrt((y * y).sum())) + EPS
    return lag, float((x * y).sum()) / (nx * ny)


def bin_spikes(st, si, n_bins, bin_s):
    n_units = len(si) - 1
    mat = np.zeros((n_units, n_bins), dtype=np.float32)
    for u in range(n_units):
        s = st[si[u]:si[u + 1]]
        if s.size == 0:
            continue
        b = np.floor(s / bin_s).astype(np.int64)
        b = b[(b >= 0) & (b < n_bins)]
        if b.size:
            mat[u] = np.bincount(b, minlength=n_bins)[:n_bins]
    return mat


def profile(counts, bin_s, profile_ms=PROFILE_MS):
    k = max(1, int(round(profile_ms / (bin_s * 1000.0))))
    nb = counts.shape[1] // k
    if nb < 10:
        return None
    p = counts.sum(axis=0)[: nb * k].reshape(nb, k).sum(axis=1)
    p = p - p.mean()
    return p / (float(np.sqrt((p * p).sum())) + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-nm", type=int, default=8)
    ap.add_argument("--session-index", type=int, default=0)
    ap.add_argument("--work-dir", default="/tmp")
    args = ap.parse_args()

    assets = dandi_assets()
    a = assets[args.session_index]
    log(f"目标 session：{a['path']}（{a['size']/1e6:.0f} MB）")
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
           f"/assets/{a['asset_id']}/download/")
    nwb = Path(args.work_dir) / "s.nwb"
    t0 = time.time()
    urllib.request.urlretrieve(url, nwb)
    log(f"下载完 {nwb.stat().st_size/1e6:.0f} MB（{time.time()-t0:.0f}s）")

    import h5py
    with h5py.File(str(nwb), "r") as f:
        u = f["units"]
        st = np.asarray(u["spike_times"][:], dtype=np.float64)
        si = np.asarray(u["spike_times_index"][:], dtype=np.int64)
        t = f["intervals"]["trials"]
        stop = np.asarray(t["stop_time"][:], dtype=np.float64)
        start = np.asarray(t["start_time"][:], dtype=np.float64)
        stim = np.asarray(t["visual_stimulus_time"][:], dtype=np.float64)
    dur = float(np.nanmax(stop))
    n_tri = len(stop)
    n_units = len(si) - 1
    our_total = float(st.size)
    log(f"\n我们这边：总放电 {our_total:.0f}；时长 {dur:.1f}s（{dur/60:.1f} 分）；"
        f"trial {n_tri}；单元 {n_units}；刺激时刻 [{np.nanmin(stim):.1f}, {np.nanmax(stim):.1f}]")

    # 我们的 10 ms 计数
    counts = bin_spikes(st, si, int(np.ceil(dur / 0.010)), 0.010)

    # Neuromatch 侧统计
    nm_stats = []
    for fp in sorted(NM_DIR.glob("spikes_session_*.npz")):
        idx = int(fp.stem.split("_")[-1])
        z = np.load(fp, allow_pickle=True)
        sp = np.asarray(z["spikes"], dtype=np.float32)
        nm_stats.append({"idx": idx, "total": float(sp.sum()), "n_bins": int(sp.shape[1]),
                         "n_units": int(sp.shape[0]), "_sp": sp})
    nm_stats.sort(key=lambda d: abs(d["total"] - our_total))
    log(f"\nNeuromatch 侧（按总放电数排序，最像的前 {args.limit_nm} 个）：")
    for d in nm_stats[: args.limit_nm]:
        log(f"  NM session {d['idx']:>2}：总放电 {d['total']:.0f}（差 {d['total']-our_total:+.0f}）"
            f"；bin {d['n_bins']}；单元 {d['n_units']}"
            f" ⇒ 若 10 ms 则 {d['n_bins']*0.01/60:.1f} 分")

    our_prof = profile(counts, 0.010)
    log(f"\n我们的 100 ms 剖面长度 {len(our_prof)}")
    best = None
    for d in nm_stats[: args.limit_nm]:
        sp = d.pop("_sp")
        for bin_ms in CAND_BIN_MS:
            k = max(1, int(round(PROFILE_MS / bin_ms)))
            nb = sp.shape[1] // k
            if nb < 10:
                continue
            q = sp.sum(axis=0)[: nb * k].reshape(nb, k).sum(axis=1)
            q = q - q.mean()
            q = q / (float(np.sqrt((q * q).sum())) + EPS)
            lag, c = best_lag_corr(our_prof, q)
            log(f"  NM{d['idx']:>2} bin={bin_ms:>2}ms：最佳偏移 {lag*PROFILE_MS/1000.0:+.1f}s"
                f"；相关 {c:+.3f}")
            if best is None or c > best[0]:
                best = (c, d["idx"], bin_ms, lag)
    if best is None:
        log("✗ 没有可比对的组合")
        return 1
    c, nm_idx, bin_ms, lag = best
    log(f"\n★ 最佳组合：NM session {nm_idx}、bin {bin_ms} ms、偏移 {lag*PROFILE_MS/1000.0:+.1f}s"
        f"、相关 {c:+.3f}")
    if c < 0.3:
        log("✗ 相关仍太低 ⇒ 假设不成立，需换思路")
        return 1

    # 逐单元匹配（用胜出的 bin 与偏移）
    d = next(x for x in nm_stats if x["idx"] == nm_idx)
    z = np.load(NM_DIR / f"spikes_session_{nm_idx:03d}.npz", allow_pickle=True)
    sp = np.asarray(z["spikes"], dtype=np.float32)
    bin_s = bin_ms / 1000.0
    k = max(1, int(round(PROFILE_MS / bin_ms)))
    # 我们的 100 ms 剖面对应他们 bin_ms 的 k' 桶：把他们的重采样到 100 ms
    our_100 = our_prof
    L = min(len(our_100), sp.shape[1] // k)
    # 直接用原始 bin 做单元级相关（更细 ⇒ 更准）
    A = counts[:, :]
    shift = lag * k      # 100 ms 单位 ⇒ 他们的 bin 单位
    b0 = max(0, -shift)
    L2 = min(A.shape[1], sp.shape[1] - max(0, shift))
    if L2 < 100:
        log("✗ 重叠区太小")
        return 1
    Aa = A[:, 0:L2]
    Bb = sp[:, b0:b0 + L2]
    var = Aa.var(axis=0)
    sel = np.argsort(var)[-min(3000, Aa.shape[1]):]
    Aa = Aa[:, sel].astype(np.float32)
    Bb = Bb[:, sel].astype(np.float32)
    Aa = Aa - Aa.mean(axis=1, keepdims=True)
    Bb = Bb - Bb.mean(axis=1, keepdims=True)
    Aa /= (np.linalg.norm(Aa, axis=1, keepdims=True) + EPS)
    Bb /= (np.linalg.norm(Bb, axis=1, keepdims=True) + EPS)
    C = Aa @ Bb.T
    bestj = C.argmax(axis=1)
    bestv = C[np.arange(C.shape[0]), bestj]
    log(f"逐单元匹配：相关中位 {np.median(bestv):.3f}；>0.5 占 {(bestv>0.5).mean():.2f}；"
        f"最大 {bestv.max():.3f}")
    log(f"⇒ 抽取器应当用：bin={bin_ms} ms，偏移 {lag*PROFILE_MS/1000.0:+.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
