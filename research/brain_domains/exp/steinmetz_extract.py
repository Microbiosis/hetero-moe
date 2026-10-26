#!/usr/bin/env python3
"""steinmetz_extract.py — 从 DANDI:000017（Steinmetz 2019）逐 session 抽
**带 trial 对齐 + 逐单元脑区标签**的视图（云端跑）。

## 为什么必须抽（而不是直接用桶里的 npz）

桶里 `/domains/steinmetz/`（Neuromatch 版）有 39 个 `spikes_session_*.npz`（10 ms 分箱）
+ `units.parquet`（逐单元 `region` / `functional_domain`）⇒ **缺的是 trial 表**
（行为标签：`choice` / 刺激侧 / 奖励）⇒ 而 DANDI 的 NWB 里 `intervals/trials` 有：
`response_choice` / `visual_stimulus_left_contrast` / `visual_stimulus_right_contrast` /
`included` / `visual_stimulus_time` / `response_time` / `feedback_type`。
⇒ ⇒ **两边拼起来**才够：行为标签取自 NWB，脑区标签取自桶里 Neuromatch 的 parquet。

## 脑区标签怎么来（DANDI 的 NWB **没有** `brain_region` 列）

⇒ 用**计数相关性**把 DANDI 的单元与桶里 Neuromatch 的单元对上：

1. **session 级**：用「每 100 ms 的总放电率」时间剖面做归一化相关，**允许 ±5 s 的位移**
   （两个版本的 bin 起点可能不同）⇒ 找对应的 Neuromatch session；
2. **单元级**：在配对的 session 内，对每个 DANDI 单元算它与每个 Neuromatch 单元的
   **计数向量余弦相似度**（在位移校正后的重叠区间上，用一批高方差 bin 加速）⇒ 取 argmax
   ⇒ 该 Neuromatch 单元的 `region` / `functional_domain` 就是这个 DANDI 单元的标签。

⇒ **对齐质量必须报出来**（session 剖面相关 + 单元级相关系数中位数 + 有标签的单元比例）
⇒ 对不齐的 session **直接丢**（不猜、不手填）。

## 输出（**只存 trial 窗口特征，不存整段分箱** ⇒ 体积小 100 倍）

`/domains/steinmetz_trials/<session>.npz`：
- `feat` (n_trials × n_units) float32 —— 每 trial 在**固定窗**内的每单元平均放电率；
- `trial` 结构化数组：`stim_time` / `resp_time` / `choice`（±1，NaN=无反应）/
  `stim_side`（±1=对比度在哪一侧）/ `included` / `reward`；
- `unit_region` / `unit_domain`（逐单元，object）；
- `align_*`（对齐质量数字）。

## 用法（云端）

    python -u /domains/scripts/steinmetz_extract.py --out /domains/steinmetz_trials
    python -u /domains/scripts/steinmetz_extract.py --limit-sessions 2 --out /tmp/probe
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

DANDI = "000017"
VERSION = "0.240329.1926"
API = (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
       f"/assets/?page_size=100&fields=asset_id,path,size")
NM_DIR = Path("/domains/steinmetz")
BIN_S = 0.010
PROFILE_BIN = 0.100          # session 剖面用的 100 ms 桶
MAX_LAG_BINS = 50            # ±5 s
N_ALIGN_BINS = 3000          # 单元级对齐用的高方差 bin 数
WIN_START = 0.05            # trial 窗起点（刺激后 50 ms）
WIN_LEN = 0.40              # trial 窗长（400 ms）
EPS = 1e-9


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# DANDI 资产清单
# ---------------------------------------------------------------------------

def dandi_assets():
    with urllib.request.urlopen(API, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for a in data.get("results", []):
        out.append({"asset_id": a["asset_id"], "path": a["path"], "size": a["size"]})
    return out


def asset_url(asset_id):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VERSION}"
            f"/assets/{asset_id}/download/")


# ---------------------------------------------------------------------------
# Neuromatch 侧：剖面 + 单元标签
# ---------------------------------------------------------------------------

def nm_profiles():
    """每个 Neuromatch session 的（100 ms 总放电率剖面, 单元数, 该 session 的标签表）。"""
    import pandas as pd
    units = pd.read_parquet(NM_DIR / "units.parquet")
    profs = {}
    for fp in sorted(NM_DIR.glob("spikes_session_*.npz")):
        idx = int(fp.stem.split("_")[-1])
        z = np.load(fp, allow_pickle=True)
        sp = np.asarray(z["spikes"], dtype=np.float32)
        tot = sp.sum(axis=0)
        k = int(PROFILE_BIN / BIN_S)
        nb = tot.size // k
        p = tot[: nb * k].reshape(nb, k).sum(axis=1)
        p = p - p.mean()
        nrm = float(np.sqrt((p * p).sum())) + EPS
        profs[idx] = {"prof": p / nrm, "n_units": int(sp.shape[0]), "shape": sp.shape}
        del sp, z
    return profs, units


def unit_labels(units):
    """(session, unit_idx) → (region, functional_domain)。"""
    lab = {}
    for sess, uidx, reg, dom in zip(units["session"].to_numpy(),
                                    units["unit_idx"].to_numpy(),
                                    units["region"].to_numpy(),
                                    units["functional_domain"].to_numpy()):
        lab[(int(sess), int(uidx))] = (str(reg), str(dom))
    return lab


# ---------------------------------------------------------------------------
# 对齐
# ---------------------------------------------------------------------------

def match_session(prof, profs):
    """在允许位移内找最匹配的 Neuromatch session ⇒ (idx, corr, lag)。"""
    best = (None, -2.0, 0)
    L = min(len(prof), max(len(p["prof"]) for p in profs.values()))
    for idx, p in profs.items():
        q = p["prof"]
        for lag in range(-MAX_LAG_BINS, MAX_LAG_BINS + 1):
            a0, a1 = max(0, lag), min(L, L + lag)
            b0, b1 = a0 - lag, a1 - lag
            if b0 < 0 or b1 > len(q) or a1 - a0 < 50:
                continue
            x, y = prof[a0:a1], q[b0:b1]
            nx = float(np.sqrt((x * x).sum())) + EPS
            ny = float(np.sqrt((y * y).sum())) + EPS
            c = float((x * y).sum()) / (nx * ny)
            if c > best[1]:
                best = (idx, c, lag)
    return best


def match_units(our_counts, nm_spikes, lag):
    """逐单元余弦相似度 ⇒ 每个我们的单元对应哪个 Neuromatch 单元。"""
    L = min(our_counts.shape[1], nm_spikes.shape[1] - lag)
    if L < 100:
        return None, None
    a0, b0 = 0, max(0, -lag)
    A = our_counts[:, a0:a0 + L]
    B = nm_spikes[:, b0:b0 + L]
    # 选高方差 bin（信息量大、加速矩阵乘）
    var = A.var(axis=0)
    sel = np.argsort(var)[-min(N_ALIGN_BINS, A.shape[1]):]
    A = A[:, sel].astype(np.float32)
    B = B[:, sel].astype(np.float32)
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + EPS)
    C = A @ B.T                       # (n_our, n_nm)
    best = C.argmax(axis=1)
    best_val = C[np.arange(C.shape[0]), best]
    return best, best_val


# ---------------------------------------------------------------------------
# NWB 抽取
# ---------------------------------------------------------------------------

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


def extract_one(asset, work_dir, profs, labels, out_dir):
    import h5py
    t0 = time.time()
    url = asset_url(asset["asset_id"])
    nwb = work_dir / "s.nwb"
    log(f"  ↓ {asset['path']} ({asset['size']/1e6:.0f} MB)")
    urllib.request.urlretrieve(url, nwb)
    dl = time.time() - t0
    try:
        with h5py.File(str(nwb), "r") as f:
            u = f["units"]
            st = np.asarray(u["spike_times"][:], dtype=np.float64)
            si = np.asarray(u["spike_times_index"][:], dtype=np.int64)
            t = f["intervals"]["trials"]
            stop = np.asarray(t["stop_time"][:], dtype=np.float64)
            dur = float(np.nanmax(stop))
            n_bins = int(np.ceil(dur / BIN_S))
            counts = bin_spikes(st, si, n_bins)
            # --- trial 表 ---
            stim = np.asarray(t["visual_stimulus_time"][:], dtype=np.float64)
            resp = np.asarray(t["response_time"][:], dtype=np.float64)
            choice = np.asarray(t["response_choice"][:], dtype=np.float64)
            lc = np.asarray(t["visual_stimulus_left_contrast"][:], dtype=np.float64)
            rc = np.asarray(t["visual_stimulus_right_contrast"][:], dtype=np.float64)
            inc = np.asarray(t["included"][:], dtype=bool)
            fb = np.asarray(t["feedback_type"][:], dtype=np.float64)
        # --- 刺激侧：哪一侧的对比度非零 ---
        side = np.where(np.abs(lc) > np.abs(rc), 1.0, -1.0)
        # --- 与 Neuromatch 对齐 ---
        k = int(PROFILE_BIN / BIN_S)
        nb = counts.shape[1] // k
        prof = counts.sum(axis=0)[: nb * k].reshape(nb, k).sum(axis=1)
        prof = prof - prof.mean()
        prof = prof / (float(np.sqrt((prof * prof).sum())) + EPS)
        idx, corr, lag = match_session(prof, profs)
        if idx is None or corr < 0.5:
            log(f"  ✗ session 对齐失败（corr={corr:.3f}）⇒ 丢")
            return None
        z = np.load(NM_DIR / f"spikes_session_{idx:03d}.npz", allow_pickle=True)
        nm_sp = np.asarray(z["spikes"], dtype=np.float32)
        best, val = match_units(counts, nm_sp, lag)
        del nm_sp, z
        med = float(np.median(val)) if val is not None else -1.0
        frac = float((val > 0.5).mean()) if val is not None else 0.0
        log(f"  ✓ 对齐 NM session {idx}：剖面相关 {corr:.3f}（位移 {lag*PROFILE_BIN:+.1f}s）"
            f"；单元相关中位 {med:.3f}；>0.5 的单元占 {frac:.2f}")
        if med < 0.3 or frac < 0.5:
            log("  ✗ 单元级对齐质量不够 ⇒ 丢（不猜）")
            return None
        # --- 逐单元标签 ---
        reg = np.array([labels.get((idx, int(b)), ("", ""))[0] for b in best], dtype=object)
        dom = np.array([labels.get((idx, int(b)), ("", ""))[1] for b in best], dtype=object)
        # --- trial 窗口特征 ---
        a = np.floor((stim + WIN_START) / BIN_S).astype(np.int64)
        b = np.floor((stim + WIN_START + WIN_LEN) / BIN_S).astype(np.int64)
        feat = np.zeros((len(stim), counts.shape[0]), dtype=np.float32)
        ok = np.isfinite(a) & np.isfinite(b) & (b > a) & (b <= counts.shape[1])
        for i in np.flatnonzero(ok):
            seg = counts[:, a[i]:b[i]]
            feat[i] = seg.mean(axis=1)
        out = {
            "session": asset["path"],
            "nm_session": idx,
            "feat": feat,
            "trial_stim_time": stim,
            "trial_resp_time": resp,
            "trial_choice": choice,
            "trial_stim_side": side,
            "trial_included": inc,
            "trial_reward": fb,
            "unit_region": reg,
            "unit_domain": dom,
            "align_session_corr": np.float32(corr),
            "align_lag_s": np.float32(lag * PROFILE_BIN),
            "align_unit_corr_median": np.float32(med),
            "align_unit_frac_gt05": np.float32(frac),
            "win_start": np.float32(WIN_START),
            "win_len": np.float32(WIN_LEN),
            "bin_s": np.float32(BIN_S),
        }
        p = out_dir / (Path(asset["path"]).stem + ".npz")
        np.savez_compressed(p, **out)
        log(f"  → {p.name}（{p.stat().st_size/1e6:.1f} MB；下载 {dl:.0f}s，总 {time.time()-t0:.0f}s）")
        return {"path": asset["path"], "nm_session": idx, "align_session_corr": float(corr),
                "align_unit_corr_median": med, "align_unit_frac_gt05": frac,
                "n_units": int(counts.shape[0]), "n_trials": int(len(stim)),
                "n_included": int(inc.sum()),
                "regions": {r: int((reg == r).sum()) for r in sorted(set(reg.tolist()))},
                "domains": {d: int((dom == d).sum()) for d in sorted(set(dom.tolist()))}}
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
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    assets = dandi_assets()
    log(f"DANDI {DANDI}@{VERSION}：{len(assets)} 个 session")
    if args.limit_sessions:
        assets = assets[: args.limit_sessions]

    log("读 Neuromatch 侧：剖面 + 单元标签 …")
    profs, units_df = nm_profiles()
    labels = unit_labels(units_df)
    log(f"  {len(profs)} 个 NM session；标签表 {len(labels)} 条")

    recs = []
    for a in assets:
        stem = Path(a["path"]).stem
        if (out_dir / (stem + ".npz")).exists():
            log(f"· {stem} 已存在 ⇒ 跳过")
            continue
        try:
            r = extract_one(a, work_dir, profs, labels, out_dir)
        except Exception as e:                       # 单个 session 失败不影响其余
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
