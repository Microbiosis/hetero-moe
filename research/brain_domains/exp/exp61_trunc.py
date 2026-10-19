#!/usr/bin/env python3
"""exp61 — 截断到相同时长后的 tau。

exp60 的问题：

  tau 与记录时长的 ρ = 0.287（越长越慢）
  猕猴片段中位 16 min，人类 2 min
  人类放电率 [0.16, 0.30] 与猕猴 [0.35, 0.63] 几乎不重叠，无法匹配

所以物种差异可能只是"人类片段太短，估不出慢成分"。

本脚本把两边的片段都截到相同长度，重算平台模型的 tau：

  3000 bin (60 s)   —— 人类中位 2 min，多数够
  1500 bin (30 s)   —— 更严格
  全部可用长度       —— 对照

若截断后物种差异消失 → 差异是时长的产物。
若仍在 → 差异与记录时长无关（但仍与记录技术混淆）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, spearmanr

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/trunc")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500
N_NULL = 2
SEED = 0
LENS = [1500, 3000, 6000, None]     # None = 全段


def n2(n):
    k = 1
    while k < n:
        k <<= 1
    return k


def rho_curve(x, max_lag):
    T, d = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    N = n2(2 * T)
    F = np.fft.rfft(xc, n=N, axis=0)
    Fs = F.sum(axis=1)
    tot = np.fft.irfft(Fs * np.conj(Fs), n=N)[:max_lag + 1]
    dia = np.fft.irfft((np.abs(F) ** 2).sum(axis=1), n=N)[:max_lag + 1]
    nrm = np.arange(T, T - max_lag - 1, -1)
    tot = tot / np.maximum(nrm, 1)
    dia = dia / np.maximum(nrm, 1)
    t0 = max(tot[0], 1e-12)
    return (tot - dia) / t0


def null_noise(x, ml, rng, n=N_NULL):
    prof = []
    for _ in range(n):
        xs = np.empty_like(x)
        for j in range(x.shape[1]):
            xs[:, j] = np.roll(x[:, j], int(rng.integers(1, x.shape[0])))
        prof.append(rho_curve(xs, ml))
    return float(np.sqrt((np.array(prof)[:, 2:] ** 2).mean()))


def m_plateau(t, A, tau, c):
    return A * np.exp(-t / np.maximum(tau, 1e-6)) + c


def fit_tau(t, y):
    try:
        p, _ = curve_fit(m_plateau, t, y, p0=[float(y[0]), 200.0, max(float(y[-1]), 0.0)],
                         bounds=([0.0, 2.0, -1.0], [10.0, 1e5, 1.0]), maxfev=60000)
        r = y - m_plateau(t, *p)
        return float(p[1]), float(p[0]), float(p[2]), float(1 - np.var(r) / max(np.var(y), 1e-20))
    except Exception:
        return None, None, None, None


def task_of(source, fname):
    if source == "human_intracortical":
        return "H1_reach_grasp" if fname.startswith("h1") else "H2_handwriting"
    if "churchland" in fname:
        return "reach"
    if "dmfc" in fname:
        return "timing"
    if "area2" in fname:
        return "bump"
    return "?"


def main():
    t0 = time.time()
    W = np.load(V3 / "counts.npy", mmap_mode="r")
    idx = pd.read_parquet(V3 / "counts_index.parquet")
    rng = np.random.default_rng(SEED)
    rows = []
    for n, (_, r) in enumerate(idx.iterrows()):
        x = np.asarray(W[r.bin_start:r.bin_end, :r.n_units], dtype=np.float32)
        rec = {"source": r.source, "task": task_of(r.source, r.file),
               "file": r.file, "n_units": int(x.shape[1]),
               "n_bins_max": int(x.shape[0]), "rate": float(x.mean())}
        got = False
        for L in LENS:
            key = "full" if L is None else str(L)
            xx = x if L is None else x[:L]
            if xx.shape[0] < 1200:
                rec[f"tau_{key}"] = None
                continue
            ml = min(MAX_LAG, xx.shape[0] - 2)
            noise = null_noise(xx, ml, rng)
            rho = rho_curve(xx, ml)
            snr = rho[1] / max(noise, 1e-12)
            usable = np.flatnonzero((rho > 2 * noise) & np.isfinite(rho))
            usable = usable[(usable >= 2) & (usable <= ml)]
            if snr < 3.0 or len(usable) < 10:
                rec[f"tau_{key}"] = None
                continue
            tv, A, c, r2 = fit_tau(usable.astype(float) * DT, rho[usable])
            rec[f"tau_{key}"] = round(tv, 1) if tv else None
            rec[f"A_{key}"] = round(A, 3) if A else None
            rec[f"c_{key}"] = round(c, 4) if c is not None else None
            rec[f"snr_{key}"] = round(snr, 1)
            got = got or tv is not None
        rows.append(rec)
        if n % 20 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} tau_full={rec.get('tau_full')} "
                  f"tau_3000={rec.get('tau_3000')} [{time.time()-t0:.0f}s]", flush=True)
        del x

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_segment.csv", index=False)
    rep = {"task": "duration_truncation", "dt_ms": DT, "n": int(len(d))}

    for L in LENS:
        key = "full" if L is None else str(L)
        col = f"tau_{key}"
        mac = d[d.source == "macaque_nlb"][col].dropna()
        hum = d[d.source == "human_intracortical"][col].dropna()
        entry = {"n_macaque": int(len(mac)), "n_human": int(len(hum)),
                 "hours": "full" if L is None else round(L * DT / 3.6e6, 3)}
        if len(mac) > 3 and len(hum) > 3:
            entry.update({
                "macaque_median": round(float(mac.median()), 1),
                "human_median": round(float(hum.median()), 1),
                "ratio": round(float(mac.median() / max(hum.median(), 1e-9)), 2),
                "p": float(mannwhitneyu(mac, hum).pvalue)})
        mac_m1 = d[(d.source == "macaque_nlb") & (d.task == "reach")][col].dropna()
        hum_h2 = d[d.task == "H2_handwriting"][col].dropna()
        if len(mac_m1) > 3 and len(hum_h2) > 3:
            entry["reach_vs_handwriting"] = {
                "macaque": round(float(mac_m1.median()), 1),
                "human": round(float(hum_h2.median()), 1),
                "ratio": round(float(mac_m1.median() / max(hum_h2.median(), 1e-9)), 2),
                "p": float(mannwhitneyu(mac_m1, hum_h2).pvalue)}
        rep[f"len_{key}"] = entry

    # tau 与时长（在同一物种内）
    rep["tau_vs_duration_within"] = {}
    for src, g in d.groupby("source"):
        m = g[["tau_full", "n_bins_max"]].dropna()
        if len(m) > 10:
            r_ = spearmanr(m.tau_full, m.n_bins_max)
            rep["tau_vs_duration_within"][src] = {
                "rho": round(float(r_[0]), 3), "p": float(r_[1]), "n": int(len(m))}
    m = d[["tau_full", "n_bins_max"]].dropna()
    r_ = spearmanr(m.tau_full, m.n_bins_max)
    rep["tau_vs_duration_pooled"] = {"rho": round(float(r_[0]), 3), "p": float(r_[1])}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
