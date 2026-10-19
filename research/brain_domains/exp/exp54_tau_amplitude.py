#!/usr/bin/env python3
"""exp54 — 把幅度从时间常数里分离出来。

exp53 用 R(τ) = tr C(τ)/tr C(0) 拟合 exp(−τ/T)，把幅度固定成 1。
但

    R(0) = 信号方差 / (信号方差 + 噪声方差)

白噪声只在 τ=0 贡献，τ>0 的 tr C(τ) 只剩信号。所以

    R(τ) ≈ A · ρ_signal(τ),    A = 信号方差占比 < 1

A 小的时候，用 A=1 拟合会把 T 压小。exp53 里猕猴 τ=159 ms、人类 τ=11 ms，
差 14.6 倍，但两者的 top-1 占比是 0.31 对 0.06，正好落在这个陷阱里。

改为拟合两个参数

    R(τ) = A · exp(−τ/T)        A, T 都自由

T 就是信号自身的时间常数，与信噪比无关。A 单独报告，它是可观测性指标。

同时报告：
  τ_int 的幅度校正版：先在 R(τ) 上除以 A 再积分
  A 与 top-1 占比、有效维度的关系
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
OUT = Path("/domains/runs/tau-amplitude")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500


def weighted_acf(x, max_lag):
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    a = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    a = a / np.maximum(norms, 1)
    return a.sum(axis=1) / max(a[0].sum(), 1e-12)


def fit_fixed(a, dt=DT, lo=2.0, hi=100000.0):
    """R(τ) = exp(−τ/T)，幅度固定为 1（旧做法）。"""
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.995)
    if ok.sum() < 5:
        return None, None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        p, _ = curve_fit(f, t, y, p0=[200.0], bounds=([lo], [hi]), maxfev=8000)
        return float(p[0]), 1.0
    except Exception:
        return None, None


def fit_free(a, dt=DT, lo=1.0, hi=100000.0):
    """R(τ) = A·exp(−τ/T)，A 与 T 都自由。返回 (T, A)。"""
    ok = np.isfinite(a) & (a > 1e-4) & (a < 0.999)
    if ok.sum() < 6:
        return None, None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, A, T):
        return A * np.exp(-tt / np.maximum(T, 1e-3))

    try:
        p, _ = curve_fit(f, t, y, p0=[float(y[0]), 200.0],
                         bounds=([1e-4, lo], [1.0, hi]), maxfev=20000)
        return float(p[1]), float(p[0])
    except Exception:
        return None, None


def tau_int_corrected(a, A, dt=DT, c=5.0):
    """先除以 A 再积分，得到信号自身的时间常数。"""
    if not A or A <= 0:
        return None
    r = np.minimum(a / A, 1.0)
    taus = 1.0 + 2.0 * np.cumsum(r[1:])
    Ms = np.arange(1, len(r))
    m = Ms >= c * np.maximum(taus, 1e-9)
    if np.any(m):
        return float(taus[int(np.argmax(m))] * dt)
    return float(taus[-1] * dt)


def main():
    t0 = time.time()
    W = np.load(V3 / "counts.npy", mmap_mode="r")
    idx = pd.read_parquet(V3 / "counts_index.parquet")
    rows = []
    for n, (_, r) in enumerate(idx.iterrows()):
        x = np.asarray(W[r.start:r.end, :r.n_units], dtype=np.float32)
        if x.shape[0] < 1000:
            continue
        a = weighted_acf(x, MAX_LAG)
        tf, _ = fit_fixed(a)
        tv, A = fit_free(a)
        rows.append({
            "source": r.source, "region": r.region, "file": r.file,
            "n_units": int(r.n_units), "n_bins": int(x.shape[0]),
            "hours": round(x.shape[0] * DT / 3.6e6, 3),
            "rate": float(x.mean()),
            "acf1": round(float(a[1]), 4),
            "acf2": round(float(a[2]), 4),
            "acf10": round(float(a[10]), 4) if len(a) > 10 else None,
            "acf100": round(float(a[100]), 4) if len(a) > 100 else None,
            "tau_fixed": tf, "tau_free": tv, "amp": A,
            "tau_int_corr": tau_int_corrected(a, A),
        })
        if n % 20 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} fixed={tf and round(tf,1)} "
                  f"free={tv and round(tv,1)} A={A and round(A,3)} [{time.time()-t0:.0f}s]",
                  flush=True)
        del x
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_segment.csv", index=False)

    rep = {"task": "tau_amplitude_separation", "dt_ms": DT, "n_segments": int(len(df))}

    def s(g, c):
        v = g[c].dropna()
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(v.median()), 2),
                "p10": round(float(v.quantile(.1)), 2),
                "p90": round(float(v.quantile(.9)), 2)}

    rep["overall"] = {c: s(df, c) for c in
                      ["tau_fixed", "tau_free", "amp", "acf1", "acf2", "acf100",
                       "tau_int_corr"]}
    rep["by_source"] = {}
    for src, g in df.groupby("source"):
        rep["by_source"][src] = {
            "n_segments": int(len(g)),
            **{c: s(g, c) for c in ["tau_fixed", "tau_free", "amp", "acf1",
                                    "tau_int_corr", "rate"]},
            "hours_median": round(float(g.hours.median()), 3),
        }

    # M1 上的物种比较：三种口径
    g1 = df[(df.region == "M1") & (df.source == "macaque_nlb")]
    g2 = df[(df.region == "M1") & (df.source == "human_intracortical")]
    cmp = {"n_macaque": int(len(g1)), "n_human": int(len(g2))}
    for c in ["tau_fixed", "tau_free", "tau_int_corr"]:
        v1 = g1[c].dropna(); v2 = g2[c].dropna()
        if len(v1) > 2 and len(v2) > 2:
            cmp[c] = {
                "macaque_median": round(float(v1.median()), 2),
                "human_median": round(float(v2.median()), 2),
                "ratio": round(float(v1.median() / v2.median()), 2),
                "mannwhitney_p": float(mannwhitneyu(v1, v2).pvalue),
            }
    rep["Q_species_M1"] = cmp

    # 幅度与谱的关系
    for c in ["amp"]:
        for d in ["acf1", "rate"]:
            m = df[[c, d]].dropna()
            if len(m) > 5:
                r = spearmanr(m[c], m[d])
                rep[f"corr_{c}_vs_{d}"] = {"rho": round(float(r[0]), 3),
                                           "p": float(r[1]), "n": int(len(m))}

    # 幅度校正是否消除了物种差异
    if "tau_fixed" in cmp and "tau_free" in cmp:
        rep["interpretation"] = {
            "fixed_ratio": cmp["tau_fixed"]["ratio"],
            "free_ratio": cmp["tau_free"]["ratio"],
            "shrinkage_explained": round(
                100 * (1 - (cmp["tau_free"]["ratio"] - 1) /
                       max(cmp["tau_fixed"]["ratio"] - 1, 1e-9)), 1),
        }

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
