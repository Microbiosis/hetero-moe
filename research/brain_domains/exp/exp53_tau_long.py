#!/usr/bin/env python3
"""exp53 — τ 在长片段上的收敛性（三个检验）。

旧版把每段切成 800 ms 的窗，而 τ ≈ 200 ms，每窗只装 4 个 τ。
v3 存了原始计数，片段长 20 s–1000 s，可以回答三个问题：

  Q1 收敛：τ 随分析窗口长度 W 如何变化？
     若 τ 是真实时间常数，W 超过几个 τ 后就该稳定。
     若随 W 单调变化，τ 是 W 的函数，不是系统的性质。

  Q2 分裂信度：同一片段的前半与后半，τ 是否一致？
     这是比「分辨率检验」更强的信度检验，因为它不改变采样率。

  Q3 物种：猕猴与人类在 M1 上的不变量 τ 是否不同？
     同时报告可观测性（top-1 占比、有效维度）作为对照。

全部用不变量估计量 tr C(τ)/tr C(0)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/tau-long")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500                  # 10 s
WS = [250, 500, 1000, 2000, 4000, 8000, 16000, 10**9]   # 5s..全段
SEED = 0


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


def fit_tau(a, dt=DT, lo=2.0, hi=100000.0):
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.995)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[200.0], bounds=([lo], [hi]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


def tau_int(a, dt=DT, c=5.0):
    taus = 1.0 + 2.0 * np.cumsum(a[1:])
    Ms = np.arange(1, len(a))
    m = Ms >= c * np.maximum(taus, 1e-9)
    if np.any(m):
        return float(taus[int(np.argmax(m))] * dt)
    return float(taus[-1] * dt)


def top1(x):
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False) ** 2
    return float(s[0] / max(s.sum(), 1e-12)), float((s.sum() ** 2) / max((s ** 2).sum(), 1e-12))


def main():
    t0 = time.time()
    W = np.load(V3 / "counts.npy", mmap_mode="r")
    idx = pd.read_parquet(V3 / "counts_index.parquet")
    print(f"counts {W.shape} | segments {len(idx)}", flush=True)

    # ---------------- 逐片段：全段 + 窗口长度扫描
    rows = []
    for n, (_, r) in enumerate(idx.iterrows()):
        x = np.asarray(W[r.start:r.end, :r.n_units], dtype=np.float32)
        if x.shape[0] < 600:
            continue
        t1, eff = top1(x)
        # 收敛曲线
        curve = {}
        for w in WS:
            L = min(w, x.shape[0])
            if L < 600:
                continue
            a = weighted_acf(x[:L], MAX_LAG)
            curve[str(w)] = {"tau_fit": fit_tau(a), "tau_int": tau_int(a),
                             "acf1": float(a[1])}
        # 分裂信度（把全段一分为二）
        h = x.shape[0] // 2
        ah = weighted_acf(x[:h], MAX_LAG)
        bh = weighted_acf(x[h:], MAX_LAG)
        a_all = weighted_acf(x, MAX_LAG)
        rows.append({
            "idx": int(n), "source": r.source, "region": r.region,
            "file": r.file, "n_units": int(r.n_units),
            "n_bins": int(x.shape[0]), "hours": round(x.shape[0] * DT / 3.6e6, 2),
            "rate": float(x.mean()), "fano": float(x.var() / max(x.mean(), 1e-9)),
            "top1": round(t1, 4), "eff_dim": round(eff, 2),
            "tau_all_fit": curve.get(str(10**9), {}).get("tau_fit"),
            "tau_all_int": curve.get(str(10**9), {}).get("tau_int"),
            "acf1_all": curve.get(str(10**9), {}).get("acf1"),
            "tau_half1": fit_tau(ah), "tau_half2": fit_tau(bh),
            "tau_int_half1": tau_int(ah), "tau_int_half2": tau_int(bh),
            "curve": json.dumps(curve),
            "n_tau_in_window": round(x.shape[0] * DT / max(curve.get(str(10**9), {}).get("tau_fit") or 1e9, 1e-9), 1),
        })
        if n % 10 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} {x.shape[0]}b {r.n_units}u "
                  f"tau={rows[-1]['tau_all_fit']} [{time.time()-t0:.0f}s]", flush=True)
        del x
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_segment.csv", index=False)

    rep = {"task": "tau_long_convergence", "dt_ms": DT,
           "n_segments": int(len(df)), "max_lag_bins": MAX_LAG,
           "windows_bins": [w for w in WS]}

    # ---------------- Q1 收敛
    conv = {}
    for w in WS:
        vals = []
        for c in df.curve:
            v = json.loads(c).get(str(w), {}).get("tau_fit")
            if v is not None:
                vals.append(v)
        if vals:
            conv[str(w)] = {"n": len(vals), "median": round(float(np.median(vals)), 1),
                            "p10": round(float(np.percentile(vals, 10)), 1),
                            "p90": round(float(np.percentile(vals, 90)), 1)}
    rep["Q1_convergence_tau_fit"] = conv

    # 逐片段：全段 tau 与短窗 tau 的比值
    rat = {}
    full = [json.loads(c).get(str(10**9), {}).get("tau_fit") for c in df.curve]
    for w in WS[:-1]:
        vals = []
        for c, f in zip(df.curve, full):
            v = json.loads(c).get(str(w), {}).get("tau_fit")
            if v and f:
                vals.append(v / f)
        if vals:
            rat[str(w)] = {"n": len(vals), "median_ratio_vs_full": round(float(np.median(vals)), 3)}
    rep["Q1_ratio_short_over_full"] = rat

    # ---------------- Q2 分裂信度
    m = df.dropna(subset=["tau_half1", "tau_half2"])
    if len(m) > 3:
        from scipy.stats import spearmanr, pearsonr
        r_p = pearsonr(m.tau_half1, m.tau_half2)
        r_s = spearmanr(m.tau_half1, m.tau_half2)
        ratio = (m.tau_half1 / m.tau_half2)
        # Spearman-Brown 校正
        sb = 2 * r_p[0] / (1 + r_p[0]) if r_p[0] > -1 else None
        rep["Q2_split_half"] = {
            "n": int(len(m)),
            "pearson_r": round(float(r_p[0]), 3), "pearson_p": float(r_p[1]),
            "spearman_rho": round(float(r_s[0]), 3), "spearman_p": float(r_s[1]),
            "spearman_brown": round(float(sb), 3) if sb is not None else None,
            "ratio_median": round(float(ratio.median()), 3),
            "ratio_iqr": round(float(ratio.quantile(.75) - ratio.quantile(.25)), 3),
        }
    mi = df.dropna(subset=["tau_int_half1", "tau_int_half2"])
    if len(mi) > 3:
        from scipy.stats import spearmanr
        r = spearmanr(mi.tau_int_half1, mi.tau_int_half2)
        rep["Q2_split_half_tau_int"] = {"n": int(len(mi)),
                                        "rho": round(float(r[0]), 3), "p": float(r[1])}

    # ---------------- Q3 物种
    by = {}
    for s, g in df.groupby("source"):
        by[s] = {
            "n": int(len(g)),
            "tau_all_fit_median": round(float(g.tau_all_fit.median()), 1),
            "tau_all_fit_p10": round(float(g.tau_all_fit.quantile(.1)), 1),
            "tau_all_fit_p90": round(float(g.tau_all_fit.quantile(.9)), 1),
            "tau_int_median": round(float(g.tau_all_int.median()), 1),
            "top1_median": round(float(g.top1.median()), 4),
            "eff_dim_median": round(float(g.eff_dim.median()), 1),
            "rate_median": round(float(g.rate.median()), 3),
            "fano_median": round(float(g.fano.median()), 3),
            "hours_median": round(float(g.hours.median()), 2),
        }
    rep["Q3_by_source"] = by
    m1 = df[df.shared_node == "M1"] if "shared_node" in df else df
    g1 = df[(df.region == "M1") & (df.source == "macaque_nlb")].tau_all_fit.dropna()
    g2 = df[(df.region == "M1") & (df.source == "human_intracortical")].tau_all_fit.dropna()
    if len(g1) > 2 and len(g2) > 2:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(g1, g2)
        rep["Q3_m1_between_species"] = {
            "macaque_n": int(len(g1)), "macaque_median": round(float(g1.median()), 1),
            "human_n": int(len(g2)), "human_median": round(float(g2.median()), 1),
            "ratio": round(float(g1.median() / g2.median()), 2),
            "mannwhitney_p": float(u.pvalue)}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
