#!/usr/bin/env python3
"""exp49b — tau 的规范依赖与噪声底污染（spike, 20 ms）。

上一步发现：逐成分 ACF 取中位数，量到的是噪声。
    churchland_00  top-5 特征值 = 0.31, 0.047, 0.029, 0.013, 0.011
                    PC1 ACF(1) = 0.87，而全部成分 ACF 中位数 = 0.055

所以 τ 的小值不是「数据没有相关性」，而是「中位数被 63 个噪声成分主导」。

本脚本比较五个量：

  T1  median_PC      逐成分 ACF 取中位数（旧方法）
  T2  mean_PC        逐成分 ACF 等权平均（我上一版的错误）
  T3  trace_weighted tr C(τ) / tr C(0)  —— 方差加权，且正交不变
  T4  PC1_only       只用第 1 主成分
  T5  sobi_median    联合对角化后的成分取中位数

并检验：
  R1  T1 在随机正交旋转下是否改变   -> 变了说明是基的产物
  R2  T3 在随机正交旋转下是否改变   -> 应保持到 1e-6
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SEG = Path("/domains/primate/macaque_nlb")
OUT = Path("/domains/runs/tau-gauge")
OUT.mkdir(parents=True, exist_ok=True)

DT, MAX_LAG, N_ROT, SEED = 20.0, 100, 5, 0


def acf_cols(x, max_lag):
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    acf = acf / np.maximum(norms, 1)
    return acf / np.maximum(acf[0:1], 1e-12)


def fit_tau(a, dt=DT):
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.995)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[200.0], bounds=([1.0], [50000.0]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


def weighted_acf(x, max_lag):
    """tr C(τ) / tr C(0)，按方差加权。正交不变。"""
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]   # (lag, d)
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    acf = acf / np.maximum(norms, 1)
    return acf.sum(axis=1) / max(acf[0].sum(), 1e-12)


def sobi(x):
    T = x.shape[0]
    C0 = x.T @ x / T
    C1 = x[:-1].T @ x[1:] / (T - 1)
    C0 = (C0 + C0.T) / 2
    try:
        w, V = np.linalg.eigh(np.linalg.solve(C0, (C1 + C1.T) / 2))
    except np.linalg.LinAlgError:
        return None
    return V[:, np.argsort(-np.abs(w))]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    files = sorted(SEG.glob("spikes_*.npz"))
    print(f"{len(files)} segments", flush=True)
    for k, f in enumerate(files):
        raw = np.load(f)["spikes"].astype(np.float32)        # (n_units, n_bins20)
        if raw.shape[1] < MAX_LAG + 50 or raw.shape[0] < 10:
            continue
        mu = raw.mean(axis=1, keepdims=True)
        sd = raw.std(axis=1, keepdims=True) + 1e-6
        zc = ((raw - mu) / sd)
        zc = zc - zc.mean(axis=1, keepdims=True)
        C = zc @ zc.T
        w, U = np.linalg.eigh(C)
        o = np.argsort(-w)
        w, U = np.clip(w[o], 0, None), U[:, o]
        d = min(64, zc.shape[0] - 1, zc.shape[1] - 1)
        proj = zc.T @ U[:, :d]                                # (n_bins, d) 方差 = 特征值
        ml = min(MAX_LAG, proj.shape[0] - 2)

        A = acf_cols(proj, ml)
        t1 = fit_tau(np.median(A, axis=1))
        t2 = fit_tau(A.mean(axis=1))
        t3 = fit_tau(weighted_acf(proj, ml))
        t4 = fit_tau(A[:, 0])

        V = sobi(proj)
        t5 = fit_tau(np.median(acf_cols(proj @ V, ml), axis=1)) if V is not None else None

        # 旋转检验
        r1, r3 = [], []
        for _ in range(N_ROT):
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            Pq = proj @ Q
            r1.append(fit_tau(np.median(acf_cols(Pq, ml), axis=1)))
            r3.append(fit_tau(weighted_acf(Pq, ml)))
        r1 = [v for v in r1 if v]
        r3 = [v for v in r3 if v]

        rows.append({
            "file": f.name,
            "n_units": int(raw.shape[0]),
            "n_bins": int(raw.shape[1]),
            "d": int(d),
            "ev_top1": float(w[0] / max(w[:d].sum(), 1e-12)),
            "ev_top5": float(w[:5].sum() / max(w[:d].sum(), 1e-12)),
            "rate": float(raw.mean()),
            "tau_T1_medianPC": t1,
            "tau_T2_meanPC": t2,
            "tau_T3_weighted": t3,
            "tau_T4_PC1": t4,
            "tau_T5_sobi": t5,
            "rot_T1_min": float(np.min(r1)) if r1 else None,
            "rot_T1_max": float(np.max(r1)) if r1 else None,
            "rot_T3_min": float(np.min(r3)) if r3 else None,
            "rot_T3_max": float(np.max(r3)) if r3 else None,
            "rot_T3_spread": float(np.ptp(r3)) if r3 else None,
            "acf1_pc1": float(A[1, 0]),
            "acf1_median": float(np.median(A[1])),
        })
        if k % 10 == 0:
            print(f"  [{k+1}/{len(files)}] {f.name} d={d} "
                  f"T1={t1} T3={t3} PC1={t4} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_spike.csv", index=False)

    def s(c):
        v = df[c].dropna()
        return {"n": int(len(v)), "median": round(float(v.median()), 1),
                "p10": round(float(v.quantile(.1)), 1),
                "p90": round(float(v.quantile(.9)), 1)}

    rep = {
        "task": "tau_gauge_spike_20ms",
        "dt_ms": DT, "n_segments": int(len(df)),
        "ev_top1_median": round(float(df.ev_top1.median()), 4),
        "tau": {k: s(f"tau_{k}") for k in
                ["T1_medianPC", "T2_meanPC", "T3_weighted", "T4_PC1", "T5_sobi"]},
        "acf1": {"pc1_median": round(float(df.acf1_pc1.median()), 4),
                 "allPC_median": round(float(df.acf1_median.median()), 4)},
        "rotation_T1": {
            "n": int(df.rot_T1_min.notna().sum()),
            "median_ratio_max_over_min": round(float(
                (df.rot_T1_max / df.rot_T1_min).replace([np.inf], np.nan).dropna().median()), 3),
            "max_ratio": round(float(
                (df.rot_T1_max / df.rot_T1_min).replace([np.inf], np.nan).dropna().max()), 3),
        },
        "rotation_T3": {
            "n": int(df.rot_T3_spread.notna().sum()),
            "median_abs_spread_ms": round(float(df.rot_T3_spread.median()), 4),
            "max_abs_spread_ms": round(float(df.rot_T3_spread.max()), 4),
        },
    }
    from scipy.stats import spearmanr
    m = df[["tau_T1_medianPC", "tau_T3_weighted", "tau_T4_PC1", "ev_top1"]].dropna()
    if len(m) > 5:
        rep["rank_corr"] = {
            "T1_vs_T3": round(float(spearmanr(m.tau_T1_medianPC, m.tau_T3_weighted)[0]), 3),
            "T1_vs_PC1": round(float(spearmanr(m.tau_T1_medianPC, m.tau_T4_PC1)[0]), 3),
            "T3_vs_PC1": round(float(spearmanr(m.tau_T3_weighted, m.tau_T4_PC1)[0]), 3),
            "ev_top1_vs_T1": round(float(spearmanr(m.ev_top1, m.tau_T1_medianPC)[0]), 3),
            "ev_top1_vs_PC1": round(float(spearmanr(m.ev_top1, m.tau_T4_PC1)[0]), 3),
            "n": int(len(m)),
        }
    (OUT / "report_spike.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
