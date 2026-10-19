#!/usr/bin/env python3
"""exp50 — 头皮 EEG 上 tau 的规范依赖（250 Hz 原生，dt = 4 ms）。

复现 exp39/40/42/43 的旧流程：
    每个图像：80 次重复沿时间拼成 (80*250, 63)
    逐通道 z-score -> SVD -> 取前 32 个成分得分 P
    a = median(acf_fft(P, 250), axis=1)          <- 逐成分 ACF 取中位数
    tau = fit_exp(a)

然后在同一批 P 上算五个量，并做旋转检验：

  T1 median per-PC     旧方法
  T2 mean per-PC       等权平均
  T3_32 trace-weighted tr C(tau)/tr C(0)，只用前 32 维子空间
  T3_63 trace-weighted 全部 63 通道（真正的不变量）
  T4 PC1
  T5 SOBI 成分的中位数

判据：
  T3_63 在旋转下必须不变（差 < 1e-9）
  T1 在旋转下的变化幅度 = 旧 tau 的不可靠程度
  T1 与 T3_63 的比值 = 旧 tau 偏离真值的倍数
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

EEG = Path("/domains/omnibrain4d/eeg")
OUT = Path("/domains/runs/tau-gauge")
OUT.mkdir(parents=True, exist_ok=True)

FS = 250.0
DT = 1000.0 / FS          # 4 ms
MAX_LAG = 250             # 1000 ms
N_PCS = 32                # exp42 的值
N_IMG = 12                # 每被试取样图像数
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
N_ROT = 5
SEED = 0


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


def weighted_acf(x, max_lag):
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    acf = acf / np.maximum(norms, 1)
    return acf.sum(axis=1) / max(acf[0].sum(), 1e-12)


def fit_tau(a, dt=DT):
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.995)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[150.0], bounds=([1.0], [50000.0]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


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
    for sm in SUBJECTS:
        f = EEG / f"{sm}.npz"
        if not f.exists():
            print(f"  [skip] {f}", flush=True)
            continue
        eeg = np.load(f)["eeg"]              # (200, 80, 63, 250) float16
        n_img, n_rep, n_ch, T = eeg.shape
        idx_img = np.linspace(0, n_img - 1, min(N_IMG, n_img)).astype(int)
        print(f"{sm}: {eeg.shape} -> {len(idx_img)} images", flush=True)
        for ii in idx_img:
            seg = eeg[ii].astype(np.float32)                  # (n_rep, 63, 250)
            X = seg.transpose(1, 0, 2).reshape(n_ch, n_rep * T)
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True) + 1e-6
            Z = ((X - mu) / sd).T                             # (n_rep*T, 63)
            Zc = Z - Z.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Zc, full_matrices=False)
            d = min(N_PCS, n_ch)
            P = (U[:, :d] * S[:d]).astype(np.float64)
            ev = (S[:d] ** 2) / np.maximum((S ** 2).sum(), 1e-12)
            ml = min(MAX_LAG, Zc.shape[0] - 2)

            A = acf_cols(P, ml)
            t1 = fit_tau(np.median(A, axis=1))
            t2 = fit_tau(A.mean(axis=1))
            t3_32 = fit_tau(weighted_acf(P, ml))
            t3_63 = fit_tau(weighted_acf(Zc, ml))
            t4 = fit_tau(A[:, 0])

            V = sobi(P)
            t5 = fit_tau(np.median(acf_cols(P @ V, ml), axis=1)) if V is not None else None

            r1, r3 = [], []
            for _ in range(N_ROT):
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                Pq = P @ Q
                r1.append(fit_tau(np.median(acf_cols(Pq, ml), axis=1)))
                r3.append(fit_tau(weighted_acf(Pq, ml)))
            r1 = [v for v in r1 if v]
            r3 = [v for v in r3 if v]

            rows.append({
                "subject": sm, "image_idx": int(ii),
                "n_ch": n_ch, "d": d, "n_samples": int(Zc.shape[0]),
                "ev_top1": float(ev[0]),
                "ev_top5": float(ev[:5].sum()),
                "ev_top32": float(ev[:d].sum()),
                "tau_T1_medianPC": t1,
                "tau_T2_meanPC": t2,
                "tau_T3_top32": t3_32,
                "tau_T3_all63": t3_63,
                "tau_T4_PC1": t4,
                "tau_T5_sobi": t5,
                "rot_T1_min": float(np.min(r1)) if r1 else None,
                "rot_T1_max": float(np.max(r1)) if r1 else None,
                "rot_T3_spread": float(np.ptp(r3)) if r3 else None,
                "acf1_pc1": float(A[1, 0]),
                "acf1_median": float(np.median(A[1])),
            })
        print(f"  done {sm} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_eeg.csv", index=False)

    def s(c):
        v = df[c].dropna()
        return {"n": int(len(v)), "median": round(float(v.median()), 1),
                "p10": round(float(v.quantile(.1)), 1),
                "p90": round(float(v.quantile(.9)), 1)}

    rep = {
        "task": "tau_gauge_eeg_250hz",
        "dt_ms": DT, "n_pcs": N_PCS, "n_records": int(len(df)),
        "n_subjects": int(df.subject.nunique()),
        "ev": {
            "top1_median": round(float(df.ev_top1.median()), 4),
            "top5_median": round(float(df.ev_top5.median()), 4),
            "top32_median": round(float(df.ev_top32.median()), 4),
        },
        "acf1": {
            "pc1_median": round(float(df.acf1_pc1.median()), 4),
            "allPC_median": round(float(df.acf1_median.median()), 4),
        },
        "tau": {k: s(f"tau_{k}") for k in
                ["T1_medianPC", "T2_meanPC", "T3_top32", "T3_all63", "T4_PC1", "T5_sobi"]},
        "rotation_T1": {
            "median_ratio_max_over_min": round(float(
                (df.rot_T1_max / df.rot_T1_min).replace([np.inf], np.nan).dropna().median()), 3),
            "max_ratio": round(float(
                (df.rot_T1_max / df.rot_T1_min).replace([np.inf], np.nan).dropna().max()), 3),
        },
        "rotation_T3_top32": {
            "median_abs_spread_ms": round(float(df.rot_T3_spread.median()), 6),
            "max_abs_spread_ms": round(float(df.rot_T3_spread.max()), 6),
        },
    }
    from scipy.stats import spearmanr
    m = df[["tau_T1_medianPC", "tau_T3_all63", "tau_T4_PC1", "ev_top1"]].dropna()
    if len(m) > 5:
        rep["rank_corr"] = {
            "T1_vs_T3all": round(float(spearmanr(m.tau_T1_medianPC, m.tau_T3_all63)[0]), 3),
            "T1_vs_PC1": round(float(spearmanr(m.tau_T1_medianPC, m.tau_T4_PC1)[0]), 3),
            "T3all_vs_PC1": round(float(spearmanr(m.tau_T3_all63, m.tau_T4_PC1)[0]), 3),
            "ev_top1_vs_T1": round(float(spearmanr(m.ev_top1, m.tau_T1_medianPC)[0]), 3),
            "n": int(len(m)),
        }
    # 逐被试
    per = df.groupby("subject").agg(
        T1=("tau_T1_medianPC", "median"), T3all=("tau_T3_all63", "median"),
        PC1=("tau_T4_PC1", "median"), ev1=("ev_top1", "median")).round(1)
    rep["per_subject"] = per.to_dict("index")
    (OUT / "report_eeg.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
