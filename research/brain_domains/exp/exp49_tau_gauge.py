#!/usr/bin/env python3
"""exp49 — tau 是否依赖基的选取（规范依赖）？

旧方法（exp39/40/42/43）：
    a = median(acf_fft(P, max_lag), axis=1)     # 逐主成分 ACF 取中位数
    tau = fit_exp(a)

median 是非线性泛函，且逐成分 ACF 在 O(64) 旋转下不不变。
因此 tau 是「分析管线」的性质，不一定是「数据」的性质。

本实验对比四种量：

  A  tau_median_PCA   逐主成分 ACF 的中位数（复现旧方法）
  B  tau_trace        tr<x(t)x(t+tau)^T> / tr<x(t)x(t)^T>  —— 正交不变
  C  tau_median_ROT   随机正交旋转后再算 A（若 A 依赖基，这里会变）
  D  tau_median_SOBI  联合对角化 lag-0 与 lag-1 后的成分（基被数据本身固定）

判据：
  B 在旋转下不变（差 < 1e-6）          -> B 是规范不变量
  C 在旋转下改变                        -> A 不是规范不变量
  A 与 D 差异大                          -> 旧 tau 是 PCA 的产物

同时报告 tau_int（Sokal 积分时间，不做指数拟合假设）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DS = Path("/ds")
OUT = Path("/domains/runs/tau-gauge")
OUT.mkdir(parents=True, exist_ok=True)

DT_MS = 20.0
MAX_LAG = 100          # 2 s
N_ROT = 5
SEED = 0


# ------------------------------------------------------------------ 重建连续序列
def stitch(W, sub):
    """把 50% 重叠的窗拼回连续序列 (n_bins, 64)。

    窗 i 覆盖 proj[s_i : s_i+40]，s_i = 20 i。相邻窗共享 20 列，
    重叠部分数值完全相同，所以拼接是精确的，不是插值。
    """
    sub = sub.sort_values("win_in_segment")
    pos = sub.index.to_numpy()
    first = np.asarray(W[pos[0]], dtype=np.float32).T          # (40, 64)
    parts = [first]
    for k in range(1, len(pos)):
        w = np.asarray(W[pos[k]], dtype=np.float32).T
        parts.append(w[20:])
    return np.concatenate(parts, axis=0)


# ------------------------------------------------------------------ ACF
def acf_cols(x, max_lag):
    """逐列 ACF（无偏归一化）。x: (T, d) -> (max_lag+1, d)"""
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


def fit_tau(a, dt=DT_MS):
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.99)
    if ok.sum() < 4:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[150.0], bounds=([1.0], [20000.0]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


def tau_int(a, dt=DT_MS, c=5.0):
    taus = 2.0 * np.cumsum(a) - 1.0
    m = np.arange(len(taus)) < c * np.maximum(taus, 1e-9)
    idx = int(np.argmin(m)) if m.any() else len(taus) - 1
    return float(taus[idx] * dt)


def sobi(x, n_lags=3):
    """联合对角化 lag-0 与 lag-1..n_lags 的协方差，返回混合矩阵 V。

    解广义特征问题 C1 v = lam C0 v（取 lag-1），再用 lag-2..n_lags
    做一次 Jacobi 微调省略——单滞后已足以固定基。
    """
    T = x.shape[0]
    C0 = x.T @ x / T
    C1 = x[:-1].T @ x[1:] / (T - 1)
    C0 = (C0 + C0.T) / 2
    try:
        w, V = np.linalg.eigh(np.linalg.solve(C0, C1 + C1.T) / 2)
    except np.linalg.LinAlgError:
        return None, None
    order = np.argsort(-np.abs(w))
    return V[:, order], w[order]


# ------------------------------------------------------------------ 主流程
def main():
    t0 = time.time()
    W = np.load(DS / "timeseries.npy", mmap_mode="r")
    idx = pd.read_parquet(DS / "index_timeseries.parquet")
    print(f"tensor {W.shape} | index {idx.shape}", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    segs = idx.groupby("segment_idx")
    for n, (sid, sub) in enumerate(segs):
        P = stitch(W, sub)
        # 去掉零填充的轴（k < 64 的片段）
        nz = np.flatnonzero(np.abs(P).sum(axis=0) > 0)
        P = P[:, nz]
        d = P.shape[1]
        if d < 4 or P.shape[0] < MAX_LAG + 10:
            continue

        ml = min(MAX_LAG, P.shape[0] - 2)
        A = acf_cols(P, ml)                       # (ml+1, d)
        tau_med_pca = fit_tau(np.median(A, axis=1))
        ti_med_pca = tau_int(np.median(A, axis=1))
        trace = A.sum(axis=1) / d                 # tr 归一化
        tau_trace = fit_tau(trace)
        ti_trace = tau_int(trace)

        # 随机正交旋转
        rot_med, rot_trace = [], []
        for _ in range(N_ROT):
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            Ar = acf_cols(P @ Q, ml)
            rot_med.append(fit_tau(np.median(Ar, axis=1)))
            rot_trace.append(fit_tau(Ar.sum(axis=1) / d))

        # SOBI
        V, lam = sobi(P)
        tau_sobi = ti_sobi = None
        if V is not None:
            As = acf_cols(P @ V, ml)
            tau_sobi = fit_tau(np.median(As, axis=1))
            ti_sobi = tau_int(np.median(As, axis=1))

        # 单个 PCA 成分的 tau 分布（看离散程度）
        per_pc = [fit_tau(A[:, j]) for j in range(d)]
        per_pc = [v for v in per_pc if v is not None]

        rows.append({
            "segment_idx": int(sid),
            "source": sub.source.iloc[0],
            "region": sub.region.iloc[0],
            "shared_node": sub.shared_node.iloc[0],
            "n_units": int(sub.n_units.iloc[0]),
            "d": int(d),
            "n_bins": int(P.shape[0]),
            "n_win": int(len(sub)),
            "tau_median_pca": tau_med_pca,
            "tau_trace": tau_trace,
            "tau_int_median_pca": ti_med_pca,
            "tau_int_trace": ti_trace,
            "tau_median_pca_rot_med": float(np.median(rot_med)),
            "tau_median_pca_rot_min": float(np.min(rot_med)),
            "tau_median_pca_rot_max": float(np.max(rot_med)),
            "tau_trace_rot_max_absdiff": float(np.max(np.abs(
                np.array(rot_trace, dtype=float) - (tau_trace or np.nan)))),
            "tau_median_sobi": tau_sobi,
            "tau_int_median_sobi": ti_sobi,
            "per_pc_tau_p10": float(np.percentile(per_pc, 10)) if per_pc else None,
            "per_pc_tau_p50": float(np.percentile(per_pc, 50)) if per_pc else None,
            "per_pc_tau_p90": float(np.percentile(per_pc, 90)) if per_pc else None,
            "tau_per_window": float((tau_trace or np.nan) / (40 * DT_MS)),
        })
        if n % 20 == 0:
            print(f"  [{n+1}/{len(segs)}] seg {sid} d={d} bins={P.shape[0]} "
                  f"tau_pca={tau_med_pca} trace={tau_trace} sobi={tau_sobi} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)

    # ------------------------------------------------ 汇总
    def stats(col):
        v = df[col].dropna()
        return {"n": int(len(v)), "median": round(float(v.median()), 1),
                "p10": round(float(v.quantile(.1)), 1),
                "p90": round(float(v.quantile(.9)), 1)}

    rot_spread = (df.tau_median_pca_rot_max - df.tau_median_pca_rot_min).dropna()
    rot_ratio = (df.tau_median_pca_rot_max / df.tau_median_pca_rot_min).dropna()
    report = {
        "task": "tau_gauge_dependence",
        "dt_ms": DT_MS,
        "max_lag_bins": MAX_LAG,
        "n_rotations": N_ROT,
        "n_segments": int(len(df)),
        "tau_median_pca": stats("tau_median_pca"),
        "tau_trace": stats("tau_trace"),
        "tau_median_sobi": stats("tau_median_sobi"),
        "tau_int_median_pca": stats("tau_int_median_pca"),
        "tau_int_trace": stats("tau_int_trace"),
        "tau_int_median_sobi": stats("tau_int_median_sobi"),
        "rotation_test": {
            "n_segments": int(len(rot_spread)),
            "median_abs_change_ms": round(float(rot_spread.median()), 2),
            "max_abs_change_ms": round(float(rot_spread.max()), 2),
            "median_ratio_max_over_min": round(float(rot_ratio.median()), 3),
            "max_ratio_max_over_min": round(float(rot_ratio.max()), 3),
            "trace_max_absdiff": round(float(df.tau_trace_rot_max_absdiff.max()), 6),
        },
        "per_pc_spread": {
            "median_p10": round(float(df.per_pc_tau_p10.median()), 1),
            "median_p50": round(float(df.per_pc_tau_p50.median()), 1),
            "median_p90": round(float(df.per_pc_tau_p90.median()), 1),
        },
        "tau_per_window": {
            "median": round(float(df.tau_per_window.median()), 2),
            "p10": round(float(df.tau_per_window.quantile(.1)), 2),
            "p90": round(float(df.tau_per_window.quantile(.9)), 2),
        },
        "by_source": {},
    }
    for s, g in df.groupby("source"):
        report["by_source"][s] = {
            "n_segments": int(len(g)),
            "tau_median_pca": round(float(g.tau_median_pca.median()), 1),
            "tau_trace": round(float(g.tau_trace.median()), 1),
            "tau_median_sobi": round(float(g.tau_median_sobi.median()), 1),
        }

    # 秩相关：旧的 tau 与不变量的 tau 是否一致
    m = df[["tau_median_pca", "tau_trace", "tau_median_sobi"]].dropna()
    if len(m) > 5:
        from scipy.stats import spearmanr
        report["rank_correlation"] = {
            "pca_vs_trace": round(float(spearmanr(m.tau_median_pca, m.tau_trace)[0]), 3),
            "pca_vs_sobi": round(float(spearmanr(m.tau_median_pca, m.tau_median_sobi)[0]), 3),
            "sobi_vs_trace": round(float(spearmanr(m.tau_median_sobi, m.tau_trace)[0]), 3),
            "n": int(len(m)),
        }

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
