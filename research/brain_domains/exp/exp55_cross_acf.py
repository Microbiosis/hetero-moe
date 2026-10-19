#!/usr/bin/env python3
"""exp55 — 把不应期从群体动力学里拆出去。

exp54 的发现：

    acf1 = 0.14    acf2 = 0.19    acf100 = 0.00

lag-2 高于 lag-1，ACF 在第一个 bin 处凹陷再回升。指数拟合对非单调曲线
没有意义，所以那些 τ 数值（15.6 / 491 / 766 ms）都是拟合器的产物。

原因：trace-ACF 把两类东西加在一起

    tr C(τ) = Σ_i ⟨x_i(t)x_i(t+τ)⟩        单单元自相关：不应期 -> lag 1 处为负
            + Σ_{i≠j} ⟨x_i(t)x_j(t+τ)⟩    跨单元互相关：群体状态 -> 慢、正

所以本脚本算**去掉对角项**的互相关：

    R_cross(τ) = Σ_{i≠j} ⟨x_i x_j(τ)⟩ / Σ_{i≠j} ⟨x_i x_j(0)⟩

只保留群体耦合。同时报告：
  - R_total 与 R_cross 的单调性（lag1..lag10 有多少次上升）
  - 各自的 lag-1、lag-2 值
  - 单指数与幂律两种拟合，以及拟合区间敏感性
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/tau-cross")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500


def acf_parts(x, max_lag):
    """返回 (total_tr, diag_tr) 两条未归一化的 ACF 曲线。"""
    T, d = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    F = np.fft.rfft(xc, n=n, axis=0)                     # (freq, d)
    # 全和必须「先按通道求和再取模方」，否则等于逐通道模方之和，与 diag 完全相同
    Fs = F.sum(axis=1)
    total = np.fft.irfft(Fs * np.conj(Fs), n=n)[:max_lag + 1]
    diag = np.fft.irfft((np.abs(F) ** 2).sum(axis=1), n=n)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)
    total = total / np.maximum(norms, 1)
    diag = diag / np.maximum(norms, 1)
    return total, diag


def cross_acf(x, max_lag):
    """Σ_{i≠j} ⟨x_i x_j(τ)⟩ / Σ_{i≠j} ⟨x_i x_j(0)⟩。"""
    total, diag = acf_parts(x, max_lag)
    cross = total - diag
    # 归一化用 cross(0) 而不是 total(0)：归一化后 R_cross(0) = 1，衡量的是群体耦合的衰减
    return (cross / max(cross[0], 1e-12),
            total / max(total[0], 1e-12),
            diag / max(diag[0], 1e-12))


def n_rises(a, k=10):
    """lag 1..k 内上升的次数。单调下降应为 0。"""
    seg = a[1:k + 1]
    return int((np.diff(seg) > 0).sum())


def fit_exp(a, dt=DT, lo=2.0, hi=100000.0):
    ok = np.isfinite(a) & (a > 0.01) & (a < 0.999)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, A, T):
        return A * np.exp(-tt / np.maximum(T, 1e-3))

    try:
        p, _ = curve_fit(f, t, y, p0=[float(y[0]), 200.0],
                         bounds=([1e-4, lo], [1.0, hi]), maxfev=20000)
        return float(p[1])
    except Exception:
        return None


def fit_pow(a, dt=DT, lo=2.0, hi=1.0):
    """R(τ) = A·τ^(−β)。返回 (beta, 有效时间常数 τ_e = dt·e^(1/β))。"""
    ok = np.isfinite(a) & (a > 0.01) & (a < 0.999)
    if ok.sum() < 6:
        return None, None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]
    try:
        b, logA = np.polyfit(np.log(t), np.log(y), 1)
        beta = -float(b)
        if beta <= 0:
            return None, None
        return beta, float(dt * np.exp(1.0 / beta))
    except Exception:
        return None, None


def fit_range_sensitivity(a, dt=DT):
    """用不同的最大 lag 拟合单指数，看 T 变化多少倍。"""
    out = {}
    for frac in (0.1, 0.25, 0.5, 1.0):
        k = max(8, int(len(a) * frac))
        out[str(frac)] = fit_exp(a[:k], dt)
    vals = [v for v in out.values() if v]
    if len(vals) >= 2:
        out["ratio_max_over_min"] = round(float(max(vals) / max(min(vals), 1e-9)), 2)
    return out


def main():
    t0 = time.time()
    W = np.load(V3 / "counts.npy", mmap_mode="r")
    idx = pd.read_parquet(V3 / "counts_index.parquet")
    rows = []
    for n, (_, r) in enumerate(idx.iterrows()):
        x = np.asarray(W[r.start:r.end, :r.n_units], dtype=np.float32)
        if x.shape[0] < 1000:
            continue
        rc, rt, rd = cross_acf(x, MAX_LAG)
        beta, tau_e = fit_pow(rc)
        sens = fit_range_sensitivity(rc)
        rows.append({
            "source": r.source, "region": r.region, "file": r.file,
            "n_units": int(r.n_units), "n_bins": int(x.shape[0]),
            "hours": round(x.shape[0] * DT / 3.6e6, 3),
            "rate": float(x.mean()),
            # 三种曲线
            "cross_lag1": round(float(rc[1]), 4),
            "cross_lag2": round(float(rc[2]), 4),
            "cross_lag50": round(float(rc[50]), 4),
            "total_lag1": round(float(rt[1]), 4),
            "total_lag2": round(float(rt[2]), 4),
            "diag_lag1": round(float(rd[1]), 4),
            "diag_lag2": round(float(rd[2]), 4),
            # 单调性
            "cross_rises": n_rises(rc), "total_rises": n_rises(rt),
            # 拟合
            "tau_exp_cross": fit_exp(rc), "tau_exp_total": fit_exp(rt),
            "pow_beta_cross": beta, "pow_tau_e_cross": tau_e,
            "sens": json.dumps(sens),
        })
        if n % 20 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} c1={rc[1]:.3f} c2={rc[2]:.3f} "
                  f"rises={rows[-1]['cross_rises']} beta={beta and round(beta,2)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        del x
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_segment.csv", index=False)

    rep = {"task": "cross_acf", "dt_ms": DT, "n_segments": int(len(df))}

    def s(c, g=df):
        v = g[c].dropna()
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(v.median()), 3),
                "p10": round(float(v.quantile(.1)), 3),
                "p90": round(float(v.quantile(.9)), 3)}

    rep["overall"] = {c: s(c) for c in
                      ["cross_lag1", "cross_lag2", "total_lag1", "total_lag2",
                       "diag_lag1", "diag_lag2", "cross_rises", "total_rises",
                       "tau_exp_cross", "tau_exp_total", "pow_beta_cross",
                       "pow_tau_e_cross"]}
    rep["monotonic_fraction"] = {
        "cross_monotone": round(float((df.cross_rises == 0).mean()), 3),
        "total_monotone": round(float((df.total_rises == 0).mean()), 3),
    }
    rep["by_source"] = {src: {c: s(c, g) for c in
                              ["cross_lag1", "cross_lag2", "tau_exp_cross",
                               "pow_beta_cross", "pow_tau_e_cross", "cross_rises", "rate"]}
                        for src, g in df.groupby("source")}
    # 拟合区间敏感性
    rr = [json.loads(c).get("ratio_max_over_min") for c in df.sens]
    rr = [v for v in rr if v]
    if rr:
        rep["fit_range_sensitivity"] = {
            "n": len(rr), "median_ratio": round(float(np.median(rr)), 2),
            "p90_ratio": round(float(np.percentile(rr, 90)), 2),
            "note": "同一曲线用 10%/25%/50%/100% 的 lag 范围拟合单指数，T 的极差倍数"}

    g1 = df[(df.region == "M1") & (df.source == "macaque_nlb")]
    g2 = df[(df.region == "M1") & (df.source == "human_intracortical")]
    cmp = {"n_macaque": int(len(g1)), "n_human": int(len(g2))}
    for c in ["cross_lag1", "tau_exp_cross", "pow_beta_cross", "pow_tau_e_cross"]:
        v1, v2 = g1[c].dropna(), g2[c].dropna()
        if len(v1) > 2 and len(v2) > 2:
            cmp[c] = {"macaque": round(float(v1.median()), 3),
                      "human": round(float(v2.median()), 3),
                      "ratio": round(float(v1.median() / max(v2.median(), 1e-9)), 2),
                      "p": float(mannwhitneyu(v1, v2).pvalue)}
    rep["Q_species_M1"] = cmp

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
