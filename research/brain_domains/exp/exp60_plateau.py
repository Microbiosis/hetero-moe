#!/usr/bin/env python3
"""exp60 — 用「快速指数 + 平台」重拟合。

exp58 的双指数里，tau_slow 在 78.6% 的片段上撞到 10 s 上界，
即 exp(−τ/tau_slow) ≈ 常数。所以真实形式是

    ρ(τ) = A·exp(−τ/t) + c

三个参数，t 可识别，c 是 10 s 内不衰减的部分。

本脚本用这个模型重拟合，并与双指数比较 AIC。若 AIC 相当（4 参 vs 3 参，
ΔAIC < 2 即认为无改进），则平台模型是正确的描述，双指数的 tau_slow 是伪参数。

同时报告：
  - A、t、c 的中位数与按物种/任务分组
  - t 与放电率、c 与放电率的相关
  - 匹配放电率后的物种差异（t 与 c）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, spearmanr

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/plateau")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500
N_NULL = 3
SEED = 0


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
    return (tot - dia) / t0, float((tot[0] - dia[0]) / t0)


def null_noise(x, ml, rng, n=N_NULL):
    prof = []
    for _ in range(n):
        xs = np.empty_like(x)
        for j in range(x.shape[1]):
            xs[:, j] = np.roll(x[:, j], int(rng.integers(1, x.shape[0])))
        r, _ = rho_curve(xs, ml)
        prof.append(r)
    return float(np.sqrt((np.array(prof)[:, 2:] ** 2).mean()))


def m_plateau(t, A, tau, c):
    return A * np.exp(-t / np.maximum(tau, 1e-6)) + c


def m_two(t, A1, t1, A2, t2):
    return A1 * np.exp(-t / np.maximum(t1, 1e-6)) + A2 * np.exp(-t / np.maximum(t2, 1e-6))


def fit_both(t, y):
    out = {}
    n = len(t)
    for name, fn, p0, bd in [
        ("plateau", m_plateau, [float(y[0]), 200.0, max(float(y[-1]), 0.0)],
         ([0.0, 2.0, -1.0], [10.0, 1e5, 1.0])),
        ("two_exp", m_two, [float(y[0]) * 0.6, 100.0, float(y[0]) * 0.4, 5000.0],
         ([0.0, 2.0, 0.0, 2.0], [10.0, 1e5, 10.0, 1e7])),
    ]:
        try:
            p, _ = curve_fit(fn, t, y, p0=p0, bounds=bd, maxfev=60000)
            r = y - fn(t, *p)
            out[name] = {"params": [float(v) for v in p],
                         "aic": float(n * np.log(np.mean(r ** 2) + 1e-20)),
                         "r2": float(1 - np.var(r) / max(np.var(y), 1e-20))}
        except Exception:
            pass
    return out


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
    import time
    t0 = time.time()
    W = np.load(V3 / "counts.npy", mmap_mode="r")
    idx = pd.read_parquet(V3 / "counts_index.parquet")
    rng = np.random.default_rng(SEED)
    rows = []
    for n, (_, r) in enumerate(idx.iterrows()):
        x = np.asarray(W[r.bin_start:r.bin_end, :r.n_units], dtype=np.float32)
        if x.shape[0] < 1200 or x.shape[1] < 8:
            continue
        ml = min(MAX_LAG, x.shape[0] - 2)
        noise = null_noise(x, ml, rng)
        rho, shared = rho_curve(x, ml)
        snr = float(rho[1] / max(noise, 1e-12))
        usable = np.flatnonzero((rho > 2 * noise) & np.isfinite(rho))
        usable = usable[(usable >= 2) & (usable <= ml)]
        if snr < 3.0 or len(usable) < 10:
            print(f"  [{n+1}/{len(idx)}] {r.file} DROP snr={snr:.1f}", flush=True)
            del x
            continue
        t = usable.astype(float) * DT
        y = rho[usable]
        f = fit_both(t, y)
        best = min(f, key=lambda k: f[k]["aic"]) if f else None
        pl = f.get("plateau")
        tw = f.get("two_exp")
        rows.append({
            "source": r.source, "task": task_of(r.source, r.file),
            "file": r.file, "region": r.region,
            "n_units": int(x.shape[1]), "n_bins": int(x.shape[0]),
            "hours": round(x.shape[0] * DT / 3.6e6, 3),
            "rate": float(x.mean()), "fano": float(x.var() / max(x.mean(), 1e-9)),
            "shared_frac": round(float(shared), 5), "snr": round(snr, 1),
            "n_points": int(len(usable)),
            "A": pl["params"][0] if pl else None,
            "tau_ms": pl["params"][1] if pl else None,
            "c": pl["params"][2] if pl else None,
            "r2_plateau": round(pl["r2"], 4) if pl else None,
            "aic_plateau": round(pl["aic"], 2) if pl else None,
            "aic_two": round(tw["aic"], 2) if tw else None,
            "best": best,
        })
        if n % 15 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} snr={snr:.0f} best={best} "
                  f"tau={rows[-1]['tau_ms'] and round(rows[-1]['tau_ms'],1)} "
                  f"c={rows[-1]['c'] and round(rows[-1]['c'],3)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        del x

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_segment.csv", index=False)
    rep = {"task": "plateau_model", "dt_ms": DT, "n": int(len(d))}

    def s(c, g=d):
        v = g[c].dropna()
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(v.median()), 2),
                "p10": round(float(v.quantile(.1)), 2),
                "p90": round(float(v.quantile(.9)), 2)}

    rep["model_wins"] = d.best.value_counts().to_dict()
    rep["aic_delta_two_minus_plateau"] = {
        "median": round(float((d.aic_two - d.aic_plateau).median()), 1),
        "note": "> 0 表示平台模型（3 参）更好或相当"}
    rep["params"] = {c: s(c) for c in ["A", "tau_ms", "c", "r2_plateau", "shared_frac"]}
    rep["by_task"] = {k: {"n": int(len(g)),
                          **{c: s(c, g) for c in ["A", "tau_ms", "c", "rate", "snr"]}}
                      for k, g in d.groupby("task")}

    mac = d[d.source == "macaque_nlb"]
    hum = d[d.source == "human_intracortical"]
    rep["species"] = {}
    for c in ["tau_ms", "c", "A", "shared_frac"]:
        a, b = mac[c].dropna(), hum[c].dropna()
        if len(a) > 5 and len(b) > 5:
            rep["species"][c] = {"macaque": round(float(a.median()), 3),
                                 "human": round(float(b.median()), 3),
                                 "ratio": round(float(a.median() / max(b.median(), 1e-9)), 2),
                                 "p": float(mannwhitneyu(a, b).pvalue)}
    # 匹配放电率（低率层：人类全部落在 0.13–0.40）
    lo, hi = float(hum.rate.quantile(.05)), float(hum.rate.quantile(.95))
    mm = mac[(mac.rate >= lo) & (mac.rate <= hi)]
    hh = hum[(hum.rate >= lo) & (hum.rate <= hi)]
    rep["rate_matched"] = {"rate_range": [round(lo, 3), round(hi, 3)],
                           "n_macaque": int(len(mm)), "n_human": int(len(hh))}
    for c in ["tau_ms", "c"]:
        a, b = mm[c].dropna(), hh[c].dropna()
        if len(a) > 3 and len(b) > 3:
            rep["rate_matched"][c] = {
                "macaque": round(float(a.median()), 2),
                "human": round(float(b.median()), 2),
                "ratio": round(float(a.median() / max(b.median(), 1e-9)), 2),
                "p": float(mannwhitneyu(a, b).pvalue)}
    # 相关
    rep["correlates"] = {}
    for c in ["rate", "snr", "shared_frac", "n_units", "hours", "fano"]:
        for y in ["tau_ms", "c"]:
            m = d[[y, c]].dropna()
            if len(m) > 10:
                r_ = spearmanr(m[y], m[c])
                rep["correlates"][f"{y}_vs_{c}"] = {"rho": round(float(r_[0]), 3),
                                                    "p": float(r_[1]), "n": int(len(m))}
    # 猕猴内部
    mt = mac[mac.task.isin(["reach", "timing", "bump"])]
    rep["macaque_by_task"] = {k: {"n": int(len(g)),
                                  "rate": round(float(g.rate.median()), 3),
                                  "tau_ms": round(float(g.tau_ms.median()), 1),
                                  "c": round(float(g.c.median()), 3)}
                              for k, g in mt.groupby("task")}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
