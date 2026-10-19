#!/usr/bin/env python3
"""exp57 — 三个模型的比较 + SNR 筛选。

exp56b 的两个发现：

1. 局域斜率随 lag 单调上升：0.112 (lag 2-5) → 0.589 (lag 50-100)。
   幂律要求斜率恒定，所以"幂律"不成立；曲线在 log-log 上是凹的。
   凹的 log-log 正是拉伸指数 exp(−(τ/τ0)^β)（β<1）的特征。

2. H1（人类七自由度抓取）的 ρ(1) = 0.0057，噪声底 0.0166，SNR < 1。
   在 SNR < 1 的数据上拟合任何模型都只是在拟合噪声。
   因此必须按 SNR 筛选，并报告被筛掉的比例——那本身是结论。

本脚本：
  - 对每个片段算噪声底（独立循环移位零模型）
  - 只保留 SNR = ρ(1)/noise ≥ 3 且可用点 ≥ 10 的片段
  - 拟合三个模型，比较 AIC
      M1 幂律       ρ = A·τ^(−α)
      M2 拉伸指数   ρ = A·exp(−(τ/τ0)^β)
      M3 双指数     ρ = A1·exp(−τ/τ1) + A2·exp(−τ/τ2)
  - 报告各模型胜出比例、参数中位数、按来源与被试分组
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kruskal, mannwhitneyu, spearmanr

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/models")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500
N_NULL = 3
SEED = 0
SNR_MIN = 3.0
MIN_PTS = 10


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
    total = np.fft.irfft(Fs * np.conj(Fs), n=N)[:max_lag + 1]
    diag = np.fft.irfft((np.abs(F) ** 2).sum(axis=1), n=N)[:max_lag + 1]
    nrm = np.arange(T, T - max_lag - 1, -1)
    total = total / np.maximum(nrm, 1)
    diag = diag / np.maximum(nrm, 1)
    t0 = max(total[0], 1e-12)
    return (total - diag) / t0, float((total[0] - diag[0]) / t0)


def null_noise(x, max_lag, rng, n=N_NULL):
    prof = []
    for _ in range(n):
        xs = np.empty_like(x)
        for j in range(x.shape[1]):
            xs[:, j] = np.roll(x[:, j], int(rng.integers(1, x.shape[0])))
        r, _ = rho_curve(xs, max_lag)
        prof.append(r)
    prof = np.array(prof)
    return float(np.sqrt((prof[:, 2:] ** 2).mean()))


# ---------------- 模型
def m_power(t, A, a):
    return A * np.power(t, -a)


def m_stretch(t, A, t0, b):
    return A * np.exp(-np.power(t / np.maximum(t0, 1e-6), b))


def m_two(t, A1, t1, A2, t2):
    return A1 * np.exp(-t / np.maximum(t1, 1e-6)) + A2 * np.exp(-t / np.maximum(t2, 1e-6))


def fit_models(t, y):
    """返回 {model: (params, aic, r2)}。"""
    out = {}
    n = len(t)
    # M1
    try:
        p, _ = curve_fit(m_power, t, y, p0=[y[0], 0.3],
                         bounds=([1e-6, 0.01], [10.0, 3.0]), maxfev=20000)
        r = y - m_power(t, *p)
        out["power"] = (list(p), float(n * np.log(np.mean(r ** 2) + 1e-20)),
                        1 - np.var(r) / max(np.var(y), 1e-20))
    except Exception:
        pass
    # M2
    try:
        p, _ = curve_fit(m_stretch, t, y, p0=[y[0], 200.0, 0.5],
                         bounds=([1e-6, 5.0, 0.05], [10.0, 1e6, 2.0]), maxfev=40000)
        r = y - m_stretch(t, *p)
        out["stretch"] = (list(p), float(n * np.log(np.mean(r ** 2) + 1e-20)),
                          1 - np.var(r) / max(np.var(y), 1e-20))
    except Exception:
        pass
    # M3
    try:
        p, _ = curve_fit(m_two, t, y,
                         p0=[y[0] * 0.6, 100.0, y[0] * 0.4, 1500.0],
                         bounds=([0.0, 5.0, 0.0, 5.0], [10.0, 1e5, 10.0, 1e7]),
                         maxfev=60000)
        r = y - m_two(t, *p)
        out["two_exp"] = (list(p), float(n * np.log(np.mean(r ** 2) + 1e-20)),
                          1 - np.var(r) / max(np.var(y), 1e-20))
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


def local_slope(rho, noise):
    out = {}
    for a, b in [(2, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500)]:
        if b >= len(rho) or rho[a] <= 1e-9 or rho[b] <= 1e-9:
            continue
        if rho[a] < 2 * noise or rho[b] < 2 * noise:
            continue
        out[f"{a}-{b}"] = round(float(
            -(np.log(rho[b]) - np.log(rho[a])) / (np.log(b) - np.log(a))), 3)
    return out


def main():
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
        base = {
            "source": r.source, "task": task_of(r.source, r.file),
            "file": r.file, "region": r.region,
            "n_units": int(x.shape[1]), "n_bins": int(x.shape[0]),
            "hours": round(x.shape[0] * DT / 3.6e6, 3),
            "rate": float(x.mean()), "fano": float(x.var() / max(x.mean(), 1e-9)),
            "shared_frac": round(float(shared), 5),
            "rho1": round(float(rho[1]), 5),
            "noise": round(noise, 6), "snr": round(snr, 1),
            "n_usable": int(len(usable)),
            "usable_max_lag": int(usable.max()) if len(usable) else None,
        }

        if snr < SNR_MIN or len(usable) < MIN_PTS:
            rows.append({**base, "kept": False, "model": None})
            if n % 10 == 0:
                print(f"  [{n+1}/{len(idx)}] {r.file} DROP snr={snr:.1f} "
                      f"n_usable={len(usable)} [{time.time()-t0:.0f}s]", flush=True)
            del x
            continue

        t = usable.astype(float) * DT
        y = rho[usable]
        fits = fit_models(t, y)
        best = min(fits, key=lambda k: fits[k][1]) if fits else None
        ls = local_slope(rho, noise)

        rows.append({
            **base, "kept": True, "model": best,
            "n_points": int(len(usable)),
            "fit": json.dumps({k: {"params": [round(v, 5) for v in v[0]],
                                   "aic": round(v[1], 2),
                                   "r2": round(float(v[2]), 4)}
                               for k, v in fits.items()}),
            "meta_2x2_stretch": round(float(
                fits["stretch"][0][1] * DT), 1) if "stretch" in fits else None,
            "beta_stretch": round(float(fits["stretch"][0][2]), 3) if "stretch" in fits else None,
            "alpha_power": round(float(fits["power"][0][1]), 3) if "power" in fits else None,
            "local_slopes": json.dumps(ls),
        })
        if n % 10 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} snr={snr:.1f} best={best} "
                  f"beta_s={rows[-1]['beta_stretch']} [{time.time()-t0:.0f}s]", flush=True)
        del x

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_segment.csv", index=False)

    kept = df[df.kept == True]
    rep = {"task": "model_comparison", "dt_ms": DT, "n_segments": int(len(df)),
           "snr_min": SNR_MIN, "min_points": MIN_PTS,
           "n_kept": int(len(kept)),
           "drop_by_source": df[df.kept == False].groupby("source").size().to_dict(),
           "keep_by_source": kept.groupby("source").size().to_dict(),
           "keep_by_task": kept.groupby("task").size().to_dict()}

    rep["snr_by_task"] = {k: {"n": int(len(g)),
                              "median": round(float(g.snr.median()), 1),
                              "p10": round(float(g.snr.quantile(.1)), 1),
                              "p90": round(float(g.snr.quantile(.9)), 1)}
                          for k, g in df.groupby("task")}

    # 模型胜出
    if len(kept):
        rep["model_wins"] = kept.model.value_counts().to_dict()
        rep["model_wins_by_task"] = {
            k: g.model.value_counts().to_dict() for k, g in kept.groupby("task")}
        rep["params"] = {}
        for c in ["beta_stretch", "meta_2x2_stretch", "alpha_power"]:
            v = kept[c].dropna()
            if len(v):
                rep["params"][c] = {"n": int(len(v)),
                                    "median": round(float(v.median()), 3),
                                    "p10": round(float(v.quantile(.1)), 3),
                                    "p90": round(float(v.quantile(.9)), 3)}
        # 局域斜率（仅保留片段）
        LS = pd.DataFrame([json.loads(s) for s in kept.local_slopes])
        rep["local_slope"] = {c: {"n": int(LS[c].notna().sum()),
                                  "median": round(float(LS[c].median()), 3)}
                              for c in LS.columns if LS[c].notna().sum() >= 5}
        # 按任务
        rep["by_task"] = {k: {
            "n": int(len(g)),
            "beta_stretch": round(float(g.beta_stretch.median()), 3) if g.beta_stretch.notna().any() else None,
            "meta_2x2_ms": round(float(g.meta_2x2_stretch.median()), 1) if g.meta_2x2_stretch.notna().any() else None,
            "alpha_power": round(float(g.alpha_power.median()), 3) if g.alpha_power.notna().any() else None,
            "snr": round(float(g.snr.median()), 1),
            "shared_frac": round(float(g.shared_frac.median()), 4),
        } for k, g in kept.groupby("task")}
        # 任务检验（猕猴内部）
        mac = kept[kept.source == "macaque_nlb"]
        gs = [g.beta_stretch.dropna().values for _, g in mac.groupby("task")
              if len(g) >= 2]
        if len(gs) >= 2:
            kw = kruskal(*gs)
            rep["macaque_task_test_beta_stretch"] = {
                "p": float(kw.pvalue),
                "medians": {k: round(float(g.beta_stretch.median()), 3)
                            for k, g in mac.groupby("task") if len(g) >= 2}}
        hum = kept[kept.source == "human_intracortical"]
        a = hum[hum.task == "H1_reach_grasp"].beta_stretch.dropna()
        b = hum[hum.task == "H2_handwriting"].beta_stretch.dropna()
        rep["human_task_beta_stretch"] = {
            "H1_median": round(float(a.median()), 3) if len(a) else None,
            "H2_median": round(float(b.median()), 3) if len(b) else None,
            "n": [int(len(a)), int(len(b))],
            "p": float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else None}
        # 物种（同任务口径：reach vs reach）
        m1 = kept[(kept.source == "macaque_nlb") & (kept.task == "reach")].beta_stretch.dropna()
        h2 = kept[kept.task == "H2_handwriting"].beta_stretch.dropna()
        rep["cross_species_beta_stretch"] = {
            "macaque_reach": round(float(m1.median()), 3) if len(m1) else None,
            "human_handwriting": round(float(h2.median()), 3) if len(h2) else None,
            "n": [int(len(m1)), int(len(h2))],
            "p": float(mannwhitneyu(m1, h2).pvalue) if len(m1) > 2 and len(h2) > 2 else None}
        # 相关物
        rep["correlates_beta_stretch"] = {}
        for c in ["n_units", "rate", "fano", "shared_frac", "hours", "snr"]:
            mm = kept[["beta_stretch", c]].dropna()
            if len(mm) > 5:
                r_ = spearmanr(mm.beta_stretch, mm[c])
                rep["correlates_beta_stretch"][c] = {
                    "rho": round(float(r_[0]), 3), "p": float(r_[1]), "n": int(len(mm))}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
