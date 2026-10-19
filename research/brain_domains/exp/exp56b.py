#!/usr/bin/env python3
"""exp56b — β 的稳定性与零模型（修正归一化）。

## 上一版的错误

exp56 用 ρ(τ) = cross(τ)/cross(0)。但零模型（每单元独立循环移位）破坏跨单元
对齐后 cross(0) 趋于 0，除一个接近零的数会放大噪声，null 的 β 变成
0.032 / 0.08 甚至负值——这不是"零模型也有 β"，而是归一化失效。

## 本版

改为按总方差归一：

    total(0) = Σ_{ij} <x_i x_j>            = 全部方差（含对角）
    cross(τ) = total(τ) − diag(τ)          = 只含 i≠j
    ρ(τ)     = cross(τ) / total(0)         ∈ [−1, 1]，有绝对尺度

    shared_frac = cross(0) / total(0)      = 总方差里跨单元共享的比例

白噪声下 ρ(τ) ≈ 0；共享潜变量下 ρ(τ) > 0 且衰减。
噪声底由零模型的 ρ(τ) 在 τ∈[2,500] 上的 RMS 给出，SNR = ρ(1)/噪声底。

β 从 log ρ(τ) 对 log τ 拟合（τ ∈ [2, min(500, T/4)]），仅在 ρ(τ) > 2×噪声底
的点上拟合，点数不足 8 个则报 None。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/beta2")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500
TRUNC = 3000
FIT_RANGES = [100, 250, 500]
LOCAL = [(2, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500)]
N_NULL = 3
SEED = 0


def n2(n):
    k = 1
    while k < n:
        k <<= 1
    return k


def acf_pair(x, max_lag):
    """返回 (total, diag) 未归一化 ACF，无偏除以 (T−τ)。"""
    T, d = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    N = n2(2 * T)
    F = np.fft.rfft(xc, n=N, axis=0)
    Fs = F.sum(axis=1)
    total = np.fft.irfft(Fs * np.conj(Fs), n=N)[:max_lag + 1]
    diag = np.fft.irfft((np.abs(F) ** 2).sum(axis=1), n=N)[:max_lag + 1]
    nrm = np.arange(T, T - max_lag - 1, -1)
    return total / np.maximum(nrm, 1), diag / np.maximum(nrm, 1)


def rho_curve(x, max_lag):
    """ρ(τ) = (total(τ) − diag(τ)) / total(0)。"""
    total, diag = acf_pair(x, max_lag)
    t0 = max(total[0], 1e-12)
    cross = total - diag
    return cross / t0, float(cross[0] / t0)


def fit_beta(rho, noise, hi=MAX_LAG, lo=2):
    hi = min(hi, len(rho) - 1)
    if hi - lo < 8:
        return None
    t = np.arange(lo, hi + 1) * DT
    y = rho[lo:hi + 1]
    ok = np.isfinite(y) & (y > max(2 * noise, 1e-9))
    if ok.sum() < 8:
        return None
    b = -np.polyfit(np.log(t[ok]), np.log(y[ok]), 1)[0]
    return float(b)


def local_slopes(rho, noise):
    out = {}
    for a, b in LOCAL:
        if b >= len(rho) or rho[a] <= 1e-9 or rho[b] <= 1e-9:
            continue
        if rho[a] < 2 * noise or rho[b] < 2 * noise:
            continue
        out[f"{a}-{b}"] = round(float(
            -(np.log(rho[b]) - np.log(rho[a])) / (np.log(b) - np.log(a))), 3)
    return out


def agg(x, f):
    T = (x.shape[0] // f) * f
    return x[:T].reshape(T // f, f, x.shape[1]).sum(axis=1)


def top_stats(x):
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False) ** 2
    return (float(s[0] / max(s.sum(), 1e-12)),
            float((s.sum() ** 2) / max((s ** 2).sum(), 1e-12)))


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
        if x.shape[0] < 1200 or x.shape[1] < 8:
            continue
        ml = min(MAX_LAG, x.shape[0] - 2)

        # ---- 零模型：独立循环移位，拿噪声底
        null_prof = []
        for _ in range(N_NULL):
            xs = np.empty_like(x)
            for j in range(x.shape[1]):
                xs[:, j] = np.roll(x[:, j], int(rng.integers(1, x.shape[0])))
            rn, _ = rho_curve(xs, ml)
            null_prof.append(rn)
        null_prof = np.array(null_prof)
        noise = float(np.sqrt((null_prof[:, 2:] ** 2).mean()))
        null_beta = [fit_beta(p, noise) for p in null_prof]

        rho, shared = rho_curve(x, ml)
        top1, eff = top_stats(x)

        b_full = fit_beta(rho, noise)
        ls = local_slopes(rho, noise)

        # 稳定性
        range_b = {str(k): fit_beta(rho, noise, hi=min(k, ml)) for k in FIT_RANGES}
        h = x.shape[0] // 2
        r1, s1 = rho_curve(x[:h], ml)
        r2, s2 = rho_curve(x[h:], ml)
        q = x.shape[0] // 4
        qs = []
        for i in range(4):
            rq, _ = rho_curve(x[i * q:(i + 1) * q], ml)
            qs.append(fit_beta(rq, noise))
        bws = {}
        for f in (1, 2, 5):
            xf = x if f == 1 else agg(x, f)
            if xf.shape[0] > 100:
                rf, _ = rho_curve(xf, min(MAX_LAG, xf.shape[0] - 2))
                bws[str(f * 20)] = fit_beta(rf, noise)
        b_tr = None
        if x.shape[0] >= TRUNC:
            rt, _ = rho_curve(x[:TRUNC], min(MAX_LAG, TRUNC - 2))
            b_tr = fit_beta(rt, noise)

        rows.append({
            "source": r.source, "task": task_of(r.source, r.file),
            "file": r.file, "region": r.region,
            "n_units": int(x.shape[1]), "n_bins": int(x.shape[0]),
            "hours": round(x.shape[0] * DT / 3.6e6, 3),
            "rate": float(x.mean()), "fano": float(x.var() / max(x.mean(), 1e-9)),
            "top1": round(top1, 4), "eff_dim": round(eff, 2),
            "shared_frac": round(float(shared), 5),
            "rho1": round(float(rho[1]), 5), "rho2": round(float(rho[2]), 5),
            "noise_floor": round(noise, 6),
            "snr_rho1": round(float(rho[1] / max(noise, 1e-12)), 1),
            "beta": b_full,
            "beta_null": json.dumps([round(v, 3) if v else None for v in null_beta]),
            "beta_h1": fit_beta(r1, noise), "beta_h2": fit_beta(r2, noise),
            "shared_h1": round(s1, 5), "shared_h2": round(s2, 5),
            "beta_q": json.dumps([round(v, 3) if v else None for v in qs]),
            "beta_rng": json.dumps({k: (round(v, 3) if v else None)
                                    for k, v in range_b.items()}),
            "beta_bw": json.dumps({k: (round(v, 3) if v else None)
                                   for k, v in bws.items()}),
            "beta_trunc": b_tr,
            "local_slopes": json.dumps(ls),
        })
        if n % 10 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} β={b_full and round(b_full,3)} "
                  f"null={[v and round(v,3) for v in null_beta]} "
                  f"shared={shared:.4f} noise={noise:.2e} [{time.time()-t0:.0f}s]",
                  flush=True)
        del x
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_segment.csv", index=False)

    rep = {"task": "beta_stability_v2", "dt_ms": DT, "n_segments": int(len(df)),
           "normalization": "rho(tau) = cross(tau)/total(0)", "n_null": N_NULL}

    def med(c, g=df):
        v = g[c].dropna()
        return None if not len(v) else {
            "n": int(len(v)), "median": round(float(v.median()), 4),
            "p10": round(float(v.quantile(.1)), 4),
            "p90": round(float(v.quantile(.9)), 4)}

    rep["shared_frac"] = med("shared_frac")
    rep["rho1"] = med("rho1")
    rep["snr_rho1"] = med("snr_rho1")
    rep["noise_floor"] = med("noise_floor")
    rep["beta"] = med("beta")

    # Q4 零模型
    NB = pd.DataFrame([json.loads(s) for s in df.beta_null])
    nbmed = NB.mean(axis=1)
    rep["Q4_null_beta"] = {
        "n": int(nbmed.notna().sum()),
        "null_median": round(float(nbmed.median()), 3),
        "null_p10": round(float(nbmed.quantile(.1)), 3),
        "null_p90": round(float(nbmed.quantile(.9)), 3),
        "real_median": round(float(df.beta.median()), 3),
        "frac_null_lt_half_real": round(float(
            (nbmed < 0.5 * df.beta).mean()), 3),
        "frac_null_ge_0.2": round(float((nbmed >= 0.2).mean()), 3)}

    # Q1
    LS = pd.DataFrame([json.loads(s) for s in df.local_slopes])
    rep["Q1_local_slope"] = {c: {"median": round(float(LS[c].median()), 3),
                                 "p10": round(float(LS[c].quantile(.1)), 3),
                                 "p90": round(float(LS[c].quantile(.9)), 3),
                                 "n": int(LS[c].notna().sum())}
                             for c in LS.columns}
    if len(LS.columns) >= 3:
        cols = list(LS.columns)
        inc = LS[cols].apply(lambda r: bool(np.all(np.diff(r.dropna().values) > -0.05))
                             if r.notna().sum() == len(cols) else False, axis=1)
        rep["Q1_slope_increases_with_lag"] = round(float(inc.mean()), 3)

    # Q2
    m = df.dropna(subset=["beta_h1", "beta_h2"])
    if len(m) > 5:
        r_ = spearmanr(m.beta_h1, m.beta_h2)
        rep["Q2_split_half_beta"] = {
            "n": int(len(m)), "rho": round(float(r_[0]), 3), "p": float(r_[1])}
    ms = df.dropna(subset=["shared_h1", "shared_h2"])
    if len(ms) > 5:
        r_ = spearmanr(ms.shared_h1, ms.shared_h2)
        rep["Q2_split_half_shared"] = {
            "n": int(len(ms)), "rho": round(float(r_[0]), 3), "p": float(r_[1]),
            "ratio_median": round(float((ms.shared_h1 / ms.shared_h2).median()), 3)}
    RNGD = pd.DataFrame([json.loads(s) for s in df.beta_rng])
    rep["Q2_fit_range"] = {c: med(c, RNGD) if c in RNGD else None for c in RNGD.columns}
    BWD = pd.DataFrame([json.loads(s) for s in df.beta_bw])
    rep["Q2_bin_width"] = {c: med(c, BWD) if c in BWD else None for c in BWD.columns}
    m2 = df.dropna(subset=["beta", "beta_trunc"])
    if len(m2) > 5:
        rep["Q2_truncation"] = {
            "n": int(len(m2)),
            "full": round(float(m2.beta.median()), 3),
            "trunc": round(float(m2.beta_trunc.median()), 3),
            "rho": round(float(spearmanr(m2.beta, m2.beta_trunc)[0]), 3)}
    QD = pd.DataFrame([json.loads(s) for s in df.beta_q])
    rep["Q2_quarters"] = {"range_median": round(float((QD.max(axis=1) - QD.min(axis=1)).median()), 3),
                          "sd_median": round(float(QD.std(axis=1).median()), 3)}

    # Q3
    rep["Q3_correlates"] = {}
    for c in ["n_units", "rate", "fano", "top1", "eff_dim", "hours", "shared_frac"]:
        mm = df[["beta", c]].dropna()
        if len(mm) > 5:
            r_ = spearmanr(mm.beta, mm[c])
            rep["Q3_correlates"][c] = {"rho": round(float(r_[0]), 3),
                                       "p": float(r_[1]), "n": int(len(mm))}
    # shared_frac 与什么相关（它是 β 的主要混淆候选）
    rep["Q3_shared_correlates"] = {}
    for c in ["n_units", "rate", "top1", "eff_dim", "hours"]:
        mm = df[["shared_frac", c]].dropna()
        if len(mm) > 5:
            r_ = spearmanr(mm.shared_frac, mm[c])
            rep["Q3_shared_correlates"][c] = {"rho": round(float(r_[0]), 3),
                                              "p": float(r_[1]), "n": int(len(mm))}

    # 任务 / 物种
    rep["by_task"] = {}
    for (src, tk), g in df.groupby(["source", "task"]):
        rep["by_task"][f"{src}:{tk}"] = {
            "n_segments": int(len(g)),
            "beta": med("beta", g), "shared_frac": med("shared_frac", g),
            "rho1": med("rho1", g), "top1": med("top1", g)}
    mac = df[df.source == "macaque_nlb"]
    groups = [g.beta.dropna().values for _, g in mac.groupby("task") if len(g) >= 2]
    if len(groups) >= 2:
        kw = kruskal(*groups)
        rep["macaque_task_test"] = {
            "kruskal_H": round(float(kw.statistic), 3), "p": float(kw.pvalue),
            "medians": {tk: round(float(g.beta.median()), 3)
                        for tk, g in mac.groupby("task") if len(g) >= 2}}
    hum = df[df.source == "human_intracortical"]
    g1 = hum[hum.task == "H1_reach_grasp"].beta.dropna()
    g2 = hum[hum.task == "H2_handwriting"].beta.dropna()
    if len(g1) > 2 and len(g2) > 2:
        rep["human_task_test"] = {
            "H1": round(float(g1.median()), 3), "H2": round(float(g2.median()), 3),
            "p": float(mannwhitneyu(g1, g2).pvalue), "n": [int(len(g1)), int(len(g2))]}
    a = df[(df.source == "macaque_nlb") & (df.region == "M1")].beta_trunc.dropna()
    b = hum.beta_trunc.dropna()
    if len(a) > 2 and len(b) > 2:
        rep["species_truncated"] = {
            "macaque": round(float(a.median()), 3), "human": round(float(b.median()), 3),
            "p": float(mannwhitneyu(a, b).pvalue), "n": [int(len(a)), int(len(b))]}
    # shared_frac 的物种比较（截断后没有 shared，用全段）
    a2 = df[(df.source == "macaque_nlb") & (df.region == "M1")].shared_frac.dropna()
    b2 = hum.shared_frac.dropna()
    if len(a2) > 2 and len(b2) > 2:
        rep["species_shared"] = {
            "macaque": round(float(a2.median()), 5), "human": round(float(b2.median()), 5),
            "p": float(mannwhitneyu(a2, b2).pvalue)}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
