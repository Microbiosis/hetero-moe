#!/usr/bin/env python3
"""exp56 — 幂律指数 β 的稳定性、相关物与零模型。

exp55 得到 β = 0.371（猕猴 0.338，人类 0.434，p = 0.0016）。
本脚本回答四个问题：

Q1 真的是幂律吗？
   算局域斜率 d log R / d log τ 在 7 个区间上的值。
   幂律要求各区间斜率大致相同；若随 lag 单调变化，则是 crossover，不是幂律。

Q2 β 稳不稳？
   - 分裂信度：前半 / 后半 / 四等分
   - 拟合区间：lag ≤ 100 / 250 / 500 / 1000
   - bin 宽：20 / 40 / 100 ms（计数可加，重采样无损）
   - 截断到相同长度：全部截到 3000 bin（60 s）
   - 随机子采样窗（只保留一半时间点）

Q3 β 与什么相关？
   n_units、放电率、Fano、有效维度、top-1 占比、片段时长

Q4 β 需要真实的跨单元结构吗？
   零模型：每个单元独立循环移位（保留各自的边际分布与自相关，
   破坏跨单元时间对齐）。若 β 仍在，则 β 不来自群体耦合。

关键对照：猕猴内部三个任务（伸手 / 计时 / 触碰）与人类内部两个任务
（抓取 / 手写）。物种与任务不混淆。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

V3 = Path("/domains/counts-v3")
OUT = Path("/domains/runs/beta")
OUT.mkdir(parents=True, exist_ok=True)

DT = 20.0
MAX_LAG = 500
TRUNC = 3000          # 60 s，用于时长控制
FIT_RANGES = [100, 250, 500, 1000]
LOCAL = [(2, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500)]
N_SHUFFLE = 2
SEED = 0


def n2(n):
    k = 1
    while k < n:
        k <<= 1
    return k


def cross_acf(x, max_lag):
    """Σ_{i≠j}<x_i x_j(τ)> / Σ_{i≠j}<x_i x_j(0)>。"""
    T, d = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    F = np.fft.rfft(xc, n=n2(2 * T), axis=0)
    Fs = F.sum(axis=1)
    total = np.fft.irfft(Fs * np.conj(Fs), n=n2(2 * T))[:max_lag + 1]
    diag = np.fft.irfft((np.abs(F) ** 2).sum(axis=1), n=n2(2 * T))[:max_lag + 1]
    nrm = np.arange(T, T - max_lag - 1, -1)
    cross = (total - diag) / np.maximum(nrm, 1)
    return cross / max(cross[0], 1e-12)


def beta_of(R, hi=MAX_LAG, lo=2):
    hi = min(hi, len(R) - 1)
    if hi - lo < 5:
        return None
    t = np.arange(lo, hi + 1) * DT
    y = R[lo:hi + 1]
    ok = np.isfinite(y) & (y > 1e-6)
    if ok.sum() < 5:
        return None
    b = -np.polyfit(np.log(t[ok]), np.log(y[ok]), 1)[0]
    return float(b)


def local_slopes(R):
    out = {}
    for a, b in LOCAL:
        if b >= len(R) or R[a] <= 1e-9 or R[b] <= 1e-9:
            continue
        out[f"{a}-{b}"] = round(float(
            -(np.log(R[b]) - np.log(R[a])) / (np.log(b) - np.log(a))), 3)
    return out


def agg(x, f):
    """时间轴按 f 聚合（计数可加）。"""
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
        if x.shape[0] < 1200 or x.shape[1] < 5:
            continue
        ml = min(MAX_LAG, x.shape[0] - 2)
        R = cross_acf(x, ml)
        top1, eff = top_stats(x)

        # Q1 局域斜率
        ls = local_slopes(R)

        # Q2 稳定性
        b_full = beta_of(R)
        # 拟合区间
        range_b = {str(k): beta_of(R, hi=min(k, ml)) for k in FIT_RANGES}
        # 分裂
        h = x.shape[0] // 2
        b_h1 = beta_of(cross_acf(x[:h], ml))
        b_h2 = beta_of(cross_acf(x[h:], ml))
        q = x.shape[0] // 4
        b_q = [beta_of(cross_acf(x[i * q:(i + 1) * q], ml)) for i in range(4)]
        # bin 宽
        bw = {}
        for f in (1, 2, 5):
            xf = x if f == 1 else agg(x, f)
            if xf.shape[0] > 60:
                Rf = cross_acf(xf, min(MAX_LAG, xf.shape[0] - 2))
                bw[str(f * 20)] = beta_of(Rf, hi=min(MAX_LAG, xf.shape[0] - 2))
        # 截断到 TRUNC
        b_tr = beta_of(cross_acf(x[:TRUNC], min(MAX_LAG, TRUNC - 2))) if x.shape[0] >= TRUNC else None
        # 半数时间点（随机抽 1/2 的 bin，破坏局部结构，作为对照）
        sel = np.sort(rng.choice(x.shape[0], x.shape[0] // 2, replace=False))
        b_sub = beta_of(cross_acf(x[sel], ml))

        # Q4 零模型：每列独立循环移位
        nulls = []
        for _ in range(N_SHUFFLE):
            xs = x.copy()
            for j in range(xs.shape[1]):
                xs[:, j] = np.roll(xs[:, j], int(rng.integers(1, x.shape[0])))
            nulls.append(beta_of(cross_acf(xs, ml)))

        rows.append({
            "source": r.source, "task": task_of(r.source, r.file),
            "file": r.file, "region": r.region,
            "n_units": int(x.shape[1]), "n_bins": int(x.shape[0]),
            "hours": round(x.shape[0] * DT / 3.6e6, 3),
            "rate": float(x.mean()), "fano": float(x.var() / max(x.mean(), 1e-9)),
            "top1": round(top1, 4), "eff_dim": round(eff, 2),
            "beta": b_full,
            "beta_h1": b_h1, "beta_h2": b_h2,
            "beta_q": json.dumps([round(v, 3) if v else None for v in b_q]),
            "beta_rng": json.dumps({k: (round(v, 3) if v else None)
                                    for k, v in range_b.items()}),
            "beta_bw": json.dumps({k: (round(v, 3) if v else None)
                                   for k, v in bw.items()}),
            "beta_trunc": b_tr, "beta_sub": b_sub,
            "beta_null": json.dumps([round(v, 3) if v else None for v in nulls]),
            "local_slopes": json.dumps(ls),
            "R1": round(float(R[1]), 4), "R2": round(float(R[2]), 4),
        })
        if n % 10 == 0:
            print(f"  [{n+1}/{len(idx)}] {r.file} β={b_full and round(b_full,3)} "
                  f"null={nulls and [round(v,3) if v else None for v in nulls]} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        del x
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_segment.csv", index=False)

    rep = {"task": "beta_stability", "dt_ms": DT, "n_segments": int(len(df)),
           "trunc_bins": TRUNC, "n_shuffle": N_SHUFFLE}

    def med(c, g=df):
        v = g[c].dropna()
        return None if not len(v) else {
            "n": int(len(v)), "median": round(float(v.median()), 3),
            "p10": round(float(v.quantile(.1)), 3),
            "p90": round(float(v.quantile(.9)), 3)}

    # ---------- Q1
    ls_cols = [f"{a}-{b}" for a, b in LOCAL]
    rep["Q1_local_slope"] = {c: med(c) for c in []}  # placeholder
    LS = pd.DataFrame([json.loads(s) for s in df.local_slopes])
    rep["Q1_local_slope"] = {c: {"median": round(float(LS[c].median()), 3),
                                 "p10": round(float(LS[c].quantile(.1)), 3),
                                 "p90": round(float(LS[c].quantile(.9)), 3),
                                 "n": int(LS[c].notna().sum())}
                             for c in LS.columns}
    rep["Q1_monotone_increase"] = round(float(
        LS[[c for c in LS.columns]].apply(
            lambda r: bool(np.all(np.diff(r.dropna().values) > -0.05)), axis=1).mean()), 3)

    # ---------- Q2
    rep["Q2_beta"] = med("beta")
    rep["Q2_split_half"] = None
    m = df.dropna(subset=["beta_h1", "beta_h2"])
    if len(m) > 5:
        r = spearmanr(m.beta_h1, m.beta_h2)
        rep["Q2_split_half"] = {
            "n": int(len(m)), "rho": round(float(r[0]), 3), "p": float(r[1]),
            "half_ratio_median": round(float((m.beta_h1 / m.beta_h2).median()), 3),
            "half_ratio_iqr": round(float((m.beta_h1 / m.beta_h2).quantile(.75)
                                          - (m.beta_h1 / m.beta_h2).quantile(.25)), 3)}
    BQ = pd.DataFrame([json.loads(s) for s in df.beta_q])
    rep["Q2_quarters"] = {"n": int(BQ.notna().any(axis=1).sum()),
                          "sd_median": round(float(BQ.std(axis=1).median()), 3),
                          "range_median": round(float((BQ.max(axis=1) - BQ.min(axis=1)).median()), 3)}
    RNG = pd.DataFrame([json.loads(s) for s in df.beta_rng])
    rep["Q2_fit_range"] = {c: {"median": round(float(RNG[c].median()), 3),
                               "spread_p90_p10": round(float(RNG[c].quantile(.9)
                                                             - RNG[c].quantile(.1)), 3)}
                           for c in RNG.columns}
    rep["Q2_fit_range_ratio"] = {
        "median_max_over_min": round(float((RNG.max(axis=1) / RNG.min(axis=1)).median()), 2)}
    BW = pd.DataFrame([json.loads(s) for s in df.beta_bw])
    rep["Q2_bin_width"] = {c: med(c, BW) if c in BW else None for c in BW.columns}
    m2 = df.dropna(subset=["beta", "beta_trunc"])
    if len(m2) > 5:
        rep["Q2_truncation"] = {
            "n": int(len(m2)),
            "beta_full_median": round(float(m2.beta.median()), 3),
            "beta_trunc_median": round(float(m2.beta_trunc.median()), 3),
            "rho": round(float(spearmanr(m2.beta, m2.beta_trunc)[0]), 3),
            "ratio_median": round(float((m2.beta_trunc / m2.beta).median()), 3)}
    m3 = df.dropna(subset=["beta", "beta_sub"])
    if len(m3) > 5:
        rep["Q2_subsample"] = {"n": int(len(m3)),
                               "ratio_median": round(float((m3.beta_sub / m3.beta).median()), 3)}

    # ---------- Q3
    rep["Q3_correlates"] = {}
    for c in ["n_units", "rate", "fano", "top1", "eff_dim", "hours"]:
        mm = df[["beta", c]].dropna()
        if len(mm) > 5:
            r = spearmanr(mm.beta, mm[c])
            rep["Q3_correlates"][c] = {"rho": round(float(r[0]), 3),
                                       "p": float(r[1]), "n": int(len(mm))}

    # ---------- Q4 零模型
    NULL = pd.DataFrame([json.loads(s) for s in df.beta_null])
    rep["Q4_null"] = {
        "beta_real_median": round(float(df.beta.median()), 3),
        "beta_null_median": round(float(NULL.mean(axis=1).median()), 3),
        "beta_null_p10": round(float(NULL.mean(axis=1).quantile(.1)), 3),
        "beta_null_p90": round(float(NULL.mean(axis=1).quantile(.9)), 3),
        "frac_null_ge_half_real": round(float(
            (NULL.mean(axis=1) >= 0.5 * df.beta).mean()), 3)}

    # ---------- 任务对照
    rep["by_task"] = {}
    for (src, tk), g in df.groupby(["source", "task"]):
        rep["by_task"][f"{src}:{tk}"] = {
            "n_segments": int(len(g)),
            "beta": med("beta", g),
            "beta_trunc": med("beta_trunc", g),
            "top1": med("top1", g),
            "rate": med("rate", g)}
    mac = df[df.source == "macaque_nlb"]
    groups = [g.beta.dropna().values for _, g in mac.groupby("task") if len(g) >= 2]
    if len(groups) >= 2:
        kw = kruskal(*groups)
        rep["macaque_task_test"] = {
            "n_groups": len(groups), "kruskal_H": round(float(kw.statistic), 3),
            "p": float(kw.pvalue),
            "group_medians": {tk: round(float(g.beta.median()), 3)
                              for tk, g in mac.groupby("task") if len(g) >= 2}}
    hum = df[df.source == "human_intracortical"]
    g1 = hum[hum.task == "H1_reach_grasp"].beta.dropna()
    g2 = hum[hum.task == "H2_handwriting"].beta.dropna()
    if len(g1) > 2 and len(g2) > 2:
        rep["human_task_test"] = {
            "H1_median": round(float(g1.median()), 3),
            "H2_median": round(float(g2.median()), 3),
            "p": float(mannwhitneyu(g1, g2).pvalue),
            "n": [int(len(g1)), int(len(g2))]}
    # 物种（截断后）
    a = df[(df.source == "macaque_nlb") & (df.region == "M1")].beta_trunc.dropna()
    b = df[(df.source == "human_intracortical")].beta_trunc.dropna()
    if len(a) > 2 and len(b) > 2:
        rep["species_truncated"] = {
            "macaque_median": round(float(a.median()), 3),
            "human_median": round(float(b.median()), 3),
            "p": float(mannwhitneyu(a, b).pvalue),
            "n": [int(len(a)), int(len(b))]}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
