#!/usr/bin/env python3
"""exp58 — 从双指数拟合里取出两个时间尺度。

exp57：145 个通过 SNR≥3 的片段中，双指数胜出 137 个（94.5%），
幂律 4 个，拉伸指数 4 个。所以"幂律"这个描述是错的，
真实结构是**两个时间尺度**的叠加。

本脚本解析 exp57 的拟合参数（A1·exp(−τ/t1) + A2·exp(−τ/t2)），
按 t 排序成 (fast, slow)，报告：

  - 两个尺度的中位数与分布
  - 两个幅度的中位数与分布
  - 交叉点 τ* （两分量相等处）
  - 物种 / 任务差异（在 fast、slow、幅度比三个量上）
  - 与 SNR、放电率、共享比例的相关
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

OUT = Path("/domains/runs/models")
OUT2 = Path("/domains/runs/twoscale")
OUT2.mkdir(parents=True, exist_ok=True)
DT = 20.0


def main():
    df = pd.read_csv(OUT / "per_segment.csv")
    df = df[df.kept == True].copy()
    print(f"kept segments: {len(df)}")

    rec = []
    for _, r in df.iterrows():
        f = json.loads(r.fit)
        if "two_exp" not in f:
            continue
        p = f["two_exp"]["params"]      # A1, t1, A2, t2  (t 单位 ms)
        A1, t1, A2, t2 = p
        if t1 > t2:
            A1, t1, A2, t2 = A2, t2, A1, t1
        # 交叉点：A1 exp(-t/t1) = A2 exp(-t/t2)
        cross_tau = None
        if A1 > 0 and A2 > 0 and t2 > t1:
            ratio = A1 / A2
            if ratio > 0:
                cross_tau = float(t1 * t2 / (t2 - t1) * np.log(ratio)) if ratio != 1 else 0.0
                if not np.isfinite(cross_tau) or cross_tau < 0:
                    cross_tau = None
        rec.append({
            "source": r.source, "task": r.task, "file": r.file, "region": r.region,
            "n_units": r.n_units, "n_bins": r.n_bins, "hours": r.hours,
            "rate": r.rate, "fano": r.fano, "snr": r.snr,
            "shared_frac": r.shared_frac,
            "A_fast": max(A1, 0.0), "tau_fast_ms": float(t1),
            "A_slow": max(A2, 0.0), "tau_slow_ms": float(t2),
            "frac_fast": float(A1 / max(A1 + A2, 1e-12)),
            "cross_tau_ms": cross_tau,
            "aic_two": f["two_exp"]["aic"],
            "aic_stretch": f.get("stretch", {}).get("aic"),
            "aic_power": f.get("power", {}).get("aic"),
            "r2_two": f["two_exp"]["r2"],
            "beta_stretch": r.beta_stretch,
            "beta_stretch_t0": r.meta_2x2_stretch,
        })
    d = pd.DataFrame(rec)
    d.to_csv(OUT2 / "twoscale.csv", index=False)
    print(f"parsed two_exp: {len(d)}")

    rep = {"task": "two_timescale", "dt_ms": DT, "n": int(len(d))}

    def s(c, g=d):
        v = g[c].dropna()
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(v.median()), 1),
                "p10": round(float(v.quantile(.1)), 1),
                "p90": round(float(v.quantile(.9)), 1)}

    rep["params"] = {c: s(c) for c in
                     ["tau_fast_ms", "tau_slow_ms", "A_fast", "A_slow",
                      "frac_fast", "cross_tau_ms", "r2_two"]}
    rep["aic_delta"] = {
        "two_minus_stretch_median": round(float(
            (d.aic_two - d.aic_stretch).median()), 1)
        if d.aic_stretch.notna().any() else None,
        "two_minus_power_median": round(float(
            (d.aic_two - d.aic_power).median()), 1)
        if d.aic_power.notna().any() else None,
        "note": "负值表示双指数更好"}

    # tau_fast 是否落在几个 bin 上（退化迹象）
    rep["tau_fast_quantized"] = {
        "frac_le_60ms": round(float((d.tau_fast_ms <= 60).mean()), 3),
        "unique_rounded": int(d.tau_fast_ms.round(-1).nunique())}
    rep["tau_slow_hits_ceiling"] = {
        "frac_ge_10000ms": round(float((d.tau_slow_ms >= 10000).mean()), 3),
        "frac_ge_100000ms": round(float((d.tau_slow_ms >= 100000).mean()), 3)}

    rep["by_task"] = {k: {"n": int(len(g)),
                          **{c: s(c, g) for c in
                             ["tau_fast_ms", "tau_slow_ms", "frac_fast", "snr"]}}
                      for k, g in d.groupby("task")}

    # 任务 / 物种
    mac = d[d.source == "macaque_nlb"]
    gs = [g.tau_fast_ms.dropna().values for _, g in mac.groupby("task") if len(g) >= 2]
    if len(gs) >= 2:
        rep["macaque_task_tau_fast"] = {"p": float(kruskal(*gs).pvalue)}
    gs2 = [g.tau_slow_ms.dropna().values for _, g in mac.groupby("task") if len(g) >= 2]
    if len(gs2) >= 2:
        rep["macaque_task_tau_slow"] = {"p": float(kruskal(*gs2).pvalue)}

    a = d[(d.source == "macaque_nlb") & (d.task == "reach")]
    b = d[d.task == "H2_handwriting"]
    if len(a) > 5 and len(b) > 5:
        rep["species_reach_vs_handwriting"] = {}
        for c in ["tau_fast_ms", "tau_slow_ms", "frac_fast", "A_fast", "A_slow"]:
            v1, v2 = a[c].dropna(), b[c].dropna()
            if len(v1) > 5 and len(v2) > 5:
                rep["species_reach_vs_handwriting"][c] = {
                    "macaque": round(float(v1.median()), 1),
                    "human": round(float(v2.median()), 1),
                    "ratio": round(float(v1.median() / max(v2.median(), 1e-9)), 2),
                    "p": float(mannwhitneyu(v1, v2).pvalue)}

    # 相关
    rep["correlates"] = {}
    for c in ["snr", "rate", "fano", "shared_frac", "n_units", "hours"]:
        for y in ["tau_fast_ms", "tau_slow_ms", "frac_fast"]:
            m = d[[y, c]].dropna()
            if len(m) > 10:
                r = spearmanr(m[y], m[c])
                rep["correlates"][f"{y}_vs_{c}"] = {
                    "rho": round(float(r[0]), 3), "p": float(r[1]), "n": int(len(m))}

    (OUT2 / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
