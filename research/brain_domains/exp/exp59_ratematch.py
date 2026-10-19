#!/usr/bin/env python3
"""exp59 — 放电率匹配后，物种差异还剩多少？

exp58 的链条：

    tau_fast 与 rate 的 ρ = 0.641
    tau_fast 与 shared_frac 的 ρ = 0.743
    shared_frac 与 rate 的 ρ = 0.883（exp56b）

而猕猴放电率中位 0.456，人类 0.23。所以"物种差异"可能是放电率差异。

本脚本三重检验：

  A. 分层匹配：按放电率分 decile，在每个 decile 内比较两个物种
  B. 最近邻匹配：每个人类片段配一个放电率最接近的猕猴片段（±0.02）
  C. 偏相关：控制 rate 后，物种与 tau_fast 的关联

同时检验：猕猴内部三个任务的 tau_fast 差异（exp58 的 p=0.0015）
是否也只是放电率差异。三个任务的放电率中位分别是
reach 0.46 / timing 0.13 / bump 0.13。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

IN = Path("/domains/runs/twoscale/twoscale.csv")
OUT = Path("/domains/runs/ratematch")
OUT.mkdir(parents=True, exist_ok=True)


def partial_spearman(x, y, z):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rz = pd.Series(z).rank().to_numpy()
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r, p = spearmanr(ex, ey)
    return float(r), float(p)


def main():
    d = pd.read_csv(IN)
    mac = d[d.source == "macaque_nlb"].copy()
    hum = d[d.source == "human_intracortical"].copy()
    print(f"macaque {len(mac)} | human {len(hum)}")
    print(f"rate: macaque {mac.rate.median():.3f} | human {hum.rate.median():.3f}")

    rep = {"task": "rate_matching", "n_macaque": int(len(mac)), "n_human": int(len(hum)),
           "rate_median": {"macaque": round(float(mac.rate.median()), 4),
                           "human": round(float(hum.rate.median()), 4)}}

    # ---------- 原始差异
    rep["raw"] = {}
    for c in ["tau_fast_ms", "tau_slow_ms", "frac_fast", "shared_frac", "fano"]:
        a, b = mac[c].dropna(), hum[c].dropna()
        if len(a) > 5 and len(b) > 5:
            rep["raw"][c] = {
                "macaque": round(float(a.median()), 3),
                "human": round(float(b.median()), 3),
                "ratio": round(float(a.median() / max(b.median(), 1e-9)), 2),
                "p": float(mannwhitneyu(a, b).pvalue)}

    # ---------- C 偏相关（控制 rate）
    rep["partial_controlling_rate"] = {}
    for c in ["tau_fast_ms", "tau_slow_ms", "frac_fast"]:
        m = d[[c, "rate", "source"]].dropna()
        if len(m) < 20:
            continue
        ismac = (m.source == "macaque_nlb").astype(float)
        r, p = partial_spearman(m[c], ismac, m.rate)
        # 只用放电率重叠的区间
        lo = max(mac.rate.min(), hum.rate.min())
        hi = min(mac.rate.max(), hum.rate.max())
        sub = m[(m.rate >= lo) & (m.rate <= hi)]
        r2, p2 = partial_spearman(sub[c], (sub.source == "macaque_nlb").astype(float),
                                  sub.rate) if len(sub) > 20 else (None, None)
        rep["partial_controlling_rate"][c] = {
            "all": {"rho": round(r, 3), "p": float(p), "n": int(len(m))},
            "rate_overlap_only": {"rho": round(r2, 3) if r2 is not None else None,
                                  "p": float(p2) if p2 is not None else None,
                                  "n": int(len(sub)),
                                  "rate_range": [round(float(lo), 3), round(float(hi), 3)]}}

    # ---------- A 分层匹配
    mac["q"] = pd.qcut(mac.rate, 5, labels=False, duplicates="drop")
    edges = mac.groupby("q").rate.agg(["min", "max"])
    strata = []
    for qi, row in edges.iterrows():
        mm = mac[(mac.rate >= row["min"]) & (mac.rate <= row["max"])]
        hh = hum[(hum.rate >= row["min"]) & (hum.rate <= row["max"])]
        if len(mm) < 3 or len(hh) < 3:
            continue
        strata.append({
            "stratum": int(qi),
            "rate_range": [round(float(row["min"]), 3), round(float(row["max"]), 3)],
            "n_macaque": int(len(mm)), "n_human": int(len(hh)),
            "tau_fast_macaque": round(float(mm.tau_fast_ms.median()), 1),
            "tau_fast_human": round(float(hh.tau_fast_ms.median()), 1),
            "ratio": round(float(mm.tau_fast_ms.median()
                                 / max(hh.tau_fast_ms.median(), 1e-9)), 2),
            "p": float(mannwhitneyu(mm.tau_fast_ms.dropna(),
                                    hh.tau_fast_ms.dropna()).pvalue)})
    rep["A_stratified"] = strata

    # ---------- B 最近邻匹配（±0.02）
    pairs = []
    used = set()
    for _, h in hum.sample(frac=1.0, random_state=0).iterrows():
        cand = mac[(mac.rate - h.rate).abs() <= 0.02]
        cand = cand[~cand.index.isin(used)]
        if not len(cand):
            continue
        j = (cand.rate - h.rate).abs().idxmin()
        used.add(j)
        pairs.append({"h_rate": h.rate, "m_rate": mac.loc[j, "rate"],
                      "h_tau": h.tau_fast_ms, "m_tau": mac.loc[j, "tau_fast_ms"],
                      "h_shared": h.shared_frac, "m_shared": mac.loc[j, "shared_frac"]})
    P = pd.DataFrame(pairs).dropna(subset=["h_tau", "m_tau"])
    if len(P) > 5:
        rep["B_nearest_match"] = {
            "n_pairs": int(len(P)),
            "rate_diff_median": round(float((P.h_rate - P.m_rate).abs().median()), 4),
            "tau_fast_macaque": round(float(P.m_tau.median()), 1),
            "tau_fast_human": round(float(P.h_tau.median()), 1),
            "ratio": round(float(P.m_tau.median() / max(P.h_tau.median(), 1e-9)), 2),
            "p_paired": float(mannwhitneyu(P.m_tau, P.h_tau).pvalue),
            "shared_macaque": round(float(P.m_shared.median()), 4),
            "shared_human": round(float(P.h_shared.median()), 4)}

    # ---------- 猕猴内部任务差异是否也是放电率
    rep["macaque_task"] = {}
    for tk, g in mac.groupby("task"):
        if len(g) < 2:
            continue
        rep["macaque_task"][tk] = {
            "n": int(len(g)),
            "rate": round(float(g.rate.median()), 3),
            "tau_fast": round(float(g.tau_fast_ms.median()), 1),
            "shared_frac": round(float(g.shared_frac.median()), 4)}
    mt = mac[mac.task.isin(["reach", "timing", "bump"])]
    if mt.task.nunique() >= 2:
        r1 = spearmanr(mt.tau_fast_ms, mt.rate)
        rep["macaque_task_tau_vs_rate"] = {"rho": round(float(r1[0]), 3),
                                           "p": float(r1[1]), "n": int(len(mt))}
        # 控制 rate 后的任务效应：比较 reach 与 (timing+bump)
        r_, p_ = partial_spearman(
            mt.tau_fast_ms.fillna(mt.tau_fast_ms.median()),
            (mt.task == "reach").astype(float), mt.rate)
        rep["macaque_reach_vs_other_ctrl_rate"] = {"rho": round(r_, 3), "p": float(p_)}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
