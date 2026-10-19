#!/usr/bin/env python3
"""exp62 — 小鼠，50 ms 原生栅格，检验 τ 是否随 bin 宽/物种变化。

现有两个成员都在 20 ms 上，τ 约 180–450 ms。
小鼠数据是 50 ms bin，205–1240 units，39 个 session。
如果 τ ≈ 400 ms 是真时间常数，它应当在 50 ms 上同样出现（400/50 = 8 bin，
足够估）。

数据注意：小鼠的 spikes 是 float32，但取值全部是 1/8 的整数倍
（max 4.375，27–29 个不同值）。所以底层是计数，被除以了 8。
这里乘回 8 得到整数，不损失信息；同时报告这一事实。

流程与 primate 相同：
  1. ρ(τ) = [tr C(τ) − diag]/tr C(0)，τ 以 50 ms 为单位
  2. 噪声底 = 每单元独立循环移位的零模型
  3. SNR ≥ 3 且可用点 ≥ 10 才拟合
  4. 三模型 AIC：幂律 / 拉伸指数 / A·exp(−τ/t)+c
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu

SEG = Path("/domains/steinmetz")
OUT = Path("/domains/runs/mouse")
OUT.mkdir(parents=True, exist_ok=True)

DT = 50.0
MAX_LAG = 400           # 20 s
N_NULL = 2
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


def m_stretch(t, A, t0, b):
    return A * np.exp(-np.power(t / np.maximum(t0, 1e-6), b))


def m_power(t, A, a):
    return A * np.power(t, -a)


def fit_all(t, y):
    n = len(t)
    out = {}
    for name, fn, p0, bd in [
        ("plateau", m_plateau, [float(y[0]), 400.0, max(float(y[-1]), 0.0)],
         ([0.0, 5.0, -1.0], [20.0, 1e6, 1.0])),
        ("stretch", m_stretch, [float(y[0]), 400.0, 0.5],
         ([1e-6, 5.0, 0.05], [20.0, 1e6, 2.0])),
        ("power", m_power, [float(y[0]), 0.3],
         ([1e-6, 0.01], [20.0, 3.0])),
    ]:
        try:
            p, _ = curve_fit(fn, t, y, p0=p0, bounds=bd, maxfev=60000)
            r = y - fn(t, *p)
            out[name] = {"p": [float(v) for v in p],
                         "aic": float(n * np.log(np.mean(r ** 2) + 1e-20)),
                         "r2": float(1 - np.var(r) / max(np.var(y), 1e-20))}
        except Exception:
            pass
    return out


def main():
    t0 = time.time()
    u = pd.read_parquet(SEG / "units.parquet")
    sess_region = (u.groupby(["session", "region"]).size()
                   .reset_index(name="n")
                   .sort_values(["session", "n"], ascending=[True, False])
                   .groupby("session").first()["region"].to_dict())
    sess_dom = (u.groupby(["session", "functional_domain"]).size()
                .reset_index(name="n")
                .sort_values(["session", "n"], ascending=[True, False])
                .groupby("session").first()["functional_domain"].to_dict())

    rng = np.random.default_rng(SEED)
    rows = []
    files = sorted(glob.glob(str(SEG / "spikes_session_*.npz")))
    print(f"{len(files)} sessions", flush=True)
    for i, f in enumerate(files):
        raw = np.load(f)["spikes"].astype(np.float32)      # (n_units, n_bins_50ms)
        # 1/8 量化检查
        frac8 = float((np.abs(raw * 8 - np.round(raw * 8)) < 1e-5).mean())
        x = (np.round(raw * 8).T).astype(np.float32)       # (n_bins, n_units) 整数计数
        del raw
        nb, nu = x.shape
        if nb < 1200 or nu < 8:
            continue
        ml = min(MAX_LAG, nb - 2)
        noise = null_noise(x, ml, rng)
        rho, shared = rho_curve(x, ml)
        snr = float(rho[1] / max(noise, 1e-12))
        usable = np.flatnonzero((rho > 2 * noise) & np.isfinite(rho))
        usable = usable[(usable >= 2) & (usable <= ml)]
        sid = int(Path(f).stem.split("_")[-1])
        rec = {"session": sid, "file": Path(f).name,
               "region": sess_region.get(sid, "?"),
               "domain": sess_dom.get(sid, "?"),
               "n_units": int(nu), "n_bins": int(nb),
               "hours": round(nb * DT / 3.6e6, 3),
               "count_per_bin": round(float(x.mean()), 4),   # 每 50 ms bin 的放电计数
               "rate_hz": round(float(x.mean() / (DT / 1000.0)), 2),
               "frac_multiple_of_1_8": round(frac8, 5),
               "shared_frac": round(float(shared), 5),
               "rho1": round(float(rho[1]), 5),
               "noise": round(noise, 6), "snr": round(snr, 1),
               "kept": bool(snr >= 3.0 and len(usable) >= 10)}
        if rec["kept"]:
            tt = usable.astype(float) * DT
            yy = rho[usable]
            fits = fit_all(tt, yy)
            best = min(fits, key=lambda k: fits[k]["aic"]) if fits else None
            rec["best"] = best
            rec["tau_ms"] = round(fits["plateau"]["p"][1], 1) if "plateau" in fits else None
            rec["c"] = round(fits["plateau"]["p"][2], 4) if "plateau" in fits else None
            rec["A"] = round(fits["plateau"]["p"][0], 3) if "plateau" in fits else None
            rec["r2"] = round(fits["plateau"]["r2"], 4) if "plateau" in fits else None
            rec["beta_stretch"] = round(fits["stretch"]["p"][2], 3) if "stretch" in fits else None
            rec["alpha_power"] = round(fits["power"]["p"][1], 3) if "power" in fits else None
            rec["n_points"] = int(len(usable))
        rows.append(rec)
        print(f"  [{i+1}/{len(files)}] sess{sid:03d} {nu}u {nb}b snr={snr:.0f} "
              f"tau={rec.get('tau_ms')} keep={rec['kept']} [{time.time()-t0:.0f}s]",
              flush=True)
        del x

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_session.csv", index=False)
    k = d[d.kept == True]
    rep = {"task": "mouse_50ms", "dt_ms": DT, "n_sessions": int(len(d)),
           "n_kept": int(len(k)),
           "quantization": {"median_multiple_of_1_8": round(float(d.frac_multiple_of_1_8.median()), 5),
                            "note": "原始 float32 全部是 1/8 的整数倍，乘 8 得整数计数"},
           "snr": {"median": round(float(d.snr.median()), 1),
                   "p10": round(float(d.snr.quantile(.1)), 1)}}

    def s(c):
        v = k[c].dropna()
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(v.median()), 2),
                "p10": round(float(v.quantile(.1)), 2),
                "p90": round(float(v.quantile(.9)), 2)}

    rep["params"] = {c: s(c) for c in ["tau_ms", "c", "A", "r2", "shared_frac",
                                       "rho1", "rate", "n_units", "hours"]}
    rep["model_wins"] = k.best.value_counts().to_dict() if "best" in k else None
    rep["by_region"] = {r: {"n": int(len(g)),
                            "tau_ms": round(float(g.tau_ms.median()), 1)
                            if g.tau_ms.notna().any() else None,
                            "rate": round(float(g.rate.median()), 3)}
                        for r, g in k.groupby("region") if len(g) >= 3}
    rep["by_domain"] = {r: {"n": int(len(g)),
                            "tau_ms": round(float(g.tau_ms.median()), 1)
                            if g.tau_ms.notna().any() else None}
                        for r, g in k.groupby("domain") if len(g) >= 3}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
