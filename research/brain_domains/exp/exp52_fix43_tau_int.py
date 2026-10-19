#!/usr/bin/env python3
"""exp52 — exp43 的逐试次 tau 用积分时间重算（不依赖指数拟合阈值）。

exp51 发现：502 个试次里只有 83 个能拟合出指数 tau。
其余 419 个的 trace-ACF 在 4 个 lag（16 ms）内就低于 2%，
指数拟合无从下手。这说明逐试次 tau 在 600 ms 的 spike 窗口上
大多不可辨识，而不是"值很小"。

改用 Sokal 积分时间 tau_int = 1 + 2*sum(rho)，对加权 ACF 施加，
它对快速衰减仍然给出有限值，不需要 2% 阈值。

同时报告：
  - 可辨识比例（指数能拟合的试次数 / 总试次）
  - 两种估计量的中位数
  - 与 |计时误差| 的 Spearman 与置换检验
  - 对照：把 tau 与行为随机配对
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

NWB = Path("/domains/primate/dmfc_rsg/sub-Haydn_desc-train_ecephys.nwb")
OUT = Path("/domains/runs/tau-fixed")
OUT.mkdir(parents=True, exist_ok=True)

BIN_MS = 4.0
WIN_AFTER_SET_MS = 600.0
N_PCS = 16
MAX_LAG_MS = 400.0


def acf_cols(x, max_lag):
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    a = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    a = a / np.maximum(norms, 1)
    return a / np.maximum(a[0:1], 1e-12)


def weighted_acf(x, max_lag):
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    a = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    a = a / np.maximum(norms, 1)
    return a.sum(axis=1) / max(a[0].sum(), 1e-12)


def tau_int(a, dt, c=5.0, floor=1e-9):
    """Sokal: tau = 1 + 2*sum(rho)，窗口 M >= c*tau。"""
    taus = 1.0 + 2.0 * np.cumsum(a[1:])
    Ms = np.arange(1, len(a))
    m = Ms >= c * np.maximum(taus, floor)
    if np.any(m):
        return float(taus[int(np.argmax(m))] * dt)
    return float(taus[-1] * dt)


def fit_tau(a, dt, lo=2.0, hi=5000.0):
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.995)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]
    f = lambda tt, T: np.exp(-tt / np.maximum(T, 1e-3))
    try:
        popt, _ = curve_fit(f, t, y, p0=[100.0], bounds=([lo], [hi]), maxfev=6000)
        return float(popt[0])
    except Exception:
        return None


def main():
    import h5py
    t0 = time.time()
    with h5py.File(NWB, "r") as f:
        tr = f["intervals"]["trials"]
        d = {k: tr[k][:] for k in ["ready_time", "set_time", "go_time",
                                   "is_outlier", "is_short"]}
        st = f["units"]["spike_times"][:]
        si = f["units"]["spike_times_index"][:]
    bounds = np.concatenate([[0], si])
    units = [st[bounds[i]:bounds[i + 1]] for i in range(len(si))]

    rt, s_, gt = d["ready_time"], d["set_time"], d["go_time"]
    t1 = s_ - rt
    t2 = gt - s_
    ok = (np.isfinite(t1) & np.isfinite(t2) & (t1 > 0.1) & (t2 > 0.05)
          & (~d["is_outlier"]) & (~d["is_short"]))
    err = (t2 - t1) * 1000.0
    print(f"trials {len(gt)} | units {len(units)} | valid {ok.sum()}", flush=True)

    bin_s = BIN_MS / 1000.0
    win_s = WIN_AFTER_SET_MS / 1000.0
    ml = int(MAX_LAG_MS / BIN_MS)

    rows = []
    for i in np.flatnonzero(ok):
        a0, a1 = s_[i], s_[i] + win_s
        edges = np.arange(a0, a1 + bin_s, bin_s)
        M = np.zeros((len(edges) - 1, len(units)), dtype=np.float32)
        for j, sp in enumerate(units):
            if len(sp) == 0:
                continue
            sel = sp[(sp >= a0) & (sp < a1)]
            if len(sel) == 0:
                continue
            bi = np.clip(((sel - a0) / bin_s).astype(int), 0, M.shape[0] - 1)
            np.add.at(M[:, j], bi, 1.0)
        if M.shape[0] < 20 or M.sum() < 10:
            continue
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, keepdims=True) + 1e-6
        Z = (M - mu) / sd
        dd = min(N_PCS, Z.shape[1] - 1)
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        P = (U[:, :dd] * S[:dd]).astype(np.float64)
        lam = S ** 2

        aw = weighted_acf(Z, ml)
        ao = np.median(acf_cols(P, ml), axis=1)
        rows.append({
            "trial": int(i),
            "abs_err_ms": float(abs(err[i])),
            "rel_err": float(abs(err[i]) / (t1[i] * 1000)),
            "tau_int_inv": tau_int(aw, BIN_MS),
            "tau_int_old": tau_int(ao, BIN_MS),
            "tau_fit_inv": fit_tau(aw, BIN_MS),
            "tau_fit_old": fit_tau(ao, BIN_MS),
            "acf1_inv": float(aw[1]),
            "ev_top1": float(lam[0] / lam[:dd].sum()),
            "n_spikes": int(M.sum()),
        })
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp43_int.csv", index=False)

    def corr(col):
        dd = df.dropna(subset=[col, "abs_err_ms"])
        r, p = spearmanr(dd[col], dd.abs_err_ms)
        rng = np.random.default_rng(0)
        perm = np.array([spearmanr(dd[col], rng.permutation(dd.abs_err_ms.to_numpy()))[0]
                         for _ in range(2000)])
        return {"n": int(len(dd)), "rho": round(float(r), 4), "p": float(p),
                "p_perm": float((np.abs(perm) >= abs(r)).mean())}

    res = {
        "task": "exp43_fixed_tau_int",
        "n_trials": int(len(df)),
        "estimable_fraction": {
            "tau_fit_inv": round(float(df.tau_fit_inv.notna().mean()), 3),
            "tau_fit_old": round(float(df.tau_fit_old.notna().mean()), 3),
            "tau_int_inv": round(float(df.tau_int_inv.notna().mean()), 3),
        },
        "medians_ms": {
            "tau_int_inv": round(float(df.tau_int_inv.median()), 2),
            "tau_int_old": round(float(df.tau_int_old.median()), 2),
            "tau_fit_inv": (round(float(df.tau_fit_inv.median()), 2)
                            if df.tau_fit_inv.notna().any() else None),
            "tau_fit_old": (round(float(df.tau_fit_old.median()), 2)
                            if df.tau_fit_old.notna().any() else None),
        },
        "acf1_inv_median": round(float(df.acf1_inv.median()), 4),
        "ev_top1_median": round(float(df.ev_top1.median()), 4),
        "correlations": {
            "tau_int_inv": corr("tau_int_inv"),
            "tau_int_old": corr("tau_int_old"),
        },
    }
    # 与单位数 / spike 总数的混淆
    for c in ["n_spikes", "ev_top1"]:
        r, p = spearmanr(df[c], df.tau_int_inv)
        res[f"confound_{c}_vs_tau_int_inv"] = {"rho": round(float(r), 3),
                                               "p": round(float(p), 4)}
    (OUT / "report_exp43_int.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(res, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
