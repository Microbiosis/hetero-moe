#!/usr/bin/env python3
"""exp51 — 用正交不变的 tau 重跑 exp42 与 exp43。

旧估计量（exp39/40/42/43）：
    a = median(acf_fft(P, max_lag), axis=1)
    tau = fit_exp(a)
exp49/50 证明它在 spike 上是噪声（6.9 vs 271.8 ms，旋转下变 65%），
在 EEG 上偏 6%（190.3 vs 202.4 ms）。

本脚本两处都换成
    tau = fit_exp( tr C(k) / tr C(0) )
即按方差加权的 trace-ACF，对任何正交混合不变。

  A. exp42 重算：10 被试，40 图像，不变量 tau；解码准确率直接沿用
     已存的 runs/tau-vs-decode/report.json（acc 与 tau 估计量无关），
     重算 Spearman / 偏相关 / LOO。
  B. exp43 重算：502 试次，逐试次不变量 tau；重算与计时误差的相关和置换检验。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

EEG = Path("/domains/omnibrain4d/eeg")
NWB = Path("/domains/primate/dmfc_rsg/sub-Haydn_desc-train_ecephys.nwb")
OLD = Path("/domains/runs/tau-vs-decode/report.json")
OUT = Path("/domains/runs/tau-fixed")
OUT.mkdir(parents=True, exist_ok=True)

FS = 250.0
DT = 1000.0 / FS
N_PCS = 32
N_IMG = 40
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
MAX_LAG_MS = 1000.0
MAX_LAG_TRIAL_MS = 400.0
WIN_AFTER_SET_MS = 600.0
N_PCS_TRIAL = 16


# ---------------------------------------------------------------- ACF
def acf_cols(x, max_lag):
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


def weighted_acf(x, max_lag):
    """tr C(k)/tr C(0)。正交不变。"""
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    acf = acf / np.maximum(norms, 1)
    return acf.sum(axis=1) / max(acf[0].sum(), 1e-12)


def fit_tau(a, dt, lo=2.0, hi=50000.0):
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.995)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[200.0], bounds=([lo], [hi]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


# ---------------------------------------------------------------- A. exp42
def part_a():
    old = json.loads(OLD.read_text())
    acc = {r["subject"]: r["acc_k200"] for r in old["per_subject"]}
    old_tau = {r["subject"]: r["tau_ms"] for r in old["per_subject"]}
    rows = []
    t0 = time.time()
    for sm in SUBJECTS:
        f = EEG / f"{sm}.npz"
        if not f.exists():
            continue
        eeg = np.load(f)["eeg"]
        n_img, n_rep, n_ch, T = eeg.shape
        ml = int(min(MAX_LAG_MS / DT, T - 2))
        taus_new, taus_old, ev1s = [], [], []
        for ii in range(min(N_IMG, n_img)):
            seg = eeg[ii].astype(np.float32)
            X = seg.transpose(1, 0, 2).reshape(n_ch, n_rep * T)
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True) + 1e-6
            Z = ((X - mu) / sd).T
            Zc = Z - Z.mean(axis=0, keepdims=True)
            tn = fit_tau(weighted_acf(Zc, ml), DT)
            if tn:
                taus_new.append(tn)
            U, S, _ = np.linalg.svd(Zc, full_matrices=False)
            d = min(N_PCS, n_ch)
            P = (U[:, :d] * S[:d]).astype(np.float64)
            to = fit_tau(np.median(acf_cols(P, ml), axis=1), DT)
            if to:
                taus_old.append(to)
            lam = S ** 2
            ev1s.append(float(lam[0] / lam[:d].sum()))
        rows.append({
            "subject": sm,
            "tau_invariant": round(float(np.median(taus_new)), 1) if taus_new else None,
            "tau_oldmedian": round(float(np.median(taus_old)), 1) if taus_old else None,
            "tau_stored": old_tau.get(sm),
            "acc_k200": acc.get(sm),
            "ev_top1": round(float(np.median(ev1s)), 4) if ev1s else None,
            "n_img": len(taus_new),
        })
        print(f"  {sm}: inv={rows[-1]['tau_invariant']} old={rows[-1]['tau_oldmedian']} "
              f"stored={old_tau.get(sm)} acc={acc.get(sm)} [{time.time()-t0:.0f}s]",
              flush=True)
        del eeg
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp42_fixed.csv", index=False)

    d = df.dropna(subset=["tau_invariant", "acc_k200"])
    n = len(d)

    def rep(x, y):
        r, p = spearmanr(x, y)
        return float(r), float(p)

    r_inv, p_inv = rep(d.tau_invariant, d.acc_k200)
    r_old, p_old = rep(d.tau_oldmedian, d.acc_k200)
    r_sto, p_sto = rep(df.dropna(subset=["tau_stored", "acc_k200"]).tau_stored,
                       df.dropna(subset=["tau_stored", "acc_k200"]).acc_k200)

    # 偏相关（控制 eff_dim 与 ev_top1）
    def partial(x, y, z):
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rz = pd.Series(z).rank().to_numpy()
        A = np.c_[np.ones(len(rz)), rz]
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        r, p = spearmanr(ex, ey)
        return float(r), float(p)

    pr_ev, pp_ev = partial(d.tau_invariant, d.acc_k200, d.ev_top1)

    loos = []
    for i in range(n):
        s = d.drop(d.index[i])
        loos.append(float(spearmanr(s.tau_invariant, s.acc_k200)[0]))

    res = {
        "task": "exp42_fixed_invariant_tau",
        "n_subjects": int(n),
        "acc_source": "runs/tau-vs-decode/report.json（acc 与 tau 估计量无关，直接沿用）",
        "tau_oldmedian_vs_acc": {"rho": round(r_old, 3), "p": round(p_old, 4)},
        "tau_stored_vs_acc": {"rho": round(r_sto, 3), "p": round(p_sto, 4)},
        "tau_invariant_vs_acc": {"rho": round(r_inv, 3), "p": round(p_inv, 4)},
        "tau_invariant_vs_acc_ctrl_evtop1": {"rho": round(pr_ev, 3), "p": round(pp_ev, 4)},
        "loo": {"min": round(min(loos), 3), "max": round(max(loos), 3),
                "median": round(float(np.median(loos)), 3),
                "sign_consistent": bool(all(np.sign(v) == np.sign(r_inv) for v in loos))},
        "tau_shift": {
            "inv_minus_old_pct_median": round(float(
                ((d.tau_invariant - d.tau_oldmedian) / d.tau_oldmedian * 100).median()), 2)},
    }
    if p_inv < 0.05 and abs(r_inv) > 0.5:
        res["verdict"] = f"显著 (rho={r_inv:+.2f}, p={p_inv:.3f})"
    elif abs(r_inv) < 0.3:
        res["verdict"] = (f"无相关 (rho={r_inv:+.2f}, p={p_inv:.3f}) — "
                          "换成不变量后结论不变：tau 不预测解码")
    else:
        res["verdict"] = (f"功效不足 (rho={r_inv:+.2f}, p={p_inv:.3f})，n=10 需 |rho|>0.65")
    return df, res


# ---------------------------------------------------------------- B. exp43
def part_b():
    import h5py
    if not NWB.exists():
        return None, {"error": f"missing {NWB}"}

    with h5py.File(NWB, "r") as f:
        t = f["intervals"]["trials"]
        tr = {k: t[k][:] for k in
              ["ready_time", "set_time", "go_time", "is_outlier", "is_short"]}
        u = f["units"]
        st = u["spike_times"][:]
        si = u["spike_times_index"][:]
    bounds = np.concatenate([[0], si])
    units = [st[bounds[i]:bounds[i + 1]] for i in range(len(si))]
    print(f"  trials {len(tr['go_time'])} | units {len(units)}", flush=True)

    rt, s_, gt = tr["ready_time"], tr["set_time"], tr["go_time"]
    t1 = s_ - rt
    t2 = gt - s_
    ok = (np.isfinite(t1) & np.isfinite(t2) & (t1 > 0.1) & (t2 > 0.05)
          & (~tr["is_outlier"]) & (~tr["is_short"]))
    err = (t2 - t1) * 1000.0

    bin_s = 4.0 / 1000.0
    win_s = WIN_AFTER_SET_MS / 1000.0
    ml = int(MAX_LAG_TRIAL_MS / 4.0)

    rows = []
    t0 = time.time()
    for i in np.flatnonzero(ok):
        a0, a1 = s_[i], s_[i] + win_s
        edges = np.arange(a0, a1 + bin_s, bin_s)
        if len(edges) < 6:
            continue
        M = np.zeros((len(edges) - 1, len(units)), dtype=np.float32)
        for j, sp in enumerate(units):
            if len(sp) == 0:
                continue
            sel = sp[(sp >= a0) & (sp < a1)]
            if len(sel) == 0:
                continue
            bi = np.clip(((sel - a0) / bin_s).astype(int), 0, M.shape[0] - 1)
            np.add.at(M[:, j], bi, 1.0)
        if M.shape[0] < 8 or M.sum() < 10:
            continue
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, keepdims=True) + 1e-6
        Z = (M - mu) / sd
        d = min(N_PCS_TRIAL, Z.shape[1] - 1)
        try:
            U, S, _ = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        P = (U[:, :d] * S[:d]).astype(np.float64)
        lam = S ** 2
        tn = fit_tau(weighted_acf(Z, ml), 4.0, lo=2.0, hi=5000.0)
        to = fit_tau(np.median(acf_cols(P, ml), axis=1), 4.0, lo=2.0, hi=5000.0)
        rows.append({
            "trial": int(i), "abs_err_ms": float(abs(err[i])),
            "rel_err": float(abs(err[i]) / (t1[i] * 1000)),
            "tau_inv": tn, "tau_old": to,
            "ev_top1": float(lam[0] / lam[:d].sum()),
            "n_spikes": int(M.sum()),
        })
        if len(rows) % 100 == 0:
            print(f"    {len(rows)} trials [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp43_fixed.csv", index=False)

    d = df.dropna(subset=["tau_inv", "abs_err_ms"])
    n = len(d)
    r_inv, p_inv = spearmanr(d.tau_inv, d.abs_err_ms)
    sub_old = df.dropna(subset=["tau_old", "abs_err_ms"])
    r_old, p_old = spearmanr(sub_old.tau_old, sub_old.abs_err_ms)

    rng = np.random.default_rng(0)
    perm = np.array([spearmanr(d.tau_inv, rng.permutation(d.abs_err_ms.to_numpy()))[0]
                     for _ in range(2000)])
    p_perm = float((np.abs(perm) >= abs(r_inv)).mean())

    res = {
        "task": "exp43_fixed_invariant_tau",
        "n_trials": int(n),
        "n_trials_old": int(len(sub_old)),
        "tau_old_vs_abs_err": {"rho": round(float(r_old), 4), "p": float(p_old)},
        "tau_invariant_vs_abs_err": {"rho": round(float(r_inv), 4), "p": float(p_inv),
                                     "p_perm": p_perm, "n_perm": 2000},
        "permutation_ci95": [round(float(np.percentile(perm, 2.5)), 4),
                             round(float(np.percentile(perm, 97.5)), 4)],
        "tau_medians": {"old": round(float(df.tau_old.median()), 1),
                        "invariant": round(float(df.tau_inv.median()), 1)},
        "ev_top1_median": round(float(df.ev_top1.median()), 4),
    }
    if p_perm < 0.05:
        res["verdict"] = f"显著 (rho={r_inv:+.3f}, perm p={p_perm:.4f})"
    else:
        res["verdict"] = (f"无相关 (rho={r_inv:+.3f}, perm p={p_perm:.4f}) — "
                          "换成不变量后结论不变")
    return df, res


def main():
    print("=" * 78, flush=True)
    print("A. exp42 重算（EEG 250 Hz，正交不变 tau）", flush=True)
    print("=" * 78, flush=True)
    df_a, res_a = part_a()
    print("\n" + json.dumps(res_a, indent=2, ensure_ascii=False), flush=True)
    (OUT / "report_exp42_fixed.json").write_text(
        json.dumps(res_a, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print("B. exp43 重算（DMFC 逐试次，正交不变 tau）", flush=True)
    print("=" * 78, flush=True)
    df_b, res_b = part_b()
    print("\n" + json.dumps(res_b, indent=2, ensure_ascii=False), flush=True)
    (OUT / "report_exp43_fixed.json").write_text(
        json.dumps(res_b, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
