"""Phase 42: τ 与解码能力是否相关？—— 构念效度检验（10 被试）。

## 要检验的假设

文献中的核心论断：τ 反映「神经整合窗口」——τ 越长，回路整合信息的时间
跨度越大，理论上应该支持更强的任务表现。

**可检验预测**：τ 长的被试，图像解码准确率应更高。

若成立 → τ 通过了**构念效度**检验（不只是"测得出"，还"有意义"）
若不成立 → τ 只是个统计量，没有功能意义

## 设计

10 个被试，每人算两个量：
    τ      —— ACF 全曲线指数拟合（已验证：5/5 通过分辨率检验）
    acc@200 —— 200 类解码准确率（已验证：15–25× 随机）

然后：
1. **Spearman 相关** τ vs acc（非参数，n=10）
2. **必须控制混淆**：τ 和解码都可能受信号质量影响
   - 控制「信噪比」：用 1/f 指数或 ACF 初值做代理
   - 控制「有效维度」：PCA 前 32 维的解释方差
3. **留一交叉验证**：n=10 太小，必须看 LOO 稳定性

## 关键：必须报告"没有"的可能

n=10 时，Spearman |ρ| > 0.65 才 p<0.05。
若 ρ 在 0.3–0.6 之间，**既不能说有、也不能说没有** —— 必须如实报告功效不足。

这是本项目的纪律：**不做"接近显著"的过度解读。**

## 输出

    runs/tau-vs-decode/
    ├── results.csv      逐被试：τ、acc、SNR代理、有效维度
    └── report.json      相关分析 + 控制分析 + LOO
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

FS = 250.0
EEG_DIR = Path("/domains/omnibrain4d/eeg")
OUT = Path("/domains/runs/tau-vs-decode")

SUBJECTS = list(range(1, 11))
N_PCS = 32
MAX_IMAGES = 40
K_DECODE = 200
N_TRIALS = 2
MAX_LAG_MS = 1000.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
WIN_MS = (50, 500)
N_TIME_BINS = 10


# ------------------------------------------------------------ ACF / τ
def acf_fft(x, max_lag):
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


def estimate_tau(P, dt_ms, max_lag):
    a = np.median(acf_fft(P, max_lag), axis=1)
    lags = np.arange(len(a))
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.99)
    if ok.sum() < 5:
        return None, None
    def f(t, tt):
        return np.exp(-t / np.maximum(tt, 1e-3))
    try:
        popt, _ = curve_fit(f, lags[ok] * dt_ms, a[ok], p0=[200.0],
                            bounds=([5.0], [5000.0]), maxfev=8000)
        return float(popt[0]), float(a[1])   # τ, ACF(lag=1) 作为 SNR 代理
    except Exception:
        return None, None


# ------------------------------------------------------------ 特征 / 解码
def build_features(eeg):
    a = int(WIN_MS[0] / 1000 * FS)
    b = min(int(WIN_MS[1] / 1000 * FS), eeg.shape[-1])
    X = eeg.astype(np.float32)
    base = X[:, :, :, :max(a, 1)].mean(axis=3, keepdims=True)
    Xw = X[:, :, :, a:b] - base
    n_img, n_rep, n_ch, _ = Xw.shape
    seg = (b - a) // N_TIME_BINS
    Tb = Xw[:, :, :, :seg * N_TIME_BINS].reshape(
        n_img, n_rep, n_ch, N_TIME_BINS, seg).mean(axis=4)
    return Tb.reshape(n_img, n_rep, n_ch * N_TIME_BINS)


def _ridge_fit_predict(Xtr, ytr, Xte, n_class, lam=1.0):
    n, d = Xtr.shape
    Y = np.zeros((n, n_class), dtype=np.float32)
    Y[np.arange(n), ytr] = 1.0
    A = Xtr.T @ Xtr
    A.flat[:: d + 1] += lam * np.trace(A) / d
    try:
        W = np.linalg.solve(A, Xtr.T @ Y)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, Xtr.T @ Y, rcond=None)[0]
    return (Xte @ W).argmax(axis=1)


def decode_once(Xf, k, seed):
    n_img, n_rep, feat = Xf.shape
    if k > n_img:
        return None
    rng = np.random.default_rng(seed)
    n_tr = n_rep // 2
    cls = rng.choice(n_img, k, replace=False)
    Xs = Xf[cls].reshape(-1, feat)
    y = np.repeat(np.arange(k), n_rep)
    tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_tr) for i in range(k)])
    te = np.concatenate([np.arange(i * n_rep + n_rep - n_tr, (i + 1) * n_rep)
                         for i in range(k)])
    Xtr, ytr = Xs[tr], y[tr]
    Xte, yte = Xs[te], y[te]
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    pred = _ridge_fit_predict((Xtr - mu) / sd, ytr, (Xte - mu) / sd, k)
    return float((pred == yte).mean())


def decode_multi(Xf, k, n_trials, seed0):
    accs = []
    for t in range(n_trials):
        a = decode_once(Xf, k, seed0 + t * 1000 + k)
        if a is not None:
            accs.append(a)
    if not accs:
        return None, None
    return float(np.mean(accs)), float(np.std(accs))


# ------------------------------------------------------------ 逐被试
def analyze(eeg, sid):
    n_img, n_rep, n_ch, T = eeg.shape
    d = min(N_PCS, n_ch)
    n_use = min(n_img, MAX_IMAGES)
    dt_ms = 1000.0 / FS
    max_lag = int(min(MAX_LAG_MS / dt_ms, T - 2))

    # ---- τ + SNR 代理 + 有效维度
    taus, acf1s, evrs = [], [], []
    for i in range(n_use):
        seg = eeg[i].astype(np.float32)
        X = seg.transpose(1, 0, 2).reshape(n_ch, n_rep * T)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-6
        Z = ((X - mu) / sd).T
        Zc = Z - Z.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(Zc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        P = (U[:, :d] * S[:d]).astype(np.float64)
        t, a1 = estimate_tau(P, dt_ms, max_lag)
        if t is not None:
            taus.append(t)
        if a1 is not None:
            acf1s.append(a1)
        # 有效维度（参与率）
        lam = S ** 2
        evrs.append(float((lam.sum() ** 2) / max((lam ** 2).sum(), 1e-12)))

    # ---- 解码
    Xf = build_features(eeg)
    acc, acc_std = decode_multi(Xf, K_DECODE, N_TRIALS, sid * 7919)

    return {
        "subject": f"sub-{sid:02d}",
        "tau_ms": round(float(np.median(taus)), 1) if taus else None,
        "tau_iqr": (round(float(np.percentile(taus, 75) - np.percentile(taus, 25)), 1)
                    if taus else None),
        "acf1": round(float(np.median(acf1s)), 4) if acf1s else None,
        "eff_dim": round(float(np.median(evrs)), 2) if evrs else None,
        "acc_k200": round(acc, 5) if acc is not None else None,
        "acc_std": round(acc_std, 5) if acc_std is not None else None,
        "chance": 1.0 / K_DECODE,
        "ratio": round(acc / (1.0 / K_DECODE), 2) if acc else None,
        "n_images": len(taus),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 82, flush=True)
    print("Phase 42 — τ 与解码能力是否相关（构念效度，10 被试）", flush=True)
    print("=" * 82, flush=True)
    print(f"  τ: ACF 全曲线 | acc: {K_DECODE} 类 × {N_TRIALS} 次 | "
          f"{N_PCS} 维 | {MAX_IMAGES} 图像", flush=True)

    rows = []
    t0 = time.time()

    for sid in SUBJECTS:
        f = EEG_DIR / f"sub-{sid:02d}.npz"
        if not f.exists():
            continue
        eeg = np.load(f)["eeg"]
        t1 = time.time()
        r = analyze(eeg, sid)
        rows.append(r)
        del eeg
        print(f"  {r['subject']}: τ={r['tau_ms']:>6} ms | "
              f"acc={r['acc_k200']:.4f} ({r['ratio']:.1f}x) | "
              f"acf1={r['acf1']:.3f} | eff_dim={r['eff_dim']:.1f} "
              f"[{time.time()-t1:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    d = df.dropna(subset=["tau_ms", "acc_k200"])
    n = len(d)
    if n < 5:
        print("样本不足", flush=True)
        return

    print("\n" + "=" * 82, flush=True)
    print(f"相关分析（n={n}）", flush=True)
    print("=" * 82, flush=True)

    def report(x, y, name):
        rho, p = spearmanr(x, y)
        print(f"  {name:<26} ρ = {rho:+.3f}  p = {p:.3f}", flush=True)
        return float(rho), float(p)

    rho_tau, p_tau = report(d.tau_ms, d.acc_k200, "τ  vs  acc")
    rho_snr, p_snr = report(d.acf1, d.acc_k200, "ACF(lag1) vs acc  [SNR代理]")
    rho_eff, p_eff = report(d.eff_dim, d.acc_k200, "eff_dim vs acc")
    rho_te, p_te = report(d.tau_ms, d.eff_dim, "τ vs eff_dim")

    # ---- 控制 SNR / 有效维度后的偏相关
    print("\n" + "-" * 82, flush=True)
    print("偏相关（控制 SNR 代理 acf1）", flush=True)
    print("-" * 82, flush=True)

    def partial_spearman(x, y, z):
        rx = pd.Series(x).rank()
        ry = pd.Series(y).rank()
        rz = pd.Series(z).rank()
        def resid(a, b):
            A = np.c_[np.ones(len(b)), b]
            coef, *_ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ coef
        ex = resid(rx.to_numpy(), rz.to_numpy())
        ey = resid(ry.to_numpy(), rz.to_numpy())
        r, p = spearmanr(ex, ey)
        return float(r), float(p)

    pr_snr, pp_snr = partial_spearman(d.tau_ms, d.acc_k200, d.acf1)
    print(f"  τ vs acc | acf1     ρ = {pr_snr:+.3f}  p = {pp_snr:.3f}", flush=True)
    pr_eff, pp_eff = partial_spearman(d.tau_ms, d.acc_k200, d.eff_dim)
    print(f"  τ vs acc | eff_dim  ρ = {pr_eff:+.3f}  p = {pp_eff:.3f}", flush=True)

    # ---- LOO 稳定性
    print("\n" + "-" * 82, flush=True)
    print("留一交叉验证（移除每个被试后重算 ρ）", flush=True)
    print("-" * 82, flush=True)
    loos = []
    for i in range(n):
        sub = d.drop(d.index[i])
        r, _ = spearmanr(sub.tau_ms, sub.acc_k200)
        loos.append(float(r))
    print(f"  ρ 范围: [{min(loos):+.3f}, {max(loos):+.3f}]  中位 {np.median(loos):+.3f}",
          flush=True)
    same_sign = all(np.sign(v) == np.sign(rho_tau) for v in loos)
    print(f"  符号一致: {'是' if same_sign else '否'}", flush=True)

    # ---- 判定（严格）
    print("\n" + "=" * 82, flush=True)
    print("判定", flush=True)
    print("=" * 82, flush=True)

    if p_tau < 0.05 and rho_tau > 0.5 and same_sign:
        verdict = (f"✅ 构念效度成立：τ 与解码准确率显著正相关 "
                   f"(ρ={rho_tau:+.2f}, p={p_tau:.3f})，LOO 符号一致")
    elif p_tau < 0.05 and rho_tau < -0.5:
        verdict = (f"⚠️ 显著负相关 (ρ={rho_tau:+.2f}) —— 与'整合窗口'假设相反，"
                   "需查是否受混淆驱动")
    elif abs(rho_tau) < 0.3:
        verdict = (f"❌ 无相关 (ρ={rho_tau:+.2f}, p={p_tau:.3f}) —— "
                   "τ 不预测解码能力，没有功能意义")
    else:
        # 关键：诚实报告功效不足
        power_needed = "|ρ| > 0.65"
        verdict = (f"⚠️ **功效不足**：ρ={rho_tau:+.2f}, p={p_tau:.3f}。"
                   f"n={n} 时需 {power_needed} 才能 p<0.05。"
                   "既不能声称有、也不能声称没有。")

    print(f"  {verdict}", flush=True)

    # 控制后的解读
    print(f"\n  控制 SNR 后: ρ={pr_snr:+.2f} (p={pp_snr:.3f})", flush=True)
    print(f"  控制 eff_dim 后: ρ={pr_eff:+.2f} (p={pp_eff:.3f})", flush=True)
    if abs(pr_snr) < abs(rho_tau) * 0.6:
        print("  → 大部分相关可由信号质量解释，不是 τ 的独立贡献", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "tau_vs_decode",
        "n_subjects": int(n),
        "hypothesis": "τ 反映神经整合窗口，应预测解码能力",
        "per_subject": rows,
        "correlations": {
            "tau_vs_acc": {"rho": round(rho_tau, 3), "p": round(p_tau, 4)},
            "acf1_vs_acc": {"rho": round(rho_snr, 3), "p": round(p_snr, 4)},
            "eff_dim_vs_acc": {"rho": round(rho_eff, 3), "p": round(p_eff, 4)},
            "tau_vs_eff_dim": {"rho": round(rho_te, 3), "p": round(p_te, 4)},
        },
        "partial": {
            "tau_acc_ctrl_acf1": {"rho": round(pr_snr, 3), "p": round(pp_snr, 4)},
            "tau_acc_ctrl_effdim": {"rho": round(pr_eff, 3), "p": round(pp_eff, 4)},
        },
        "loo": {"min": round(min(loos), 3), "max": round(max(loos), 3),
                "median": round(float(np.median(loos)), 3),
                "sign_consistent": bool(same_sign)},
        "power_note": f"n={n} 时 Spearman 需 |rho|>0.65 才 p<0.05",
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
