"""Phase 40: 正确的 ACF 估计 + 抗混叠对照 + 1/f 指数（三法交叉验证）。

## 纠正 exp39 的理解错误

exp39 里 dt = 1000/FS = 4ms，所以：
    250 Hz: τ = 148.7 ms = **37.2 bins**
    125 Hz: τ = 166.1 ms = 20.8 bins
     62.5 Hz: τ = 237.3 ms = 14.8 bins

所以 τ 有 37 个 bin，不是 1–3 个。分辨率不是问题。

真正的问题是：**为什么降采样后 τ 变大？**

两个候选解释：
  H1 **混叠**：无抗混叠滤波，31–125 Hz 折回低频 → 看起来更慢
  H2 **多时间尺度**：真实 ACF 不是单指数，粗采样偏向慢成分

## 本实验：三法交叉验证

### 方法 1: 直接 ACF（不做 VAR）
在 4ms 原始序列上算 ACF（用 FFT），最多到 lag=250（1秒），
拟合单指数 R(τ)=exp(-τ/T)，得 T。

同时算 **Sokal 积分自相关时间**：
    τ_int = 1 + 2 Σ_{t=1}^{M} ρ(t),  M 取最小满足 M ≥ 5·τ_int

这个量不依赖指数拟合假设。

### 方法 2: 抗混叠 vs 朴素降采样
同一序列，两条路降到 125 / 62.5 Hz：
  a) 朴素 `[::dec]`（exp39 的做法）
  b) `scipy.signal.decimate`（先 8 阶 Butterworth 低通）
若 b 的 τ 比 a 稳定 → 混叠是主因（H1）
若 b 仍随降采样变大 → 多尺度是主因（H2）

### 方法 3: 1/f 非周期指数
Welch PSD → log-log 拟合斜率。
用三个拟合范围（1–40 / 2–30 / 4–20 Hz）看其稳定性。
文献（Boncompte 2026）指出指数依赖拟合范围，必须报告。

## 判据

  方法 1 的 T 与 τ_int 接近 → 单指数近似尚可
  方法 2 中 filtered 稳定、naive 不稳 → H1（混叠）成立，可修复
  方法 2 中两者都随降采样变大     → H2（多尺度）成立，τ 概念本身有问题
  方法 3 的指数在三个范围内一致     → 1/f 是更稳健的替代量

> ## ⚠️ 估计量已弃用（2026-10-19）
>
> 本脚本用 `np.median(acf_fft(P), axis=1)` 估 τ，这是**基依赖**的：
> 换一组正交基 τ 就变。实测 20 ms spike 上旋转 5 次变化 1.65 倍。
>
> 正交不变的替代是 `tr C(τ) / tr C(0)`，见 `lib/tau.py` 与
> `FINDINGS_TAU_GAUGE.md`。用不变量重跑的结果：`exp51_fix.py`、`exp52_fix43.py`。
>
> 本文件保留以记录原始流程，结论以 FINDINGS_TAU_GAUGE.md 为准。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit

FS = 250.0
EEG_DIR = Path("/domains/omnibrain4d/eeg")
OUT = Path("/domains/runs/acf-proper")

N_PCS = 32
MAX_IMAGES = 30
SUBJECTS = [1, 2, 3]
MAX_LAG_MS = 1000.0
DECS = [1, 2, 4]
FIT_RANGES = [(1, 40), (2, 30), (4, 20)]


# ------------------------------------------------------------ ACF
def acf_fft(x, max_lag):
    """逐列 ACF（用 FFT），x: (T, d)。返回 (max_lag+1, d) 归一化 ACF。"""
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    f = np.fft.rfft(xc, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag + 1]
    # 无偏归一化：除以 (T - lag)
    norms = np.arange(T, T - max_lag - 1, -1)[:, None]
    acf = acf / np.maximum(norms, 1)
    acf = acf / np.maximum(acf[0:1], 1e-12)
    return acf


def fit_exp_tau(lags, acf_vals, dt_ms):
    """拟合 R(t) = exp(-t/T)，返回 T（ms）。"""
    ok = np.isfinite(acf_vals) & (acf_vals > 0.02) & (acf_vals < 0.99)
    if ok.sum() < 4:
        return None
    t = lags[ok] * dt_ms
    y = acf_vals[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[100.0],
                            bounds=([1.0], [10000.0]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


def integrated_tau(acf_vals, dt_ms, c=5.0):
    """Sokal 积分自相关时间：τ_int = 1 + 2Σρ，窗口 M ≥ c·τ_int。"""
    rho = acf_vals
    taus = 2.0 * np.cumsum(rho) - 1.0
    m = np.arange(len(taus)) < c * taus
    idx = np.argmin(m) if np.any(m) else len(taus) - 1
    return float(taus[idx] * dt_ms)


# ------------------------------------------------------------ 1/f
def aperiodic_exponent(x, fs, fmin, fmax):
    """Welch PSD → log-log 斜率（负值）。"""
    f, P = signal.welch(x, fs=fs, nperseg=min(256, x.shape[0]),
                        axis=0, detrend="constant")
    m = (f >= fmin) & (f <= fmax)
    if m.sum() < 5:
        return None
    lf = np.log10(f[m])
    lp = np.log10(P[m] + 1e-20)
    # 逐列拟合后取中位
    sl = []
    for j in range(lp.shape[1]):
        y = lp[:, j]
        if not np.isfinite(y).all():
            continue
        sl.append(np.polyfit(lf, y, 1)[0])
    return float(np.median(sl)) if sl else None


# ------------------------------------------------------------ 主分析
def analyze_subject(eeg, sid, max_images=MAX_IMAGES):
    n_img, n_rep, n_ch, T = eeg.shape
    d = min(N_PCS, n_ch)
    n_use = min(n_img, max_images)

    rows = []
    max_lag = int(min(MAX_LAG_MS / 1000 * FS, T - 2))

    for i in range(n_use):
        seg = eeg[i].astype(np.float32)                 # (n_rep, n_ch, T)
        X = seg.transpose(1, 0, 2).reshape(n_ch, n_rep * T)

        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-6
        Z = ((X - mu) / sd).T                           # (time, ch)

        Zc = Z - Z.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(Zc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        P = (U[:, :d] * S[:d]).astype(np.float64)       # (time, d)

        # ---------- 方法 1: 原始 4ms 上的 ACF
        a = acf_fft(P, max_lag)
        acf0 = np.median(a, axis=1)
        tau_exp = fit_exp_tau(np.arange(len(acf0)), acf0, 1000.0 / FS)
        tau_int = integrated_tau(acf0, 1000.0 / FS)

        # ---------- 方法 2: 朴素 vs 抗混叠降采样
        tau_naive, tau_filt = {}, {}
        for dec in DECS:
            if dec == 1:
                tau_naive[1] = tau_filt[1] = tau_exp
                continue
            if P.shape[0] < 2 * dec + 10:
                continue
            # 朴素
            Pn = P[::dec]
            ml = max_lag // dec
            an = np.median(acf_fft(Pn, ml), axis=1)
            tau_naive[dec] = fit_exp_tau(np.arange(len(an)), an, 1000.0 * dec / FS)
            # 抗混叠（8 阶 Butterworth，截止 0.8*Nyquist）
            try:
                Pf = signal.decimate(P, dec, axis=0, ftype="iir",
                                     zero_phase=True)
            except Exception:
                continue
            mlf = max_lag // dec
            af = np.median(acf_fft(Pf, mlf), axis=1)
            tau_filt[dec] = fit_exp_tau(np.arange(len(af)), af, 1000.0 * dec / FS)

        # ---------- 方法 3: 1/f 指数（原始 4ms）
        exps = {}
        for lo, hi in FIT_RANGES:
            exps[f"{lo}-{hi}Hz"] = aperiodic_exponent(P, FS, lo, hi)

        rows.append({
            "subject": sid, "image": i,
            "tau_exp_ms": tau_exp, "tau_int_ms": tau_int,
            "tau_naive_2x": tau_naive.get(2), "tau_naive_4x": tau_naive.get(4),
            "tau_filt_2x": tau_filt.get(2), "tau_filt_4x": tau_filt.get(4),
            **{f"exp_{k}": v for k, v in exps.items()},
        })

    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 84, flush=True)
    print("Phase 40 — 正确 ACF + 抗混叠对照 + 1/f 指数", flush=True)
    print("=" * 84, flush=True)
    print(f"  被试 {SUBJECTS} | {N_PCS} PCA 维 | {MAX_IMAGES} 图像 "
          f"| 最大 lag {MAX_LAG_MS:.0f}ms", flush=True)

    all_rows = []
    t0 = time.time()

    for sid in SUBJECTS:
        f = EEG_DIR / f"sub-{sid:02d}.npz"
        if not f.exists():
            continue
        print(f"\n  --- sub-{sid:02d} ---", flush=True)
        eeg = np.load(f)["eeg"]
        t1 = time.time()
        rows = analyze_subject(eeg, f"sub-{sid:02d}")
        all_rows += rows
        del eeg

        df = pd.DataFrame(rows)
        print(f"    τ_exp 中位 {df.tau_exp_ms.median():.1f} ms | "
              f"τ_int 中位 {df.tau_int_ms.median():.1f} ms "
              f"[{time.time()-t1:.1f}s]", flush=True)
        print(f"    naive 2x {df.tau_naive_2x.median():.1f} | "
              f"filt 2x {df.tau_filt_2x.median():.1f} | "
              f"naive 4x {df.tau_naive_4x.median():.1f} | "
              f"filt 4x {df.tau_filt_4x.median():.1f}", flush=True)
        ec = [c for c in df.columns if c.startswith("exp_")]
        print("    1/f 指数: " +
              ", ".join(f"{c[4:]}={df[c].median():+.2f}" for c in ec), flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    if not len(df):
        return

    # ============ 汇总
    print("\n" + "=" * 84, flush=True)
    print("方法 1: 直接 ACF（4ms，无 VAR）", flush=True)
    print("=" * 84, flush=True)
    print(f"  指数拟合 τ_exp 中位 : {df.tau_exp_ms.median():7.1f} ms "
          f"(IQR {df.tau_exp_ms.quantile(.75)-df.tau_exp_ms.quantile(.25):.1f})",
          flush=True)
    print(f"  积分自相关 τ_int 中位: {df.tau_int_ms.median():7.1f} ms "
          f"(IQR {df.tau_int_ms.quantile(.75)-df.tau_int_ms.quantile(.25):.1f})",
          flush=True)
    ratio = df.tau_int_ms.median() / max(df.tau_exp_ms.median(), 1e-9)
    print(f"  比值 τ_int / τ_exp  : {ratio:.2f}  "
          f"({'单指数近似尚可' if 0.7 <= ratio <= 1.5 else '偏离单指数 → 多尺度'})",
          flush=True)

    print("\n" + "=" * 84, flush=True)
    print("方法 2: 抗混叠对照（关键）", flush=True)
    print("=" * 84, flush=True)
    base = df.tau_exp_ms.median()
    print(f"{'条件':<16} {'τ(ms)':>9} {'相对 4ms':>10}", flush=True)
    print(f"{'4ms (基准)':<16} {base:>9.1f} {1.0:>10.3f}", flush=True)
    for name, col in [("naive 2x", "tau_naive_2x"), ("filt  2x", "tau_filt_2x"),
                      ("naive 4x", "tau_naive_4x"), ("filt  4x", "tau_filt_4x")]:
        v = df[col].median()
        if np.isfinite(v):
            print(f"{name:<16} {v:>9.1f} {v/base:>10.3f}", flush=True)

    r_n2 = df.tau_naive_2x.median() / base
    r_f2 = df.tau_filt_2x.median() / base
    r_n4 = df.tau_naive_4x.median() / base
    r_f4 = df.tau_filt_4x.median() / base

    aliasing_fixed = (abs(r_f4 - 1) < abs(r_n4 - 1) * 0.5)
    if aliasing_fixed:
        v2 = ("✅ 抗混叠显著改善 → **混叠是主因**，τ 可以修复"
              f"（naive 4x 比值 {r_n4:.2f} → filtered {r_f4:.2f}）")
    elif abs(r_f4 - 1) < 0.25:
        v2 = f"✅ 抗混叠后 τ 稳定（比值 {r_f4:.2f}）"
    else:
        v2 = (f"❌ 抗混叠后仍变化（naive {r_n4:.2f} → filtered {r_f4:.2f}）→ "
              "**多时间尺度是主因**，单一 τ 概念本身有问题")
    print(f"\n  {v2}", flush=True)

    print("\n" + "=" * 84, flush=True)
    print("方法 3: 1/f 非周期指数（依赖拟合范围）", flush=True)
    print("=" * 84, flush=True)
    ec = [c for c in df.columns if c.startswith("exp_")]
    vals = {}
    for c in ec:
        v = df[c].median()
        vals[c[4:]] = round(float(v), 3) if np.isfinite(v) else None
        print(f"  {c[4:]:<10} {v:+.3f}", flush=True)
    num = [v for v in vals.values() if v is not None]
    spread = max(num) - min(num) if num else np.nan
    print(f"\n  范围间跨度: {spread:.3f}", flush=True)
    if np.isfinite(spread) and spread < 0.15:
        v3 = f"✅ 指数在三个范围内一致（跨度 {spread:.3f}）→ 1/f 是稳健替代量"
    else:
        v3 = f"⚠️ 指数依赖拟合范围（跨度 {spread:.3f}）→ 必须报告范围"
    print(f"  {v3}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "acf_proper",
        "fs": FS, "n_pcs": N_PCS, "max_images": MAX_IMAGES,
        "subjects": SUBJECTS,
        "method1": {
            "tau_exp_ms": round(float(df.tau_exp_ms.median()), 1),
            "tau_int_ms": round(float(df.tau_int_ms.median()), 1),
            "ratio": round(float(ratio), 3),
        },
        "method2": {
            "base_4ms": round(float(base), 1),
            "naive_2x": round(float(df.tau_naive_2x.median()), 1),
            "filt_2x": round(float(df.tau_filt_2x.median()), 1),
            "naive_4x": round(float(df.tau_naive_4x.median()), 1),
            "filt_4x": round(float(df.tau_filt_4x.median()), 1),
            "ratios": {"naive_2x": round(float(r_n2), 3),
                       "filt_2x": round(float(r_f2), 3),
                       "naive_4x": round(float(r_n4), 3),
                       "filt_4x": round(float(r_f4), 3)},
            "verdict": v2,
        },
        "method3": {"exponents": vals, "spread": round(float(spread), 3),
                    "verdict": v3},
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  完成，耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
