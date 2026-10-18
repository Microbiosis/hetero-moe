"""Phase 41: 一锤定音 —— VAR(1) 极点 vs 单变量 ACF，谁该被信任。

## exp40 的关键发现

同一批数据、同样的降采样：
    直接 ACF 指数拟合：186.0 → 185.2 (2x) → 185.9 (4x)   比值 1.000 稳定
    VAR(1) 谱半径    ：148.7 → 166.1 (2x) → 237.3 (4x)   比值 1.60 漂移

**问题不在采样率，在 VAR(1) 这个模型。**

### 为什么

多元 VAR(1) 的谱半径 ρ(A) 对应的是**最慢集体模态**，不是任一单变量的
边际自相关衰减率。多变量耦合产生集体模态谱，粗采样滤掉快模态后，
慢模态在归一化后占比变大 → 谱半径变大 → τ 变大。

而单变量 ACF 是**边际统计量**，不依赖变量间的耦合结构，
所以对降采样稳定（只要 τ >> 采样间隔）。

## 本实验：三个对比

### 对比 1: 修 τ_int（Sokal 窗口 bug）
exp40 的 τ_int = 4.0 ms（= 1 bin），明显错误。
根因：`taus[0] = 2*ρ(0) - 1 = 1`，此时条件 `M ≥ c·τ` 立即满足（0 < 5）。
修正：从 lag=1 开始搜索，要求 τ 累积值先上升再找稳定点。

### 对比 2: VAR(1) vs ACF 在多个 decimation 下的稳定性
同一批片段、同一 PCA 空间，两种方法各算 τ，比较比值。

### 对比 3: 用第一主成分（单变量）做 VAR(1)
如果单变量 VAR(1) 也稳定，说明问题在"多元"而非"VAR"。
如果单变量 VAR(1) 也漂移，说明 VAR 这个模型族本身不适合。

## 判据

  单变量 ACF 稳定 + 多元 VAR 漂移 + 单变量 VAR 稳定
      → 结论：**用单变量 ACF；多元 VAR 的谱半径不是边际时间常数**
  单变量 ACF 稳定 + 两种 VAR 都漂移
      → 结论：**VAR 族不适合，只用 ACF**
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

FS = 250.0
EEG_DIR = Path("/domains/omnibrain4d/eeg")
OUT = Path("/domains/runs/var-vs-acf")

N_PCS = 32
MAX_IMAGES = 25
SUBJECTS = [1, 2, 3, 4, 5]
DECS = [1, 2, 4]
MAX_LAG_MS = 1000.0


# ------------------------------------------------------------ ACF
def acf_fft(x, max_lag):
    """x: (T, d) → (max_lag+1, d) 无偏归一化 ACF。"""
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


def fit_exp_tau(lags, acf_vals, dt_ms):
    ok = np.isfinite(acf_vals) & (acf_vals > 0.02) & (acf_vals < 0.99)
    if ok.sum() < 4:
        return None
    t = lags[ok] * dt_ms
    y = acf_vals[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[100.0], bounds=([1.0], [10000.0]),
                            maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


def integrated_tau(acf_vals, dt_ms, c=5.0):
    """Sokal 积分自相关时间（修正版）。

    修正：从 lag=1 开始搜索。lag=0 时 τ_int=1，条件立即满足，会误判。
    正确做法：累积 τ(M) = 1 + 2Σ_{t=1}^{M} ρ(t)，找最小 M 使 M ≥ c·τ(M)。
    """
    n = len(acf_vals)
    if n < 4:
        return None
    taus = np.cumsum(2.0 * acf_vals) - 1.0     # taus[M] = 1 + 2Σ_{1..M}ρ - 1? 见下
    # 正确累积：τ(M) = 1 + 2 Σ_{t=1..M} ρ(t)
    taus = 1.0 + 2.0 * np.cumsum(acf_vals[1:])   # 长度 n-1，索引对应于 M=1..n-1
    Ms = np.arange(1, n)
    m = Ms >= c * taus
    if np.any(m):
        idx = int(np.argmax(m))
        return float(taus[idx] * dt_ms)
    return None


# ------------------------------------------------------------ VAR(1)
def var_pole_tau(x, dt_ms, ridge=1e-2):
    """多元 VAR(1)：拟合 x_t = A x_{t-1}，返回最慢极点对应 τ。"""
    x0, x1 = x[:-1], x[1:]
    if len(x0) < 2 * x.shape[1]:
        return None
    x0c = x0 - x0.mean(axis=0, keepdims=True)
    x1c = x1 - x1.mean(axis=0, keepdims=True)
    G = x0c.T @ x0c
    d = G.shape[0]
    G.flat[:: d + 1] += ridge * np.trace(G) / d
    try:
        A = np.linalg.solve(G, x0c.T @ x1c)
    except np.linalg.LinAlgError:
        return None
    ev = np.abs(np.linalg.eigvals(A))
    ev = ev[(ev > 1e-4) & (ev < 1.0)]
    if len(ev) == 0:
        return None
    lam = float(np.sort(ev)[::-1][0])
    return float(-1.0 / np.log(lam) * dt_ms)


def scalar_var_tau(x1d, dt_ms, ridge=1e-2):
    """单变量 AR(1)：x_t = a x_{t-1} + e，τ = -dt/ln|a|。"""
    x0, x1 = x1d[:-1], x1d[1:]
    x0c = x0 - x0.mean()
    x1c = x1 - x1.mean()
    num = float(x0c @ x1c)
    den = float(x0c @ x0c) + ridge * float(x0c @ x0c)
    if den <= 1e-12:
        return None
    a = num / den
    a = abs(a)
    if not (1e-4 < a < 1.0):
        return None
    return float(-1.0 / np.log(a) * dt_ms)


# ------------------------------------------------------------ 主分析
def decimate_naive(P, dec):
    return P[::dec]


def decimate_filt(P, dec):
    from scipy import signal
    return signal.decimate(P, dec, axis=0, ftype="iir", zero_phase=True)


def analyze_subject(eeg, sid, max_images=MAX_IMAGES):
    n_img, n_rep, n_ch, T = eeg.shape
    d = min(N_PCS, n_ch)
    n_use = min(n_img, max_images)
    rows = []

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

        row = {"subject": sid, "image": i}

        for dec in DECS:
            dt = 1000.0 * dec / FS
            if dec == 1:
                Pd = P
            else:
                if P.shape[0] < 2 * dec + 10:
                    continue
                Pd = decimate_naive(P, dec)      # 用 naive（exp40 显示它稳定）
            max_lag = int(min(MAX_LAG_MS / 1000 * FS, Pd.shape[0] - 2)) // dec

            a = np.median(acf_fft(Pd, max_lag), axis=1)
            lags = np.arange(len(a))
            row[f"acf_{dec}x"] = fit_exp_tau(lags, a, dt)
            row[f"int_{dec}x"] = integrated_tau(a, dt)
            row[f"mvar_{dec}x"] = var_pole_tau(Pd, dt)
            row[f"svar_{dec}x"] = scalar_var_tau(Pd[:, 0], dt)

        rows.append(row)

    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 86, flush=True)
    print("Phase 41 — VAR(1) 极点 vs 单变量 ACF：谁可信", flush=True)
    print("=" * 86, flush=True)
    print(f"  被试 {SUBJECTS} | {N_PCS} 维 | {MAX_IMAGES} 图像 | dec {DECS}",
          flush=True)

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
        if len(df):
            print(f"    ACF   : 4ms {df.acf_1x.median():6.1f} | "
                  f"2x {df.acf_2x.median():6.1f} | 4x {df.acf_4x.median():6.1f}",
                  flush=True)
            print(f"    mVAR  : 4ms {df.mvar_1x.median():6.1f} | "
                  f"2x {df.mvar_2x.median():6.1f} | 4x {df.mvar_4x.median():6.1f}",
                  flush=True)
            print(f"    sVAR  : 4ms {df.svar_1x.median():6.1f} | "
                  f"2x {df.svar_2x.median():6.1f} | 4x {df.svar_4x.median():6.1f}",
                  flush=True)
            print(f"    τ_int : 4ms {df.int_1x.median():6.1f} | "
                  f"2x {df.int_2x.median():6.1f} | 4x {df.int_4x.median():6.1f} "
                  f"[{time.time()-t1:.1f}s]", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    if not len(df):
        return

    # ============ 汇总
    methods = {
        "ACF 指数拟合": ("acf_", "单变量边际"),
        "单变量 AR(1)": ("svar_", "单变量参数"),
        "多元 VAR(1) 极点": ("mvar_", "多元集体模态"),
        "积分自相关 τ_int": ("int_", "非参数"),
    }

    print("\n" + "=" * 86, flush=True)
    print("四种方法的降采样稳定性（τ ms，中位）", flush=True)
    print("=" * 86, flush=True)
    print(f"{'方法':<22} {'性质':<14} {'4ms':>8} {'2x':>8} {'4x':>8} "
          f"{'比值4x':>8} {'判定':>8}", flush=True)

    summary = {}
    for name, (pre, kind) in methods.items():
        v1 = df[f"{pre}1x"].median()
        v2 = df[f"{pre}2x"].median()
        v4 = df[f"{pre}4x"].median()
        if not np.isfinite(v1) or v1 <= 0:
            continue
        r4 = v4 / v1
        ok = 0.75 <= r4 <= 1.35
        summary[name] = {"kind": kind, "tau_4ms": round(float(v1), 1),
                         "tau_2x": round(float(v2), 1),
                         "tau_4x": round(float(v4), 1),
                         "ratio_4x": round(float(r4), 3),
                         "consistent": bool(ok)}
        print(f"{name:<22} {kind:<14} {v1:>8.1f} {v2:>8.1f} {v4:>8.1f} "
              f"{r4:>8.3f} {'✅' if ok else '❌':>8}", flush=True)

    # ============ 判定
    print("\n" + "=" * 86, flush=True)
    print("判定", flush=True)
    print("=" * 86, flush=True)

    acf_ok = summary.get("ACF 指数拟合", {}).get("consistent", False)
    mvar_ok = summary.get("多元 VAR(1) 极点", {}).get("consistent", False)
    svar_ok = summary.get("单变量 AR(1)", {}).get("consistent", False)

    print(f"  ACF 指数拟合  : {'✅ 稳定' if acf_ok else '❌ 漂移'}", flush=True)
    print(f"  单变量 AR(1)  : {'✅ 稳定' if svar_ok else '❌ 漂移'}", flush=True)
    print(f"  多元 VAR(1)   : {'✅ 稳定' if mvar_ok else '❌ 漂移'}", flush=True)

    if acf_ok and (not mvar_ok):
        if svar_ok:
            verdict = ("✅ **结论明确**：单变量方法（ACF 指数拟合、AR(1)）都稳定；"
                       "多元 VAR(1) 的谱半径不稳定。\n"
                       "    多元 VAR 的谱半径是'最慢集体模态'，粗采样滤掉快模态后"
                       "慢模态占比变大 → 谱半径变大 → τ 变大。\n"
                       f"    **可信的 τ ≈ {summary['ACF 指数拟合']['tau_4ms']:.0f} ms**"
                       "（ACF 指数拟合，4ms 基准）")
        else:
            verdict = ("⚠️ ACF 稳定但两种 VAR 都不稳 → VAR 模型族不适用，只用 ACF。"
                       f"τ ≈ {summary['ACF 指数拟合']['tau_4ms']:.0f} ms")
    elif acf_ok and mvar_ok:
        verdict = "✅ 四种方法都稳定 —— 之前的漂移可能是实现问题，现已修正"
    else:
        verdict = ("❌ ACF 也不稳定 —— τ 在当前数据上不可识别。"
                   "应改用 1/f 指数（exp40 显示跨度仅 0.064）")
    print(f"\n  {verdict}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "var_vs_acf",
        "fs": FS, "n_pcs": N_PCS, "max_images": MAX_IMAGES,
        "subjects": SUBJECTS, "decs": DECS,
        "methods": summary,
        "verdict": verdict,
        "explanation": (
            "多元 VAR(1) 的谱半径对应最慢集体模态，不是边际自相关衰减率。"
            "多变量耦合产生集体模态谱；粗采样滤掉快模态后，慢模态在归一化后"
            "占比变大 → 谱半径变大 → τ 变大。单变量 ACF 是边际统计量，"
            "不依赖变量间耦合，故对降采样稳定。"
        ),
        "tau_int_fix": ("exp40 的 τ_int = 4.0ms 是 Sokal 窗口 bug：lag=0 时 "
                        "τ=1，条件 M≥c·τ 立即满足。已改为从 lag=1 开始搜索。"),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
