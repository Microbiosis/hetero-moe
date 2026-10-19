"""τ 的正交不变估计量。

## 为什么需要

PCA 给出的基是「每一叶内部的典范基」，但片段之间没有任何关系。
任何对逐成分 ACF 取统计量（median / mean / max）的做法都会依赖这组基。

对任意正交 Q：

    tr C(τ) Qᵀ = tr(Qᵀ C(τ) Q) = tr C(τ)

所以 `tr C(τ) / tr C(0)` 与基的选取无关。它是按方差加权的平均 ACF。

## 什么时候必须用它

先用 `top1_fraction(x)` 看信号分布：

    top-1 > 0.2  ->  必须用 weighted_acf（逐成分统计会被噪声或单成分主导）
    top-1 < 0.2  ->  逐成分中位数与不变量接近，但仍推荐 weighted_acf

实测：头皮 EEG top-1 = 0.052，两者差 6%；
      20 ms spike top-1 = 0.438，两者差 39 倍。

## 用法

    from lib.tau import weighted_acf, fit_tau, tau_int, top1_fraction

    R = weighted_acf(zscored, max_lag=100)        # (101,) 无量纲
    tau_ms = fit_tau(R, dt_ms=20.0)
    t_int = tau_int(R, dt_ms=20.0)                # 不依赖指数假设
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["acf_cols", "weighted_acf", "fit_tau", "tau_int",
           "gauge_test", "top1_fraction"]


def _acf_core(x: np.ndarray, max_lag: int) -> np.ndarray:
    """逐列 ACF，无偏归一化。x: (T, d) -> (max_lag+1, d)"""
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


def acf_cols(x: np.ndarray, max_lag: int) -> np.ndarray:
    """逐列 ACF。**基依赖**，仅供诊断，不要在其上取 median/mean 后当结论。"""
    return _acf_core(x, max_lag)


def weighted_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """tr C(τ) / tr C(0)。正交不变。

    注意：不能复用 `_acf_core`，它对每列除以各自的 lag-0 方差。
    旋转后每列方差改变，那个归一化会破坏不变性（实测极差 0.01 而非 0）。
    必须先在未归一化的 ACF 上按列求和，再除以 lag-0 的总和。
    """
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


def fit_tau(a: np.ndarray, dt_ms: float, lo: float = 2.0,
            hi: float = 50000.0, thresh: float = 0.02) -> float | None:
    """拟合 R(t) = exp(-t/T)。返回 T（ms）或 None（可辨识点不足）。"""
    ok = np.isfinite(a) & (a > thresh) & (a < 0.995)
    if ok.sum() < 5:
        return None
    t = np.arange(len(a))[ok] * dt_ms
    y = a[ok]

    def f(tt, T):
        return np.exp(-tt / np.maximum(T, 1e-3))

    try:
        popt, _ = curve_fit(f, t, y, p0=[200.0], bounds=([lo], [hi]), maxfev=8000)
        return float(popt[0])
    except Exception:
        return None


def tau_int(a: np.ndarray, dt_ms: float, c: float = 5.0) -> float:
    """Sokal 积分自相关时间：τ = 1 + 2Σρ，窗口 M ≥ c·τ。不做指数假设。"""
    taus = 1.0 + 2.0 * np.cumsum(a[1:])
    Ms = np.arange(1, len(a))
    m = Ms >= c * np.maximum(taus, 1e-9)
    if np.any(m):
        return float(taus[int(np.argmax(m))] * dt_ms)
    return float(taus[-1] * dt_ms)


def gauge_test(x: np.ndarray, max_lag: int, n_rot: int = 5,
               seed: int = 0) -> dict:
    """检验统计量在随机正交旋转下的稳定性。

    返回
        inv_spread  不变量 weighted_acf 的极差（应为 0，到浮点精度）
        med_spread  逐成分中位数的极差（不为 0 说明基依赖）
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    inv, med = [], []
    for _ in range(n_rot):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        xq = x @ Q
        inv.append(weighted_acf(xq, max_lag))
        med.append(np.median(_acf_core(xq, max_lag), axis=1))
    return {
        "inv_spread": float(np.ptp(np.array(inv), axis=0).max()),
        "med_spread": float(np.ptp(np.array(med), axis=0).max()),
        "n_rot": n_rot,
    }


def top1_fraction(x: np.ndarray) -> float:
    """top-1 特征值占比。> 0.2 时不要用逐成分统计。"""
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False) ** 2
    return float(s[0] / max(s.sum(), 1e-12))
