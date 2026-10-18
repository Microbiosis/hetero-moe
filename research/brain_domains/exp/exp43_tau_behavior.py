"""Phase 43: τ 与行为的相关 —— 逐试次设计（DMFC ready-set-go）。

## 为什么换设计

exp42 用跨被试相关（n=10）检验 τ 与解码准确率，结果 ρ=−0.10。
但那个设计有两个问题：

1. **用解码代替行为** —— 文献的"整合窗口"是关于行为的
2. **n=10 功效不足** —— 需 |ρ|>0.65

DMFC_RSG 数据集解决了第一个问题：它有**真实的计时行为**。
但只有 1 个被试（sub-Haydn），所以不能做跨被试。

改用**逐试次设计**：
    每个试次单独估 τ（该试次神经活动的衰减速率）
    每个试次有计时误差 |t2 - t1|
    看 τ 是否预测该试次的计时准确度（n = 1029）

## 数据（DANDI 000130）

    intervals/trials 里的字段：
      ready_time, set_time, go_time
      → 目标间隔 t1 = set - ready
      → 产生间隔 t2 = go - set
      → 计时误差 = t2 - t1（中位 −2 ms，SD 147 ms）
      → |误差| 中位 69 ms（9% 相对误差）

    units/spike_times: 54 个 unit 的绝对时间戳

## 分析窗口

ready-set-go 的关键是 **set→go 的等待期**（被试必须在这段时间内
维持 t1 的记忆并倒计时）。取 set 之后到 go 之前作为主要窗口。

但注意：不同试次的这个窗口长度不同（因为 t1 不同）。
所以必须做 **时间归一化**，或者只取固定长度（如 set 后 400ms）。

## 三种 τ 估计

    1. ACF 指数拟合（exp41 验证过，5/5 通过分辨率检验）
    2. 积分自相关 τ_int（若窗口足够）
    3. 1/f 指数（作为对照，非 τ）

## 判据

    Spearman ρ(τ, |计时误差|) 显著负 → τ 长 = 更准 → 支持"整合窗口"假设
    ρ ≈ 0                            → τ 与行为无关
    显著正                            → 反方向，需解释

## 必须做的对照

1. **试次级打乱**：把 τ 与行为随机配对，看 ρ 的零分布
2. **控制反应时**：t2 本身与 τ 可能都受"唤醒度"影响
3. **控制 unit 数变异**：某些试次 spike 计数少，τ 估不准
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

OUT = Path("/domains/runs/tau-behavior")
NWB_TRAIN = "/domains/primate/dmfc_rsg/sub-Haydn_desc-train_ecephys.nwb"

FS = 250.0                 # 统一到 250 Hz（4ms bin），与 exp41 一致
BIN_MS = 4.0
WIN_AFTER_SET_MS = 600.0   # set 之后的窗口（固定长度，避免随 t1 变）
N_PCS = 16
MAX_LAG_MS = 400.0


# ------------------------------------------------------------ NWB 读取
def load_trials(nwb_path):
    import h5py
    with h5py.File(nwb_path, "r") as f:
        t = f["intervals"]["trials"]
        d = {k: t[k][:] for k in
             ["ready_time", "set_time", "go_time", "start_time", "stop_time",
              "is_outlier", "is_short"]}
        u = f["units"]
        spike_times = u["spike_times"][:]
        st_idx = u["spike_times_index"][:]
    # 按 unit 切分
    bounds = np.concatenate([[0], st_idx])
    units = [spike_times[bounds[i]:bounds[i + 1]] for i in range(len(st_idx))]
    return d, units


def bin_spikes(units, t0, t1, bin_ms=BIN_MS):
    """把 [t0, t1) 内的 spike 转成 (n_bins, n_units) 计数矩阵。"""
    edges = np.arange(t0, t1 + bin_ms / 1000, bin_ms / 1000)
    if len(edges) < 4:
        return None
    M = np.zeros((len(edges) - 1, len(units)), dtype=np.float32)
    for j, sp in enumerate(units):
        if len(sp) == 0:
            continue
        s = sp[(sp >= t0) & (sp < t1)]
        if len(s) == 0:
            continue
        idx = np.clip(((s - t0) / (bin_ms / 1000)).astype(int), 0, M.shape[0] - 1)
        np.add.at(M[:, j], idx, 1.0)
    return M


# ------------------------------------------------------------ τ 估计
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
    if P.shape[0] < 8:
        return None
    a = np.median(acf_fft(P, max_lag), axis=1)
    lags = np.arange(len(a))
    ok = np.isfinite(a) & (a > 0.02) & (a < 0.99)
    if ok.sum() < 4:
        return None

    def f(t, tau):
        return np.exp(-t / np.maximum(tau, 1e-3))

    try:
        popt, _ = curve_fit(f, lags[ok] * dt_ms, a[ok], p0=[100.0],
                            bounds=([2.0], [5000.0]), maxfev=6000)
        return float(popt[0])
    except Exception:
        return None


def integrated_tau(a, dt_ms, c=5.0):
    if len(a) < 6:
        return None
    taus = 1.0 + 2.0 * np.cumsum(a[1:])
    Ms = np.arange(1, len(a))
    m = Ms >= c * taus
    if np.any(m):
        return float(taus[int(np.argmax(m))] * dt_ms)
    return None


# ------------------------------------------------------------ 主分析
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 82, flush=True)
    print("Phase 43 — τ vs 计时行为（逐试次，DMFC ready-set-go）", flush=True)
    print("=" * 82, flush=True)

    if not Path(NWB_TRAIN).exists():
        print(f"缺 {NWB_TRAIN}", flush=True)
        return

    t0 = time.time()
    trials, units = load_trials(NWB_TRAIN)
    n_trial = len(trials["go_time"])
    print(f"  试次 {n_trial} | unit {len(units)} | 载入 {time.time()-t0:.1f}s",
          flush=True)

    # ---- 行为
    rt = trials["ready_time"]; st_ = trials["set_time"]; gt = trials["go_time"]
    t1 = st_ - rt          # 目标间隔
    t2 = gt - st_          # 产生间隔
    ok_beh = (np.isfinite(t1) & np.isfinite(t2) & (t1 > 0.1) &
              (t2 > 0.05) & (~trials["is_outlier"]) & (~trials["is_short"]))
    print(f"  行为有效试次 {ok_beh.sum()}", flush=True)
    err = t2 - t1
    print(f"  计时误差: 中位 {np.nanmedian(err[ok_beh]):+.3f}s, "
          f"|误差| 中位 {np.nanmedian(np.abs(err[ok_beh])):.3f}s", flush=True)

    # ---- 逐试次 τ
    rows = []
    win_s = WIN_AFTER_SET_MS / 1000.0
    dt_ms = BIN_MS
    max_lag = int(MAX_LAG_MS / BIN_MS)

    for i in range(n_trial):
        if not ok_beh[i]:
            continue
        a0 = st_[i]
        a1 = a0 + win_s
        M = bin_spikes(units, a0, a1)
        if M is None or M.shape[0] < 20:
            continue
        # 逐 unit z-score 后 PCA
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, keepdims=True) + 1e-6
        Z = (M - mu) / sd
        d = min(N_PCS, Z.shape[1] - 1)
        try:
            U, S, _ = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        P = (U[:, :d] * S[:d]).astype(np.float64)

        a = np.median(acf_fft(P, max_lag), axis=1)
        tau = estimate_tau(P, dt_ms, max_lag)
        ti = integrated_tau(a, dt_ms)

        rows.append({
            "trial": i,
            "t1_ms": round(float(t1[i] * 1000), 1),
            "t2_ms": round(float(t2[i] * 1000), 1),
            "err_ms": round(float(err[i] * 1000), 1),
            "abs_err_ms": round(float(abs(err[i]) * 1000), 1),
            "rel_err": round(float(abs(err[i]) / t1[i]), 4),
            "tau_ms": round(tau, 1) if tau else None,
            "tau_int_ms": round(ti, 1) if ti else None,
            "n_spikes": int(M.sum()),
            "acf1": round(float(a[1]), 4) if len(a) > 1 else None,
            "n_bins": int(M.shape[0]),
        })
        if len(rows) % 100 == 0:
            print(f"    {len(rows)} 试次 ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(f"\n  完成 {len(df)} 试次 | 耗时 {time.time()-t0:.0f}s", flush=True)

    d = df.dropna(subset=["tau_ms", "abs_err_ms"])
    if len(d) < 30:
        print("有效试次不足", flush=True)
        return

    print("\n" + "=" * 82, flush=True)
    print(f"相关分析（n={len(d)} 试次）", flush=True)
    print("=" * 82, flush=True)

    def rep(x, y, name):
        r, p = spearmanr(x, y)
        print(f"  {name:<34} ρ = {r:+.4f}  p = {p:.2e}", flush=True)
        return float(r), float(p)

    r1, p1 = rep(d.tau_ms, d.abs_err_ms, "τ vs |计时误差|")
    r2, p2 = rep(d.tau_ms, d.rel_err, "τ vs 相对误差")
    r3, p3 = rep(d.tau_ms, d.t2_ms, "τ vs 产生间隔 t2")
    r4, p4 = rep(d.tau_ms, d.n_spikes, "τ vs spike 总数")
    r5, p5 = rep(d.n_spikes, d.abs_err_ms, "spike 总数 vs |误差|")

    # ---- 置换对照
    print("\n" + "-" * 82, flush=True)
    print("置换对照（1000 次打乱 τ 与行为的配对）", flush=True)
    print("-" * 82, flush=True)
    rng = np.random.default_rng(0)
    null = []
    x = d.tau_ms.to_numpy()
    y = d.abs_err_ms.to_numpy()
    for _ in range(1000):
        null.append(spearmanr(rng.permutation(x), y)[0])
    null = np.array(null)
    p_perm = float((np.abs(null) >= abs(r1)).mean())
    print(f"  零分布: 中位 {np.median(null):+.4f}  95%区间 "
          f"[{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]",
          flush=True)
    print(f"  观测 ρ={r1:+.4f} 的置换 p = {p_perm:.4f}", flush=True)

    # ---- 分 t1 层（控制目标间隔）
    print("\n" + "-" * 82, flush=True)
    print("按目标间隔 t1 分层（控制任务难度）", flush=True)
    print("-" * 82, flush=True)
    d2 = d.copy()
    d2["t1_bin"] = pd.qcut(d2.t1_ms, 4, labels=["Q1(短)", "Q2", "Q3", "Q4(长)"],
                           duplicates="drop")
    for lab, g in d2.groupby("t1_bin", observed=True):
        if len(g) < 20:
            continue
        rr, pp = spearmanr(g.tau_ms, g.abs_err_ms)
        print(f"  {lab:<10} n={len(g):>4}  t1中位={g.t1_ms.median():>6.0f}ms  "
              f"ρ = {rr:+.4f}  p = {pp:.3f}", flush=True)

    # ---- 判定
    print("\n" + "=" * 82, flush=True)
    print("判定", flush=True)
    print("=" * 82, flush=True)

    if p_perm < 0.05 and r1 < 0:
        verdict = (f"✅ τ 与计时误差显著负相关 (ρ={r1:+.3f}, 置换 p={p_perm:.4f}) "
                   f"—— τ 长 = 更准，支持「整合窗口」假设")
    elif p_perm < 0.05 and r1 > 0:
        verdict = (f"⚠️ τ 与计时误差显著正相关 (ρ={r1:+.3f}) "
                   f"—— 与「整合窗口」假设方向相反")
    elif abs(r1) < 0.05:
        verdict = (f"❌ 无相关 (ρ={r1:+.3f}, 置换 p={p_perm:.4f}) "
                   f"—— τ 不预测计时准确度")
    else:
        verdict = (f"⚠️ 弱相关但不显著 (ρ={r1:+.3f}, p={p_perm:.4f}) "
                   f"—— n={len(d)} 下功效不足")

    print(f"  {verdict}", flush=True)
    print(f"\n  注：单被试（sub-Haydn），只能做逐试次分析。", flush=True)
    print(f"      窗口：set 后 {WIN_AFTER_SET_MS:.0f} ms（固定，避免随 t1 变化）",
          flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "tau_vs_behavior",
        "dataset": "DANDI 000130 DMFC_RSG (sub-Haydn)",
        "design": "single-subject, trial-wise",
        "n_trials_total": int(n_trial),
        "n_trials_used": int(len(d)),
        "window_after_set_ms": WIN_AFTER_SET_MS,
        "n_pcs": N_PCS, "bin_ms": BIN_MS,
        "behavior": {
            "t1_median_ms": round(float(d.t1_ms.median()), 1),
            "t2_median_ms": round(float(d.t2_ms.median()), 1),
            "abs_err_median_ms": round(float(d.abs_err_ms.median()), 1),
            "rel_err_median": round(float(d.rel_err.median()), 4),
        },
        "correlations": {
            "tau_vs_abs_err": {"rho": round(r1, 4), "p": float(p1), "p_perm": p_perm},
            "tau_vs_rel_err": {"rho": round(r2, 4), "p": float(p2)},
            "tau_vs_t2": {"rho": round(r3, 4), "p": float(p3)},
            "tau_vs_nspikes": {"rho": round(r4, 4), "p": float(p4)},
            "nspikes_vs_abs_err": {"rho": round(r5, 4), "p": float(p5)},
        },
        "permutation": {"median": round(float(np.median(null)), 4),
                        "ci95": [round(float(np.percentile(null, 2.5)), 4),
                                 round(float(np.percentile(null, 97.5)), 4)],
                        "n": 1000},
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
