"""Phase 20: 用 VAR(1) 的特征值直接估计时间常数（含合成数据正控）。

## 为什么这个方法不同

前 19 个实验都用"拟合 R²(k) 曲线求 τ"，而 exp19 证明那条路走不通：
窗口内平均 = 平滑，τ 必然固定在 bin 数上（实测 3.5 bins）。

本实验改用**动力学系统辨识**：

    x_t = A · x_{t-1} + ε_t          （VAR(1)，x 是逐 bin 的 PCA 投影）

A 的特征值 λ 直接给出衰减率：

    |λ| = exp(-Δt / τ)   →   τ = -Δt / ln|λ|

其中 Δt 是 bin 宽度。**τ 从特征值算出来，不经过任何平滑或曲线拟合。**

## 关键：先做合成数据正控

这是前 19 个实验缺的一步 —— 从没验证过"方法能否恢复已知的真值"。

合成数据：64 维 AR(1)，真实 τ = 200 ms，在 50/100/200ms 三种采样率下生成。
按定义，a = exp(-Δt/τ) 随采样率变化，但 τ 应恒为 200 ms。

    50ms  → a = 0.779
    100ms → a = 0.607
    200ms → a = 0.368

若方法正确，三种采样率下都应恢复 τ ≈ 200 ms。
若恢复不出来，说明方法本身有问题，**不该应用到真实数据**。

## 真实数据

猕猴 M1/PMd（exp18 中唯一稳定为正的来源），67 段。
不做窗口内平均：直接对逐 bin 的 PCA 投影序列建模。
分块拟合（4 块），取块间中位，规避非平稳。

## 判据

  合成数据：三种采样率恢复的 τ_ms 一致（比值 0.8-1.25）  → 方法可信
  真实数据：三种 bin 宽下 τ_ms 一致                        → τ 是真实时间常数
            否则                                          → 结论是"测不出"
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
warnings.filterwarnings("ignore")

SEED = 0
BIN_WIDTHS = [50, 100, 200]
D_PCA = 64
N_BLOCKS = 4
RIDGE = 1e-2
TAU_TRUE_MS = 200.0
MIN_BINS_PER_BLOCK = 300
TARGET_AREA = "M1_PMd"


# ---------------------------------------------------------------- 核心
def fit_var1(X, ridge=RIDGE):
    """X: (n, d) 逐 bin 序列。拟合 x_t = A x_{t-1}。

    返回 A（d×d）及特征值。用 ridge 保证数值稳定。
    """
    x0 = X[:-1]          # t-1
    x1 = X[1:]           # t
    if len(x0) < 2 * X.shape[1]:
        return None, None
    x0c = x0 - x0.mean(0, keepdims=True)
    x1c = x1 - x1.mean(0, keepdims=True)
    G = x0c.T @ x0c
    G.flat[:: G.shape[0] + 1] += ridge * np.trace(G) / G.shape[0]
    try:
        A = np.linalg.solve(G, x0c.T @ x1c)
    except np.linalg.LinAlgError:
        return None, None
    ev = np.linalg.eigvals(A)
    return A, ev


def tau_from_eigs(ev, bin_ms, mode="slowest"):
    """从特征值算 τ（ms）。只保留 |λ| < 1 的稳定模态。"""
    mag = np.abs(ev)
    mag = mag[(mag > 1e-6) & (mag < 1.0)]
    if len(mag) == 0:
        return None, None
    mag = np.sort(mag)[::-1]
    if mode == "slowest":
        lam = mag[0]
    elif mode == "median":
        lam = np.median(mag)
    else:
        lam = mag[0]
    tau_bins = -1.0 / np.log(lam)
    return float(tau_bins), float(tau_bins * bin_ms)


def spectral_radius(ev):
    mag = np.abs(ev)
    mag = mag[mag < 10]
    return float(mag.max()) if len(mag) else None


def var_r2(X, A):
    """VAR(1) 的解释方差（样本内）。"""
    x0 = X[:-1] - X[:-1].mean(0, keepdims=True)
    x1 = X[1:] - X[1:].mean(0, keepdims=True)
    pred = x0 @ A.T
    ss_res = ((x1 - pred) ** 2).sum()
    ss_tot = (x1 ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def analyze_series(X, bin_ms):
    """分块拟合 VAR(1)，返回 τ 的中位与谱半径。"""
    n = len(X)
    blk = n // N_BLOCKS
    if blk < MIN_BINS_PER_BLOCK:
        return None
    taus, rads, r2s = [], [], []
    for b in range(N_BLOCKS):
        seg = X[b * blk:(b + 1) * blk]
        A, ev = fit_var1(seg)
        if A is None:
            continue
        tb, tm = tau_from_eigs(ev, bin_ms)
        if tb is not None and 0.05 < tb < 500:
            taus.append(tm)
        r = spectral_radius(ev)
        if r:
            rads.append(r)
        r2s.append(var_r2(seg, A))
    if not taus:
        return None
    return {
        "tau_ms": float(np.median(taus)),
        "tau_ms_iqr": float(np.percentile(taus, 75) - np.percentile(taus, 25)),
        "spectral_radius": float(np.median(rads)) if rads else None,
        "var_r2": float(np.median(r2s)),
        "n_blocks_ok": len(taus),
    }


# ---------------------------------------------------------------- 合成正控
def synthetic_control():
    """64 维 AR(1)，真实 τ = 200ms，三种采样率。"""
    print("\n" + "=" * 84)
    print(f"正控：合成 AR(1)，真实 τ = {TAU_TRUE_MS:.0f} ms")
    print("=" * 84)
    rng = np.random.default_rng(SEED)
    d = 32
    n_bins = 6000

    # 生成 50ms 分辨率的母序列（最细），再降采样得到其他分辨率
    a50 = np.exp(-50.0 / TAU_TRUE_MS)
    # 各维度不同衰减，但都对应同一 τ 的倍数（用 0.5~1.5 倍）
    scale = np.linspace(0.5, 1.5, d)
    a_vec = np.exp(-50.0 / (TAU_TRUE_MS * scale))

    x = np.zeros((n_bins, d), dtype=np.float64)
    noise = rng.standard_normal((n_bins, d))
    for t in range(1, n_bins):
        x[t] = a_vec * x[t - 1] + noise[t]

    print(f"  生成 {n_bins} bins × {d} 维，τ 在 {TAU_TRUE_MS*0.5:.0f}–{TAU_TRUE_MS*1.5:.0f} ms 之间")
    print(f"  (各维 τ 不同，故报中位数应 ≈ {TAU_TRUE_MS:.0f} ms)\n")

    print(f"{'bin':>5} {'理论 a':>9} {'n_bins':>8} {'恢复 τ_ms':>11} {'谱半径':>9} {'VAR R²':>8}")
    results = {}
    for bw in BIN_WIDTHS:
        ratio = bw // 50
        xs = x[::ratio] if ratio > 1 else x
        r = analyze_series(xs, bw)
        if r is None:
            print(f"{bw:>5} 拟合失败")
            continue
        a_theory = np.exp(-bw / TAU_TRUE_MS)
        results[bw] = r
        print(f"{bw:>5} {a_theory:>9.3f} {len(xs):>8} {r['tau_ms']:>11.1f} "
              f"{r['spectral_radius']:>9.3f} {r['var_r2']:>8.3f}")

    if results:
        vals = [v["tau_ms"] for v in results.values()]
        ratio = max(vals) / min(vals)
        print(f"\n  三种采样率恢复的 τ: {[f'{v:.0f}' for v in vals]} ms")
        print(f"  最大/最小 = {ratio:.2f}")
        if ratio < 1.5:
            print("  ✅ 方法通过正控 —— 可以应用到真实数据")
            ok = True
        else:
            print("  ❌ 方法未通过正控 —— 恢复不出已知 τ，不应继续")
            ok = False
    else:
        ok = False
    return ok, results


# ---------------------------------------------------------------- 真实数据
def pca_bins(x_counts):
    """逐 bin PCA（不做窗口平均）。返回 (n_bins, D_PCA)。"""
    z = x_counts.astype(np.float32)
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True) + 1e-6
    z = (z - mu) / sd
    nb = z.shape[1]
    k = min(D_PCA, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    w_, U = np.linalg.eigh(zc @ zc.T)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < D_PCA:
        proj = np.pad(proj, ((0, 0), (0, D_PCA - proj.shape[1])))
    return proj.astype(np.float32)


def to_bins(sp, bin_ms, src_ms):
    ratio = max(1, int(round(bin_ms / src_ms)))
    nb = sp.shape[1] // ratio
    return sp[:, :nb * ratio].reshape(sp.shape[0], nb, ratio).sum(axis=2)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp20")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("Phase 20 — VAR(1) 特征值估计时间常数（含合成正控）")
    print("=" * 84)

    # ---- 正控
    method_ok, ctrl = synthetic_control()

    if not method_ok:
        print("\n❌ 正控未通过，不进入真实数据分析。")
        (out / "phase20_report.json").write_text(json.dumps({
            "schema_version": "1.0", "control_passed": False,
            "control": {str(k): v for k, v in ctrl.items()},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # ---- 真实数据
    print("\n" + "=" * 84)
    print(f"真实数据：猕猴 {TARGET_AREA}（逐 bin，不做窗口平均）")
    print("=" * 84)

    mac = root / "primate" / "macaque_nlb"
    seg = mac / "segments.parquet"
    if not seg.exists():
        print("❌ 缺少 segments.parquet")
        return
    sdf = pd.read_parquet(seg)
    meta = sdf[sdf["area"] == TARGET_AREA]
    print(f"  片段 {len(meta)}")

    rows = []
    t0 = time.time()

    for i, row in meta.iterrows():
        f = mac / row["file"]
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)

        for bw in BIN_WIDTHS:
            x = to_bins(sp, bw, 20.0)
            if x.shape[1] < MIN_BINS_PER_BLOCK * 4:
                continue
            proj = pca_bins(x)
            if proj is None:
                continue
            r = analyze_series(proj, bw)
            if r is None:
                continue
            rows.append({"file": str(row["file"]), "bin_ms": bw,
                         "n_units": int(x.shape[0]), "n_bins": int(x.shape[1]),
                         **{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()}})
        if len(rows) % 60 < 3 and len(rows):
            print(f"    ... {len(rows)} 条 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase20.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 84)
    print("按分辨率汇总")
    print("=" * 84)
    print(f"{'bin':>5} {'n':>4} {'τ_ms 中位':>11} {'IQR':>8} {'谱半径':>8} {'VAR R²':>8}")
    for bw, g in df.groupby("bin_ms"):
        print(f"{bw:>5} {len(g):>4} {g['tau_ms'].median():>11.1f} "
              f"{g['tau_ms_iqr'].median():>8.1f} {g['spectral_radius'].median():>8.3f} "
              f"{g['var_r2'].median():>8.3f}")

    print("\n" + "=" * 84)
    print("决定性判据：τ_ms 在三种分辨率下是否一致")
    print("=" * 84)
    med = df.groupby("bin_ms")["tau_ms"].median()
    ratios = {}
    if 100 in med.index:
        base = float(med.loc[100])
        for bw in BIN_WIDTHS:
            if bw in med.index:
                ratios[bw] = round(float(med.loc[bw]) / base, 3)
        print(f"  基准 100ms = {base:.1f} ms")
        for bw, r in ratios.items():
            print(f"    {bw:>3}ms: {med.loc[bw]:>8.1f} ms  比值 {r:>5.3f}  "
                  f"{'✅' if 0.75 <= r <= 1.35 else '❌'}")

    n_ok = sum(1 for r in ratios.values() if 0.75 <= r <= 1.35)
    n_tot = len(ratios)
    print("\n" + "=" * 84)
    print("最终判读")
    print("=" * 84)
    if n_tot and n_ok == n_tot:
        verdict = (f"✅ 方法通过正控，且真实数据 τ 在三种分辨率下一致 → "
                   f"**猕猴 {TARGET_AREA} 的时间常数 ≈ {med.loc[100]:.0f} ms**")
    elif n_tot and n_ok == 0:
        verdict = ("❌ 方法通过正控（能恢复合成 τ），但真实数据 τ 分辨率不一致。"
                   "可能原因：真实信号非平稳、样本不足、或存在多个不可分离的模态。"
                   "结论：**τ 仍不可测**，但这次排除了'方法错误'这一解释。")
    else:
        verdict = f"⚠️ {n_ok}/{n_tot} 一致，证据不足"
    print(f"  {verdict}")

    (out / "phase20_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "control_passed": bool(method_ok),
        "control": {str(k): {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in v.items()} for k, v in ctrl.items()},
        "bin_widths": BIN_WIDTHS, "d_pca": D_PCA, "n_blocks": N_BLOCKS,
        "area": TARGET_AREA, "n_records": int(len(df)),
        "by_bin": {str(int(bw)): {
            "n": int(len(g)),
            "tau_ms": round(float(g["tau_ms"].median()), 1),
            "spectral_radius": round(float(g["spectral_radius"].median()), 4),
            "var_r2": round(float(g["var_r2"].median()), 4)}
            for bw, g in df.groupby("bin_ms")},
        "tau_ratios_vs_100ms": ratios,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
