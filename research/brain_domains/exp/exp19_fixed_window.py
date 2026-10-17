"""Phase 19: 决定性检验 —— 窗口固定在 800ms（而非固定 8 个 bins）。

## 为什么这是最后一步

exp18 的分辨率检验失败（τ_ms 比值 0.594，应 ≈1.0）。原因：

    WIN = 8 固定的是 **bin 数**，不是物理时间。
        100ms bin → 窗口 800 ms
        200ms bin → 窗口 1600 ms    ← 物理窗口也变了

窗口内做时间平均是一种平滑，平滑的物理尺度变了，τ 自然随 bin 数恒定。

**正确检验必须固定物理窗口。**

## 三种配置（窗口都是 800ms）

    50ms  → WIN=16, STRIDE=16
    100ms → WIN=8,  STRIDE=8
    200ms → WIN=4,  STRIDE=4

若 τ 是真实时间常数，三种配置下 τ_ms 应一致。

## 聚焦猕猴 M1/PMd

exp18 中这是**唯一稳定为正**的来源（21/24 段净 R²(1) > 0，中位 ≈ +0.25），
其余（小鼠全部区域）净 R²(1) ≈ 0 或为负。用全部 67 段，不抽样。

## 同时修的问题

exp18 出现 net R² = -5.6e7 的记录（ridge 病态）。加入 |R²| > 10 过滤。
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
CTX = 8
KS = [1, 2, 4, 8]
WINDOW_MS = 800.0
BIN_WIDTHS = [50, 100, 200]
ALPHA = 1000.0
MAX_WIN = 400
N_BLOCKS = 4
R2_VALID_MAX = 10.0
TARGET_AREA = "M1_PMd"


def pca_proj(z):
    nb = z.shape[1]
    k = min(64, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    w_, U = np.linalg.eigh(zc @ zc.T)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < 64:
        proj = np.pad(proj, ((0, 0), (0, 64 - proj.shape[1])))
    return proj


def make_windows(proj, win, max_win=MAX_WIN):
    nb = proj.shape[0]
    if nb < win * 40:
        return None
    starts = list(range(0, nb - win + 1, win))        # 零重叠
    if len(starts) < 40:
        return None
    if len(starts) > max_win:
        s0 = (len(starts) - max_win) // 2
        starts = starts[s0:s0 + max_win]
    return np.stack([proj[s:s + win].mean(axis=0) for s in starts]).astype(np.float32)


def make_pairs(x, k, ctx=CTX):
    n = x.shape[0]
    idx = np.arange(ctx - 1, n - k)
    if len(idx) < 60:
        return None, None
    X = np.stack([x[i - ctx + 1:i + 1].reshape(-1) for i in idx]).astype(np.float32)
    Y = x[idx + k].astype(np.float32)
    return X, Y


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=ALPHA):
    d = Xtr.shape[1]
    xm = Xtr.mean(0, keepdims=True)
    xs = Xtr.std(0, keepdims=True) + 1e-8
    Xtr_ = (Xtr - xm) / xs
    Xte_ = (Xte - xm) / xs
    ym = Ytr.mean(0, keepdims=True)
    A = Xtr_.T @ Xtr_ + alpha * np.eye(d, dtype=np.float32)
    try:
        W = np.linalg.solve(A, Xtr_.T @ (Ytr - ym))
    except np.linalg.LinAlgError:
        return np.nan
    pred = Xte_ @ W + ym
    return float(1.0 - ((Yte - pred) ** 2).sum() /
                 max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def r2_curve_blocked(x, n_blocks=N_BLOCKS):
    blk = len(x) // n_blocks
    if blk < 70:
        return {}
    out = {}
    for k in KS:
        vals = []
        for b in range(n_blocks):
            xb = x[b * blk:(b + 1) * blk]
            X, Y = make_pairs(xb, k)
            if X is None:
                continue
            cut = int(len(X) * 0.7)
            if cut < 30 or len(X) - cut < 15:
                continue
            v = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
            if np.isfinite(v) and abs(v) <= R2_VALID_MAX:
                vals.append(v)
        if vals:
            out[k] = float(np.median(vals))
    return out


def fit_tau(prof):
    from scipy.optimize import curve_fit
    if len(prof) < 3:
        return None
    ks = np.array(sorted(prof), float)
    r2 = np.array([prof[k] for k in sorted(prof)], float)
    if r2[0] <= 0.02:
        return None

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.02), 2.0, min(r2[-1], 0.0)]
        popt, _ = curve_fit(f, ks, r2, p0=p0,
                            bounds=([0, 0.05, -2.0], [5, 100, 1.0]), maxfev=12000)
        return float(popt[1])
    except Exception:
        return None


def to_bins(sp, bin_ms, src_ms):
    ratio = max(1, int(round(bin_ms / src_ms)))
    nb = sp.shape[1] // ratio
    return sp[:, :nb * ratio].reshape(sp.shape[0], nb, ratio).sum(axis=2)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp19")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("Phase 19 — 决定性检验：窗口固定在 800ms")
    print("=" * 84)
    for bw in BIN_WIDTHS:
        print(f"  {bw:>3}ms bin → WIN={int(round(WINDOW_MS/bw)):>2} bins = {WINDOW_MS:.0f}ms (零重叠)")
    print(f"  目标: macaque {TARGET_AREA}（exp18 中唯一稳定为正的来源）")

    mac = root / "primate" / "macaque_nlb"
    seg = mac / "segments.parquet"
    if not seg.exists():
        print("❌ 缺少 segments.parquet")
        return
    sdf = pd.read_parquet(seg)
    meta = sdf[sdf["area"] == TARGET_AREA]
    print(f"  片段: {len(meta)}")

    rows = []
    t0 = time.time()

    for i, row in meta.iterrows():
        f = mac / row["file"]
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)

        for bw in BIN_WIDTHS:
            win = int(round(WINDOW_MS / bw))
            x = to_bins(sp, bw, 20.0)
            if x.shape[1] < win * 40:
                continue
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-6
            proj = pca_proj((x - mu) / sd)
            if proj is None:
                continue
            w = make_windows(proj, win)
            if w is None:
                continue
            prof = r2_curve_blocked(w)
            if len(prof) < 3:
                continue
            ws = np.stack([rng.permutation(w[:, j]) for j in range(w.shape[1])], axis=1)
            prof_s = r2_curve_blocked(ws)
            floor = float(np.median(list(prof_s.values()))) if prof_s else 0.0
            tau_b = fit_tau(prof)
            rows.append({
                "file": str(row["file"]), "bin_ms": bw, "win_bins": win,
                "n_units": int(x.shape[0]), "n_win": int(len(w)),
                "r2_1": round(prof.get(1, np.nan), 4),
                "r2_4": round(prof.get(4, np.nan), 4),
                "floor": round(floor, 4),
                "net_1": round(prof.get(1, np.nan) - floor, 4),
                "net_4": round(prof.get(4, np.nan) - floor, 4),
                "tau_bins": (round(tau_b, 3) if tau_b else None),
                "tau_ms": (round(tau_b * bw, 1) if tau_b else None),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase19.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 84)
    print("结果 1: 净可预测性（窗口都是 800ms）")
    print("=" * 84)
    print(f"{'bin':>5} {'WIN':>5} {'n':>4} {'R²(1)':>9} {'floor':>9} {'净R²(1)':>10} {'净R²(4)':>9}")
    for bw, g in df.groupby("bin_ms"):
        print(f"{bw:>5} {int(g['win_bins'].iloc[0]):>5} {len(g):>4} "
              f"{g['r2_1'].median():>9.4f} {g['floor'].median():>9.4f} "
              f"{g['net_1'].median():>10.4f} {g['net_4'].median():>9.4f}")

    print("\n" + "=" * 84)
    print("结果 2: 决定性判据 —— τ_ms 在三种 bin 宽度下是否一致")
    print("=" * 84)
    real = df.dropna(subset=["tau_ms"])
    print(f"  可拟合记录: {len(real)}/{len(df)}")
    med = real.groupby("bin_ms")["tau_ms"].agg(["size", "median"]).round(1)
    if len(med):
        print("\n" + med.to_string())

    ratios = {}
    if len(med) and 100 in med.index:
        base = float(med.loc[100, "median"])
        for bw in BIN_WIDTHS:
            if bw in med.index:
                ratios[bw] = round(float(med.loc[bw, "median"]) / base, 3)
        print("\n  以 100ms 为基准:")
        for bw, r in ratios.items():
            print(f"    {bw:>3}ms: {med.loc[bw,'median']:>8.1f} ms  比值 {r:>5.3f}  "
                  f"{'✅' if 0.75 <= r <= 1.35 else '❌'}")

    print("\n" + "=" * 84)
    print("最终判读")
    print("=" * 84)
    n_ok = sum(1 for r in ratios.values() if 0.75 <= r <= 1.35)
    n_tot = len(ratios)
    if n_tot and n_ok == n_tot:
        verdict = (f"✅ τ 在三种分辨率下一致 → 真实时间常数约 "
                   f"{med.loc[100,'median']:.0f} ms（猕猴 M1/PMd）")
    elif n_tot and n_ok == 0:
        verdict = ("❌ τ_ms 随 bin 宽度系统变化 —— τ 仍与方法学绑定，"
                   "不是神经动力学时间常数。当前框架下无法测量。")
    else:
        verdict = f"⚠️ {n_ok}/{n_tot} 个分辨率一致，证据不充分"
    print(f"  {verdict}")
    print(f"\n  参考: 净 R²(1) 中位 {df['net_1'].median():+.4f}"
          "（若显著为正，可预测性本身真实，只是 τ 测不准）")

    (out / "phase19_report.json").write_text(json.dumps({
        "schema_version": "1.0", "window_ms": WINDOW_MS,
        "bin_widths": BIN_WIDTHS, "ks": KS, "area": TARGET_AREA,
        "n_records": int(len(df)),
        "by_bin": {str(int(bw)): {
            "win_bins": int(g["win_bins"].iloc[0]), "n": int(len(g)),
            "r2_1": round(float(g["r2_1"].median()), 4),
            "floor": round(float(g["floor"].median()), 4),
            "net_1": round(float(g["net_1"].median()), 4),
            "net_4": round(float(g["net_4"].median()), 4),
            "tau_ms": (round(float(g["tau_ms"].median()), 1)
                       if g["tau_ms"].notna().any() else None)}
            for bw, g in df.groupby("bin_ms")},
        "tau_ratios_vs_100ms": ratios,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
