"""Phase 11: 完整的可预测性衰减曲线（k = 1…16 bins）。

## 前情

Phase 9 因 alpha 太小（1.0，应为 1000）得到全负 R²。
Phase 10 又因随机抽窗口破坏了时间连续性（小鼠 10,193 窗抽 800，实际间隔 1.3s）。
Phase 10b 修正后得到可信结果：
    k=1  R²: macaque 0.410 / mouse 0.307 / human 0.144
    k=8  R²: macaque −0.053 / mouse −0.177 / human −0.092
且**线性优于 MLP**（同一批数据，线性 R² 更高），故本实验只用 Ridge。

## 本实验

在连续窗口上，逐个 k ∈ {1,2,3,4,6,8,12,16} 测 R²，得到完整衰减曲线，
并拟合指数衰减  R²(k) = a·exp(−k/τ) + c  求时间常数 τ。

τ 的物理含义：**神经活动携带的"记忆"能维持多少个 100ms**。
这比单个 R² 更能刻画动力学特征，且对绝对幅度、单元数不敏感。

## 设计要点（都是前几轮的教训）

1. 连续取中段窗口（exp10 的教训）
2. alpha=1000 + 特征 z-score（exp9 的教训）
3. 每个 k 都配打乱对照，报告净信号（exp8b 的教训）
4. 只用线性模型（exp10b 已证明线性更好）

## 判据

  若 τ 在物种间差异显著（ε²>0.3）→ 动力学时间尺度是物种特异的
  若 τ 相近（ε²<0.1）→ 可能存在跨物种普适的时间尺度
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
CTX = 8
KS = [1, 2, 3, 4, 6, 8, 12, 16]
WIN_LEN = 1500
N_GROUPS = 3
ALPHA = 1000.0


def contiguous_slice(w, n=WIN_LEN):
    if len(w) <= n:
        return w
    s = (len(w) - n) // 2
    return w[s:s + n]


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
    W = np.linalg.solve(A, Xtr_.T @ (Ytr - ym))
    pred = Xte_ @ W + ym
    ss_res = ((Yte - pred) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def fit_tau(ks, r2):
    """拟合 R²(k) = a·exp(-k/τ) + c，返回 τ（单位：bin）。"""
    from scipy.optimize import curve_fit

    ks = np.asarray(ks, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    ok = np.isfinite(r2)
    if ok.sum() < 4:
        return None
    ks, r2 = ks[ok], r2[ok]

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.05), 3.0, min(r2[-1], 0.0)]
        popt, _ = curve_fit(f, ks, r2, p0=p0, maxfev=8000)
        tau = float(popt[1])
        return tau if 0.05 < tau < 200 else None
    except Exception:
        return None


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp11")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 80)
    print("Phase 11 — 可预测性衰减曲线 R²(k)")
    print(f"  ctx={CTX} bins | k∈{KS} | 连续 {WIN_LEN} 窗 | alpha={ALPHA}")
    print("=" * 80)

    specs = [("mouse", root / "spike_view", "session"),
             ("macaque", root / "spike_view_macaque", "file"),
             ("human", root / "spike_view_human", "file")]

    rows = []
    t0 = time.time()

    for species, d, gcol in specs:
        if not d.exists():
            continue
        meta = pd.read_parquet(d / "meta.parquet")
        groups = sorted(meta[gcol].unique())[:N_GROUPS]

        for gval in groups:
            f = d / (f"windows_sess{int(gval):03d}.npz" if gcol == "session" else str(gval))
            if not f.exists():
                continue
            w = np.load(f)["windows"].mean(axis=2)      # (n_win, 64)
            if len(w) < 400:
                continue
            w = contiguous_slice(w)
            ws = np.stack([rng.permutation(w[:, j]) for j in range(w.shape[1])], axis=1)

            prof, prof_s = {}, {}
            for k in KS:
                X, Y = make_pairs(w, k)
                Xs, Ys = make_pairs(ws, k)
                if X is None or Xs is None:
                    continue
                cut = int(len(X) * 0.7)
                cuts = int(len(Xs) * 0.7)
                if cut < 100 or cuts < 100:
                    continue
                prof[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
                prof_s[k] = ridge_r2(Xs[:cuts], Ys[:cuts], Xs[cuts:], Ys[cuts:])

            if len(prof) < 4:
                continue
            ks_ok = sorted(prof)
            tau = fit_tau(ks_ok, [prof[k] for k in ks_ok])
            floor = float(np.median(list(prof_s.values())))
            net = {k: prof[k] - floor for k in ks_ok}
            tau_net = fit_tau(ks_ok, [net[k] for k in ks_ok])

            row = {"species": species, "group": str(gval), "n_win": int(len(w)),
                   "noise_floor": round(floor, 4), "tau": tau, "tau_net": tau_net}
            for k in KS:
                row[f"r2_{k}"] = round(prof.get(k, float("nan")), 4)
                row[f"net_{k}"] = round(net.get(k, float("nan")), 4)
            rows.append(row)
            print(f"  {species:8s} {str(gval)[:22]:<22} "
                  f"R²(1)={prof.get(1, float('nan')):+.3f} "
                  f"R²(8)={prof.get(8, float('nan')):+.3f} "
                  f"floor={floor:+.3f} τ={tau if tau is None else round(tau,2)}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase11.csv", index=False)

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 80)
    print("R²(k) 曲线（中位数）")
    print("=" * 80)
    cols = [f"r2_{k}" for k in KS]
    print(df.groupby("species")[["noise_floor"] + cols].median().round(4).to_string())

    print("\n" + "=" * 80)
    print("时间常数 τ（单位：100ms bin）")
    print("=" * 80)
    tstat = df.groupby("species")["tau_net"].agg(["median", "size", "min", "max"])
    print(tstat.round(3).to_string())

    from scipy.stats import kruskal
    s = df[["species", "tau_net"]].dropna()
    if s["species"].nunique() >= 2:
        g = [s[s.species == sp]["tau_net"].to_numpy() for sp in s.species.unique()]
        g = [x for x in g if len(x) >= 2]
        if len(g) >= 2:
            H, p = kruskal(*g)
            n = sum(len(x) for x in g)
            k_ = len(g)
            e2 = max(0.0, (H - k_ + 1) / (n - k_))
            print(f"\n  物种间 τ 差异: ε²={e2:.4f} p={p:.3g} (n={n})")
        else:
            e2, p = None, None
    else:
        e2, p = None, None

    # 半衰视界（R² 降到 R²(1) 一半的 k）
    print("\n" + "=" * 80)
    print("半衰视界（R² 降至峰值一半处的 k）")
    print("=" * 80)
    hl = {}
    for _, r in df.iterrows():
        vals = [(k, r.get(f"r2_{k}", np.nan)) for k in KS]
        vals = [(k, v) for k, v in vals if np.isfinite(v)]
        if not vals:
            continue
        k0, v0 = vals[0]
        if v0 <= 0:
            hl.setdefault(r["species"], []).append(np.nan)
            continue
        tgt = v0 / 2
        h = np.nan
        for i in range(1, len(vals)):
            k1, v1 = vals[i - 1]
            k2, v2 = vals[i]
            if v1 >= tgt >= v2:
                h = k1 + (v1 - tgt) / (v1 - v2) * (k2 - k1)
                break
        hl.setdefault(r["species"], []).append(h)
    for sp, v in hl.items():
        arr = np.array(v, dtype=float)
        print(f"  {sp:8s} 中位 {np.nanmedian(arr) if np.isfinite(arr).any() else float('nan'):.2f} bins "
              f"({np.nanmedian(arr)*100 if np.isfinite(arr).any() else float('nan'):.0f} ms)")

    (out / "phase11_report.json").write_text(json.dumps({
        "schema_version": "1.0", "ctx": CTX, "ks": KS,
        "win_len": WIN_LEN, "alpha": ALPHA,
        "median_curve": {sp: {f"k{k}": (round(float(row[f"r2_{k}"]), 4)
                                        if pd.notna(row[f"r2_{k}"]) else None)
                              for k in KS}
                         for sp, row in df.groupby("species")[cols].median().iterrows()},
        "tau_net": {sp: float(v) if pd.notna(v) else None
                    for sp, v in df.groupby("species")["tau_net"].median().items()},
        "tau_eps2": round(float(e2), 4) if e2 is not None else None,
        "tau_p": float(p) if p is not None else None,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
