"""Phase 14: 多区域 × 多时间尺度分解。

## 为什么这是目前最干净的实验设计

前面所有跨物种比较都栽在混淆上（模态 / 记录规模 / 脑区 / 任务 / 被试）。
本实验改用**同一 session 内的多区域对照**：

    同一只动物、同一时刻、同一根 Neuropixels 探针、同一行为状态，
    只是 **unit 所属解剖区不同**。

因此 session 内比较可以控制掉除"区域"以外的一切变量。
若 τ 在 session 内随区域显著变化，则 τ 的区域特异性是真实的。

## 两个维度

### 维度 1: 多区域
Steinmetz 数据：39 session × 平均 8.4 区域 = 326 个 (session, region) 组合，
其中 315 个有 >= 8 个 unit（中位 58）。

对每个组合**单独**做 PCA → 分窗 → R²(k) → τ。而不是像之前那样把
整个 session 的所有 unit 混在一起。

### 维度 2: 多时间尺度
单指数拟合 R²(k)=a·exp(-k/τ)+c 隐含"只有一个时间尺度"的假设，
但神经动力学已知是**多尺度**的（幂律/无标度）。

改为双指数：
    R²(k) = a_fast·exp(-k/τ_fast) + a_slow·exp(-k/τ_slow) + c

提取三个量：
    τ_fast   快成分时间常数
    τ_slow   慢成分时间常数
    slow_frac = a_slow / (a_fast + a_slow)   慢成分的权重

**slow_frac 是二维指纹的核心**：两个区域可以有相同的中位 τ，
但一个由快速瞬态主导、另一个由慢速整合主导。

## 检验

1. **区域内 vs 区域间**（session 内）：方差分解
      τ_ij = μ + α_i(session) + β_j(region|session) + ε
2. **置换检验**：在 session 内打乱 region 标签，看 observed 的区域间方差
   是否超出 null
3. **双指数 vs 单指数**：ΔAIC，判断多尺度结构是否真的存在
4. **slow_frac 能否区分区域**（在控制了 session 之后）

## 判据

  置换 p<0.01 且区域解释方差 > 20%  → τ 的区域特异性成立
  ΔAIC 显著优于单指数               → 多时间尺度结构真实存在
  slow_frac 在区域内一致、区域间不同 → 它是区域的功能指纹
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
KS = [1, 2, 3, 4, 6, 8, 12]
WIN_LEN = 800
WIN, STRIDE = 8, 4
PROJ_DIM = 64
ALPHA = 1000.0
MIN_UNITS = 10
N_PERM = 200


def contiguous_slice(w, n=WIN_LEN):
    if len(w) <= n:
        return w
    s = (len(w) - n) // 2
    return w[s:s + n]


def pca_windows(z, max_win=WIN_LEN):
    """z: (n_units, n_bins) 已 z-score → (n_win, PROJ_DIM) 每窗时间均值。"""
    nb = z.shape[1]
    k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    C = zc @ zc.T
    w_, U = np.linalg.eigh(C)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]                    # (nb, k)
    if proj.shape[1] < PROJ_DIM:
        proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
    starts = list(range(0, nb - WIN + 1, STRIDE))
    if len(starts) < 100:
        return None
    # 取中段连续窗口
    if len(starts) > max_win:
        s0 = (len(starts) - max_win) // 2
        starts = starts[s0:s0 + max_win]
    return np.stack([proj[s:s + WIN].mean(axis=0) for s in starts]).astype(np.float32)


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
    return float(1.0 - ((Yte - pred) ** 2).sum() /
                 max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def r2_curve(x, rng, shuffle=False):
    if shuffle:
        x = np.stack([rng.permutation(x[:, j]) for j in range(x.shape[1])], axis=1)
    out = {}
    for k in KS:
        X, Y = make_pairs(x, k)
        if X is None:
            continue
        cut = int(len(X) * 0.7)
        if cut < 100 or len(X) - cut < 60:
            continue
        out[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
    return out


# ---------------------------------------------------- 多尺度拟合
def fit_single(ks, r2):
    from scipy.optimize import curve_fit
    ks = np.asarray(ks, float); r2 = np.asarray(r2, float)
    ok = np.isfinite(r2)
    if ok.sum() < 4:
        return None
    ks, r2 = ks[ok], r2[ok]

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.05), 3.0, min(r2[-1], 0.0)]
        popt, _ = curve_fit(f, ks, r2, p0=p0, maxfev=8000)
        rss = float(((r2 - f(ks, *popt)) ** 2).sum())
        return {"tau": float(popt[1]), "a": float(popt[0]), "c": float(popt[2]),
                "rss": rss, "k": 3}
    except Exception:
        return None


def fit_double(ks, r2):
    """双指数: a_fast·exp(-k/τ_fast) + a_slow·exp(-k/τ_slow) + c"""
    from scipy.optimize import curve_fit
    ks = np.asarray(ks, float); r2 = np.asarray(r2, float)
    ok = np.isfinite(r2)
    if ok.sum() < 5:
        return None
    ks, r2 = ks[ok], r2[ok]

    def f(k, a1, t1, a2, t2, c):
        return a1 * np.exp(-k / max(t1, 1e-3)) + a2 * np.exp(-k / max(t2, 1e-3)) + c

    best = None
    for a1_0, t1_0, a2_0, t2_0 in [
        (0.4, 0.8, 0.15, 5.0),
        (0.5, 1.0, 0.2, 8.0),
        (0.3, 0.5, 0.25, 4.0),
        (0.6, 1.5, 0.1, 10.0),
    ]:
        try:
            p0 = [a1_0, t1_0, a2_0, t2_0, min(r2[-1], 0.0)]
            bounds = ([0, 0.2, 0, 1.0, -1.0], [3, 30, 3, 60, 1.0])
            popt, _ = curve_fit(f, ks, r2, p0=p0, bounds=bounds, maxfev=20000)
            rss = float(((r2 - f(ks, *popt)) ** 2).sum())
            if best is None or rss < best["rss"]:
                a1, t1, a2, t2, c = popt
                # 保证 t1 < t2
                if t1 > t2:
                    a1, a2 = a2, a1
                    t1, t2 = t2, t1
                best = {"a_fast": float(a1), "tau_fast": float(t1),
                        "a_slow": float(a2), "tau_slow": float(t2),
                        "c": float(c), "rss": rss, "k": 5}
        except Exception:
            continue
    if best is None:
        return None
    tot = best["a_fast"] + best["a_slow"]
    best["slow_frac"] = float(best["a_slow"] / tot) if tot > 1e-9 else np.nan
    return best


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp14")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    stein = root / "steinmetz"
    units = pd.read_parquet(stein / "units.parquet")

    print("=" * 80)
    print("Phase 14 — 多区域 × 多时间尺度分解")
    print(f"  session 内按区域单独 PCA | 双指数拟合 | 窗 {WIN_LEN} | k∈{KS}")
    print("=" * 80)

    rows = []
    t0 = time.time()

    for sess in sorted(units["session"].unique()):
        f = stein / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)      # (n_units, n_bins_50ms)
        u = units[units["session"] == sess].reset_index(drop=True)
        if sp.shape[0] != len(u):
            continue

        # 50ms → 100ms
        nb = sp.shape[1] // 2
        x = sp[:, :nb * 2].reshape(sp.shape[0], nb, 2).sum(axis=2)

        for region, sub in u.groupby("region"):
            idx = sub["unit_idx"].to_numpy()
            idx = idx[(idx >= 0) & (idx < x.shape[0])]
            if len(idx) < MIN_UNITS:
                continue

            z = x[idx]
            mu = z.mean(axis=1, keepdims=True)
            sd = z.std(axis=1, keepdims=True) + 1e-6
            z = (z - mu) / sd

            w = pca_windows(z)
            if w is None:
                continue

            prof = r2_curve(w, rng)
            prof_s = r2_curve(w, rng, shuffle=True)
            if len(prof) < 4:
                continue

            ks_ok = sorted(prof)
            floor = float(np.median(list(prof_s.values()))) if prof_s else float("nan")
            net = {k: prof[k] - floor for k in ks_ok}
            y = [net[k] for k in ks_ok]

            s1 = fit_single(ks_ok, y)
            s2 = fit_double(ks_ok, y)

            row = {
                "session": int(sess), "region": region, "n_units": int(len(idx)),
                "n_win": int(len(w)),
                "r2_1": round(prof.get(1, float("nan")), 4),
                "floor": round(floor, 4),
                "tau_single": s1["tau"] if s1 else None,
                "tau_fast": s2["tau_fast"] if s2 else None,
                "tau_slow": s2["tau_slow"] if s2 else None,
                "slow_frac": s2["slow_frac"] if s2 else None,
                "aic_single": (len(ks_ok) * np.log(s1["rss"] / len(ks_ok)) + 2 * 3) if s1 and s1["rss"] > 0 else None,
                "aic_double": (len(ks_ok) * np.log(s2["rss"] / len(ks_ok)) + 2 * 5) if s2 and s2["rss"] > 0 else None,
            }
            rows.append(row)

        n_done = sum(1 for r in rows if r["session"] == sess)
        if n_done:
            print(f"  sess {sess:02d}: {n_done} 个区域完成")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase14_regions.csv", index=False)
    print(f"\n  共 {len(df)} 个 (session, region) 组合，耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    # ============ 分析 1: 多尺度结构是否真实
    print("\n" + "=" * 80)
    print("分析 1: 单指数 vs 双指数")
    print("=" * 80)
    both = df.dropna(subset=["aic_single", "aic_double"])
    if len(both):
        d_aic = both["aic_single"] - both["aic_double"]      # >0 表示双指数更好
        print(f"  n={len(both)}  ΔAIC(单−双) 中位 {d_aic.median():+.2f}")
        print(f"  双指数更好: {(d_aic > 0).sum()}/{len(both)} ({(d_aic>0).mean()*100:.0f}%)")
        print(f"  ΔAIC > 10 (强证据): {(d_aic > 10).sum()}/{len(both)}")
        print(f"\n  τ_fast 中位 {both['tau_fast'].median():.2f} bins "
              f"({both['tau_fast'].median()*100:.0f} ms)")
        print(f"  τ_slow 中位 {both['tau_slow'].median():.2f} bins "
              f"({both['tau_slow'].median()*100:.0f} ms)")
        print(f"  slow_frac 中位 {both['slow_frac'].median():.3f} "
              f"(IQR {both['slow_frac'].quantile(.25):.3f}–{both['slow_frac'].quantile(.75):.3f})")

    # ============ 分析 2: session 内区域差异
    print("\n" + "=" * 80)
    print("分析 2: session 内区域差异（控制动物/时间/探针/状态）")
    print("=" * 80)

    def within_session_spread(data, col):
        """每个 session 内，该指标跨区域的标准差（只算有>=2区域的session）。"""
        vals = []
        for s, g in data.groupby("session"):
            v = g[col].dropna()
            if len(v) >= 2:
                vals.append(float(v.std()))
        return np.array(vals) if vals else np.array([])

    results = {}
    for col in ["tau_single", "tau_fast", "tau_slow", "slow_frac"]:
        obs = within_session_spread(df, col)
        if len(obs) == 0:
            continue
        # 置换: session 内打乱 region 标签
        null = []
        for _ in range(N_PERM):
            sh = df.copy()
            for s, idxs in sh.groupby("session").groups.items():
                vals = sh.loc[idxs, col].to_numpy()
                sh.loc[idxs, col] = rng.permutation(vals)
            nv = within_session_spread(sh, col)
            if len(nv):
                null.append(float(np.median(nv)))
        null = np.array(null)
        obs_med = float(np.median(obs))
        p = float((null >= obs_med).mean())
        results[col] = {"observed_spread": round(obs_med, 4),
                        "null_spread": round(float(null.mean()), 4),
                        "ratio": round(obs_med / max(null.mean(), 1e-9), 2),
                        "p": p, "n_sessions": int(len(obs))}
        print(f"  {col:<12} 区域内标准差中位 {obs_med:7.3f} | "
              f"打乱后 {null.mean():7.3f} | 比值 {obs_med/max(null.mean(),1e-9):5.2f} "
              f"| p={p:.3f}")

    # ============ 分析 3: 区域级汇总（跨 session 聚合）
    print("\n" + "=" * 80)
    print("分析 3: 区域特征表（按 τ_slow 排序，只列 n>=3 的区域）")
    print("=" * 80)
    g = df.groupby("region").agg(
        n_session=("session", "nunique"),
        n_units_med=("n_units", "median"),
        tau_single=("tau_single", "median"),
        tau_fast=("tau_fast", "median"),
        tau_slow=("tau_slow", "median"),
        slow_frac=("slow_frac", "median"),
        r2_1=("r2_1", "median"),
    )
    g = g[g["n_session"] >= 3].sort_values("tau_slow", ascending=False)
    print(g.round(3).to_string())

    # ============ 分析 4: slow_frac 是否独立于 τ
    print("\n" + "=" * 80)
    print("分析 4: slow_frac 与 τ 的关系（是否提供独立信息）")
    print("=" * 80)
    sub = df.dropna(subset=["slow_frac", "tau_single"])
    if len(sub) > 10:
        from scipy.stats import spearmanr
        for c in ["tau_single", "tau_fast", "tau_slow", "r2_1"]:
            s2 = df.dropna(subset=["slow_frac", c])
            if len(s2) < 10:
                continue
            r, p = spearmanr(s2["slow_frac"], s2[c])
            print(f"  slow_frac vs {c:<12}: ρ={r:+.3f} (p={p:.2g}, n={len(s2)})")
        print("\n  → |ρ|<0.5 说明 slow_frac 携带 τ 之外的独立信息")

    (out / "phase14_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_combos": int(len(df)),
        "n_sessions": int(df["session"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "ks": KS, "win_len": WIN_LEN, "alpha": ALPHA,
        "multi_scale": {
            "n": int(len(both)) if len(both) else 0,
            "frac_double_better": float((d_aic > 0).mean()) if len(both) else None,
            "delta_aic_median": float(d_aic.median()) if len(both) else None,
            "tau_fast_median": float(both["tau_fast"].median()) if len(both) else None,
            "tau_slow_median": float(both["tau_slow"].median()) if len(both) else None,
            "slow_frac_median": float(both["slow_frac"].median()) if len(both) else None,
        },
        "within_session": results,
        "region_table": {k: {c: (None if pd.isna(v) else round(float(v), 4))
                             for c, v in row.items()}
                         for k, row in g.iterrows()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  结果 → {out}")


if __name__ == "__main__":
    main()
