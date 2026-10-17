"""Phase 14b: 修正的置换检验 —— 用"伪区域"作零假设。

## 修什么

exp14 的置换检验写错了：

    for s, idxs in sh.groupby("session").groups.items():
        sh.loc[idxs, col] = rng.permutation(sh.loc[idxs, col])
    # 然后算 session 内跨区域的标准差

**标准差对组内置换是不变量**。打乱同一 session 内各区域的 τ 值后，
这组数的标准差完全不变。所以 observed ≡ null，ratio 恒为 1.00，
p 恒为 1.00 —— 这个检验没有测任何东西。

## 正确的零假设

"解剖区域标签"若只是把**同一批 unit 任意分组**，是否也能得到同样的
区域间离散度？

实现：对每个 session，把该 session 的所有 unit **随机重新分配到伪区域**，
**保持各区域原有的 unit 数量**（这样控制住了"小样本噪声更大"的影响），
然后重新做 PCA → 分窗 → R²(k) → τ。

若观测的离散度显著大于伪区域，则解剖标签确实捕捉到了真实结构。

## 计算控制

全量 200 次置换 × 305 组合不可行。折中：
  - 取 12 个 session（有 >= 4 个区域的优先）
  - 每 session 做 20 次置换
  - 只用 τ_single（不做双指数拟合）
  - KS 缩到 [1,2,4,8]，WIN_LEN=500
预计 ~10 分钟。
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
WIN_LEN = 500
WIN, STRIDE = 8, 4
PROJ_DIM = 64
ALPHA = 1000.0
MIN_UNITS = 10
N_PERM = 20
N_SESSIONS = 12


def pca_windows(z, max_win=WIN_LEN):
    nb = z.shape[1]
    k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    C = zc @ zc.T
    w_, U = np.linalg.eigh(C)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < PROJ_DIM:
        proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
    starts = list(range(0, nb - WIN + 1, STRIDE))
    if len(starts) < 100:
        return None
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


def tau_of(z):
    """z: (n_units, n_bins) 原始计数 → τ 或 None"""
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True) + 1e-6
    zz = (z - mu) / sd
    w = pca_windows(zz)
    if w is None:
        return None
    prof = {}
    for k in KS:
        X, Y = make_pairs(w, k)
        if X is None:
            continue
        cut = int(len(X) * 0.7)
        if cut < 100 or len(X) - cut < 60:
            continue
        prof[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
    if len(prof) < 3:
        return None

    from scipy.optimize import curve_fit
    ks = np.array(sorted(prof), float)
    r2 = np.array([prof[k] for k in sorted(prof)], float)

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.05), 3.0, min(r2[-1], 0.0)]
        popt, _ = curve_fit(f, ks, r2, p0=p0, maxfev=8000)
        t = float(popt[1])
        return t if 0.05 < t < 200 else None
    except Exception:
        return None


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp14b")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    stein = root / "steinmetz"
    units = pd.read_parquet(stein / "units.parquet")

    print("=" * 78)
    print("Phase 14b — 伪区域零假设（修正 exp14 的无效置换）")
    print(f"  KS={KS} | WIN_LEN={WIN_LEN} | {N_PERM} 次置换 | {N_SESSIONS} 个 session")
    print("=" * 78)

    # 挑 region 数最多的 session
    region_counts = units.groupby("session")["region"].nunique()
    cand = region_counts[region_counts >= 4].sort_values(ascending=False)
    sessions = list(cand.index[:N_SESSIONS])
    print(f"  选中 session: {sessions}")
    print(f"  各自区域数: {[int(region_counts[s]) for s in sessions]}")

    rows = []
    t0 = time.time()

    for sess in sessions:
        f = stein / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)
        u = units[units["session"] == sess].reset_index(drop=True)
        if sp.shape[0] != len(u):
            continue
        nb = sp.shape[1] // 2
        x = sp[:, :nb * 2].reshape(sp.shape[0], nb, 2).sum(axis=2)

        # 实际区域
        real_groups = []
        for region, sub in u.groupby("region"):
            idx = sub["unit_idx"].to_numpy()
            idx = idx[(idx >= 0) & (idx < x.shape[0])]
            if len(idx) >= MIN_UNITS:
                real_groups.append((region, idx))
        if len(real_groups) < 3:
            print(f"  sess {sess:02d}: 有效区域 {len(real_groups)} < 3，跳过")
            continue

        # 观测 τ
        obs_taus = []
        for region, idx in real_groups:
            t = tau_of(x[idx])
            if t is not None:
                obs_taus.append(t)
        if len(obs_taus) < 3:
            continue
        obs_sd = float(np.std(obs_taus))

        # 伪区域: 保持各区域 unit 数量
        sizes = [len(idx) for _, idx in real_groups]
        all_idx = np.arange(x.shape[0])
        null_sds = []
        for _ in range(N_PERM):
            perm = rng.permutation(all_idx)
            null_taus = []
            pos = 0
            for sz in sizes:
                idx = perm[pos:pos + sz]
                pos += sz
                t = tau_of(x[idx])
                if t is not None:
                    null_taus.append(t)
            if len(null_taus) >= 3:
                null_sds.append(float(np.std(null_taus)))

        if null_sds:
            null_med = float(np.median(null_sds))
            p = float((np.array(null_sds) >= obs_sd).mean())
            rows.append({"session": int(sess), "n_regions": len(obs_taus),
                         "obs_sd": round(obs_sd, 4), "null_sd_median": round(null_med, 4),
                         "ratio": round(obs_sd / max(null_med, 1e-9), 3),
                         "p": p})
            print(f"  sess {sess:02d}: {len(obs_taus)} 区域 | 观测 SD {obs_sd:.3f} | "
                  f"伪区域 SD {null_med:.3f} | 比值 {obs_sd/max(null_med,1e-9):.2f} | p={p:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase14b.csv", index=False)

    if not len(df):
        print("无结果")
        return

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  session 数: {len(df)}")
    print(f"  比值 中位: {df['ratio'].median():.3f}")
    print(f"  显著 (p<0.05) 的 session: {(df['p'] < 0.05).sum()}/{len(df)}")

    med_ratio = float(df["ratio"].median())
    n_sig = int((df["p"] < 0.05).sum())
    if med_ratio > 1.5 and n_sig >= len(df) * 0.6:
        v = "✅ 解剖区域标签捕捉到真实结构（观测离散度显著大于伪区域）"
    elif med_ratio > 1.15:
        v = "⚠️ 有弱结构，但大部分离散度来自非解剖因素"
    else:
        v = "❌ 区域标签不优于随机分组 —— τ 的区域特异性不成立"
    print(f"\n  判定: {v}")

    (out / "phase14b_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_sessions": int(len(df)),
        "n_perm": N_PERM, "ks": KS, "win_len": WIN_LEN,
        "median_ratio": round(med_ratio, 3),
        "n_significant": n_sig,
        "note": ("exp14 的置换检验无效: 标准差对组内置换是不变量, "
                 "故 observed≡null、ratio 恒为 1.00。本实验改用"
                 "『随机把 unit 分到伪区域(保持区域大小)』作为零假设。"),
        "verdict": v, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
