"""Phase 12: τ 的差异是物种造成的，还是脑区/任务造成的？

## 问题

exp11（n=157）得到极强信号：
    τ: macaque 5.41 bins (541ms) / mouse 1.86 (186ms) / human 0.68 (68ms)
    ε² = 0.663, p = 2.5e-23

但这三个物种用的数据在**脑区**和**任务**上完全不同：

    macaque   churchland(67/71 段) = M1/PMd 运动皮层, 伸手运动
              area2-bump(2/71)     = Brodmann 2 躯体感觉
    mouse     Steinmetz           = 多区域(含皮层下), 决策任务
    human     h1/h2               = 颞/额叶癫痫监测电极, 静息或语言

**运动皮层在运动执行期具有极强的序列动力学**（文献共识：M1 的
"神经轨迹"可高度预测）。因此 τ=541ms 可能完全来自脑区/任务，
而非物种。

## 设计

1. 在**猕猴内部**按区域切开：
     - churchland (M1/PMd)
     - area2-bump (Area2, 躯体感觉)
   若两者 τ 差异就很大 → 区域是关键因子

2. 在**小鼠内部**按主导域切开（sensory/motor/memory/...）
   若差异大 → 区域是关键因子

3. 用**方差分解**把 τ 的总方差分给：
     - 物种
     - 区域（物种内）
     - 残差
   看谁解释得多。

## 判据

  区域内方差 ≥ 区域间方差 → 物种不是主导，脑区/任务才是
  区域间方差 ≫ 区域内   → 物种效应（或至少是物种绑定的因素）主导
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
CTX = 8
KS = [1, 2, 3, 4, 6, 8, 12, 16]
WIN_LEN = 1500
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
    return float(1.0 - ((Yte - pred) ** 2).sum() /
                 max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def fit_tau(ks, r2):
    from scipy.optimize import curve_fit
    ks = np.asarray(ks, float)
    r2 = np.asarray(r2, float)
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


def tau_one(w, rng):
    w = contiguous_slice(w)
    ws = np.stack([rng.permutation(w[:, j]) for j in range(w.shape[1])], axis=1)
    prof, prof_s = {}, {}
    for k in KS:
        X, Y = make_pairs(w, k)
        Xs, Ys = make_pairs(ws, k)
        if X is None or Xs is None:
            continue
        cut, cuts = int(len(X) * 0.7), int(len(Xs) * 0.7)
        if cut < 100 or cuts < 100:
            continue
        prof[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
        prof_s[k] = ridge_r2(Xs[:cuts], Ys[:cuts], Xs[cuts:], Ys[cuts:])
    if len(prof) < 4:
        return None, None, None
    ks_ok = sorted(prof)
    floor = float(np.median(list(prof_s.values())))
    net = [prof[k] - floor for k in ks_ok]
    return (fit_tau(ks_ok, net), round(prof.get(1, float("nan")), 4), round(floor, 4))


def omega_sq(df, val, group):
    s = df[[group, val]].dropna()
    if len(s) < 5 or s[group].nunique() < 2:
        return None
    grand = s[val].mean()
    ss_b = (s.groupby(group)[val].agg(["mean", "size"])
            .assign(ss=lambda d: d["size"] * (d["mean"] - grand) ** 2)["ss"].sum())
    ss_w = ((s[val] - s.groupby(group)[val].transform("mean")) ** 2).sum()
    tot = ss_b + ss_w
    return float(ss_b / tot) if tot > 0 else 0.0


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp12")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 80)
    print("Phase 12 — τ 的差异: 物种 vs 脑区/任务")
    print("=" * 80)

    rows = []

    # ---- 小鼠: 记录主导域
    d = root / "spike_view"
    if d.exists():
        meta = pd.read_parquet(d / "meta.parquet")
        for sess, sub in meta.groupby("session"):
            f = d / f"windows_sess{int(sess):03d}.npz"
            if not f.exists():
                continue
            w = np.load(f)["windows"].mean(axis=2)
            if len(w) < 400:
                continue
            tau, r2_1, floor = tau_one(w, rng)
            if tau is None:
                continue
            dom = sub["domain_l1"].value_counts().index[0]
            rows.append({"species": "mouse", "region": f"mouse_{dom}",
                         "group": f"sess{int(sess)}", "tau": tau,
                         "r2_1": r2_1, "floor": floor})
            print(f"  mouse  sess{int(sess):02d} [{dom:<12}] τ={tau:5.2f}")

    # ---- 猕猴: 记录数据集(=区域)
    d = root / "spike_view_macaque"
    if d.exists():
        meta = pd.read_parquet(d / "meta.parquet")
        for fname, sub in meta.groupby("file"):
            f = d / str(fname)
            if not f.exists():
                continue
            w = np.load(f)["windows"].mean(axis=2)
            if len(w) < 400:
                continue
            tau, r2_1, floor = tau_one(w, rng)
            if tau is None:
                continue
            area = sub["area"].iloc[0]
            ds = sub["dataset"].iloc[0]
            rows.append({"species": "macaque", "region": f"macaque_{area}",
                         "group": str(fname), "tau": tau, "r2_1": r2_1, "floor": floor})
            print(f"  macaque {area:<8} τ={tau:5.2f}  ({ds})")

    # ---- 人类
    d = root / "spike_view_human"
    if d.exists():
        meta = pd.read_parquet(d / "meta.parquet")
        for fname, sub in meta.groupby("file"):
            f = d / str(fname)
            if not f.exists():
                continue
            w = np.load(f)["windows"].mean(axis=2)
            if len(w) < 400:
                continue
            tau, r2_1, floor = tau_one(w, rng)
            if tau is None:
                continue
            ds = sub["dataset"].iloc[0]
            rows.append({"species": "human", "region": f"human_{ds}",
                         "group": str(fname), "tau": tau, "r2_1": r2_1, "floor": floor})

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase12.csv", index=False)
    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 80)
    print("按 物种 × 区域 的 τ")
    print("=" * 80)
    t = df.groupby(["species", "region"])["tau"].agg(["size", "median", "std"]).round(3)
    print(t.to_string())

    print("\n" + "=" * 80)
    print("方差分解: τ 的方差由谁解释?")
    print("=" * 80)
    w_sp = omega_sq(df, "tau", "species")
    w_rg = omega_sq(df, "tau", "region")
    # 物种内残差（剥掉物种均值后，区域的解释力）
    df2 = df.dropna(subset=["tau"]).copy()
    df2["tau_within_sp"] = df2["tau"] - df2.groupby("species")["tau"].transform("mean")
    w_rg_within = omega_sq(df2, "tau_within_sp", "region")
    print(f"  按物种分组        ω² = {w_sp:.4f}" if w_sp else "")
    print(f"  按区域分组        ω² = {w_rg:.4f}" if w_rg else "")
    print(f"  物种内按区域      ω² = {w_rg_within:.4f}" if w_rg_within else "")

    # 区域内 vs 区域间 的 τ 跨度
    print("\n" + "=" * 80)
    print("关键对照: 区域内部的 τ 跨度")
    print("=" * 80)
    for (sp, rg), g in df.groupby(["species", "region"]):
        if len(g) < 2:
            continue
        print(f"  {rg:<22} n={len(g):3d}  τ {g['tau'].min():5.2f} – {g['tau'].max():5.2f} "
              f"(中位 {g['tau'].median():5.2f})")
    within_spread = float(df.groupby("species")["tau"].apply(
        lambda s: s.max() - s.min()).median())
    print(f"\n  物种内 τ 总跨度（中位）: {within_spread:.2f} bins")
    print(f"  物种间 τ 中位差: {df.groupby('species')['tau'].median().max() - df.groupby('species')['tau'].median().min():.2f} bins")

    # 判读
    if w_rg and w_sp and w_rg >= w_sp * 0.8:
        v = "⚠️ 区域/任务解释力接近或超过物种 —— τ 差异主要不是物种效应"
    elif w_sp and w_rg and w_sp > w_rg * 1.5:
        v = "✅ 物种解释力显著高于区域 —— τ 差异更可能是物种特异的"
    else:
        v = "混合，两者都有贡献"
    print(f"\n  判定: {v}")

    (out / "phase12_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_groups": int(len(df)),
        "by_region": {f"{a}|{b}": {"n": int(c["size"]), "tau_median": round(float(c["median"]), 3)}
                      for (a, b), c in t.iterrows()},
        "omega2_species": round(w_sp, 4) if w_sp else None,
        "omega2_region": round(w_rg, 4) if w_rg else None,
        "omega2_region_within_species": round(w_rg_within, 4) if w_rg_within else None,
        "verdict": v,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
