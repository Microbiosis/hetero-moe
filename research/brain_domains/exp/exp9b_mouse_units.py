"""Phase 9b: 小鼠可预测性为何最低? 检查是否与 unit 数相关。

## 观察

exp9 结果（净信号 = R² - 噪声地板）:
  macaque  +0.255
  human    +0.031
  mouse    -0.112   ← 最低, 且为负

小鼠与猕猴同为侵入式 spike, 不该差这么多。

## 假说

小鼠的 64 维来自**逐 session PCA**，而每个 session 的 unit 数差异巨大
(205-1240)。unit 越多, PCA 前 64 个成分捕捉的方差比例越高、越"干净";
unit 少时成分更噪。

若如此, 小鼠内部 R² 应与 n_units 正相关。

## 检验

1. 小鼠各 session 的 R²(k=1) vs n_units 的 Spearman 相关
2. 取 unit 数最高/最低的三分之一, 分别看 R² 中位
3. 对照: 猕猴内部同样做（其 unit 数 40-192, 范围也大）

## 若成立的含义

小鼠的低可预测性不是生物学性质, 而是 **PCA 在低 unit 数时质量差**的
方法学伪影。这会让 exp9 的跨物种比较失去意义。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
CTX, HORIZONS, MAX_WIN = 8, [1, 4, 16], 1200


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1000.0):
    d = Xtr.shape[1]
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr_ = (Xtr - xm) / xs
    Xte_ = (Xte - xm) / xs
    ym = Ytr.mean(0, keepdims=True)
    A = Xtr_.T @ Xtr_ + alpha * np.eye(d, dtype=np.float32)
    B = Xtr_.T @ (Ytr - ym)
    W = np.linalg.solve(A, B)
    pred = Xte_ @ W + ym
    return float(1.0 - ((Yte - pred) ** 2).sum() / max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def pairs(x, k, ctx=CTX):
    n = x.shape[0]
    idx = np.arange(ctx - 1, n - k)
    if len(idx) < 50:
        return None, None
    X = np.stack([x[i - ctx + 1:i + 1].reshape(-1) for i in idx]).astype(np.float32)
    Y = x[idx + k].astype(np.float32)
    return X, Y


def profile_one(x, rng):
    if len(x) > MAX_WIN:
        idx = rng.choice(len(x), MAX_WIN, replace=False)
        x = x[np.sort(idx)]
    blk = len(x) // 4
    if blk < 150:
        return None
    out = {}
    for k in HORIZONS:
        vals = []
        for b in range(4):
            xb = x[b*blk:(b+1)*blk]
            X, Y = pairs(xb, k)
            if X is None:
                continue
            cut = int(len(X) * 0.7)
            if cut < 40 or len(X) - cut < 20:
                continue
            vals.append(ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:]))
        if vals:
            out[k] = float(np.median(vals))
    return out if out else None


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    stein = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data/steinmetz")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/out/exp9b")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 76)
    print("Phase 9b — 小鼠可预测性与 unit 数的关系")
    print("=" * 76)

    units = pd.read_parquet(stein / "units.parquet")
    cnt = units.groupby("session").size().to_dict()

    rows = []
    # 小鼠
    d = root / "spike_view"
    meta = pd.read_parquet(d / "meta.parquet")
    for sess in sorted(meta["session"].unique()):
        f = d / f"windows_sess{int(sess):03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < 200:
            continue
        prof = profile_one(w.mean(axis=2), rng)
        if prof is None:
            continue
        n_u = cnt.get(int(sess), 0)
        rows.append({"species": "mouse", "group": f"sess{int(sess)}", "n_units": n_u,
                     **{f"r2_k{k}": round(v, 4) for k, v in prof.items()}})
        print(f"  mouse sess {sess:02d}: {n_u:4d} units → R²(k1)={prof.get(1, float('nan')):+.4f}")

    # 猕猴对照
    d = root / "spike_view_macaque"
    meta = pd.read_parquet(d / "meta.parquet")
    for fname, sub in meta.groupby("file"):
        w = np.load(d / fname)["windows"]
        if len(w) < 200:
            continue
        prof = profile_one(w.mean(axis=2), rng)
        if prof is None:
            continue
        n_u = int(sub["n_units"].iloc[0])
        rows.append({"species": "macaque", "group": fname, "n_units": n_u,
                     **{f"r2_k{k}": round(v, 4) for k, v in prof.items()}})

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase9b.csv", index=False)

    print("\n" + "=" * 76)
    print("相关性检验: R²(k=1) vs n_units")
    print("=" * 76)
    result = {"schema_version": "1.0"}
    for sp in df["species"].unique():
        s = df[(df["species"] == sp) & df["r2_k1"].notna()]
        if len(s) < 5:
            continue
        r, p = spearmanr(s["n_units"], s["r2_k1"])
        result[sp] = {"rho": round(float(r), 3), "p": float(p), "n": int(len(s)),
                      "n_units_range": [int(s.n_units.min()), int(s.n_units.max())]}
        print(f"  {sp}: ρ = {r:+.3f} (p={p:.3g}, n={len(s)}, "
              f"units {s.n_units.min()}-{s.n_units.max()})")
        print(f"    {'← 强正相关: unit 越多越好预测' if r > 0.4 else ''}")

    # 三分位对照
    print("\n" + "=" * 76)
    print("按 unit 数三分位分组")
    print("=" * 76)
    for sp in df["species"].unique():
        s = df[(df["species"] == sp) & df["r2_k1"].notna()].copy()
        if len(s) < 6:
            continue
        s["tercile"] = pd.qcut(s["n_units"], 3, labels=["低", "中", "高"], duplicates="drop")
        g = s.groupby("tercile", observed=True).agg(
            n=("r2_k1", "size"), units=("n_units", "median"), r2=("r2_k1", "median"))
        print(f"\n  {sp}:")
        print(g.to_string())

    (out / "phase9b_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
