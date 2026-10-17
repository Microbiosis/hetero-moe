"""Phase 6b: 检验 eff_dim 的物种差异是否只是单元数伪影。

## 问题

exp6 显示小鼠 eff_dim=4.74 vs 猕猴 2.57（ε²=0.589，最大效应）。
但两者单元数差一个数量级：
    小鼠   326 - 1,240 units
    猕猴    49 -   192 units

参与率 (participation ratio) 受**可测维度数**限制。49 个 unit 的 PCA
最多只能撑起 49 个成分，而小鼠 700+ unit 可以填满全部 64 维。
因此 observed eff_dim 差异可能完全是 recording 规模的伪影。

## 检验

1. 在**每个物种内部**，算 eff_dim 与 n_units 的 Spearman 相关
   - 若强正相关 → eff_dim 主要由 unit 数决定
2. 把两个物种**限制在相同 unit 数区间**（如 150-200）再比较
   - 若差异消失 → 原差异是伪影
   - 若差异仍在 → 物种效应真实

## 数据来源

需重新计算 eff_dim 并保留 n_units（exp6 的中间表没存这个）。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
MAX_WIN = 500


def eff_dim_of(w):
    """逐窗参与率的中位数。w: (n_win, C, T)"""
    x = w.astype(np.float64)
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        xc = x[i] - x[i].mean(axis=1, keepdims=True)
        try:
            s = np.linalg.svd(xc, compute_uv=False)
            lam = s ** 2
            out[i] = (lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12)
        except np.linalg.LinAlgError:
            out[i] = np.nan
    return float(np.nanmedian(out))


def collect(root):
    rng = np.random.default_rng(SEED)
    rows = []

    # 小鼠
    d = root / "spike_view"
    meta = pd.read_parquet(d / "meta.parquet")
    for sess, sub in meta.groupby("session"):
        f = d / f"windows_sess{int(sess):03d}.npz"
        if not f.exists(): continue
        w = np.load(f)["windows"]
        if len(w) > MAX_WIN:
            idx = rng.choice(len(w), MAX_WIN, replace=False); w = w[np.sort(idx)]
        rows.append({"species": "mouse", "group": f"sess{int(sess)}",
                     "n_units_proxy": int(sub["session"].size and 0) or None,
                     "eff_dim": eff_dim_of(w), "n_win": len(w)})

    # 猕猴
    d = root / "spike_view_macaque"
    meta = pd.read_parquet(d / "meta.parquet")
    for fname, sub in meta.groupby("file"):
        w = np.load(d / fname)["windows"]
        if len(w) > MAX_WIN:
            idx = rng.choice(len(w), MAX_WIN, replace=False); w = w[np.sort(idx)]
        rows.append({"species": "macaque", "group": fname,
                     "n_units": int(sub["n_units"].iloc[0]),
                     "eff_dim": eff_dim_of(w), "n_win": len(w)})
    return pd.DataFrame(rows)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp6b")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Phase 6b — eff_dim 的物种差异是否只是单元数伪影?")
    print("=" * 74)

    df = collect(root)
    df.to_csv(out / "phase6b.csv", index=False)

    # 小鼠也需要 n_units
    stein_units = Path("/data/steinmetz/units.parquet")
    if stein_units.exists():
        u = pd.read_parquet(stein_units)
        cnt = u.groupby("session").size()
        df["n_units"] = df.apply(
            lambda r: int(cnt.get(int(r["group"].replace("sess", "")), 0))
            if r["species"] == "mouse" else r["n_units"], axis=1)

    print(f"\n  样本: {len(df)} 个片段")
    print(df.groupby("species").agg(
        n=("eff_dim", "size"), eff_dim_med=("eff_dim", "median"),
        units_min=("n_units", "min"), units_max=("n_units", "max")).to_string())

    # 1) 种内相关
    print("\n=== 1. 种内: eff_dim 与 n_units 的相关 ===")
    corr = {}
    for sp in df["species"].unique():
        s = df[df["species"] == sp].dropna(subset=["eff_dim", "n_units"])
        if len(s) < 5:
            print(f"  {sp}: 样本不足 ({len(s)})")
            continue
        r, p = spearmanr(s["n_units"], s["eff_dim"])
        corr[sp] = {"rho": round(float(r), 3), "p": float(p), "n": int(len(s))}
        print(f"  {sp}: ρ = {r:+.3f} (p={p:.3g}, n={len(s)}) "
              f"{'← 强相关' if abs(r) > 0.5 else ''}")

    # 2) 限制到相同 unit 区间
    print("\n=== 2. 限制到重叠 unit 区间后再比较 ===")
    lo = max(df.groupby("species")["n_units"].quantile(0.1))
    hi = min(df.groupby("species")["n_units"].quantile(0.9))
    print(f"  重叠区间: {lo:.0f} - {hi:.0f} units")
    sub = df[(df["n_units"] >= lo) & (df["n_units"] <= hi)]
    print(f"  区间内样本: {len(sub)}")
    print(sub.groupby("species")["eff_dim"].agg(["size", "median"]).to_string())

    res = {"schema_version": "1.0", "within_species_corr": corr,
           "overlap_range": [float(lo), float(hi)]}
    groups = [sub[sub["species"] == s]["eff_dim"].dropna().to_numpy()
              for s in sub["species"].unique()]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        n = sum(len(g) for g in groups); k = len(groups)
        e2 = max(0.0, (H - k + 1) / (n - k))
        res["overlap_eps2"] = round(float(e2), 4)
        res["overlap_p"] = float(p)
        print(f"\n  区间内 ε² = {e2:.4f} (p={p:.3g})")
        if e2 < 0.05:
            v = "✅ 差异基本消失 —— 原 eff_dim 物种差异是单元数伪影"
        else:
            v = "⚠️ 差异仍在 —— 物种效应部分真实"
    else:
        v = "样本不足，无法判断"
    res["verdict"] = v
    print(f"  判定: {v}")

    (out / "phase6b_report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
