"""Phase 24: 正确的 Schmidt 纠缠熵 + HRP 准对角化。

## 修正 exp23 的错误

exp23 用时间二分算 I(A:B)=½[logdet Σ_A + logdet Σ_B − logdet Σ_AB]，
得到**全负值**（−7 到 −52）。互信息不可能为负。

错误原因：Σ_A、Σ_B 是分段协方差，迹远小于全段 Σ_AB，
det 不满足互信息不等式。**时间二分不能这样构造联合高斯。**

## 正确做法：矩阵二分的 Schmidt 分解

把 X（d 成分 × T 时间）视为纯态 |ψ⟩ = Σ_{ij} X_{ij}|i⟩_d|j⟩_T，
**行与列是天然的两个子系统**。Schmidt 分解即 SVD：

    X = U S Vᵀ,   p_i = s_i² / Σ_j s_j²

    S_ent = −Σ_i p_i log p_i        （纠缠熵）
    S_max = log min(d, T) = log 64  （最大可能值）

归一化纠缠度：  S_ent / log 64

这与密度矩阵谱熵（ρ = Σ/tr Σ 的特征值）**数学等价**，
但含义清楚：它衡量"行与列两个子系统之间的纠缠"。

### 面积律的正确表述

对矩阵而言，"低纠缠"= 少数奇异值占主导 = 低有效秩 = **可用小键维数压缩**。

判据：
    有效秩（= e^S_ent）远小于 64  → 低纠缠 → 层次压缩可行
    有效秩 ≈ 64                   → 最大纠缠 → 不可压缩

## 附带：金融 HRP 的准对角化

López de Prado 的 Hierarchical Risk Parity 核心是**准对角化**：
用层次聚类把相关矩阵重排成块对角结构，使高相关项相邻。

若神经数据的相关矩阵能被重排成清晰的块结构 → 存在层次
若重排后仍是弥散的 → 不存在层次

量化：重排前后，相关矩阵的"块性"（相邻元素相关均值 / 全局均值）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_WIN = 600
N_PER_SPECIES = 10


# ------------------------------------------------- Schmidt / 纠缠熵
def schmidt_entropy(X):
    """矩阵二分（行 vs 列）的纠缠熵。

    X: (d, T)。返回 S_ent、有效秩、归一化纠缠度、谱。
    """
    d = min(X.shape)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return None
    p = s ** 2
    p = p / p.sum()
    S = float(-(p * np.log(p)).sum())
    eff_rank = float(np.exp(S))
    S_max = float(np.log(d)) if d > 1 else 1.0
    return {
        "S_ent": S,
        "S_norm": S / S_max,
        "eff_rank": eff_rank,
        "rank_frac": eff_rank / d,
        "top1": float(p[0]),
        "top5": float(p[:5].sum()),
        "n_sv": int(len(s)),
        "n_components": int(d),
    }


# ------------------------------------------------- 压缩效率（键维数）
def bond_dim_needed(X, tol=0.01):
    """达到 tol 的截断误差所需的最小键维数 χ。"""
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    cum = np.cumsum(p)
    idx = np.searchsorted(cum, 1 - tol)
    return int(min(idx + 1, len(s)))


# ------------------------------------------------- HRP 准对角化
def quasi_diag_blockiness(X):
    """准对角化前后的块性。"""
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    d = C.shape[0]
    if d < 4:
        return None
    # 距离 → 层次聚类 → 叶序
    dist = 1.0 - C
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
    except Exception:
        return None

    def adj_mean(M):
        a = np.abs(M[np.arange(d - 1), np.arange(1, d)])
        return float(a.mean())

    Cr = C[np.ix_(order, order)]
    # 用 |corr| 的相邻均值（含对角线为 1，需排除）
    Cabs = np.abs(C)
    np.fill_diagonal(Cabs, 0.0)
    Cabsr = np.abs(Cr)
    np.fill_diagonal(Cabsr, 0.0)

    adj_orig = float((Cabs[np.arange(d-1), np.arange(1,d)]).mean())
    adj_reord = float((Cabsr[np.arange(d-1), np.arange(1,d)]).mean())
    glob = float(Cabs[np.triu_indices(d, 1)].mean())

    return {
        "adj_orig": round(adj_orig, 4),
        "adj_reord": round(adj_reord, 4),
        "global_mean": round(glob, 4),
        "blockiness": round(adj_reord / max(glob, 1e-9), 3),
        "improvement": round(adj_reord / max(adj_orig, 1e-9), 3),
    }


def load_matched(root: Path, view: str, n_win=N_WIN):
    d = root / view
    if not (d / "meta.parquet").exists():
        return []
    meta = pd.read_parquet(d / "meta.parquet")
    if view == "spike_view":
        gcol = "session"
    elif "file" in meta.columns:
        gcol = "file"
    else:
        gcol = "session"
    out = []
    for gval, _ in list(meta.groupby(gcol))[:N_PER_SPECIES]:
        if gcol == "session" and view == "spike_view":
            f = d / f"windows_sess{int(gval):03d}.npz"
        else:
            f = d / str(gval)
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < n_win * 2:
            continue
        X = w.mean(axis=2).T.astype(np.float64)
        X = X[:, ::2]
        if X.shape[1] < n_win:
            continue
        s = (X.shape[1] - n_win) // 2
        X = X[:, s:s + n_win]
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-9
        out.append((str(gval), (X - mu) / sd))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp24")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("Phase 24 — Schmidt 纠缠熵（正确）+ HRP 准对角化")
    print("=" * 82)
    print(f"  n_win={N_WIN} | S_max = ln(64) = {np.log(64):.4f}")
    print("  Schmidt: X = U S Vᵀ,  p_i = s_i²/Σs_j²,  S_ent = −Σ p_i ln p_i")

    rows = []
    t0 = time.time()

    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_matched(root, view)
        print(f"\n  {species}: {len(items)} 组")
        if not items:
            continue
        for key, X in items:
            sc = schmidt_entropy(X)
            if sc is None:
                continue
            chi1 = bond_dim_needed(X, 0.01)
            chi5 = bond_dim_needed(X, 0.05)
            hrp = quasi_diag_blockiness(X)
            rows.append({"species": species, "group": key,
                         **{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in sc.items()},
                         "chi_1pct": chi1, "chi_5pct": chi5,
                         **(hrp or {})})
            print(f"    {str(key)[:22]:<22} S={sc['S_ent']:.3f} "
                  f"({sc['S_norm']:.2f}) eff_rank={sc['eff_rank']:.1f} "
                  f"χ(1%)={chi1} χ(5%)={chi5}"
                  + (f" 块性={hrp['blockiness']:.2f}" if hrp else ""))

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase24.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    # ============ Schmidt 纠缠
    print("\n" + "=" * 82)
    print("Schmidt 纠缠熵（行 vs 列二分）")
    print("=" * 82)
    cols = ["S_ent", "S_norm", "eff_rank", "rank_frac", "top1", "top5",
            "chi_1pct", "chi_5pct"]
    g = df.groupby("species")[cols].median()
    print(g.round(4).to_string())
    print(f"\n  参考: S_max = {np.log(64):.4f}, 最大有效秩 = 64")

    # ============ HRP
    print("\n" + "=" * 82)
    print("HRP 准对角化（相关性块结构）")
    print("=" * 82)
    hcols = [c for c in ["adj_orig", "adj_reord", "global_mean", "blockiness", "improvement"]
             if c in df.columns]
    if hcols:
        print(df.groupby("species")[hcols].median().round(4).to_string())
        print("\n  blockiness = 重排后相邻相关 / 全局平均相关")
        print("  improvement = 重排后 / 重排前（>1 表示准对角化有效）")

    # ============ 判定
    print("\n" + "=" * 82)
    print("判定")
    print("=" * 82)
    sn = float(g["S_norm"].median())
    er = float(g["eff_rank"].median())
    chi = float(g["chi_1pct"].median())
    blk = float(df["blockiness"].median()) if "blockiness" in df.columns else np.nan

    print(f"  归一化纠缠度 S/ln64 中位 = {sn:.3f}  (1.0 = 最大纠缠)")
    print(f"  有效秩 中位 = {er:.1f} / 64")
    print(f"  达 1% 截断误差所需键维数 χ = {chi:.0f} / 64")
    if np.isfinite(blk):
        print(f"  相关矩阵块性 = {blk:.2f}  (>1.5 表示有清晰块结构)")

    if sn > 0.85 and chi > 40:
        v_ent = ("❌ **接近最大纠缠** —— 没有低维结构，"
                 "张量网络压缩不可行，数据驱动层次在信息论上不成立")
    elif sn < 0.6 and chi < 20:
        v_ent = "✅ **低纠缠** —— 存在小键维数表示，层次压缩可行"
    else:
        v_ent = f"⚠️ 中等纠缠（S/ln64={sn:.2f}, χ={chi:.0f}）"

    if np.isfinite(blk):
        if blk > 1.5:
            v_hrp = "✅ 相关矩阵有清晰块结构 —— 层次划分有依据"
        elif blk > 1.2:
            v_hrp = "⚠️ 块结构弱"
        else:
            v_hrp = "❌ 无块结构 —— 任何层次划分都是任意的"
    else:
        v_hrp = "无法判定"

    print(f"\n  纠缠: {v_ent}")
    print(f"  块结构: {v_hrp}")

    (out / "phase24_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN,
        "method": "Schmidt decomposition (SVD) of (d,T) matrix; p_i=s_i^2/sum(s^2)",
        "fix_note": ("exp23 的时间二分互信息公式错误（Σ_A、Σ_B 迹小于全段，"
                     "det 不满足互信息不等式，得到全负值）。本实验改用矩阵"
                     "行/列二分的 Schmidt 分解，这是标准且数学正确的做法。"),
        "n_records": int(len(df)),
        "by_species": {sp: {c: round(float(v_), 4) for c, v_ in r.items()}
                       for sp, r in g.iterrows()},
        "hrp": ({sp: {c: round(float(v_), 4) for c, v_ in r.items()}
                 for sp, r in df.groupby("species")[hcols].median().iterrows()}
                if hcols else None),
        "entanglement_verdict": v_ent,
        "blockiness_verdict": v_hrp,
        "S_max": round(float(np.log(64)), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
