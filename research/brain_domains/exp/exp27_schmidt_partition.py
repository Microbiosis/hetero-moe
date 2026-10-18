"""任务 B: 用纠缠谱分组，与 HRP 分组比稳定性（交叉验证"层次是否客观"）。

## 要验证什么

exp24 用 HRP（基于相关矩阵的层次聚类）得到块性 2.0，说明"有块结构"。
但块结构是否**稳定**？换一种分组方法（纠缠谱）是否得到一致的划分？

若两种完全不同的数学构造给出类似的划分 → 层次是**客观的**
若差异很大 → 层次是**方法依赖的**，不能声称客观

## 两种分组方法

### 方法 1: HRP（金融）
相关矩阵 → 距离 1−|r| → average linkage → 层次聚类 → 切 k 组

### 方法 2: 纠缠谱（量子）
对 X (d, T) 做 SVD：X = USVᵀ
- 左奇异向量 U 的列 = 成分空间的 Schmidt 基
- 用 U 的前 r 列作为每个成分的"纠缠特征"
- 对这些特征做 k-means（或层次聚类）→ 切 k 组

两种方法都只输入同一份 X，但**用的是完全不同的信息**：
HRP 用瞬时相关，纠缠谱用时间上的联合模式。

## 稳定性度量

用 **bootstrap**：对时间点重采样 50 次，每次重算两种分组，
计算 Adjusted Rand Index (ARI) 的均值。

    ARI = 1  → 完全一致
    ARI = 0  → 随机水平

## 判据

  方法间 ARI > 0.5 且 两者 bootstrap 稳定性都 > 0.6  → 层次客观
  方法间 ARI < 0.2                                 → 方法依赖，不客观
  某方法 bootstrap 稳定性 < 0.4                     → 该方法本身不可靠
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_WIN = 600
N_PER_SPECIES = 6
N_BOOT = 30
KS = [4, 8]


# ---------------------------------------------------- 方法 1: HRP
def hrp_partition(X, k):
    """相关矩阵 → 层次聚类 → 切 k 组。"""
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    dist = np.clip(1.0 - np.abs(C), 0, 2)      # 用 |r| 保持与纠缠谱同样"看强度"
    np.fill_diagonal(dist, 0.0)
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        return fcluster(Z, t=k, criterion="maxclust")
    except Exception:
        return None


# ---------------------------------------------------- 方法 2: 纠缠谱
def schmidt_partition(X, k, r=None):
    """SVD 左奇异向量作为特征 → 层次聚类 → 切 k 组。"""
    s = np.linalg.svd(X, compute_uv=False)
    if r is None:
        # 取解释 95% 能量的秩
        p = s ** 2
        p = p / p.sum()
        r = int(np.searchsorted(np.cumsum(p), 0.95) + 1)
    r = max(2, min(r, min(X.shape) - 1))
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    F = U[:, :r]                                # (d, r) 每个成分的纠缠特征
    # 用相关距离（特征方向敏感，幅度不敏感）
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - Fn @ Fn.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, 2)
    try:
        Z = linkage(squareform(D, checks=False), method="average")
        return fcluster(Z, t=k, criterion="maxclust")
    except Exception:
        return None


def normalize(X, mode="raw"):
    if mode == "raw":
        return X.copy()
    if mode == "percomp":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mu) / sd
    return X


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
    # 按分组大小降序取前 N 个（不是按文件名排序取前 N —— 那会系统性地
    # 取到最小的分组，导致人类等数据源静默消失）
    sizes = meta.groupby(gcol).size().sort_values(ascending=False)
    for gval in list(sizes.index)[:N_PER_SPECIES]:
        if gcol == "session" and view == "spike_view":
            f = d / f"windows_sess{int(gval):03d}.npz"
        else:
            f = d / str(gval)
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < n_win:
            continue
        X = w.mean(axis=2).T.astype(np.float64)
        X = X[:, ::2]
        if X.shape[1] < n_win:
            continue
        s = (X.shape[1] - n_win) // 2
        out.append((str(gval), X[:, s:s + n_win]))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp27")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("任务 B — 纠缠谱分组 vs HRP 分组：稳定性与一致性")
    print("=" * 84)
    print(f"  n_win={N_WIN} | k∈{KS} | bootstrap={N_BOOT}")

    rows = []
    t0 = time.time()

    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_matched(root, view)
        print(f"\n  {species}: {len(items)} 组")
        if not items:
            continue
        for key, X0 in items:
            for mode in ["raw", "percomp"]:
                X = normalize(X0, mode)
                for k in KS:
                    h0 = hrp_partition(X, k)
                    s0 = schmidt_partition(X, k)
                    if h0 is None or s0 is None:
                        continue
                    ari_cross = float(adjusted_rand_score(h0, s0))

                    # bootstrap 稳定性
                    h_stab, s_stab = [], []
                    T = X.shape[1]
                    for _ in range(N_BOOT):
                        idx = rng.choice(T, T, replace=True)
                        idx.sort()
                        Xb = X[:, idx]
                        hb = hrp_partition(Xb, k)
                        sb = schmidt_partition(Xb, k)
                        if hb is not None and len(np.unique(hb)) == len(np.unique(h0)):
                            h_stab.append(adjusted_rand_score(h0, hb))
                        if sb is not None and len(np.unique(sb)) == len(np.unique(s0)):
                            s_stab.append(adjusted_rand_score(s0, sb))

                    rows.append({
                        "species": species, "group": key, "norm": mode, "k": k,
                        "ari_cross": round(ari_cross, 4),
                        "hrp_stability": round(float(np.mean(h_stab)), 4) if h_stab else None,
                        "schmidt_stability": round(float(np.mean(s_stab)), 4) if s_stab else None,
                        "n_boot_hrp": len(h_stab), "n_boot_schmidt": len(s_stab),
                    })
            print(f"    {str(key)[:24]:<24} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase27.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 84)
    print("一致性（方法间 ARI）与稳定性（bootstrap ARI）")
    print("=" * 84)
    cols = ["ari_cross", "hrp_stability", "schmidt_stability"]
    g = df.groupby(["norm", "k"])[cols].median()
    print(g.round(4).to_string())

    print("\n  按物种（raw 归一化，k=8）:")
    s = df[(df.norm == "raw") & (df.k == 8)]
    if len(s):
        print(s.groupby("species")[cols].median().round(4).to_string())

    # ============ 判定
    print("\n" + "=" * 84)
    print("判定")
    print("=" * 84)

    for mode in ["raw", "percomp"]:
        s = df[df.norm == mode]
        if not len(s):
            continue
        ari = float(s["ari_cross"].median())
        hs = float(s["hrp_stability"].median())
        ss_ = float(s["schmidt_stability"].median())
        print(f"  [{mode}] 方法间 ARI={ari:.3f} | HRP 稳定性={hs:.3f} | "
              f"纠缠谱稳定性={ss_:.3f}")

    s_raw = df[df.norm == "raw"]
    ari_raw = float(s_raw["ari_cross"].median()) if len(s_raw) else np.nan
    hs_raw = float(s_raw["hrp_stability"].median()) if len(s_raw) else np.nan
    ss_raw = float(s_raw["schmidt_stability"].median()) if len(s_raw) else np.nan

    if ari_raw > 0.5 and min(hs_raw, ss_raw) > 0.6:
        v = ("✅ **层次客观** — 两种完全不同的数学构造给出相似划分，"
             "且各自稳定")
    elif ari_raw < 0.2:
        v = ("❌ **方法依赖** — 两种构造给出几乎无关的划分，"
             "不能声称层次是客观的")
    elif min(hs_raw, ss_raw) < 0.4:
        v = "⚠️ 至少一种方法本身不稳定，划分不可靠"
    else:
        v = f"⚠️ 部分一致（ARI={ari_raw:.2f}, 稳定性={hs_raw:.2f}/{ss_raw:.2f}）"

    print(f"\n  {v}")

    (out / "phase27_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN, "ks": KS, "n_boot": N_BOOT,
        "methods": {"HRP": "corr distance (1-|r|) + average linkage",
                    "Schmidt": "SVD left singular vectors (95% energy) + corr distance + average linkage"},
        "n_records": int(len(df)),
        "by_norm_k": {f"{m}|k{int(k)}": {c: round(float(v_), 4) for c, v_ in r.items()}
                      for (m, k), r in g.iterrows()},
        "raw_median": {"ari_cross": round(ari_raw, 4),
                       "hrp_stability": round(hs_raw, 4),
                       "schmidt_stability": round(ss_raw, 4)},
        "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
