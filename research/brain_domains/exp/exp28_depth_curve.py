"""任务 C: 跨物种统一 k-阶梯的深度-方差曲线（普适性检验）。

## 要验证什么

exp-hierarchy 得到三物种的 v(d) 曲线，r = +1.000（完全相等）。
但那可能是"64 维 PCA 数据的聚类几何"的必然结果，不是生物普适性。

本实验用**三个独立控制**来判断曲线是否真的普适：

### 控制 1: 随机高斯数据（基线）
若随机数据也给出同样的 v(d) → 曲线是数学必然，无信息
若真实数据偏离随机 → 有信息

### 控制 2: 匹配窗口数 + 匹配维度
三物种统一 n_win=600，维度统一 64（已是）
人类之前因窗口不足被跳过，本实验修正

### 控制 3: 用超额而非原始 v(d)
v(d) 的绝对值由划分算法决定；超额（真实 − 时间打乱）才携带生物信息
比较超额曲线的形状

## 判据

  高三物种 v(d) 相关（r>0.95）且 随机数据也如此  → 无信息（数学必然）
  三物种超额曲线形状一致（r>0.7）且 都显著>0    → 存在跨物种不变结构
  超额曲线形状不一致                            → 无普适性
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
from scipy.spatial.distance import pdist

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_WIN = 600
N_PER_SPECIES = 8
KS = [2, 4, 8, 16, 32]
N_SHUFFLE = 6


def proj_group_mean(x, labels):
    out = np.empty_like(x)
    for g in np.unique(labels):
        m = labels == g
        if m.sum():
            out[m] = x[m].mean(axis=0, keepdims=True)
    return out


def var_frac(x, comp):
    den = float((x ** 2).sum())
    return float((comp ** 2).sum()) / den if den > 1e-12 else 0.0


def build_tree(x):
    if x.shape[0] < 4:
        return None
    d = pdist(x, metric="correlation")
    d = np.nan_to_num(d, nan=1.0, posinf=2.0)
    return linkage(d, method="average")


def nested_labels(Z, n, ks):
    heights = np.sort(Z[:, 2])[::-1]
    out = {}
    for k in ks:
        if k >= n:
            continue
        out[k] = fcluster(Z, t=heights[min(k - 1, len(heights) - 1)] - 1e-9,
                          criterion="distance")
    return out


def depth_curve(x, ks=KS, rng=None, n_shuffle=N_SHUFFLE):
    """返回 {k: (var, null, excess)}"""
    Z = build_tree(x)
    if Z is None:
        return None
    parts = nested_labels(Z, x.shape[0], ks)
    if len(parts) < 2:
        return None
    order = sorted(parts)

    projs = {k: proj_group_mean(x, parts[k]) for k in order}
    res = {}
    prev = None
    for k in order:
        c = projs[k] if prev is None else (projs[k] - prev)
        res[k] = {"var": var_frac(x, c), "null": None, "excess": None}
        prev = projs[k]

    # 时间打乱零模型
    if rng is not None and n_shuffle > 0:
        nulls = {k: [] for k in order}
        for _ in range(n_shuffle):
            xs = np.stack([rng.permutation(x[i]) for i in range(x.shape[0])])
            Zs = build_tree(xs)
            if Zs is None:
                continue
            ps = nested_labels(Zs, xs.shape[0], ks)
            prev2 = None
            for k in sorted(ps):
                pj = proj_group_mean(xs, ps[k])
                c = pj if prev2 is None else (pj - prev2)
                if k in nulls:
                    nulls[k].append(var_frac(xs, c))
                prev2 = pj
        for k in order:
            if nulls[k]:
                res[k]["null"] = float(np.mean(nulls[k]))
                res[k]["excess"] = res[k]["var"] - res[k]["null"]
    return res


def gaussian_control(n_items=64, n_time=N_WIN, rng=None):
    """随机高斯数据：iid N(0,1)。这是纯数学基线。"""
    return rng.standard_normal((n_items, n_time))


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
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp28")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("任务 C — 跨物种深度-方差曲线 + 高斯基线")
    print("=" * 84)
    print(f"  n_win={N_WIN} | k∈{KS} | 零模型 {N_SHUFFLE} 次")

    rows = []
    t0 = time.time()

    # ---- 高斯基线（关键对照）
    print("\n  [基线] 随机高斯数据")
    for i in range(6):
        Xg = gaussian_control(rng=rng)
        mu = Xg.mean(axis=1, keepdims=True)
        sd = Xg.std(axis=1, keepdims=True) + 1e-9
        Xg = (Xg - mu) / sd                 # 与真实数据同样处理
        dc = depth_curve(Xg, rng=rng)
        if dc is None:
            continue
        for k, v in dc.items():
            rows.append({"species": "GAUSSIAN", "group": f"g{i}", "k": k,
                         "var": round(v["var"], 5),
                         "null": (round(v["null"], 5) if v["null"] is not None else None),
                         "excess": (round(v["excess"], 5) if v["excess"] is not None else None)})
    print(f"    → {sum(1 for r in rows if r['species']=='GAUSSIAN')} 条")

    # ---- 三物种
    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_matched(root, view)
        print(f"\n  {species}: {len(items)} 组")
        if not items:
            continue
        for key, X0 in items:
            mu = X0.mean(axis=1, keepdims=True)
            sd = X0.std(axis=1, keepdims=True) + 1e-9
            X = (X0 - mu) / sd              # 与高斯基线相同处理（公平对比）
            dc = depth_curve(X, rng=rng)
            if dc is None:
                continue
            for k, v in dc.items():
                rows.append({"species": species, "group": key, "k": k,
                             "var": round(v["var"], 5),
                             "null": (round(v["null"], 5) if v["null"] is not None else None),
                             "excess": (round(v["excess"], 5) if v["excess"] is not None else None)})
            print(f"    {str(key)[:22]:<22} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase28.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    # ============ 原始 v(k)
    print("\n" + "=" * 84)
    print("原始 v(k)（含高斯基线）")
    print("=" * 84)
    piv = df.pivot_table(index="k", columns="species", values="var", aggfunc="median")
    print(piv.round(4).to_string())

    # ============ 超额
    print("\n" + "=" * 84)
    print("超额（真实 − 时间打乱）")
    print("=" * 84)
    pe = df.pivot_table(index="k", columns="species", values="excess", aggfunc="median")
    print(pe.round(4).to_string())

    # ============ 曲线形状相关
    print("\n" + "=" * 84)
    print("曲线形状两两相关")
    print("=" * 84)

    def corr_pair(table, a, b):
        if a not in table.columns or b not in table.columns:
            return None
        x = table[a].dropna()
        y = table[b].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 3:
            return None
        return float(np.corrcoef(x[idx], y[idx])[0, 1])

    sps = [s for s in piv.columns]
    print("  原始 v(k):")
    shape_corr = {}
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            c = corr_pair(piv, sps[i], sps[j])
            if c is not None:
                shape_corr[f"{sps[i]}-{sps[j]}"] = round(c, 3)
                print(f"    {sps[i]:<9} vs {sps[j]:<9}: r = {c:+.3f}")

    print("\n  超额曲线:")
    exc_corr = {}
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            c = corr_pair(pe, sps[i], sps[j])
            if c is not None:
                exc_corr[f"{sps[i]}-{sps[j]}"] = round(c, 3)
                print(f"    {sps[i]:<9} vs {sps[j]:<9}: r = {c:+.3f}")

    # ============ 判定
    print("\n" + "=" * 84)
    print("判定")
    print("=" * 84)

    # 高斯基线 vs 真实
    g_corrs = [v for k, v in shape_corr.items() if "GAUSSIAN" in k]
    real_corrs = [v for k, v in shape_corr.items() if "GAUSSIAN" not in k]
    print(f"  高斯 vs 真实的 v(k) 相关: "
          f"{[f'{c:+.2f}' for c in g_corrs]}")
    print(f"  真实物种间的 v(k) 相关: "
          f"{[f'{c:+.2f}' for c in real_corrs]}")

    g_mean_excess = df[df.species == "GAUSSIAN"]["excess"].median()
    real_excess = df[df.species != "GAUSSIAN"].groupby("species")["excess"].median()
    print(f"\n  高斯基线超额中位 = {g_mean_excess:+.4f}")
    print(f"  真实物种超额中位: {real_excess.round(4).to_dict()}")

    exc_corrs = list(exc_corr.values())
    if g_corrs and min(g_corrs) > 0.9:
        v = ("❌ 高斯随机数据给出与真实数据几乎相同的 v(k) 曲线 —— "
             "该曲线是聚类算法的数学必然，不携带生物信息")
    elif exc_corrs and np.median(exc_corrs) > 0.7:
        v = ("✅ 原始曲线接近数学必然，但**超额曲线在三物种间形状一致** —— "
             "存在跨物种不变结构")
    elif exc_corrs and np.median(exc_corrs) < 0.3:
        v = "❌ 超额曲线在三物种间形状不一致 —— 无普适性"
    else:
        v = "⚠️ 证据混合，需更多物种/数据"
    print(f"\n  {v}")

    (out / "phase28_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN, "ks": KS, "n_shuffle": N_SHUFFLE,
        "gaussian_baseline": True,
        "n_records": int(len(df)),
        "var_by_k": {str(int(k)): {sp: (None if pd.isna(v_) else round(float(v_), 4))
                                   for sp, v_ in row.items()}
                     for k, row in piv.iterrows()},
        "excess_by_k": {str(int(k)): {sp: (None if pd.isna(v_) else round(float(v_), 4))
                                      for sp, v_ in row.items()}
                        for k, row in pe.iterrows()},
        "shape_corr_var": shape_corr,
        "shape_corr_excess": exc_corr,
        "gaussian_excess_median": round(float(g_mean_excess), 4) if pd.notna(g_mean_excess) else None,
        "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
