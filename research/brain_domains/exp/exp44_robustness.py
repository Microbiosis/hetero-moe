"""Phase 44: 超额曲线跨物种一致的稳健性检验。

## 要检验什么

exp29 发现：超额曲线（数据驱动层次划分相对时间打乱零模型的方差份额增量）
在三个物种间形状一致（r = 0.72–0.96），而高斯随机数据与之无关（r ≈ 0）。

**风险**：三个数据集都用了同一套预处理（相关距离 + average linkage + 64 维 PCA）。
若结论依赖这套选择，它只是"预处理的人为一致性"，不是生物普适性。

## 八个配置（4 聚类 × 2 距离）

    聚类方法: average / ward / complete / single
    距离度量: correlation / euclidean

每个配置下重新算超额曲线，看三物种两两相关。

## 判据（严格）

    所有 8 个配置下，三物种两两 r 的**最小值** > 0.5
        → 结论稳健
    任一配置下 r 掉到 0.3 以下
        → 结论依赖预处理，不成立
    中间
        → 部分稳健，需报告哪些配置失效

## 计算控制

    每物种 12 组 | k 阶梯 10 点 | shuffle 5 次
    8 配置 × 3 物种 × 12 组 × 10 k × 6 次计算
    ≈ 17,000 次层次聚类，Space 上约 2-4 分钟
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

OUT = Path("/domains/runs/robustness")
EEG_ROOT = Path("/domains/omnibrain4d")   # 若不在这里，用下面的备用

KS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
N_PER_SPECIES = 12
N_WIN = 600
N_SHUFFLE = 5

METHODS = ["average", "ward", "complete", "single"]
METRICS = ["correlation", "euclidean"]

# OmniBrain 三个 spike 视图的位置（在桶的 omnibrain/ 下）
VIEWS = {
    "mouse": "omnibrain/spike_view",
    "macaque": "omnibrain/spike_view_macaque",
    "human": "omnibrain/spike_view_human",
}


# ------------------------------------------------------------ 核心
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


def build_tree(x, method, metric):
    if x.shape[0] < 4:
        return None
    d = pdist(x, metric=metric)
    d = np.nan_to_num(d, nan=1.0, posinf=2.0)
    try:
        return linkage(d, method=method)
    except Exception:
        return None


def nested_labels(Z, n, ks):
    """从同一棵树切出嵌套划分（按高度降序）。"""
    heights = np.sort(Z[:, 2])[::-1]
    out = {}
    for k in ks:
        if k >= n:
            continue
        h = heights[min(k - 1, len(heights) - 1)]
        try:
            out[k] = fcluster(Z, t=h - 1e-9, criterion="distance")
        except Exception:
            continue
    return out


def excess_curve(x, method, metric, rng, n_shuffle=N_SHUFFLE, ks=KS):
    """返回 {k: excess}。"""
    Z = build_tree(x, method, metric)
    if Z is None:
        return None
    parts = nested_labels(Z, x.shape[0], ks)
    if len(parts) < 3:
        return None
    order = sorted(parts)

    projs = {k: proj_group_mean(x, parts[k]) for k in order}
    real = {}
    prev = None
    for k in order:
        c = projs[k] if prev is None else (projs[k] - prev)
        real[k] = var_frac(x, c)
        prev = projs[k]

    nulls = {k: [] for k in order}
    for _ in range(n_shuffle):
        xs = np.stack([rng.permutation(x[i]) for i in range(x.shape[0])])
        Zs = build_tree(xs, method, metric)
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

    out = {}
    for k in order:
        if nulls[k]:
            out[k] = real[k] - float(np.mean(nulls[k]))
    return out if len(out) >= 5 else None


def load_view(root: Path, view: str, n_win=N_WIN, n_groups=N_PER_SPECIES):
    d = root / view
    if not (d / "meta.parquet").exists():
        return []
    meta = pd.read_parquet(d / "meta.parquet")
    if view.endswith("spike_view"):
        gcol = "session"
    elif "file" in meta.columns:
        gcol = "file"
    else:
        gcol = "session"

    sizes = meta.groupby(gcol).size().sort_values(ascending=False)
    out = []
    for gval in list(sizes.index)[:n_groups]:
        if gcol == "session" and view.endswith("spike_view"):
            f = d / f"windows_sess{int(gval):03d}.npz"
        else:
            f = d / str(gval)
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < n_win:
            continue
        X = w.mean(axis=2).T.astype(np.float64)     # (64, n_win_full)
        X = X[:, ::2]                                # 零重叠
        if X.shape[1] < n_win:
            continue
        s = (X.shape[1] - n_win) // 2
        X = X[:, s:s + n_win]
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-9
        out.append((str(gval), (X - mu) / sd))
    return out


def gaussian_control(n_items=64, n_time=N_WIN, rng=None, n=8):
    return [rng.standard_normal((n_items, n_time)) for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("=" * 84, flush=True)
    print("Phase 44 — 超额曲线跨物种一致的稳健性检验", flush=True)
    print("=" * 84, flush=True)
    print(f"  {len(METHODS)} 聚类 × {len(METRICS)} 距离 = {len(METHODS)*len(METRICS)} 配置",
          flush=True)
    print(f"  k={KS} | 每物种 {N_PER_SPECIES} 组 | shuffle {N_SHUFFLE}", flush=True)

    # ---- 载入数据
    t0 = time.time()
    root = Path("/domains")
    data = {}
    for sp, view in VIEWS.items():
        items = load_view(root, view)
        data[sp] = items
        print(f"  {sp}: {len(items)} 组", flush=True)
    data["GAUSSIAN"] = gaussian_control(rng=rng)

    if any(len(v) == 0 for k, v in data.items() if k != "GAUSSIAN"):
        print("⚠️ 某个物种无数据，检查路径", flush=True)

    # ---- 逐配置算超额曲线
    all_results = {}
    for method in METHODS:
        for metric in METRICS:
            key = f"{method}+{metric}"
            t1 = time.time()
            curves = {}
            for sp, items in data.items():
                vals = []
                for name, x in (items if sp != "GAUSSIAN" else [(f"g{i}", a) for i, a in enumerate(items)]):
                    c = excess_curve(x, method, metric, rng)
                    if c is not None:
                        vals.append(c)
                if not vals:
                    continue
                # 每组对齐到共同 k
                common = sorted(set.intersection(*[set(v) for v in vals]))
                if len(common) < 5:
                    continue
                curves[sp] = {k: float(np.median([v[k] for v in vals])) for k in common}
            all_results[key] = curves
            sps = list(curves)
            print(f"  [{key}] 物种 {sps} ({time.time()-t1:.0f}s)", flush=True)

    # ---- 算两两相关
    print("\n" + "=" * 84, flush=True)
    print("各配置下三物种两两相关", flush=True)
    print("=" * 84, flush=True)
    print(f"{'配置':<22} {'人-鼠':>8} {'猕-鼠':>8} {'人-猕':>8} {'高-人':>8} "
          f"{'高-鼠':>8} {'最小真实':>9}", flush=True)

    table = []
    for key, curves in all_results.items():
        sps = [s for s in ["human", "macaque", "mouse", "GAUSSIAN"] if s in curves]
        if len(sps) < 3:
            continue

        def corr(a, b):
            if a not in curves or b not in curves:
                return np.nan
            ka = sorted(curves[a]); kb = sorted(curves[b])
            common = sorted(set(ka) & set(kb))
            if len(common) < 4:
                return np.nan
            x = np.array([curves[a][k] for k in common])
            y = np.array([curves[b][k] for k in common])
            return float(np.corrcoef(x, y)[0, 1])

        r_hm = corr("human", "mouse")
        r_mm = corr("macaque", "mouse")
        r_hmk = corr("human", "macaque")
        r_gh = corr("GAUSSIAN", "human")
        r_gm = corr("GAUSSIAN", "mouse")

        real = [v for v in [r_hm, r_mm, r_hmk] if np.isfinite(v)]
        min_real = min(real) if real else np.nan

        table.append({"config": key, "human_mouse": r_hm, "macaque_mouse": r_mm,
                      "human_macaque": r_hmk, "gauss_human": r_gh,
                      "gauss_mouse": r_gm, "min_real": min_real})
        print(f"{key:<22} {r_hm:>8.3f} {r_mm:>8.3f} {r_hmk:>8.3f} {r_gh:>8.3f} "
              f"{r_gm:>8.3f} {min_real:>9.3f}", flush=True)

    df = pd.DataFrame(table)
    df.to_csv(OUT / "robustness.csv", index=False)

    # ---- 汇总
    print("\n" + "=" * 84, flush=True)
    print("汇总", flush=True)
    print("=" * 84, flush=True)

    valid = df.dropna(subset=["min_real"])
    if not len(valid):
        print("无有效配置", flush=True)
        return

    print(f"  配置数: {len(valid)}", flush=True)
    print(f"  最小真实相关（三物种两两）: "
          f"中位 {valid.min_real.median():.3f}  "
          f"范围 [{valid.min_real.min():.3f}, {valid.min_real.max():.3f}]", flush=True)
    print(f"  高斯 vs 真实: 中位 {valid.gauss_human.median():.3f} "
          f"/ {valid.gauss_mouse.median():.3f}", flush=True)

    n_strong = int((valid.min_real > 0.5).sum())
    n_weak = int((valid.min_real < 0.3).sum())
    print(f"\n  最小真实相关 > 0.5 的配置: {n_strong}/{len(valid)}", flush=True)
    print(f"  最小真实相关 < 0.3 的配置: {n_weak}/{len(valid)}", flush=True)

    # 按方法/距离分开看
    print("\n  按聚类方法:", flush=True)
    for m in METHODS:
        sub = valid[valid.config.str.startswith(m + "+")]
        if len(sub):
            print(f"    {m:<10} 最小相关中位 {sub.min_real.median():.3f}", flush=True)
    print("\n  按距离度量:", flush=True)
    for mt in METRICS:
        sub = valid[valid.config.str.endswith("+" + mt)]
        if len(sub):
            print(f"    {mt:<12} 最小相关中位 {sub.min_real.median():.3f}", flush=True)

    # ---- 判定
    print("\n" + "=" * 84, flush=True)
    print("判定", flush=True)
    print("=" * 84, flush=True)

    if n_strong == len(valid):
        verdict = (f"✅ 稳健：全部 {len(valid)} 个配置下三物种两两相关都 > 0.5"
                   f"（最小 {valid.min_real.min():.3f}）"
                   f"，且高斯 < 0.3。结论不依赖预处理选择。")
    elif n_weak > 0:
        worst = valid.loc[valid.min_real.idxmin()]
        verdict = (f"❌ 不稳健：{n_weak}/{len(valid)} 个配置下相关掉到 0.3 以下"
                   f"（最差 {worst.config}: {worst.min_real:.3f}）。"
                   f"原结论部分依赖预处理选择。")
    else:
        verdict = (f"⚠️ 部分稳健：{n_strong}/{len(valid)} 个配置 > 0.5，"
                   f"最小 {valid.min_real.min():.3f}")

    print(f"  {verdict}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "excess_curve_robustness",
        "methods": METHODS, "metrics": METRICS, "ks": KS,
        "n_per_species": N_PER_SPECIES, "n_shuffle": N_SHUFFLE,
        "table": table,
        "summary": {
            "n_configs": int(len(valid)),
            "min_real_median": round(float(valid.min_real.median()), 3),
            "min_real_min": round(float(valid.min_real.min()), 3),
            "min_real_max": round(float(valid.min_real.max()), 3),
            "gauss_human_median": round(float(valid.gauss_human.median()), 3),
            "gauss_mouse_median": round(float(valid.gauss_mouse.median()), 3),
            "n_above_0.5": n_strong, "n_below_0.3": n_weak,
        },
        "by_method": {m: round(float(valid[valid.config.str.startswith(m + "+")].min_real.median()), 3)
                      for m in METHODS
                      if len(valid[valid.config.str.startswith(m + "+")])},
        "by_metric": {mt: round(float(valid[valid.config.str.endswith("+" + mt)].min_real.median()), 3)
                      for mt in METRICS
                      if len(valid[valid.config.str.endswith("+" + mt)])},
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
