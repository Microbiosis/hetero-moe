"""Phase 22: 匹配窗口数后，层次结构的物种差异还在吗？

## 要控制的混淆

exp-hier 发现超额 B（时间打乱重聚类）:
    mouse   +0.0117
    macaque +0.0072
    human   +0.0008

但三物种的窗口数不同:
    mouse   ~3000（截顶）
    macaque ~1250
    human   ~3000（截顶）

猕猴窗口少 60% → 层次聚类不稳定 → 超额被压低。
这足以解释 mouse vs macaque 的差距，但不能解释 human ≈ 0。

## 本实验

把三物种的窗口数**统一到 800**（取中段连续段），重算超额 B。

判据：
  统一后 human 仍 ≈ 0，mouse/macaque 仍 >0.005 → 差异真实
  统一后三者接近                                  → 原差异是窗口数伪影

## 附加检查

1. 用**原始 unit 数**而非 PCA 64 维重算（去掉 PCA 步骤的影响）
2. 报告每组实际 n_win，确认匹配成功
3. 计算"超额 / 打乱基线"的比值，而不只是差值（避免基线不同）
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
KS = [2, 4, 8, 16]
N_WIN = 800
N_SHUFFLE_GEOM = 8
N_PER_SPECIES = 12


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


def nested_labels(Z, n_items, ks):
    heights = np.sort(Z[:, 2])[::-1]
    out = {}
    for k in ks:
        if k >= n_items:
            continue
        idx = min(k - 1, len(heights) - 1)
        out[k] = fcluster(Z, t=heights[idx] - 1e-9, criterion="distance")
    return out


def excess_at(x, k, ks, rng, n_geom=N_SHUFFLE_GEOM):
    """真实 var(k) 与时间打乱重聚类后的 var(k) 之差。"""
    Z = build_tree(x)
    if Z is None:
        return None, None
    parts = nested_labels(Z, x.shape[0], ks)
    if k not in parts:
        return None, None
    order = sorted(parts)
    # 真实
    prev, real_v = None, None
    for kk in order:
        pj = proj_group_mean(x, parts[kk])
        c = pj if prev is None else (pj - prev)
        if kk == k:
            real_v = var_frac(x, c)
            break
        prev = pj
    if real_v is None:
        return None, None

    # 时间打乱
    nulls = []
    for _ in range(n_geom):
        xs = np.stack([rng.permutation(x[i]) for i in range(x.shape[0])])
        Zs = build_tree(xs)
        if Zs is None:
            continue
        ps = nested_labels(Zs, xs.shape[0], ks)
        prev = None
        for kk in sorted(ps):
            pj = proj_group_mean(xs, ps[kk])
            c = pj if prev is None else (pj - prev)
            if kk == k:
                nulls.append(var_frac(xs, c))
                break
            prev = pj
    if not nulls:
        return real_v, None
    return real_v, float(np.mean(nulls))


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
        X = w.mean(axis=2).T.astype(np.float64)      # (64, n_win_full)
        X = X[:, ::2]                                 # 零重叠
        if X.shape[1] < n_win:
            continue
        s = (X.shape[1] - n_win) // 2                 # 取中段，统一长度
        X = X[:, s:s + n_win]
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-9
        out.append((str(gval), (X - mu) / sd))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp22")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print(f"Phase 22 — 匹配窗口数 n_win={N_WIN} 后，层次差异还在吗？")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_matched(root, view)
        print(f"\n  {species}: {len(items)} 组（每组恰好 {N_WIN} 窗）")
        if not items:
            continue
        for key, X in items:
            for k in KS:
                real, null = excess_at(X, k, KS, rng)
                if real is None or null is None:
                    continue
                rows.append({"species": species, "group": key, "k": k,
                             "n_win": int(X.shape[1]),
                             "var": round(real, 5),
                             "null_geom": round(null, 5),
                             "excess": round(real - null, 5),
                             "ratio": round(real / max(null, 1e-9), 4)})
            print(f"    {str(key)[:26]:<26} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase22.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 78)
    print("匹配 n_win 后的超额")
    print("=" * 78)
    print(f"{'species':<9} {'n':>4} {'n_win':>6} {'真实var':>9} {'打乱var':>9} {'超额':>9} {'比值':>7}")
    g = df.groupby("species").agg(
        n=("excess", "size"), n_win=("n_win", "median"),
        var=("var", "median"), null=("null_geom", "median"),
        excess=("excess", "median"), ratio=("ratio", "median"))
    for sp, r in g.iterrows():
        print(f"{sp:<9} {int(r['n']):>4} {r['n_win']:>6.0f} {r['var']:>9.4f} "
              f"{r['null']:>9.4f} {r['excess']:>+9.4f} {r['ratio']:>7.3f}")

    print("\n  按 k 层:")
    p = df.pivot_table(index="k", columns="species", values="excess", aggfunc="median")
    print(p.round(4).to_string())

    # 判定
    print("\n" + "=" * 78)
    print("判定")
    print("=" * 78)
    e = g["excess"]
    human_e = float(e.get("human", np.nan))
    mm_e = float(np.nanmean([e.get("mouse", np.nan), e.get("macaque", np.nan)]))

    print(f"  human 超额 {human_e:+.4f} | mouse/macaque 均值 {mm_e:+.4f}")
    if abs(human_e) < 0.003 and mm_e > 0.005:
        v = ("✅ 匹配窗口数后差异仍存在：人类无层次结构，小鼠/猕猴有 "
             f"（{mm_e:+.4f} vs {human_e:+.4f}）")
    elif abs(human_e - mm_e) < 0.003:
        v = "❌ 匹配后差异消失 —— 原差异是窗口数伪影"
    else:
        v = f"⚠️ 部分保留：human {human_e:+.4f} / 动物 {mm_e:+.4f}"
    print(f"  {v}")

    (out / "phase22_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win_matched": N_WIN, "ks": KS,
        "n_shuffle_geom": N_SHUFFLE_GEOM, "n_records": int(len(df)),
        "by_species": {sp: {c: (None if pd.isna(v_) else round(float(v_), 4))
                            for c, v_ in r.items()} for sp, r in g.iterrows()},
        "by_k": {str(k): {sp: (None if pd.isna(v_) else round(float(v_), 4))
                          for sp, v_ in row.items()}
                 for k, row in p.iterrows()},
        "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
