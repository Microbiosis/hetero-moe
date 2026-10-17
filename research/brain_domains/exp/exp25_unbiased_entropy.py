"""Phase 25: 去掉 z-score 归一化偏差后的纠缠熵（含人类修正）。

## 修正 exp24 的两个问题

### 问题 1: 人类一直是 0 组
    if len(w) < n_win * 2:  continue      # 要求 1200 窗
人类每片段平均 843 窗 → 全部被跳过。
改为：只需要 n_win 个窗口（取中段），条件放宽到 len(w) >= n_win。

### 问题 2（关键）: z-score 人为抬高纠缠熵
    X = (X - mu) / sd        # 每个成分方差强制为 1
这使协方差矩阵对角全为 1，谱必然接近均匀 → 纠缠熵必然接近 ln(64)。
**测到的高纠缠是归一化造成的，不是数据性质。**

## 本实验：三种归一化对照

  A_raw      不做任何归一化（原始 PCA 投影）
  B_global   全局缩放（除以所有元素的标准差），保留相对方差
  C_percomp  逐成分 z-score（exp24 的做法，作为对照）

只有 A/B 的谱结构才反映真实的方差分布。
若 A/B 下纠缠熵显著低于 C，说明 exp24 的结论是归一化伪影。

## 附带：协方差谱的有效秩（不依赖熵）

    PR = (Σλ)² / Σλ²        （参与率，对尺度不变）

这个量在 A/B/C 三种归一化下**应当一致**（因为它是尺度的比值），
可作为交叉验证。
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


def schmidt(X):
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return None
    p = s ** 2
    p = p / p.sum()
    S = float(-(p * np.log(p)).sum())
    d = min(X.shape)
    Smax = float(np.log(d)) if d > 1 else 1.0
    return {"S_ent": S, "S_norm": S / Smax,
            "eff_rank": float(np.exp(S)), "top1": float(p[0])}


def participation_ratio(X):
    """协方差特征值的参与率，对整体尺度不变。"""
    C = np.cov(X)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0, None)
    if ev.sum() <= 1e-12:
        return np.nan
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def blockiness(X):
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    d = C.shape[0]
    if d < 4:
        return np.nan
    dist = np.clip(1.0 - C, 0, 2)
    np.fill_diagonal(dist, 0.0)
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
    except Exception:
        return np.nan
    Ca = np.abs(C)
    np.fill_diagonal(Ca, 0.0)
    Cr = Ca[np.ix_(order, order)]
    adj = float((Cr[np.arange(d - 1), np.arange(1, d)]).mean())
    glob = float(Ca[np.triu_indices(d, 1)].mean())
    return adj / max(glob, 1e-9)


def normalize(X, mode):
    if mode == "A_raw":
        return X.copy()
    if mode == "B_global":
        s = X.std() + 1e-12
        return X / s
    if mode == "C_percomp":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mu) / sd
    raise ValueError(mode)


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
        # 修正: 只需 n_win 个窗口，不再要求 2×
        if len(w) < n_win:
            continue
        X = w.mean(axis=2).T.astype(np.float64)      # (64, n_win_full)
        X = X[:, ::2]
        if X.shape[1] < n_win:
            continue
        s = (X.shape[1] - n_win) // 2
        out.append((str(gval), X[:, s:s + n_win]))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp25")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("Phase 25 — 去掉 z-score 偏差后的纠缠熵")
    print("=" * 84)
    print(f"  n_win={N_WIN} | 三种归一化对照")

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
            for mode in ["A_raw", "B_global", "C_percomp"]:
                Xn = normalize(X, mode)
                sc = schmidt(Xn)
                if sc is None:
                    continue
                rows.append({"species": species, "group": key, "norm": mode,
                             **{k: round(v, 4) for k, v in sc.items()},
                             "PR": round(participation_ratio(Xn), 3),
                             "blockiness": round(blockiness(Xn), 3)})
            print(f"    {str(key)[:24]:<24} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase25.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 84)
    print("纠缠熵 vs 归一化方式（关键对照）")
    print("=" * 84)
    p = df.pivot_table(index="norm", columns="species",
                       values=["S_norm", "eff_rank", "PR"], aggfunc="median")
    for metric in ["S_norm", "eff_rank", "PR"]:
        print(f"\n  {metric}:")
        print(p[metric].round(3).to_string())

    print("\n" + "=" * 84)
    print("块性（应当与归一化无关，作为一致性检查）")
    print("=" * 84)
    print(df.pivot_table(index="norm", columns="species",
                         values="blockiness", aggfunc="median").round(3).to_string())

    # ============ 判定
    print("\n" + "=" * 84)
    print("判定")
    print("=" * 84)

    a = df[df.norm == "A_raw"].groupby("species")["S_norm"].median()
    c = df[df.norm == "C_percomp"].groupby("species")["S_norm"].median()
    a_all = float(a.median()) if len(a) else np.nan
    c_all = float(c.median()) if len(c) else np.nan

    print(f"  A_raw      S/ln64 中位 = {a_all:.3f}")
    print(f"  C_percomp  S/ln64 中位 = {c_all:.3f}")
    print(f"  差值 = {c_all - a_all:+.3f}")

    if c_all - a_all > 0.10:
        v_norm = ("⚠️ z-score 显著抬高纠缠熵（+{:.3f}）—— exp24 的高纠缠结论是归一化伪影"
                  .format(c_all - a_all))
    else:
        v_norm = "z-score 影响不大，exp24 的结论基本成立"
    print(f"  → {v_norm}")

    pr = df.groupby("norm")["PR"].median()
    print(f"\n  参与率 PR（尺度不变，三者应一致）:")
    for m, v in pr.items():
        print(f"    {m:<10} {v:.2f}")
    pr_spread = float(pr.max() - pr.min())
    print(f"    跨度 {pr_spread:.3f} → "
          + ("✅ 一致（PR 确实尺度不变）" if pr_spread < 0.5 else "⚠️ 不一致，需查"))

    blk = df.groupby("norm")["blockiness"].median()
    print(f"\n  块性（三者应一致）: {blk.round(3).to_dict()}")
    blk_spread = float(blk.max() - blk.min())
    print(f"    跨度 {blk_spread:.3f} → "
          + ("✅ 一致" if blk_spread < 0.1 else "⚠️ 不一致，需查"))

    # 结论
    s_final = min(a_all, c_all)
    if s_final > 0.85:
        v_final = ("❌ 即使去掉归一化偏差，纠缠度仍接近最大（{}）—— "
                   "线性框架下不存在低维压缩结构".format(f"{s_final:.2f}"))
    elif s_final < 0.6:
        v_final = f"✅ 低纠缠（{s_final:.2f}）—— 存在压缩结构"
    else:
        v_final = f"⚠️ 中等纠缠（{s_final:.2f}）"
    print(f"\n  最终: {v_final}")

    (out / "phase25_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN,
        "fixes": ["人类窗口数要求 2×n_win → n_win",
                  "加入 A_raw / B_global / C_percomp 三种归一化对照"],
        "n_records": int(len(df)),
        "by_norm_species": {
            m: {sp: {"S_norm": round(float(df[(df.norm == m) & (df.species == sp)]["S_norm"].median()), 4),
                     "eff_rank": round(float(df[(df.norm == m) & (df.species == sp)]["eff_rank"].median()), 2),
                     "PR": round(float(df[(df.norm == m) & (df.species == sp)]["PR"].median()), 3),
                     "blockiness": round(float(df[(df.norm == m) & (df.species == sp)]["blockiness"].median()), 3)}
                 for sp in df.species.unique()}
            for m in df.norm.unique()},
        "S_norm_raw": round(a_all, 4),
        "S_norm_percomp": round(c_all, 4),
        "normalization_effect": round(c_all - a_all, 4),
        "normalization_verdict": v_norm,
        "PR_spread": round(pr_spread, 4),
        "blockiness_spread": round(blk_spread, 4),
        "final_verdict": v_final,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
