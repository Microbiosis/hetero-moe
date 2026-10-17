"""Phase 5: L2 细粒度解剖区是否比 L1 六大类携带更多信息?

# ⚠️ 本文件有循环论证 bug，结果无效。修正版见 exp5b_l2_proper.py

bug 位置：
    q = pd.qcut(vv, 4)              # 用 mean_abs 的 qcut 分组
    groups = [vv[q == g_] for ...]  # 再测这些组的 mean_abs 差异
    epsilon_sq(groups)              # → 必然得到 ε² ≈ 0.937

症状：39 个 session 的"L1 内部异质性 ε²" 几乎完全相同（0.9372–0.9374）。
真实生物数据不会如此一致 —— 这个"过于完美"本身就是 bug 的信号。

保留本文件的原因：它是方法学失误的实证记录，配合 FINDINGS.md 第五节阅读。
正确做法（直接从未经 PCA 的原始 spike 计算区域级活动）见 exp5b。

## 原始动机（仍然成立）

L1 的失败可能源于**过粗的归并**:
  VISp / VISl / VISal / VISam / VISpm / VISrl / VISa → "sensory" (7 个区被合并)
  CA1 / CA2 / CA3 / DG / SUB / ProS            → "memory"  (6 个区被合并)

如果不同解剖区确实有不同的动力学/功能, L1 的合并就把它们抹平了。

## 三层粒度的效应量对比

  L1   functional_domain  (6 类)
  L2   region             (原始解剖名, 每 session 数十个)
  L2+  更细: 用 cell_sub_class / layer 等（若数据支持）

用 Phase 3 的同一套统计量 + ε² 检验, 但注意:
**类别越多, ε² 天然越大**（自由度增加）。因此必须做**打乱对照**:
若 L2 的 ε² 提升只是"类别多了", 打乱后也会同样提升。

判据:
  ε²(L2) / ε²(L1) 显著 > 1  且  打乱后该比值 ≈ 1
  → L2 携带真实增量信息

## 预期风险

spike_view 的 64 个成分是**逐 session PCA** 的结果，成分编号在不同
session 间不对应。因此 L2 的解剖名只能来自 units.parquet，
而窗口的域标签是按"窗内主导域"由 L1 投票得来 —— **L2 标签需要重新计算**。

本实验直接从 units.parquet 拿 region，重新按 region 投票。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

SEED = 0
MIN_PER_CLASS = 40


def epsilon_sq(groups):
    groups = [g for g in groups if len(g) >= MIN_PER_CLASS]
    k = len(groups)
    if k < 2:
        return None, None, 0
    n = sum(len(g) for g in groups)
    if n <= k:
        return None, None, 0
    try:
        H, p = kruskal(*groups)
    except ValueError:
        return None, None, 0
    return float(max(0.0, (H - k + 1) / (n - k))), float(p), k


def vote_labels(w_units_dom, regions, n_region):
    """按窗内活动强度投票出**区域级**标签。

    这里用简化做法: 用 units 的 L2 区域索引替代 L1 域索引,
    在 window_stats 相同的活动向量上做加权投票。
    """
    # 由调用方按 region 索引传入, 此处留接口
    raise NotImplementedError


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp5")
    out.mkdir(parents=True, exist_ok=True)

    # 需要 units.parquet (含 region) 与 windows
    units_f = Path("/data/steinmetz/units.parquet")
    if not units_f.exists():
        units_f = Path("/data/brain-domains/steinmetz/units.parquet")

    print("=" * 76)
    print("Phase 5 — L2 细粒度解剖区 vs L1 六大类")
    print("=" * 76)
    print(f"  数据源 {root}")
    print(f"  units  {units_f} (存在: {units_f.exists()})")

    from exp3_domain_fingerprint import window_stats

    sp = root / "spike_view"
    meta = pd.read_parquet(sp / "meta.parquet")

    if not units_f.exists():
        print("  ❌ 找不到 units.parquet, 无法获取 L2 标签")
        return

    units = pd.read_parquet(units_f)
    print(f"  units: {len(units):,} 行, region 数 {units['region'].nunique()}")

    rows = []
    t0 = time.time()

    for sess in sorted(meta["session"].unique()):
        f = sp / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        m = meta[meta["session"] == sess].reset_index(drop=True)
        u = units[units["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(u) == 0:
            continue

        st = window_stats(w)
        # 以 mean_abs 为探针（Phase 3 最强信号）
        v = st["mean_abs"]
        y_l1 = pd.Categorical(m["domain_l1"]).codes

        # L2: 窗级 region 标签需从 unit 级聚合, 这里用**近似**:
        # 每个窗的 L2 = 该 session 内活动最强的 region。
        # 由于 window_stats 已丢了 unit 级信息, 只能用 session 级统计做粗对照:
        # 计算每个 region 的 unit 在两两窗口上的平均活动, 取其最大者。
        reg_codes = pd.Categorical(u["region"]).codes
        n_reg = len(np.unique(reg_codes))
        if n_reg < 2:
            continue
        # 用 region 的单元索引在 w 上的投影活动（近似）
        # w 的形状是 (n_win, 64, 8)，64 是 PCA 成分，不与 unit 对应。
        # 因此这里只能做 L1 内部的分层检验：在 L1 同一类内, 看 L2 是否仍有差异。
        sub_stats = []
        for l1_name in m["domain_l1"].unique():
            sel = (m["domain_l1"] == l1_name).to_numpy()
            if sel.sum() < MIN_PER_CLASS * 2:
                continue
            # 该 L1 类别内, 用 region 的**窗口占比**做代理
            # （无法直接算 region 级活动, 退而检验 L1 内部的异质性）
            vv = v[sel]
            # 用四分位分组代替 region（保守做法：若 L1 内部完全同质, ε²≈0）
            q = pd.qcut(vv, 4, labels=False, duplicates="drop")
            groups = [vv[q == g_] for g_ in np.unique(q)]
            e2, p, k = epsilon_sq(groups)
            if e2 is not None:
                sub_stats.append({"l1": l1_name, "eps2_within": e2, "n": int(sel.sum())})

        if not sub_stats:
            continue

        # L1 整体
        g_l1 = [v[y_l1 == c] for c in np.unique(y_l1)]
        e2_l1, p_l1, k_l1 = epsilon_sq(g_l1)

        rows.append({
            "session": int(sess),
            "n_windows": int(len(w)),
            "n_l1_classes": int(k_l1),
            "n_l2_regions": int(n_reg),
            "eps2_l1": round(e2_l1, 4) if e2_l1 is not None else None,
            "mean_within_l1_eps2": round(float(np.mean([s["eps2_within"] for s in sub_stats])), 4),
            "mean_within_n": int(np.mean([s["n"] for s in sub_stats])),
        })
        print(f"  sess {sess:02d}: {len(w):5d} 窗 | L1 {k_l1} 类 ε²={e2_l1:.4f} | "
              f"L1 内部异质性 ε²={rows[-1]['mean_within_l1_eps2']:.4f} | "
              f"L2 区数 {n_reg}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase5_l2.csv", index=False)

    if not len(df):
        print("无可用 session")
        return

    print("\n" + "=" * 76)
    print("汇总")
    print("=" * 76)
    e1 = df["eps2_l1"].median()
    ew = df["mean_within_l1_eps2"].median()
    print(f"  L1 整体 ε² 中位:          {e1:.4f}")
    print(f"  L1 内部异质性 ε² 中位:    {ew:.4f}")
    print(f"  比值 (内部/整体):         {ew/e1:.2f}")

    if ew / e1 > 0.5:
        verdict = "⚠️ L1 内部仍有大量异质性 —— L2 可能携带增量信息"
    else:
        verdict = "❌ L1 内部基本同质 —— 细分到 L2 不会有实质增量"
    print(f"\n  判定: {verdict}")

    (out / "phase5_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "median_eps2_l1": round(float(e1), 4),
        "median_within_l1_eps2": round(float(ew), 4),
        "ratio": round(float(ew / e1), 3),
        "n_sessions": int(len(df)),
        "verdict": verdict,
        "note": ("由于 spike_view 的 64 维是逐 session PCA 成分, 与原始 unit 不对应, "
                 "无法直接计算 region 级活动。本实验用『L1 类内的活动异质性』作为 "
                 "L2 增量的上界估计: 若 L1 内部已高度同质, 则 L2 无从细分出更多信息。"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
