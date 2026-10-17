"""Phase 5b: L2 解剖区 vs L1 功能类 —— 修正版。

## 修正什么

exp5 有**循环论证 bug**: 用 mean_abs 的 qcut 分组, 再测这些组的 mean_abs 差异,
必然得到 ε² ≈ 0.94。39 个 session 数值几乎相同就是证据。

## 正确做法

直接从未经 PCA 的原始数据计算区域级活动:

  1. 读 `steinmetz/spikes_session_XXX.npz`  → (n_units, n_bins) @50ms
  2. 读 `units.parquet`                     → 每 unit 的 region (L2) 与 functional_domain (L1)
  3. 重采样到 100ms, z-score
  4. 对每个窗口 (8 bins), 计算**每个 L2 区域**的平均活动
     → L2 标签 = argmax
     → L1 标签 = 用 L1 归并后同样计算
  5. 在**同一批窗口**上比较两种标签的 ε²

这样 L1 与 L2 用的是完全相同的窗口与特征, 唯一差别是标签粒度。

## 判据

  ε²(L2) / ε²(L1) > 1.3  且  打乱后该比值 ≈ 1
  → L2 携带真实增量信息

## 注意

L2 类别数远多于 L1 (每 session 3-14 vs 2-6), 而 ε² 随类别数增加。
因此必须报告**打乱对照**: 打乱后若 L2 的 ε² 也按比例增加, 说明只是自由度效应。
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
N_SHUFFLE = 10
WIN, STRIDE = 8, 4


def epsilon_sq(groups, min_per=MIN_PER_CLASS):
    groups = [g for g in groups if len(g) >= min_per]
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


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp5b")
    out.mkdir(parents=True, exist_ok=True)

    stein = root / "steinmetz"
    units_f = stein / "units.parquet"

    print("=" * 76)
    print("Phase 5b — L2 vs L1 标签粒度对比（修正循环论证）")
    print("=" * 76)
    print(f"  steinmetz 目录: {stein} (存在 {stein.exists()})")
    print(f"  units: {units_f} (存在 {units_f.exists()})")

    if not units_f.exists():
        print("❌ 缺少 units.parquet")
        return

    units = pd.read_parquet(units_f)
    print(f"  units {len(units):,}, L2 区数 {units['region'].nunique()}, "
          f"L1 类数 {units['functional_domain'].nunique()}")

    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()

    for sess in sorted(units["session"].unique()):
        f = stein / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)       # (n_units, n_bins_50ms)
        u = units[units["session"] == sess].reset_index(drop=True)
        if sp.shape[0] != len(u):
            print(f"  sess {sess:02d}: 行数不匹配 ({sp.shape[0]} vs {len(u)}), 跳过")
            continue

        # 50ms → 100ms（相邻两 bin 求和）
        nb = sp.shape[1] // 2
        x = sp[:, :nb * 2].reshape(sp.shape[0], nb, 2).sum(axis=2)   # (units, bins100)
        # 逐 unit z-score
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        z = (x - mu) / sd

        # 分窗
        starts = list(range(0, nb - WIN + 1, STRIDE))
        if len(starts) < 50:
            continue

        # 每窗、每 unit 的平均 |z|
        n_win = len(starts)
        act = np.empty((n_win, sp.shape[0]), dtype=np.float32)
        for i, s in enumerate(starts):
            act[i] = np.abs(z[:, s:s + WIN]).mean(axis=1)     # (units,)

        l2_ids = pd.Categorical(u["region"]).codes               # (units,)
        l1_ids = pd.Categorical(u["functional_domain"]).codes    # (units,)
        n_l2 = l2_ids.max() + 1
        n_l1 = l1_ids.max() + 1
        if n_l2 < 2 or n_l1 < 2:
            continue

        # 每窗的区域级活动 → 标签 = argmax
        def region_scores(ids, n_cls):
            sc = np.zeros((n_win, n_cls), dtype=np.float32)
            for c in range(n_cls):
                m = ids == c
                if m.any():
                    sc[:, c] = act[:, m].mean(axis=1)
            return sc

        sc_l2 = region_scores(l2_ids, n_l2)
        sc_l1 = region_scores(l1_ids, n_l1)

        # 用同一特征（sc_l2 的最大值序列）比较两种标签的 ε²
        # —— 关键: 对 L1 也要重新算标签, 不能用 exp3 的旧标签, 否则比较不公平
        y_l2 = sc_l2.argmax(axis=1)
        y_l1 = sc_l1.argmax(axis=1)

        probe = act.max(axis=1)        # 窗级活动强度（中性特征, 与标签无关）

        e2_l2, p_l2, k_l2 = epsilon_sq([probe[y_l2 == c] for c in np.unique(y_l2)])
        e2_l1, p_l1, k_l1 = epsilon_sq([probe[y_l1 == c] for c in np.unique(y_l1)])

        if e2_l2 is None or e2_l1 is None:
            continue

        # 打乱对照: 各自打乱标签
        e2_l2_sh, e2_l1_sh = [], []
        for _ in range(N_SHUFFLE):
            a = y_l2.copy(); rng.shuffle(a)
            e, _, _ = epsilon_sq([probe[a == c] for c in np.unique(a)])
            if e is not None:
                e2_l2_sh.append(e)
            b = y_l1.copy(); rng.shuffle(b)
            e, _, _ = epsilon_sq([probe[b == c] for c in np.unique(b)])
            if e is not None:
                e2_l1_sh.append(e)

        ratio = e2_l2 / e2_l1 if e2_l1 > 1e-6 else float("nan")
        ratio_sh = (np.mean(e2_l2_sh) / np.mean(e2_l1_sh)) if e2_l1_sh and np.mean(e2_l1_sh) > 1e-6 else float("nan")

        rows.append({
            "session": int(sess), "n_windows": n_win,
            "k_l1": int(k_l1), "k_l2": int(k_l2),
            "eps2_l1": round(e2_l1, 5), "eps2_l2": round(e2_l2, 5),
            "eps2_l1_shuf": round(float(np.mean(e2_l1_sh)), 5) if e2_l1_sh else None,
            "eps2_l2_shuf": round(float(np.mean(e2_l2_sh)), 5) if e2_l2_sh else None,
            "ratio_l2_over_l1": round(float(ratio), 3) if math.isfinite(ratio) else None,
            "ratio_shuf": round(float(ratio_sh), 3) if math.isfinite(ratio_sh) else None,
        })
        print(f"  sess {sess:02d}: {n_win:5d} 窗 | L1 {k_l1} 类 ε²={e2_l1:.4f} | "
              f"L2 {k_l2} 类 ε²={e2_l2:.4f} | 比值 {ratio:.2f} "
              f"(打乱后 {ratio_sh:.2f})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase5b_l2.csv", index=False)

    if not len(df):
        print("无可用 session")
        return

    print("\n" + "=" * 76)
    print("汇总")
    print("=" * 76)
    r = df["ratio_l2_over_l1"].median()
    rs = df["ratio_shuf"].median()
    print(f"  ε²(L2)/ε²(L1) 中位:        {r:.3f}")
    print(f"  打乱后的同一比值:           {rs:.3f}")
    print(f"  真实信号比 (r - rs + 1):    {r - rs + 1:.3f}")
    print(f"  L1 中位 ε²: {df['eps2_l1'].median():.4f} | "
          f"L2 中位 ε²: {df['eps2_l2'].median():.4f}")

    if (r - rs) > 0.3 and r > 1.3:
        v = "✅ L2 携带真实增量信息（超出自由度效应）"
    elif abs(r - rs) < 0.15:
        v = "❌ L2 的提升完全是类别数增加导致，无真实增量信息"
    else:
        v = "⚠️ 弱增量，需更多数据"
    print(f"\n  判定: {v}")

    (out / "phase5b_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "median_ratio_l2_over_l1": round(float(r), 4),
        "median_ratio_shuffled": round(float(rs), 4),
        "median_eps2_l1": round(float(df["eps2_l1"].median()), 4),
        "median_eps2_l2": round(float(df["eps2_l2"].median()), 4),
        "n_sessions": int(len(df)),
        "verdict": v,
        "method": ("直接从原始 spike 计算区域级活动, 同一批窗口上比较 L1/L2 标签的 ε²; "
                   "打乱对照量化'类别数增加'带来的基线提升。"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
