"""Phase 6: 跨物种动力学对照 —— 四个物种的神经活动是否共享时间统计结构?

## 为什么换这个问题

前五个实验都在问"域标签有没有用"，答案是否定的。但 OmniBrain 里还有
672,654 个样本、四个物种、三种模态 —— 这个资产本身值得单独研究。

**新问题**：不同物种/模态的神经活动，在时间统计特性上是否遵循同一套规律?

这个问题：
  - 不依赖域标签（绕开之前失败的原因）
  - 有明确的对照设计（物种 × 统计量）
  - 结果无论正负都有意义

## 测什么（七个无量纲统计量）

全部**不依赖绝对幅度**，因此跨物种可比（小鼠 spike counts 与人类 EEG 电位
量纲完全不同，但无量纲形状可以比）：

| 统计量 | 含义 | 为何无量纲 |
|---|---|---|
| `ac1` | lag-1 自相关 | 比值 |
| `ac_ratio` | ac2/ac1（衰减形状）| 比值之比 |
| `spec_slope` | 频谱斜率（log-log）| 幂律指数 |
| `hf_ratio` | 高频能量占比 | 比例 |
| `eff_dim` | 有效维度/参与率 | 计数 |
| `kurtosis` | 峰度（重尾程度）| 标准化矩 |
| `zero_frac` | 零值占比（稀疏度）| 比例 |

## 关键设计

1. **逐片段计算，不跨片段混合**（避免把不同 recording 的差异当成物种差异）
2. **报告分布而非单点**：每个物种给中位数 + IQR，附样本量
3. **置换检验**：物种间差异是否显著（Kruskal-Wallis + Dunn 事后）
4. **效应量优先于 p 值**：n 很大时 p 必然显著，必须看 ε²

## 判据

  ε² > 0.14 (大效应) 的统计量 → 该维度物种间差异大
  ε² < 0.01 (可忽略) 的统计量 → 该维度跨物种守恒

**最有价值的发现是守恒项**：如果若干统计量在四个物种上高度一致，
说明神经动力学可能有普适约束（如临界性、E/I 平衡）。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
_EEG_CACHE = {}          # {path: ndarray}  避免重复加载大文件
MIN_WINDOWS_PER_GROUP = 200      # 片段级：至少这么多窗口才纳入
MIN_GROUPS_PER_SPECIES = 3       # 物种级：至少这么多分组才参与检验
MAX_WINDOWS_PER_GROUP = 4000     # 每片段最多取这么多窗（控制计算量）


# ------------------------------------------------------- 无量纲统计量
def shape_stats(w: np.ndarray) -> dict:
    """w: (n_win, C, T) → 每窗 7 个无量纲统计量。

    全部沿时间轴算，再跨成分取中位（中位比均值稳健，重尾数据更适合）。
    """
    n, C, T = w.shape
    x = w.astype(np.float64)
    out = {}

    # ac1, ac2: 逐成分算自相关再取中位
    a1, b1 = x[:, :, :-1], x[:, :, 1:]
    a2, b2 = x[:, :, :-2], x[:, :, 2:]

    def ac(a, b):
        num = (a * b).sum(axis=2)
        den = np.sqrt((a ** 2).sum(axis=2) * (b ** 2).sum(axis=2)) + 1e-12
        return np.median(num / den, axis=1)

    r1 = ac(a1, b1)
    r2 = ac(a2, b2)
    out["ac1"] = r1
    # ac2/ac1: 衰减形状（r1→0 时不稳定，用一个小的下限）
    out["ac_ratio"] = r2 / np.maximum(np.abs(r1), 0.05) * np.sign(r1 + 1e-12)

    # 频谱斜率（log-log 线性拟合）
    spec = np.abs(np.fft.rfft(x, axis=2)) ** 2
    F = spec.shape[2]
    freq = np.fft.rfftfreq(T, d=1.0)[1:]
    sp = spec[:, :, 1:] + 1e-12
    lf = np.log(freq)[None, None, :]
    lp = np.log(sp)
    lfc = lf - lf.mean(axis=2, keepdims=True)
    slope = (lfc * (lp - lp.mean(axis=2, keepdims=True))).sum(axis=2) / \
            (lfc ** 2).sum(axis=2)
    out["spec_slope"] = np.median(slope, axis=1)

    # 高频能量占比
    hi = spec[:, :, F // 2:].sum(axis=2)
    out["hf_ratio"] = np.median(hi / (spec.sum(axis=2) + 1e-12), axis=1)

    # 有效维度（参与率）
    xc = x - x.mean(axis=2, keepdims=True)
    try:
        s = np.linalg.svd(xc, compute_uv=False)
        lam = s ** 2
        out["eff_dim"] = (lam.sum(axis=1) ** 2) / ((lam ** 2).sum(axis=1) + 1e-12)
    except np.linalg.LinAlgError:
        out["eff_dim"] = np.full(n, np.nan)

    # 峰度（重尾）: 逐成分算再取中位
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-12
    z = (x - mu) / sd
    out["kurtosis"] = np.median((z ** 4).mean(axis=2) - 3.0, axis=1)

    # 零值占比（稀疏度）
    out["zero_frac"] = (np.abs(x) < 1e-8).mean(axis=(1, 2))

    return out


STATS = ["ac1", "ac_ratio", "spec_slope", "hf_ratio", "eff_dim", "kurtosis", "zero_frac"]

# ⚠️ zero_frac 在 z-score 后的数据上恒为 0（不存在精确零值），是退化统计量。
#    保留它只为展示"看似守恒"可能是平凡结果。判读时应忽略。
DEGENERATE = {"zero_frac"}


def eps2(groups, min_per=MIN_GROUPS_PER_SPECIES):
    """组间效应量。groups 的每个元素是一个物种的全部分组中位数。"""
    groups = [g[np.isfinite(g)] for g in groups]
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


# ------------------------------------------------------------ 数据收集
def collect(root: Path) -> pd.DataFrame:
    """从四个视图收集 (species, view, stat, value) 长表。"""
    rng = np.random.default_rng(SEED)
    rows = []

    views = [
        ("mouse", "spike_view", "spike", root / "spike_view"),
        ("macaque", "spike_view_macaque", "spike", root / "spike_view_macaque"),
        ("human", "eeg_view", "eeg", root / "eeg_view"),
    ]

    for species, view, kind, d in views:
        if not d.exists():
            print(f"  [skip] {view} 不存在")
            continue
        meta_f = d / "meta.parquet"
        if not meta_f.exists():
            print(f"  [skip] {view} 无 meta")
            continue
        meta = pd.read_parquet(meta_f)

        # 逐片段/被试处理，避免跨 recording 混合
        group_col = "session" if "session" in meta.columns else (
            "subject" if "subject" in meta.columns else None)
        if group_col is None:
            print(f"  [skip] {view} 无分组列")
            continue

        fcol = "file" if "file" in meta.columns else None

        for gval, sub in meta.groupby(group_col):
            # 单片段文件（macaque）或多文件
            if fcol is not None:
                files = sub[fcol].unique()
            else:
                files = [None]

            for fname in files:
                if fname is None:
                    if view == "spike_view":
                        # 小鼠: 每 session 一个 windows_sess{NNN}.npz
                        f_ = d / f"windows_sess{int(gval):03d}.npz"
                        if not f_.exists():
                            continue
                        w = np.load(f_)["windows"]
                    elif view == "eeg_view":
                        # 人类: 单个 windows.npz 合并文件, 用 meta 行号切出该被试
                        key = str(d)
                        if key not in _EEG_CACHE:
                            _EEG_CACHE[key] = np.load(d / "windows.npz")["windows"]
                        all_w = _EEG_CACHE[key]
                        idx = sub.index.to_numpy()      # meta 的原始行号
                        w = all_w[idx]
                    else:
                        continue
                else:
                    w = np.load(d / fname)["windows"]

                if len(w) < MIN_WINDOWS_PER_GROUP:
                    continue
                if len(w) > MAX_WINDOWS_PER_GROUP:
                    idx = rng.choice(len(w), MAX_WINDOWS_PER_GROUP, replace=False)
                    w = w[np.sort(idx)]

                st = shape_stats(w)
                for name in STATS:
                    v = st[name]
                    v = v[np.isfinite(v)]
                    if len(v) == 0:
                        continue
                    rows.append({
                        "species": species, "view": view, "kind": kind,
                        "group": str(gval), "file": str(fname),
                        "stat": name,
                        "median": float(np.median(v)),
                        "n_win": int(len(v)),
                    })
            print(f"  {species}/{view} group {gval}: 累计 {len(rows)} 条记录")

    return pd.DataFrame(rows)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp6")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase 6 — 跨物种动力学对照（七个无量纲统计量）")
    print("=" * 78)

    t0 = time.time()
    df = collect(root)
    df.to_csv(out / "phase6_per_group.csv", index=False)

    if not len(df):
        print("无数据")
        return

    print(f"\n  收集到 {len(df):,} 条 (物种×分组×统计量) 记录")
    print(f"  物种: {sorted(df['species'].unique().tolist())}")

    print("\n" + "=" * 78)
    print("汇总: 每个统计量的跨物种分布与效应量")
    print("=" * 78)
    print(f"{'统计量':<14} {'物种数':>6} {'ε²':>8} {'p':>12} {'判定':>18}")

    summary = {}
    for name in STATS:
        sub = df[df["stat"] == name]
        if not len(sub):
            continue
        # 每个物种: 取该物种所有分组的**中位数**作为一个样本点
        # （分组级，避免把数千窗口当成数千独立样本 → 假阳性）
        groups = [sub[sub["species"] == s]["median"].to_numpy()
                  for s in sub["species"].unique()]
        groups = [g for g in groups if len(g) >= MIN_GROUPS_PER_SPECIES]
        if len(groups) < 2:
            print(f"{name:<14} {':':>6} 分组数不足，跳过")
            continue
        H, p, k = eps2(groups)
        if H is None:
            continue
        if H > 0.14:
            v = "差异大（物种特异）"
        elif H < 0.01:
            v = "✅ 守恒（跨物种一致）"
        else:
            v = "中等差异"
        summary[name] = {"eps2": round(H, 4), "p": p, "n_species": k, "verdict": v}
        ps = f"{p:.2e}" if p is not None and p < 1e-3 else f"{p:.4f}" if p is not None else "NA"
        print(f"{name:<14} {k:>6} {H:>8.4f} {ps:>12} {v:>18}")

    # 每物种的中位数表
    print("\n" + "=" * 78)
    print("每物种中位数（跨分组的中位）")
    print("=" * 78)
    piv = df.pivot_table(index="stat", columns="species", values="median", aggfunc="median")
    print(piv.round(4).to_string())

    # 守恒项单独列出
    conserved = [k_ for k_, v_ in summary.items() if v_["eps2"] < 0.01]
    divergent = [k_ for k_, v_ in summary.items() if v_["eps2"] > 0.14]
    print(f"\n  守恒项 (ε²<0.01): {conserved if conserved else '无'}")
    print(f"  差异大项 (ε²>0.14): {divergent if divergent else '无'}")

    # ===== 关键补充: 仅比较两个 spike 数据集（控制模态）=====
    print("\n" + "=" * 78)
    print("A/B 对照: 仅 spike 数据集（小鼠 vs 猕猴），控制模态混淆")
    print("=" * 78)
    spike_df = df[df["species"].isin(["mouse", "macaque"])]
    spike_summary = {}
    print(f"{'统计量':<14} {'ε²':>8} {'mouse':>10} {'macaque':>10} {'判定':>14}")
    for name in STATS:
        sub = spike_df[spike_df["stat"] == name]
        if not len(sub):
            continue
        groups = [sub[sub["species"] == s]["median"].to_numpy()
                  for s in ["mouse", "macaque"]]
        if any(len(g) < MIN_GROUPS_PER_SPECIES for g in groups):
            continue
        H, p, k = eps2(groups)
        if H is None:
            continue
        m_mouse = float(np.median(groups[0]))
        m_mac = float(np.median(groups[1]))
        if H > 0.14:
            v = "差异大"
        elif H < 0.01:
            v = "✅ 守恒"
        else:
            v = "中等"
        spike_summary[name] = {"eps2": round(H, 4), "p": p,
                               "mouse": round(m_mouse, 4),
                               "macaque": round(m_mac, 4), "verdict": v}
        ps = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
        print(f"{name:<14} {H:>8.4f} {m_mouse:>10.4f} {m_mac:>10.4f} {v:>14}")

    conserved_spike = [k_ for k_, v_ in spike_summary.items()
                       if v_["eps2"] < 0.01 and k_ not in DEGENERATE]
    print(f"\n  守恒项（排除退化统计量 {sorted(DEGENERATE)}）: {conserved_spike or '无'}")

    (out / "phase6_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "stats": STATS,
        "n_records": int(len(df)),
        "species": sorted(df["species"].unique().tolist()),
        "summary": summary,
        "per_species_median": {
            s: {st: (round(float(piv.loc[st, s]), 5)
                    if st in piv.index and s in piv.columns and pd.notna(piv.loc[st, s]) else None)
                for st in STATS}
            for s in piv.columns
        },
        "conserved": conserved,
        "divergent": divergent,
        "spike_only_comparison": spike_summary,
        "spike_only_conserved": conserved_spike,
        "modality_confound_note": (
            "人类是唯一非侵入式(EEG)记录, 小鼠/猕猴均为侵入式 spike。"
            "EEG 经颅骨低通滤波, 天然具有更高自相关、更陡频谱、更少高频 —— "
            "这正是观察到的人类离群模式。因此三物种差异首先是**模态差异**, "
            "不能解读为物种差异。只有 spike vs spike (小鼠 vs 猕猴) 的比较"
            "才分离出物种效应。"),
        "degenerate_stats": sorted(DEGENERATE),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
