"""Phase 23: 纠缠面积律检验 + 密度矩阵层次构造。

## 数学框架（借用自量子多体）

把神经群体活动矩阵 X（d 个成分 × T 个时间点）视为一个**二分纯态**的表示：

    系统 A = 时间前半 [0, t)
    系统 B = 时间后半 [t, T)

高斯态近似下，纠缠熵与互信息可由协方差矩阵直接算出（忽略常数项）：

    S(Σ) = ½ log det Σ
    I(A:B) = S_A + S_B − S_AB

其中 Σ_A = cov(X[:, :t])，Σ_B = cov(X[:, t:])，Σ_AB = cov(X)。

## 为什么这是关键检验

**面积律**（area law）：I(A:B) 随切点 t 增长后**饱和**，只取决于边界
    → 低纠缠 → 存在高效的张量网络（MPS）表示 → 层次构造可行
**体积律**（volume law）：I(A:B) ∝ min(t, T−t) 线性增长
    → 高纠缠 → 无法用小键维数压缩 → 层次构造只能靠外部标签

这决定了"数据驱动的层次"是否在信息论上成立。

## 附带：密度矩阵层次

由协方差矩阵构造密度矩阵 ρ = Σ / tr(Σ)，其特征值给出"本征模式"。
按特征值分组即得到一个**有信息论意义的层次**（每层是一个熵预算）。

## 三个量

1. `I(t)`            时间二分互信息，检验面积律
2. `S_spec`          密度矩阵谱熵，衡量有效维度
3. `slope`           I(t) 在对数坐标下的斜率（面积律应 → 0）
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_WIN = 600
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_PER_SPECIES = 10
JITTER = 1e-6


def logdet(S):
    """数值稳定的 log det（加 jitter，用 slogdet）。"""
    d = S.shape[0]
    S = S + JITTER * np.trace(S) / d * np.eye(d)
    sign, val = np.linalg.slogdet(S)
    return float(val) if sign > 0 else np.nan


def gauss_entropy(S):
    """高斯态熵（忽略常数）：½ log det Σ。"""
    return 0.5 * logdet(S)


def mutual_info(X, frac):
    """时间二分互信息 I(A:B) = S_A + S_B − S_AB。"""
    d, T = X.shape
    t = max(int(T * frac), d + 2)
    if T - t < d + 2:
        return np.nan
    SA = gauss_entropy(np.cov(X[:, :t]))
    SB = gauss_entropy(np.cov(X[:, t:]))
    SAB = gauss_entropy(np.cov(X))
    return float(SA + SB - SAB)


def density_matrix_stats(X):
    """密度矩阵 ρ = Σ/tr(Σ) 的谱熵与有效维度。"""
    C = np.cov(X)
    tr = np.trace(C)
    if tr <= 1e-12:
        return None
    rho = C / tr
    ev = np.linalg.eigvalsh(rho)
    ev = np.clip(ev, 1e-12, None)
    ev = ev / ev.sum()
    S = float(-(ev * np.log(ev)).sum())
    eff = float(np.exp(S))                    # 有效维度 = e^S
    purity = float((ev ** 2).sum())           # 纯度
    return {"S_spec": S, "eff_dim": eff, "purity": purity,
            "top1": float(ev.max())}


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
        X = X[:, ::2]                              # 零重叠
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
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp23")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 23 — 纠缠面积律检验（量子多体框架）")
    print("=" * 80)
    print(f"  n_win={N_WIN} | 切点比例 {FRACS}")
    print("  I(A:B) = ½[logdet Σ_A + logdet Σ_B − logdet Σ_AB]")

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
            dm = density_matrix_stats(X)
            for frac in FRACS:
                mi = mutual_info(X, frac)
                if not np.isfinite(mi):
                    continue
                rows.append({"species": species, "group": key,
                             "frac": frac, "t_bins": int(X.shape[1] * frac),
                             "mi": round(float(mi), 4),
                             **({k: round(v, 4) for k, v in dm.items()} if dm else {})})
            print(f"    {str(key)[:24]:<24} I(0.5)={mutual_info(X,0.5):.3f} "
                  f"eff_dim={dm['eff_dim'] if dm else float('nan'):.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase23.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    # ============ 面积律检验
    print("\n" + "=" * 80)
    print("纠缠面积律：I(t) 是否饱和？")
    print("=" * 80)
    piv = df.pivot_table(index="frac", columns="species", values="mi", aggfunc="median")
    print(piv.round(3).to_string())

    print("\n  判据：面积律 → I 在 mid-frac 后趋于平台；体积律 → 线性上升")
    summary = {}
    for sp in piv.columns:
        y = piv[sp].dropna()
        if len(y) < 5:
            continue
        x = y.index.to_numpy()
        # 平台检测：后半段相对前半段的增幅
        half = len(y) // 2
        rise = float(y.iloc[-1] - y.iloc[half])
        total = float(y.iloc[-1] - y.iloc[0])
        plat = 1.0 - (rise / total) if abs(total) > 1e-9 else np.nan
        # 线性拟合斜率（对数坐标）
        slope = float(np.polyfit(np.log(x), y.to_numpy(), 1)[0])
        summary[sp] = {"I_mid": round(float(y.iloc[len(y)//2]), 3),
                       "I_end": round(float(y.iloc[-1]), 3),
                       "plateau_score": round(float(plat), 3),
                       "log_slope": round(slope, 3)}
        print(f"  {sp:8s} I(mid)={y.iloc[len(y)//2]:.3f} I(end)={y.iloc[-1]:.3f} "
              f"平台度={plat:.2f} 对数斜率={slope:+.3f}")

    # ============ 密度矩阵谱
    print("\n" + "=" * 80)
    print("密度矩阵谱（ρ = Σ/tr Σ）")
    print("=" * 80)
    print(f"{'species':<9} {'谱熵 S':>9} {'有效维度':>9} {'纯度':>8} {'top1':>7}")
    dm = df.groupby("species")[["S_spec", "eff_dim", "purity", "top1"]].median()
    for sp, r in dm.iterrows():
        print(f"{sp:<9} {r['S_spec']:>9.3f} {r['eff_dim']:>9.2f} "
              f"{r['purity']:>8.4f} {r['top1']:>7.4f}")

    print(f"\n  参考：64 维最大熵 S = ln64 = {np.log(64):.3f}")

    # ============ 判定
    print("\n" + "=" * 80)
    print("判定")
    print("=" * 80)
    plateaus = [v["plateau_score"] for v in summary.values() if np.isfinite(v["plateau_score"])]
    slopes = [v["log_slope"] for v in summary.values() if np.isfinite(v["log_slope"])]
    med_plat = float(np.median(plateaus)) if plateaus else np.nan
    med_slope = float(np.median(slopes)) if slopes else np.nan

    print(f"  平台度中位 {med_plat:.2f}（1.0 = 完全饱和）")
    print(f"  对数斜率中位 {med_slope:+.3f}")

    if med_plat > 0.7 and med_slope < 0.5:
        verdict = ("✅ **面积律成立** —— 神经活动是低纠缠的，"
                   "存在高效张量网络表示，数据驱动的层次构造在信息论上可行")
        feasible = True
    elif med_plat < 0.3:
        verdict = ("❌ **体积律** —— 高纠缠，无法用小键维数压缩。"
                   "层次构造只能依赖外部标签（而标签已被证明冗余）")
        feasible = False
    else:
        verdict = (f"⚠️ 中间情形（平台度 {med_plat:.2f}）—— "
                   "部分可压缩，层次构造需中等键维数")
        feasible = None
    print(f"  {verdict}")

    (out / "phase23_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN, "fracs": FRACS,
        "framework": "gaussian entanglement entropy; I(A:B)=1/2[logdet SA + logdet SB - logdet SAB]",
        "n_records": int(len(df)),
        "mi_by_frac": {sp: {str(f): (None if pd.isna(v) else round(float(v), 4))
                            for f, v in piv[sp].items()} for sp in piv.columns},
        "area_law": summary,
        "density_matrix": {sp: {c: round(float(v), 4) for c, v in r.items()}
                           for sp, r in dm.iterrows()},
        "median_plateau": round(med_plat, 3),
        "median_log_slope": round(med_slope, 3),
        "area_law_holds": feasible,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
