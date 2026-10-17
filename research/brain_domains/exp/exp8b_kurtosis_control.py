"""Phase 8b: kurtosis ≈ -0.80 是生物性质，还是 PCA 流程的产物?

## 为什么必须查

exp8 发现 kurtosis 在三物种上高度一致 (-0.797 / -0.810 / -0.803)。
但这条"守恒"很可疑:

  - 三物种的 ac1 / ac_ratio / eff_dim 差异巨大 (ε² 0.57-0.83)
  - 唯独 kurtosis 几乎相同 —— 太整齐了

可能原因: **PCA 投影 + z-score 的必然结果**。

PCA 成分是多个变量的加权和。中心极限定理倾向于把分布拉向高斯
(峰度 0)。若数据本身有效维度低，投影后的成分可能呈"平顶"分布,
峰度稳定在某个负值附近, 与原始生物学无关。

## 检验方法: 三组对照

  A. 真实数据 → 现有流程 → kurtosis
  B. **高斯噪声** → 同一流程 → kurtosis   ← 关键
  C. **打乱每个 unit 的时间顺序** (保留振幅分布, 破坏时间结构) → kurtosis

判读:
  若 B ≈ A (-0.80)  → kurtosis 是流程产物, 不是生物性质
  若 C ≈ A           → kurtosis 只依赖振幅分布, 不依赖时间结构
  若 A 显著不同于 B、C → kurtosis 是真实的生物信号

同时测几个对照流水线的峰度:
  - 不做 PCA, 直接用第一个 unit 的 z-score 序列
  - 不做 z-score, 用原始活动
"""

from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
PROJ_DIM, WIN = 64, 8


def kurt_of(w):
    """逐窗峰度中位数（当前流程用的算法）。"""
    x = w.astype(np.float64)
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-12
    z = (x - mu) / sd
    return float(np.median(((z ** 4).mean(axis=2) - 3.0)))


def pipeline_from_units(z, rng, shuffle=False):
    """z: (n_units, n_bins) 已 z-score。返回 (n_win, 64, 8) 窗口。"""
    if shuffle:
        z = np.stack([rng.permutation(row) for row in z])
    nb = z.shape[1]
    k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
    zc = z - z.mean(axis=1, keepdims=True)
    C = zc @ zc.T
    w_, U = np.linalg.eigh(C)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < PROJ_DIM:
        proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
    starts = list(range(0, nb - WIN + 1, 4))
    if not starts:
        return None
    return np.stack([proj[s:s + WIN].T for s in starts])


def main():
    stein = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/steinmetz")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp8b")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 76)
    print("Phase 8b — kurtosis ≈ -0.80 是生物性质还是流程产物?")
    print("=" * 76)

    results = {}

    # 取 3 个真实小鼠 session
    units = pd.read_parquet(stein / "units.parquet")
    sess_list = sorted(units["session"].unique())[:3]

    for sess in sess_list:
        f = stein / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)
        nb = sp.shape[1] // 2
        x = sp[:, :nb * 2].reshape(sp.shape[0], nb, 2).sum(axis=2)
        mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
        z = (x - mu) / sd
        n_units = z.shape[0]

        print(f"\n--- session {sess} ({n_units} units, {nb} bins) ---")

        # A. 真实
        wA = pipeline_from_units(z, rng)
        kA = kurt_of(wA) if wA is not None else None
        print(f"  A 真实数据           kurtosis = {kA:.4f}")

        # B. 高斯噪声（同形状）
        zg = rng.standard_normal(z.shape).astype(np.float32)
        wB = pipeline_from_units(zg, rng)
        kB = kurt_of(wB) if wB is not None else None
        print(f"  B 高斯噪声           kurtosis = {kB:.4f}  ← 若 ≈A 则是流程产物")

        # C. 打乱时间顺序（保振幅分布，破时间结构）
        wC = pipeline_from_units(z, rng, shuffle=True)
        kC = kurt_of(wC) if wC is not None else None
        print(f"  C 打乱时间顺序       kurtosis = {kC:.4f}  ← 若 ≈A 则只依赖振幅分布")

        # D. 不做 PCA，直接第一个 unit
        kD = float(np.median(((z[0] - z[0].mean()) / (z[0].std() + 1e-12)) ** 4 - 3.0))
        print(f"  D 单 unit 原始序列   kurtosis = {kD:.4f}  ← 无 PCA")

        # E. 全 unit 堆叠
        allz = z.reshape(-1)
        kE = float(np.mean(((allz - allz.mean()) / (allz.std() + 1e-12)) ** 4) - 3.0)
        print(f"  E 全 unit 池化       kurtosis = {kE:.4f}")

        results[sess] = {"n_units": int(n_units), "A_real": kA, "B_gaussian": kB,
                         "C_shuffled": kC, "D_single_unit": kD, "E_pooled": kE}

    # 汇总
    print("\n" + "=" * 76)
    print("汇总")
    print("=" * 76)
    keys = ["A_real", "B_gaussian", "C_shuffled", "D_single_unit", "E_pooled"]
    for k in keys:
        vals = [r[k] for r in results.values() if r[k] is not None]
        if vals:
            print(f"  {k:<16} 均值 {np.mean(vals):+.4f}  (范围 {min(vals):+.3f} ~ {max(vals):+.3f})")

    # 判定
    A = np.mean([r["A_real"] for r in results.values() if r["A_real"] is not None])
    B = np.mean([r["B_gaussian"] for r in results.values() if r["B_gaussian"] is not None])
    C = np.mean([r["C_shuffled"] for r in results.values() if r["C_shuffled"] is not None])

    print(f"\n  A(真实) - B(高斯) = {A - B:+.4f}")
    print(f"  A(真实) - C(打乱) = {A - C:+.4f}")

    if abs(A - B) < 0.05:
        v = "❌ kurtosis 是 PCA 流程的产物，不是生物性质。三物种一致无意义。"
    elif abs(A - C) < 0.05:
        v = "⚠️ kurtosis 只依赖振幅分布，不依赖时间结构 —— 仍可能是记录条件的产物"
    else:
        v = "✅ kurtosis 是真实的生物信号（既不同于噪声，也依赖时间结构）"
    print(f"\n  判定: {v}")

    (out / "phase8b_report.json").write_text(json.dumps({
        "schema_version": "1.0", "sessions": {str(k): v for k, v in results.items()},
        "means": {k: (round(float(np.mean([r[k] for r in results.values() if r[k] is not None])), 4)
                      if any(r[k] is not None for r in results.values()) else None)
                  for k in keys},
        "verdict": v,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
