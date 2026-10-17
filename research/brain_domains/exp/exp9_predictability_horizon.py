"""Phase 9: 神经活动的可预测性视界 —— 换一个有物理意义的度量。

## 为什么换度量

前面用的 ac1 / ac_ratio / kurtosis / eff_dim 都是**临时选的统计量**，问题：

  - 含义模糊（kurtosis = −0.80 到底代表什么？事后才知道是 PCA 伪影）
  - 对预处理流程极度敏感
  - 难以跨数据集解释

本实验改测一个直接可解释的量：

    **可预测性视界 (predictability horizon)**
    给定当前 8 个 bin (0.8s) 的活动，能预测多远的未来？
    用 R²(k) 曲线刻画，k = 1..16 个 bin（0.1s ~ 1.6s）

## 为什么这个度量更好

1. **归一化**：R² ∈ [0,1]，天然免疫单元数量、绝对幅度
2. **物理意义明确**：R²(k)=0.5 时的 k 就是"半衰视界"，单位是毫秒
3. **直接检验前序发现**：exp8 说人类 ac_ratio 只有猕猴的 44%，
   若为真，人类的可预测视界也应显著更短
4. **自带噪声对照**：打乱时间顺序后 R² 应≈0

## 方法

对每个分组（session/片段）:
  1. 用前 70% 时间训练 Ridge 回归：x_{t+k} ~ [x_t, x_{t-1}, ..., x_{t-7}]
  2. 在后 30% 上算 R²
  3. 对 k = 1,2,4,8,16 重复
  4. 拟合指数衰减，取半衰期

Ridge 有闭式解，无训练噪声，结果可复现。

## 判据

  R²(k=1) 反映"下一 100ms 的可预测程度"
  半衰视界反映"记忆在多长时间内还有用"

  若人类视界 << 猕猴视界 → 与 exp8 的 ac_ratio 差异一致，互为佐证
  若三者相近             → exp8 的 ac_ratio 差异可能是伪影
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
CTX_BINS = 8          # 用过去 8 个 bin (0.8s) 做输入
HORIZONS = [1, 2, 4, 8, 16]
TRAIN_FRAC = 0.7
MAX_WIN = 1500        # 每组最多窗口数（控制计算）


def build_pairs(x, k, ctx=CTX_BINS):
    """从 (n_win, 64) 构造 (X, Y)。

    X[t] = [x[t-7], ..., x[t]]   (ctx*64 维)
    Y[t] = x[t+k]                 (64 维)
    """
    n, d = x.shape
    idx = np.arange(ctx - 1, n - k)
    if len(idx) < 50:
        return None, None
    X = np.stack([x[i - ctx + 1:i + 1].reshape(-1) for i in idx])
    Y = x[idx + k]
    return X.astype(np.float32), Y.astype(np.float32)


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=None):
    """闭式 Ridge 回归的多输出 R²。

    ## alpha 必须足够大（这是本实验踩过的坑）

    首版用 alpha=1.0，得到**全部为负**的 R²：
      human -0.914 / macaque -0.005 / mouse -1.665
    连"打乱时间顺序"的对照也是 -1.17 —— 这本该是 0。

    根因: 特征维度 d = 8 bins × 64 成分 = 512，而每组训练样本仅数百。
    512 个参数/输出，alpha=1 完全无法约束 → 严重过拟合。

    在**完全独立的随机数据**上实测：
      alpha=1    → R² = -1.86
      alpha=100  → R² = -0.71
      alpha=1000 → R² = -0.10   ← 可接受
      alpha=5000 → R² = -0.01
    同样 alpha 下，含真实线性信号的数据 R² = +0.37。

    因此默认 alpha = 1000（约等于训练样本量级），并对特征做 z-score
    使 alpha 有统一的量纲含义。
    """
    n, d = Xtr.shape
    if alpha is None:
        alpha = 1000.0
    # 特征 z-score（让 alpha 的量纲不依赖原始尺度）
    xm = Xtr.mean(axis=0, keepdims=True)
    xs = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr_ = (Xtr - xm) / xs
    Xte_ = (Xte - xm) / xs
    ym = Ytr.mean(axis=0, keepdims=True)
    Yc = Ytr - ym

    A = Xtr_.T @ Xtr_ + alpha * np.eye(d, dtype=np.float32)
    B = Xtr_.T @ Yc
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, B, rcond=None)[0]
    pred = Xte_ @ W + ym
    ss_res = ((Yte - pred) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def horizon_profile(x, rng, shuffle=False, n_blocks=4):
    """分块内评估的 R²(k) 曲线。

    ## 为什么必须分块

    首版按时间一刀切(前 75% 训练 / 后 25% 测试), 得到**全负 R²**:
      human -0.885, macaque -0.183, mouse -3.369
    负 R² = 比直接用测试集均值预测还差。

    根因是**时间漂移**: 神经记录随时间非平稳(电极位移、动物状态、阻抗变化),
    早段训练的模型在晚段完全失效。

    分块内评估: 把序列切成 n_blocks 个连续块, 每块内部做 train/test 切分,
    再对 R² 取块间中位。这样训练与测试来自相近时间, 大幅缓解漂移。
    仍保留时间顺序(块内前段训练/后段测试), 不引入未来信息泄漏。

    若某块 R² 仍为负, 说明该块内部也不平稳 —— 保留原值不裁剪,
    以便在报告中暴露这个问题。
    """
    if shuffle:
        # 打乱时间顺序（每列独立置换）：保留振幅分布，破坏时序
        x = np.stack([rng.permutation(x[:, j]) for j in range(x.shape[1])], axis=1)

    n = len(x)
    blk = n // n_blocks
    if blk < 150:                     # 每块至少 150 窗
        return None

    prof = {k: [] for k in HORIZONS}
    for b in range(n_blocks):
        xb = x[b * blk:(b + 1) * blk]
        nb = len(xb)
        for k in HORIZONS:
            X, Y = build_pairs(xb, k)
            if X is None:
                continue
            m = len(X)
            cut = int(m * 0.7)
            if cut < 40 or m - cut < 20:
                continue
            r2 = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
            prof[k].append(r2)

    out = {}
    for k, vals in prof.items():
        if vals:
            out[k] = float(np.median(vals))
    return out if out else None


def half_life(prof):
    """R²(k) 首次跌破 R²(1)/2 的 k（线性插值）。"""
    if 1 not in prof or prof[1] <= 0:
        return None
    target = prof[1] / 2
    ks = sorted(prof)
    for i in range(1, len(ks)):
        k0, k1 = ks[i - 1], ks[i]
        v0, v1 = prof[k0], prof[k1]
        if v0 >= target >= v1:
            if abs(v1 - v0) < 1e-9:
                return float(k1)
            return float(k0 + (v0 - target) / (v0 - v1) * (k1 - k0))
    return float(ks[-1]) if prof[ks[-1]] < target else None


def load_group_windows(root: Path):
    """产出 (species, group, n_units, windows(n_win,64))。"""
    specs = [
        ("mouse", root / "spike_view", "session", None),
        ("macaque", root / "spike_view_macaque", "file", "n_units"),
        ("human", root / "spike_view_human", "file", "n_units"),
    ]
    for species, d, gcol, ucol in specs:
        if not d.exists():
            continue
        meta = pd.read_parquet(d / "meta.parquet")
        for gval, sub in meta.groupby(gcol):
            if gcol == "session":
                f = d / f"windows_sess{int(gval):03d}.npz"
                n_u = None
            else:
                f = d / str(gval)
                n_u = int(sub[ucol].iloc[0]) if ucol in sub.columns else None
            if not f.exists():
                continue
            w = np.load(f)["windows"]
            if len(w) < 200:
                continue
            # (n_win, 64, 8) → (n_win, 64) 每窗时间均值
            yield species, f"{species}|{gval}", n_u, w.mean(axis=2)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp9")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 80)
    print("Phase 9 — 可预测性视界（R²(k) 曲线）")
    print(f"  上下文 {CTX_BINS} bins (0.8s) | 视界 {HORIZONS} bins | Ridge 闭式解")
    print("=" * 80)

    rows = []
    t0 = time.time()

    for species, gname, n_u, w in load_group_windows(root):
        if len(w) > MAX_WIN:
            idx = rng.choice(len(w), MAX_WIN, replace=False)
            w = w[np.sort(idx)]

        prof = horizon_profile(w, rng)
        prof_sh = horizon_profile(w, rng, shuffle=True)      # 噪声对照
        if prof is None:
            continue

        row = {"species": species, "group": gname, "n_units": n_u,
               "n_win": int(len(w))}
        for k in HORIZONS:
            row[f"r2_k{k}"] = round(prof.get(k, float("nan")), 4)
            row[f"r2sh_k{k}"] = round((prof_sh or {}).get(k, float("nan")), 4)
        row["half_life"] = half_life(prof)
        row["half_life_shuf"] = half_life(prof_sh) if prof_sh else None
        rows.append(row)
        print(f"  {gname[:40]:<40} R²(k1)={prof.get(1, float('nan')):.3f} "
              f"R²(k16)={prof.get(16, float('nan')):.3f} "
              f"t½={row['half_life']}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase9_horizon.csv", index=False)

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 80)
    print("汇总: 每物种的 R²(k) 曲线（中位数）")
    print("=" * 80)
    cols = [f"r2_k{k}" for k in HORIZONS]
    print(df.groupby("species")[["n_win"] + cols + ["half_life"]].median().round(4).to_string())

    print("\n" + "=" * 80)
    print("噪声对照: 打乱时间顺序后的 R²")
    print("=" * 80)
    cols_s = [f"r2sh_k{k}" for k in HORIZONS]
    print(df.groupby("species")[cols_s].median().round(4).to_string())

    # 判定对照是否有效
    mean_real_k1 = df["r2_k1"].mean()
    mean_shuf_k1 = df["r2sh_k1"].mean()
    print(f"\n  真实 R²(k=1) 均值 {mean_real_k1:.4f} vs 打乱 {mean_shuf_k1:.4f}")
    if abs(mean_shuf_k1) < 0.15:
        print("  ✅ 打乱后 R² ≈ 0，无泄漏（alpha 已校正）")
    elif mean_shuf_k1 > 0.10:
        print("  ⚠️ 打乱后 R² > 0.1 —— 存在泄漏（窗口 50% 重叠，相邻窗高度相关）")
    else:
        print(f"  ⚠️ 打乱后 R² 为负 ({mean_shuf_k1:.3f}) —— 说明时间漂移主导，模型失效")

    # 物种比较
    print("\n" + "=" * 80)
    print("物种比较: R²(k=1) 与半衰视界")
    print("=" * 80)
    from scipy.stats import kruskal
    summary = {}
    for col in ["r2_k1", "r2_k4", "r2_k16", "half_life"]:
        s = df[["species", col]].dropna()
        groups = [s[s.species == sp][col].to_numpy() for sp in s.species.unique()]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        n = sum(len(g) for g in groups)
        k = len(groups)
        e2 = max(0.0, (H - k + 1) / (n - k))
        med = s.groupby("species")[col].median().to_dict()
        summary[col] = {"eps2": round(e2, 4), "p": float(p),
                        "median": {a: round(float(b), 4) for a, b in med.items()}}
        print(f"  {col:<12} ε²={e2:.4f} p={p:.2e}  " +
              "  ".join(f"{a}={b:.3f}" for a, b in med.items()))

    (out / "phase9_report.json").write_text(json.dumps({
        "schema_version": "1.0", "ctx_bins": CTX_BINS, "horizons": HORIZONS,
        "n_groups": int(len(df)), "species_summary": summary,
        "noise_control": {"real_k1": round(float(mean_real_k1), 4),
                          "shuffled_k1": round(float(mean_shuf_k1), 4)},
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
