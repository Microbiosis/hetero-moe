"""Phase 3: 域标签到底在哪些统计量上有差异?

## 为什么要做这个

前三个实验给出了一个清晰但令人困惑的图景:
  Phase 0  域可被解码 (real 0.745 vs shuf 0.317)     → 域标签携带信息
  Phase 1  域作为输入特征无用 (域 ≈ 随机)              → 但对预测无用
  exp2     域调制动力学有效, 但前提是"域真有不同的时间常数"
  exp2b    真实数据上域调制无效 (域 vs 随机 = -0.04%)   → 域不携带时间常数信息

矛盾在于: 域标签**明明可解码**, 为什么对任何下游任务都无用?

可能的解释: 可解码的是**空间信息**(哪个电极群活跃), 而下游任务需要的是
**动力学信息**(接下来怎么变)。二者正交。

本实验直接测量: 域之间在哪些量上真的有差异?

## 测量什么

对每个窗口 (64 成分 × 8 bins) 计算七类统计量:

| 统计量 | 含义 | 与"时间常数"的关系 |
|---|---|---|
| `mean_abs`      | 平均活动幅度 | 无关 (空间/强度) |
| `std`           | 时间波动 | 弱 |
| `ac1`           | 窗内 lag-1 自相关 | **强** (时间常数代理) |
| `spectral_slope`| 频谱斜率 (1/f 特性) | **强** |
| `hf_ratio`      | 高频能量占比 | 中 (快慢) |
| `sparsity`      | 近零成分占比 | 无关 (编码稀疏度) |
| `eff_dim`       | 有效维度 (参与率) | 无关 (表征复杂度) |

## 统计检验

对每个 session:
  - 按域分组, 对每个统计量做 Kruskal-Wallis H 检验 (非参数, 不假设正态)
  - 计算效应量 ε² = (H − k + 1) / (n − k)   ← 类似 η², 0~1
  - 记录 p 值与 ε²

聚合: 每个统计量在所有 session 上的中位 ε² 和"显著 session 占比"。

**对照**: 打乱域标签后重做。若 ε² 显著下降 → 该统计量的域差异是真实的;
若 ε² 不变 → 是样本量足够大导致的假阳性。

## 判据

  某统计量 ε² > 0.06 (中等效应) 且 显著 session > 80% 且 打乱后 ε² 下降 > 50%
  → 这就是"域真正携带的信息", 后续建模应该条件在这上面。

若全部统计量的 ε² 都很小 → 域标签只有"可解码性"而没有"结构性差异",
即它编码的是电极拓扑本身, 不是任何动力学属性。
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

L1 = ["sensory", "motor", "memory", "association", "subcortical", "other"]
SEED = 0
MIN_PER_DOMAIN = 30        # 每域至少这么多窗口才参与检验
N_SHUFFLE = 20             # 打乱重复次数（取 ε² 均值）


# ------------------------------------------------------- 窗口统计量
def window_stats(w: np.ndarray) -> dict:
    """w: (N, 64, T) → 每窗口 7 个标量统计量。

    全部沿时间轴 (axis=2) 计算, 再跨成分平均。
    """
    N, C, T = w.shape
    x = w.astype(np.float32)

    # 1) 平均幅度
    mean_abs = np.abs(x).mean(axis=(1, 2))

    # 2) 时间标准差
    std = x.std(axis=2).mean(axis=1)

    # 3) 窗内 lag-1 自相关（逐成分算再平均）
    a, b = x[:, :, :-1], x[:, :, 1:]
    ac1 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        aa, bb = a[i], b[i]
        num = (aa * bb).sum(axis=1)
        den = np.sqrt((aa ** 2).sum(axis=1) * (bb ** 2).sum(axis=1)) + 1e-8
        ac1[i] = float(np.mean(num / den))

    # 4) 频谱斜率: 对每个成分做 rfft, 拟合 log power vs log freq 的斜率
    spec = np.abs(np.fft.rfft(x, axis=2)) ** 2          # (N, C, F)
    freqs = np.fft.rfftfreq(T, d=1.0)[1:]                # 去掉 DC
    sp = spec[:, :, 1:] + 1e-12
    lf = np.log(freqs)[None, None, :]
    lp = np.log(sp)
    # 最小二乘斜率 (逐 N,C)
    lf_c = lf - lf.mean(axis=2, keepdims=True)
    slope = (lf_c * (lp - lp.mean(axis=2, keepdims=True))).sum(axis=2) / \
            (lf_c ** 2).sum(axis=2)
    spectral_slope = slope.mean(axis=1)

    # 5) 高频能量占比（上半频谱 / 总能量）
    F = spec.shape[2]
    hi = spec[:, :, F // 2:].sum(axis=2)
    tot = spec.sum(axis=2) + 1e-12
    hf_ratio = (hi / tot).mean(axis=1)

    # 6) 稀疏度: 幅度低于该窗 20% 分位的成分占比
    thr = np.percentile(np.abs(x), 20, axis=(1, 2), keepdims=True)
    sparsity = (np.abs(x) < thr).mean(axis=(1, 2))

    # 7) 有效维度: 参与率 = (Σλ)² / Σλ²，λ 为协方差特征值
    #    用 SVD 的奇异值平方代替特征值
    xc = x - x.mean(axis=2, keepdims=True)
    try:
        s = np.linalg.svd(xc, compute_uv=False)        # (N, min(C,T))
        lam = s ** 2
        eff_dim = (lam.sum(axis=1) ** 2) / ((lam ** 2).sum(axis=1) + 1e-12)
    except np.linalg.LinAlgError:
        eff_dim = np.full(N, np.nan)

    return {
        "mean_abs": mean_abs,
        "std": std,
        "ac1": ac1,
        "spectral_slope": spectral_slope,
        "hf_ratio": hf_ratio,
        "sparsity": sparsity,
        "eff_dim": eff_dim,
    }


def epsilon_sq(groups):
    """Kruskal-Wallis 的效应量 ε² = (H − k + 1) / (n − k)。"""
    groups = [g for g in groups if len(g) >= MIN_PER_DOMAIN]
    k = len(groups)
    if k < 2:
        return None, None
    n = sum(len(g) for g in groups)
    if n <= k:
        return None, None
    try:
        H, p = kruskal(*groups)
    except ValueError:
        return None, None
    eps2 = (H - k + 1) / (n - k)
    return float(max(0.0, eps2)), float(p)


def measure_session(w, y, stat_names, rng):
    """返回 {stat: (eps2_real, p_real, eps2_shuf_mean)}。"""
    stats = window_stats(w)
    present = [d for d in np.unique(y) if (y == d).sum() >= MIN_PER_DOMAIN]
    if len(present) < 2:
        return None

    out = {}
    for name in stat_names:
        v = stats[name]
        if not np.isfinite(v).all():
            out[name] = (None, None, None)
            continue
        real_groups = [v[y == d] for d in present]
        e2, p = epsilon_sq(real_groups)

        # 对照: 打乱域标签
        e2s = []
        for _ in range(N_SHUFFLE):
            ys = y.copy()
            rng.shuffle(ys)
            gs = [v[ys == d] for d in present]
            e2s_tmp, _ = epsilon_sq(gs)
            if e2s_tmp is not None:
                e2s.append(e2s_tmp)
        out[name] = (e2, p, float(np.mean(e2s)) if e2s else None)
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp3")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase 3 — 域标签在哪些统计量上真的有差异?")
    print("=" * 78)

    sp = root / "spike_view"
    meta = pd.read_parquet(sp / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}
    stat_names = ["mean_abs", "std", "ac1", "spectral_slope",
                  "hf_ratio", "sparsity", "eff_dim"]

    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()

    for sess in sorted(meta["session"].unique()):
        f = sp / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        m = meta[meta["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(w) < 200:
            continue
        y = np.array([dom2i[d] for d in m["domain_l1"]], dtype=np.int64)

        res = measure_session(w, y, stat_names, rng)
        if res is None:
            continue
        for name, (e2, p, e2s) in res.items():
            if e2 is None:
                continue
            rows.append({"session": int(sess), "stat": name,
                         "eps2": round(e2, 5),
                         "eps2_shuf": round(e2s, 5) if e2s is not None else None,
                         "p": p,
                         "significant": bool(p < 0.001) if p is not None else None,
                         "n_domains": int(m["domain_l1"].nunique())})
        print(f"  sess {sess:02d}: " + " | ".join(
            f"{n}={res[n][0]:.3f}" if res[n][0] is not None else f"{n}=NA"
            for n in stat_names))

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase3_fingerprint.csv", index=False)

    print("\n" + "=" * 78)
    print("汇总: 每个统计量的跨 session 中位效应量")
    print("=" * 78)
    print(f"{'统计量':<16} {'ε²中位':>8} {'打乱后':>8} {'降幅':>8} "
          f"{'显著session':>12} {'判定':>10}")

    summary = {}
    for name in stat_names:
        sub = df[df["stat"] == name]
        if not len(sub):
            continue
        e2 = sub["eps2"].median()
        e2s = sub["eps2_shuf"].median() if sub["eps2_shuf"].notna().any() else float("nan")
        drop = (e2 - e2s) / e2 * 100 if (e2 and math.isfinite(e2s) and e2 > 0) else float("nan")
        sig = (sub["significant"] == True).mean() * 100
        if e2 > 0.06 and sig > 80 and drop > 50:
            verdict = "✅ 真实差异"
        elif e2 > 0.06 and sig > 80:
            verdict = "⚠️ 差异但打乱未降"
        else:
            verdict = "—"
        summary[name] = {"median_eps2": round(float(e2), 4),
                         "median_eps2_shuf": round(float(e2s), 4) if math.isfinite(e2s) else None,
                         "drop_pct": round(float(drop), 1) if math.isfinite(drop) else None,
                         "pct_sessions_significant": round(float(sig), 1),
                         "verdict": verdict}
        print(f"{name:<16} {e2:>8.4f} {e2s:>8.4f} {drop:>7.1f}% "
              f"{sig:>11.0f}% {verdict:>10}")

    best = max(summary.items(), key=lambda kv: kv[1]["median_eps2"]) if summary else None
    print(f"\n  最强差异统计量: {best[0]} (ε²={best[1]['median_eps2']})" if best else "")

    report = {"schema_version": "1.0", "n_sessions": int(df["session"].nunique()),
              "n_shuffle": N_SHUFFLE, "summary": summary,
              "strongest": best[0] if best else None,
              "elapsed_s": round(time.time() - t0, 1)}
    (out / "phase3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n耗时 {report['elapsed_s']}s")


if __name__ == "__main__":
    main()
