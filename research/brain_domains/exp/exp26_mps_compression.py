"""任务 A: MPS / 张量列实际的压缩可行性（交叉验证"有效秩小"）。

## 要验证什么

exp25 发现原始尺度下有效秩只有 2.8–17 / 64，暗示"低纠缠、可压缩"。
但**有效秩小只是必要条件，不充分**。必须做实际压缩：

    达到 5% 重构误差，需要多大的键维数 χ？

## 方法：矩阵秩截断（= 两体 Schmidt 分解）

X = U S Vᵀ，保留前 χ 个奇异值，误差 = sqrt(Σ_{i>χ} s_i² / Σ_i s_i²)。

χ 就是 exp24/25 里算的"键维数"，误差直接由 Schmidt 谱给出。
（首版试图做多 site TT，但 site 维数乘积必须等于 T，构造不成立，已改成秩截断。）

## 关键设计（吸取前 25 个实验的教训）

1. **三种归一化对照**：raw / global / percomp
   —— 因为 exp25 证明逐成分 z-score 会彻底改变谱结构
2. **匹配窗口数** n_win=600，人类只需 600 窗（不是 1200）
3. **报告拟合质量**：除了 χ，还要报告实际达到的误差曲线
4. **零模型**：对打乱时间轴的数据做同样压缩，若误差曲线相同
   → 说明压缩能力来自时间平滑性，不是跨成分结构

## 判据

  χ(5%) 远小于 64 且 raw 与 percomp 差异大  → 压缩能力依赖归一化（伪影）
  χ(5%) 小而打乱后 χ(5%) 大                 → 真实的时间结构可压缩
  χ(5%) ≈ 64                                → 不可压缩
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
N_PER_SPECIES = 8
N_SITES = 4                      # 时间轴切成 4 段
CHIS = [1, 2, 4, 8, 16, 32, 64]


def rank_truncate_error(X, chi):
    """矩阵秩截断（两体 MPS / Schmidt 分解）的重构误差。

    X = U S Vᵀ，保留前 chi 个奇异值：
        X_chi = U[:, :χ] diag(S[:χ]) Vᵀ[:χ]

    相对 Frobenius 误差 = sqrt(Σ_{i>χ} s_i² / Σ_i s_i²)

    ## 为什么用这个而不是多 site TT

    首版试图把 (64, 600) 重塑为 (64, 150,150,150,150)，
    但 150⁴ ≠ 600 —— site 维数的乘积必须等于 T，这个构造根本不成立。

    而秩截断正好是**两体 Schmidt 分解的直接对应**：
    χ 就是 exp24/25 里算的"键维数"，误差直接由 Schmidt 谱给出。
    这是最可解释、最不可能算错的压缩检验。
    """
    s = np.linalg.svd(X, compute_uv=False)
    if len(s) == 0:
        return None
    p2 = s ** 2
    tot = p2.sum()
    if tot <= 1e-12:
        return np.nan
    r = min(chi, len(s))
    tail = p2[r:].sum() / tot
    return float(np.sqrt(max(tail, 0.0)))


def normalize(X, mode):
    if mode == "raw":
        return X.copy()
    if mode == "global":
        return X / (X.std() + 1e-12)
    if mode == "percomp":
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
    # 按分组大小降序取前 N 个（不是按文件名排序取前 N —— 那会系统性地
    # 取到最小的分组，导致人类等数据源静默消失）
    sizes = meta.groupby(gcol).size().sort_values(ascending=False)
    for gval in list(sizes.index)[:N_PER_SPECIES]:
        if gcol == "session" and view == "spike_view":
            f = d / f"windows_sess{int(gval):03d}.npz"
        else:
            f = d / str(gval)
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < n_win:                 # 修正: 只要 n_win
            continue
        X = w.mean(axis=2).T.astype(np.float64)
        X = X[:, ::2]
        if X.shape[1] < n_win:
            continue
        s = (X.shape[1] - n_win) // 2
        out.append((str(gval), X[:, s:s + n_win]))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp26")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("任务 A — MPS/TT-SVD 实际压缩可行性")
    print("=" * 84)
    print(f"  n_win={N_WIN} | n_sites={N_SITES} | χ∈{CHIS}")
    print(f"  归一化: raw / global / percomp（三种对照）")

    rows = []
    t0 = time.time()

    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_matched(root, view)
        print(f"\n  {species}: {len(items)} 组")
        if not items:
            continue
        for key, X0 in items:
            # 时间打乱对照
            X_shuf = np.stack([rng.permutation(X0[i]) for i in range(X0.shape[0])])
            for mode in ["raw", "global", "percomp"]:
                for tag, X in [("real", X0), ("shuf", X_shuf)]:
                    Xn = normalize(X, mode)
                    for chi in CHIS:
                        err = rank_truncate_error(Xn, chi)
                        if err is None or not np.isfinite(err):
                            continue
                        rows.append({"species": species, "group": key,
                                     "norm": mode, "data": tag,
                                     "chi": chi, "err": round(err, 5)})
            print(f"    {str(key)[:24]:<24} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase26.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    # ============ 误差曲线
    print("\n" + "=" * 84)
    print("重构误差 vs 键维数 χ（real 数据，按归一化）")
    print("=" * 84)
    for mode in ["raw", "global", "percomp"]:
        s = df[(df.norm == mode) & (df.data == "real")]
        if not len(s):
            continue
        print(f"\n  [{mode}]")
        piv = s.pivot_table(index="chi", columns="species", values="err", aggfunc="median")
        print(piv.round(4).to_string())

    # ============ 关键量: χ(5%)
    print("\n" + "=" * 84)
    print("关键: 达到 5% / 10% 重构误差所需 χ")
    print("=" * 84)

    def chi_at(s, target):
        vals = []
        for sp in s["species"].unique():
            ss = s[s.species == sp].groupby("chi")["err"].median().sort_index()
            hit = ss[ss <= target]
            vals.append((sp, int(hit.index[0]) if len(hit) else int(ss.index.max()) + 1))
        return dict(vals)

    print(f"{'norm':<10} {'data':<6} {'5% 阈值':>22} {'10% 阈值':>22}")
    summary = {}
    for mode in ["raw", "global", "percomp"]:
        for tag in ["real", "shuf"]:
            s = df[(df.norm == mode) & (df.data == tag)]
            if not len(s):
                continue
            a = chi_at(s, 0.05)
            b = chi_at(s, 0.10)
            summary[f"{mode}|{tag}"] = {"chi5": a, "chi10": b}
            astr = ", ".join(f"{k[:4]}={v}" for k, v in a.items())
            bstr = ", ".join(f"{k[:4]}={v}" for k, v in b.items())
            print(f"{mode:<10} {tag:<6} {astr:>22} {bstr:>22}")

    # ============ 判定
    print("\n" + "=" * 84)
    print("判定")
    print("=" * 84)

    def get(mode, tag, sp):
        return summary.get(f"{mode}|{tag}", {}).get("chi5", {}).get(sp, 99)

    print("  χ(5%) 对比:")
    for sp in ["mouse", "macaque", "human"]:
        r_raw = get("raw", "real", sp)
        s_raw = get("raw", "shuf", sp)
        r_pc = get("percomp", "real", sp)
        s_pc = get("percomp", "shuf", sp)
        print(f"    {sp:<8} raw: real={r_raw:>3} shuf={s_raw:>3} | "
              f"percomp: real={r_pc:>3} shuf={s_pc:>3}")

    # 判据 1: 压缩能力是否依赖归一化
    diffs = []
    for sp in ["mouse", "macaque", "human"]:
        diffs.append(abs(get("raw", "real", sp) - get("percomp", "real", sp)))
    dep_norm = float(np.mean(diffs)) > 8

    # 判据 2: 真实 vs 打乱是否有差异
    gaps = []
    for sp in ["mouse", "macaque", "human"]:
        gaps.append(get("raw", "shuf", sp) - get("raw", "real", sp))
    mean_gap = float(np.mean(gaps))

    # 判据 3: 是否真的可压缩
    best = min(get("raw", "real", sp) for sp in ["mouse", "macaque", "human"])

    print(f"\n  归一化依赖度（χ 差均值）= {np.mean(diffs):.1f} → "
          + ("⚠️ 强依赖，压缩能力主要是归一化效应" if dep_norm else "✅ 依赖小"))
    print(f"  真实 vs 打乱的 χ 差均值 = {mean_gap:+.1f} → "
          + ("✅ 真实数据更可压缩（时间结构起作用）" if mean_gap > 3
             else "⚠️ 无差异（压缩能力不来自时间结构）"))
    print(f"  最佳 χ(5%) = {best} / 64")

    if best <= 16 and mean_gap > 3 and not dep_norm:
        v = "✅ **可压缩** — χ(5%) 小、真实优于打乱、且不依赖归一化"
    elif dep_norm:
        v = "⚠️ 压缩能力主要来自归一化选择，不能作为'低纠缠'的证据"
    elif mean_gap <= 3:
        v = "⚠️ 压缩能力不来自跨成分时间结构（打乱后同样可压缩）"
    else:
        v = f"❌ 需要 χ={best}，接近满秩，不可有效压缩"
    print(f"\n  {v}")

    (out / "phase26_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN, "n_sites": N_SITES, "chis": CHIS,
        "method": "matrix rank truncation (two-site Schmidt), err=sqrt(sum_{i>chi} s_i^2 / sum s_i^2)",
        "n_records": int(len(df)),
        "chi_at_threshold": summary,
        "normalization_dependence": round(float(np.mean(diffs)), 2),
        "real_vs_shuffle_gap": round(mean_gap, 2),
        "best_chi5": best,
        "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
