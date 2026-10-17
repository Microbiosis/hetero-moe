"""Phase 18: 最终定论版 —— 分块 CV + 零重叠 + 按区域分层。

## 修 exp17 的问题

exp17 用了单一 70/30 时间切分（train 早段 / test 晚段），这是 exp9 已经
踩过的坑：神经记录非平稳（电极漂移、动物状态变化），早段训练的模型在
晚段失效 → R² 为负。

exp17 的症状：
    mouse 100ms  R²(1) = -0.0709（负！比直接用测试集均值还差）
    macaque 100ms R²(1) = +0.3683（正）

这个差异**可能只是漂移程度的差异**，不能直接读成"物种差异"。

## 本版的三处修正

1. **分块交叉验证**：把时间轴切成 4 个连续块，每块内做 train/test 切分，
   取块间中位。这样训练与测试来自相近时间，规避漂移。
   仍保留时间顺序（块内前段训练、后段测试），不引入未来信息。

2. **按区域分层**：猕猴按 area（M1_PMd / Area2 / DMFC）报告，
   小鼠按 session 的主导域报告。这样能看出"物种差异"是否其实是"区域差异"。

3. **双分辨率 + 一致性判据**（沿用 exp17）。

## 判据（不预设结论）

  若净 R²(1) 在分块 CV 下仍普遍 ≈0
      → 100ms 分辨率下确实测不到跨窗可预测性，τ 无从谈起
  若某些区域净 R²(1) 稳定 >0.1
      → 这些区域有真实可预测性；再谈 τ 才有意义
  若分辨率比值落在 0.75-1.35
      → τ 是真实时间常数
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
warnings.filterwarnings("ignore")

SEED = 0
CTX = 8
KS = [1, 2, 4, 8]
WIN = 8
STRIDE = 8              # 零重叠
BIN_WIDTHS = [100, 200]
ALPHA = 1000.0
MIN_UNITS = 20
MAX_WIN = 500
N_BLOCKS = 4            # 分块 CV
MAX_PER_SPECIES = 24


def pca_proj(z):
    nb = z.shape[1]
    k = min(64, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    w_, U = np.linalg.eigh(zc @ zc.T)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < 64:
        proj = np.pad(proj, ((0, 0), (0, 64 - proj.shape[1])))
    return proj


def make_windows(proj, max_win=MAX_WIN):
    nb = proj.shape[0]
    starts = list(range(0, nb - WIN + 1, STRIDE))
    if len(starts) < 60:
        return None
    if len(starts) > max_win:
        s0 = (len(starts) - max_win) // 2
        starts = starts[s0:s0 + max_win]
    return np.stack([proj[s:s + WIN].mean(axis=0) for s in starts]).astype(np.float32)


def make_pairs(x, k, ctx=CTX):
    n = x.shape[0]
    idx = np.arange(ctx - 1, n - k)
    if len(idx) < 60:
        return None, None
    X = np.stack([x[i - ctx + 1:i + 1].reshape(-1) for i in idx]).astype(np.float32)
    Y = x[idx + k].astype(np.float32)
    return X, Y


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=ALPHA):
    d = Xtr.shape[1]
    xm = Xtr.mean(0, keepdims=True)
    xs = Xtr.std(0, keepdims=True) + 1e-8
    Xtr_ = (Xtr - xm) / xs
    Xte_ = (Xte - xm) / xs
    ym = Ytr.mean(0, keepdims=True)
    A = Xtr_.T @ Xtr_ + alpha * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, Xtr_.T @ (Ytr - ym))
    pred = Xte_ @ W + ym
    return float(1.0 - ((Yte - pred) ** 2).sum() /
                 max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def r2_curve_blocked(x, n_blocks=N_BLOCKS):
    """分块 CV 的 R²(k) 曲线：每块内 train/test，取块间中位。"""
    n = len(x)
    blk = n // n_blocks
    if blk < 80:
        return {}
    out = {}
    for k in KS:
        vals = []
        for b in range(n_blocks):
            xb = x[b * blk:(b + 1) * blk]
            X, Y = make_pairs(xb, k)
            if X is None:
                continue
            cut = int(len(X) * 0.7)
            if cut < 30 or len(X) - cut < 15:
                continue
            vals.append(ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:]))
        if vals:
            out[k] = float(np.median(vals))
    return out


def fit_tau(prof):
    from scipy.optimize import curve_fit
    if len(prof) < 3:
        return None
    ks = np.array(sorted(prof), float)
    r2 = np.array([prof[k] for k in sorted(prof)], float)
    if r2[0] <= 0.02:
        return None

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.02), 2.0, min(r2[-1], 0.0)]
        popt, _ = curve_fit(f, ks, r2, p0=p0,
                            bounds=([0, 0.05, -2.0], [5, 100, 1.0]), maxfev=12000)
        return float(popt[1])
    except Exception:
        return None


def to_bins(sp, bin_ms, src_ms):
    ratio = max(1, int(round(bin_ms / src_ms)))
    nb = sp.shape[1] // ratio
    return sp[:, :nb * ratio].reshape(sp.shape[0], nb, ratio).sum(axis=2)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp18")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 86)
    print("Phase 18 — 最终定论版（分块 CV + 零重叠 + 按区域分层）")
    print(f"  WIN={WIN} STRIDE={STRIDE}（零重叠）| {N_BLOCKS} 块 CV | bin {BIN_WIDTHS} | k∈{KS}")
    print("=" * 86)

    # ---------- 小鼠：按 session 的主导域分层
    tasks = []
    stein = root / "steinmetz"
    up = stein / "units.parquet"
    if up.exists():
        units = pd.read_parquet(up)
        for sess in sorted(units["session"].unique()):
            f = stein / f"spikes_session_{sess:03d}.npz"
            if not f.exists():
                continue
            u = units[units["session"] == sess]
            sp = np.load(f)["spikes"].astype(np.float32)
            if sp.shape[0] != len(u):
                continue
            dom = u["functional_domain"].value_counts().index[0] if len(u) else "na"
            tasks.append(("mouse", f"mouse_{dom}_{sess:02d}", sp, 50.0))

    # ---------- 猕猴：按 area 分层
    mac = root / "primate" / "macaque_nlb"
    seg = mac / "segments.parquet"
    if seg.exists():
        sdf = pd.read_parquet(seg)
        for i, row in sdf.iterrows():
            f = mac / row["file"]
            if not f.exists():
                continue
            sp = np.load(f)["spikes"].astype(np.float32)
            tasks.append(("macaque", f"macaque_{row['area']}_{i:02d}", sp, 20.0))

    print(f"  原始片段: {len(tasks)}")
    from collections import Counter
    print("  " + str(dict(Counter(t[0] for t in tasks))))

    # 抽样（保持每个区域的代表性）
    by_sp = {}
    for t in tasks:
        by_sp.setdefault(t[0], []).append(t)
    sampled = []
    for sp, lst in by_sp.items():
        if len(lst) > MAX_PER_SPECIES:
            idx = rng.choice(len(lst), MAX_PER_SPECIES, replace=False)
            lst = [lst[i] for i in np.sort(idx)]
        sampled += lst
    print(f"  抽样后: {len(sampled)} 片段")

    rows = []
    t0 = time.time()

    for sp, key, arr, src_ms in sampled:
        for bw in BIN_WIDTHS:
            if bw < src_ms * 2:
                continue
            x = to_bins(arr, bw, src_ms)
            if x.shape[1] < WIN * 60:
                continue
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-6
            proj = pca_proj((x - mu) / sd)
            if proj is None:
                continue
            w = make_windows(proj)
            if w is None:
                continue

            prof = r2_curve_blocked(w)
            if len(prof) < 3:
                continue

            # 结构对照：每列独立打乱
            ws = np.stack([rng.permutation(w[:, j]) for j in range(w.shape[1])], axis=1)
            prof_s = r2_curve_blocked(ws)
            floor = float(np.median(list(prof_s.values()))) if prof_s else 0.0

            tau = fit_tau(prof)
            rows.append({
                "species": sp, "group": key, "bin_ms": bw,
                "n_units": int(x.shape[0]), "n_win": int(len(w)),
                "r2_1": round(prof.get(1, np.nan), 4),
                "r2_2": round(prof.get(2, np.nan), 4),
                "r2_4": round(prof.get(4, np.nan), 4),
                "r2_8": round(prof.get(8, np.nan), 4),
                "floor": round(floor, 4),
                "net_1": round(prof.get(1, np.nan) - floor, 4),
                "net_4": round(prof.get(4, np.nan) - floor, 4),
                "tau_bins": (round(tau, 4) if tau else None),
                "tau_ms": (round(tau * bw, 1) if tau else None),
            })
        if len(rows) % 20 == 0 and len(rows):
            print(f"    ... {len(rows)} 条 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase18.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    # ===== 结果 1: 分块 CV 下的净可预测性
    print("\n" + "=" * 86)
    print("结果 1: 分块 CV 下的净 R²(1)（这是地基）")
    print("=" * 86)
    print(f"{'物种':<9} {'bin':>5} {'n':>4} {'R²(1)':>9} {'floor':>9} {'净R²(1)':>10} {'净R²(4)':>9}")
    for (sp, bw), g in df.groupby(["species", "bin_ms"]):
        print(f"{sp:<9} {bw:>5} {len(g):>4} {g['r2_1'].median():>9.4f} "
              f"{g['floor'].median():>9.4f} {g['net_1'].median():>10.4f} "
              f"{g['net_4'].median():>9.4f}")

    # 按具体区域/数据集
    print("\n  按具体来源（净 R²(1)，100ms）:")
    s100 = df[df.bin_ms == 100]
    for gname, g in s100.groupby("group"):
        short = gname.split("_")[1] if "_" in gname else gname
        pass
    agg = (s100.assign(region=s100["group"].str.extract(r"^(?:mouse|macaque)_([a-z A-Z]+)_")[0])
           .groupby(["species", "region"])["net_1"]
           .agg(["size", "median"]).round(4))
    print(agg.to_string())

    # ===== 结果 2: 分辨率不变性
    print("\n" + "=" * 86)
    print("结果 2: 分辨率不变性（τ_ms 在 100 与 200ms 下应一致）")
    print("=" * 86)
    real = df.dropna(subset=["tau_ms"])
    summary = {}
    if len(real):
        print(f"{'物种':<9} {'n配对':>6} {'τ@100ms':>10} {'τ@200ms':>10} {'比值':>8} {'判定':>24}")
        for sp in real["species"].unique():
            s = real[real.species == sp]
            a = s[s.bin_ms == 100].set_index("group")["tau_ms"]
            b = s[s.bin_ms == 200].set_index("group")["tau_ms"]
            m = pd.concat([a.rename("t100"), b.rename("t200")], axis=1).dropna()
            if len(m) < 4:
                continue
            ratio = float((m.t100 / m.t200).median())
            v = ("✅ 一致" if 0.75 <= ratio <= 1.35 else "❌ 不一致")
            summary[sp] = {"n": int(len(m)), "t100": round(float(m.t100.median()), 1),
                           "t200": round(float(m.t200.median()), 1),
                           "ratio": round(ratio, 3), "verdict": v}
            print(f"{sp:<9} {len(m):>6} {m.t100.median():>10.1f} {m.t200.median():>10.1f} "
                  f"{ratio:>8.3f} {v:>24}")
    else:
        print("  无可用 τ（起始 R² 太低，拟合被拒绝）")

    # ===== 最终判读
    print("\n" + "=" * 86)
    print("最终判读")
    print("=" * 86)

    best = df.groupby(["species", "bin_ms"])["net_1"].median()
    any_strong = (best > 0.10).any()
    all_weak = (best < 0.05).all()

    if all_weak:
        verdict = ("❌ 分块 CV 下所有物种净 R²(1) < 0.05 —— "
                   "100ms 分辨率下测不到跨窗可预测性。不应报告任何 τ。")
    elif any_strong:
        strong = [f"{sp}@{bw}ms={v:.3f}" for (sp, bw), v in best.items() if v > 0.10]
        verdict = (f"✅ 部分来源存在真实可预测性: {', '.join(strong)}。"
                   "但需按区域归因（见上表），不能直接归为物种。")
    else:
        verdict = "⚠️ 净信号存在但弱（0.05-0.10），证据不足"

    print(f"  {verdict}")

    (out / "phase18_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "win": WIN, "stride": STRIDE, "n_blocks": N_BLOCKS,
        "bin_widths": BIN_WIDTHS, "ks": KS,
        "n_records": int(len(df)),
        "by_species_bin": {f"{a}|{b}": {
            "n": int(len(g)),
            "r2_1": round(float(g["r2_1"].median()), 4),
            "floor": round(float(g["floor"].median()), 4),
            "net_1": round(float(g["net_1"].median()), 4),
            "net_4": round(float(g["net_4"].median()), 4)}
            for (a, b), g in df.groupby(["species", "bin_ms"])},
        "by_region": {f"{a}|{b}": {"n": int(r["size"]), "net_1": float(r["median"])}
                      for (a, b), r in agg.iterrows()},
        "resolution_invariance": summary,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
