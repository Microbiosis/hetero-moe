"""Phase 17: 干净的跨物种 τ 比较（零重叠 + 分辨率不变性 + 结构对照）。

## 三个必须同时满足的条件

### 1. 窗口零重叠
STRIDE = WIN（不是之前的一半）。相邻窗口不共享任何 bin。
预测窗口 t+k 与输入窗口 t 之间至少隔 k 个完整窗。

### 2. 分辨率不变性
在 100ms 与 200ms 两种分辨率下各算一次 τ。
    τ(bins) × bin_ms = τ(ms)
若 τ 是真实时间常数，两种分辨率下的 τ(ms) 应一致（比值 ≈ 1.0）。
若 τ(bins) 不变而 τ(ms) 差 2 倍 → 仍是伪影。

**注意符号**：这次的判据与 exp15 相反 ——
    exp15 比较 τ(bins) 的比值（真实常数应得 2.0）
    本次直接比较 τ(ms)，真实常数应得 1.0
两种写法等价，但 τ(ms) 更直观。

### 3. 结构对照
  A  real        真实数据
  B  shuffled   每列独立打乱（破坏时序，保留边缘分布）
  C  stride2x    步长加倍到 16（若信号仍来自残余结构，会变化）

## 为什么用 100/200ms 而非 50/100ms

三种数据源的原始分辨率不同：
    小鼠 Steinmetz 50ms | 猕猴 NLB 20ms | 人类 neural-pile 20ms
只有 100ms 与 200ms 能被三者整除，才能做**跨物种**的分辨率检验。
（50ms 只能用于小鼠，作为补充。）

## 预期

按 exp16 的结果，零重叠下净 R²(1) 可能 ≈0 或为负。
若如此，诚实结论是：**在 100ms 分辨率下，跨窗可预测性不可测量**。
那就不该报告任何 τ，而应报告"测不到"。
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
STRIDES = {"real": 8, "stride2x": 16}     # 均为 WIN 的整数倍 → 零重叠
BIN_WIDTHS = [100, 200]
ALPHA = 1000.0
MIN_UNITS = 20
MAX_WIN = 500
MAX_GROUPS_PER_SPECIES = 20


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


def make_windows(proj, stride, max_win=MAX_WIN):
    """零重叠窗口：STRIDE == WIN。"""
    nb = proj.shape[0]
    starts = list(range(0, nb - WIN + 1, stride))
    if len(starts) < 40:
        return None
    if len(starts) > max_win:
        s0 = (len(starts) - max_win) // 2
        starts = starts[s0:s0 + max_win]
    return np.stack([proj[s:s + WIN].mean(axis=0) for s in starts]).astype(np.float32)


def make_pairs(x, k, ctx=CTX):
    n = x.shape[0]
    idx = np.arange(ctx - 1, n - k)
    if len(idx) < 40:
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


def r2_curve(x):
    out = {}
    for k in KS:
        X, Y = make_pairs(x, k)
        if X is None:
            continue
        cut = int(len(X) * 0.7)
        if cut < 50 or len(X) - cut < 25:
            continue
        out[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
    return out


def fit_tau(prof):
    from scipy.optimize import curve_fit
    if len(prof) < 3:
        return None
    ks = np.array(sorted(prof), float)
    r2 = np.array([prof[k] for k in sorted(prof)], float)
    if r2[0] <= 0.01:            # 起始值太小，拟合无意义
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


def collect_sources(root: Path):
    """产出 (species, group_key, spikes(n_units, n_bins), src_ms)。"""
    out = []

    # 小鼠 Steinmetz (50ms)
    stein = root / "steinmetz"
    up = stein / "units.parquet"
    if up.exists():
        units = pd.read_parquet(up)
        for sess in sorted(units["session"].unique()):
            f = stein / f"spikes_session_{sess:03d}.npz"
            if not f.exists():
                continue
            sp = np.load(f)["spikes"].astype(np.float32)
            if sp.shape[0] != len(units[units["session"] == sess]):
                continue
            out.append(("mouse", f"sess{int(sess)}", sp, 50.0))

    # 猕猴：读 macaque 的 npz（20ms）
    mac = root / "primate" / "macaque_nlb"
    seg = mac / "segments.parquet"
    if seg.exists():
        sdf = pd.read_parquet(seg)
        for i, row in sdf.iterrows():
            f = mac / row["file"]
            if not f.exists():
                continue
            sp = np.load(f)["spikes"].astype(np.float32)
            out.append(("macaque", f"{row['dataset']}#{i:02d}", sp, 20.0))

    # 人类侵入式：原始在 HF，这里改用 omnibrain 的 windows（已 100ms）
    # → 无法做 100/200 双分辨率，单独处理
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp17")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("Phase 17 — 干净的跨物种 τ（零重叠 + 分辨率不变性 + 结构对照）")
    print(f"  WIN={WIN} | 步长 {STRIDES}（均≥WIN，零重叠）| bin {BIN_WIDTHS} | k∈{KS}")
    print("=" * 84)

    srcs = collect_sources(root)
    print(f"  数据源: {len(srcs)} 个片段")
    by_sp = {}
    for sp, key, arr, sm in srcs:
        by_sp.setdefault(sp, []).append((key, arr, sm))
    for sp, lst in by_sp.items():
        print(f"    {sp}: {len(lst)} 个片段")

    # 每物种抽样
    for sp in by_sp:
        if len(by_sp[sp]) > MAX_GROUPS_PER_SPECIES:
            idx = rng.choice(len(by_sp[sp]), MAX_GROUPS_PER_SPECIES, replace=False)
            by_sp[sp] = [by_sp[sp][i] for i in np.sort(idx)]
            print(f"    → {sp} 抽样至 {len(by_sp[sp])}")

    rows = []
    t0 = time.time()

    for sp, lst in by_sp.items():
        for key, arr, src_ms in lst:
            for bw in BIN_WIDTHS:
                if bw < src_ms * 2:
                    continue
                x = to_bins(arr, bw, src_ms)
                if x.shape[1] < WIN * 40:
                    continue

                # 逐 unit z-score（每次重算，因为分箱变了）
                mu = x.mean(axis=1, keepdims=True)
                sd = x.std(axis=1, keepdims=True) + 1e-6
                proj = pca_proj((x - mu) / sd)
                if proj is None:
                    continue

                for cfg, stride in STRIDES.items():
                    w = make_windows(proj, stride)
                    if w is None:
                        continue
                    prof = r2_curve(w)
                    if len(prof) < 3:
                        continue

                    # 结构对照：每列独立打乱
                    ws = np.stack([rng.permutation(w[:, j])
                                   for j in range(w.shape[1])], axis=1)
                    prof_s = r2_curve(ws)
                    floor = float(np.median(list(prof_s.values()))) if prof_s else 0.0

                    tau_bins = fit_tau(prof)
                    rows.append({
                        "species": sp, "group": key, "bin_ms": bw, "config": cfg,
                        "n_units": int(x.shape[0]), "n_win": int(len(w)),
                        "r2_1": round(prof.get(1, np.nan), 4),
                        "r2_2": round(prof.get(2, np.nan), 4),
                        "r2_4": round(prof.get(4, np.nan), 4),
                        "floor": round(floor, 4),
                        "net_1": round(prof.get(1, np.nan) - floor, 4),
                        "tau_bins": (round(tau_bins, 4) if tau_bins else None),
                        "tau_ms": (round(tau_bins * bw, 1) if tau_bins else None),
                    })
        print(f"  {sp} 完成，累计 {len(rows)} 条 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase17.csv", index=False)

    if not len(df):
        print("无数据")
        return

    # ===== 结果 1: 净可预测性
    print("\n" + "=" * 84)
    print("结果 1: 零重叠下的净可预测性（这是全部结论的地基）")
    print("=" * 84)
    print(f"{'物种':<9} {'bin':>5} {'构造':<10} {'n':>4} {'R²(1)':>8} {'floor':>8} {'净R²(1)':>9}")
    for (sp, bw, cfg), g in df.groupby(["species", "bin_ms", "config"]):
        print(f"{sp:<9} {bw:>5} {cfg:<10} {len(g):>4} {g['r2_1'].median():>8.4f} "
              f"{g['floor'].median():>8.4f} {g['net_1'].median():>9.4f}")

    overall_net = float(df[df.config == "real"]["net_1"].median())
    print(f"\n  全局净 R²(1) 中位: {overall_net:+.4f}")

    if overall_net < 0.02:
        print("  ⚠️ 净可预测性 ≈ 0 —— 零重叠下**测不到跨窗可预测性**")
        print("     这意味着: 不应报告任何 τ; 只能报告'在 100ms 分辨率下不可测量'")

    # ===== 结果 2: 分辨率不变性
    print("\n" + "=" * 84)
    print("结果 2: 分辨率不变性（τ_ms 在 100 与 200ms 下应一致）")
    print("=" * 84)
    real = df[df.config == "real"].dropna(subset=["tau_ms"])
    summary = {}
    if len(real):
        print(f"{'物种':<9} {'n配对':>6} {'τ_ms@100':>10} {'τ_ms@200':>10} {'比值':>8} {'判定':>22}")
        for sp in real["species"].unique():
            s = real[real.species == sp]
            a = s[s.bin_ms == 100].set_index("group")["tau_ms"]
            b = s[s.bin_ms == 200].set_index("group")["tau_ms"]
            m = pd.concat([a.rename("t100"), b.rename("t200")], axis=1).dropna()
            if len(m) < 5:
                continue
            ratio = float((m.t100 / m.t200).median())
            v = ("✅ 一致（真实时间常数）" if 0.75 <= ratio <= 1.35
                 else "❌ 不一致（伪影）")
            summary[sp] = {"n": int(len(m)), "t100": round(float(m.t100.median()), 1),
                           "t200": round(float(m.t200.median()), 1),
                           "ratio": round(ratio, 3), "verdict": v}
            print(f"{sp:<9} {len(m):>6} {m.t100.median():>10.1f} {m.t200.median():>10.1f} "
                  f"{ratio:>8.3f} {v:>22}")
    else:
        print("  无可用 τ（起值过低，均被拒绝拟合）")

    # ===== 结果 3: 步长加倍的影响
    print("\n" + "=" * 84)
    print("结果 3: 步长加倍（8→16）对净 R²(1) 的影响")
    print("=" * 84)
    for bw in BIN_WIDTHS:
        a = df[(df.config == "real") & (df.bin_ms == bw)]["net_1"].median()
        b = df[(df.config == "stride2x") & (df.bin_ms == bw)]["net_1"].median()
        print(f"  {bw:>3}ms: 步长8 净 {a:+.4f} → 步长16 净 {b:+.4f} "
              f"(变化 {b-a:+.4f})")
    print("  若两者都 ≈0，说明即使步长不改变结论：测不到")

    # ===== 最终判读
    print("\n" + "=" * 84)
    print("最终判读")
    print("=" * 84)

    ratios = [v["ratio"] for v in summary.values()]
    consistent = all(0.75 <= r <= 1.35 for r in ratios) if ratios else False

    if overall_net < 0.02:
        verdict = ("❌ **测不到可预测性**。零重叠 + 100ms 分辨率下，"
                   "跨窗净 R²(1) ≈ 0。任何 τ 都不应报告。"
                   "前序 16 个实验的 τ 结论全部作废，"
                   "根因确认为窗口重叠。")
    elif consistent:
        verdict = ("✅ τ 在两种分辨率下一致 → 真实时间常数。"
                   f"跨物种: " + ", ".join(f"{k}={v['t100']:.0f}ms" for k, v in summary.items()))
    else:
        verdict = "⚠️ 净信号存在但 τ 分辨率不一致 → 仍有未识别的结构性因素"

    print(f"  {verdict}")

    (out / "phase17_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "win": WIN, "strides": STRIDES, "bin_widths": BIN_WIDTHS, "ks": KS,
        "n_records": int(len(df)),
        "overall_net_r2_1": round(overall_net, 4),
        "by_group": {f"{a}|{b}|{c}": {
            "n": int(len(g)),
            "r2_1": round(float(g["r2_1"].median()), 4),
            "net_1": round(float(g["net_1"].median()), 4),
            "tau_ms": (round(float(g["tau_ms"].median()), 1)
                       if g["tau_ms"].notna().any() else None)}
            for (a, b, c), g in df.groupby(["species", "bin_ms", "config"])},
        "resolution_invariance": summary,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
