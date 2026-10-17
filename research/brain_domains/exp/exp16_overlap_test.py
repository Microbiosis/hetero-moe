"""Phase 16: 窗口重叠是不是 τ 的来源?

## 起因

exp15 的双分辨率检验显示：三个 τ（fast / single / slow）
在 50ms 与 100ms 分箱下的比值全部 ≈ 1.0，而非真实时间常数应有的 2.0。
即 τ 是"固定 bin 数"，随分析步长缩放，不随真实时间缩放。

## 最可能的原因: 窗口重叠

窗口构造（所有前序实验共用）：

    WIN = 8 bins, STRIDE = 4 bins     # 相邻窗口共享 50% 的 bin
    window_t = proj[t*4 : t*4+8].mean(axis=0)

则 window_t 与 window_{t+1} 共享 4 个原始 bin（一半）。
因此 R²(k=1) 里有相当部分来自"两个窗口共享同一批 bin"，
这是分析流程的机械相关，而非神经动力学。

若如此，τ 自然"固定 bins"——它度量的是窗口重叠导致的去相关，
与 bin 宽度无关。

## 检验

在同一批数据上比较三种窗口构造：

  A  重叠    WIN=8, STRIDE=4   （现有做法，50% 重叠）
  B  不重叠  WIN=8, STRIDE=8   （0% 重叠）
  C  大间隔  WIN=8, STRIDE=16  （间隔 8 bin，更严格）

对每种构造，在 50ms 与 100ms 分箱下算 R²(k=1) 与 τ_single。

判读：
  若 B/C 的 R²(1) 远低于 A        → A 的可预测性主要来自重叠
  若 B/C 的 τ 比值接近 2.0        → 去掉重叠后 τ 才成为真实时间常数
  若 B/C 的 R²(1) 接近 0          → 说明真实 bin-to-bin 可预测性极低，
                                     之前的"可预测性"大多是伪影
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
KS = [1, 2, 4, 8, 16]
WIN = 8
ALPHA = 1000.0
MIN_UNITS = 20
N_SESSIONS = 8
BIN_WIDTHS = [50, 100]

# (名称, stride, 说明)
CONFIGS = [
    ("A_overlap50", 4, "WIN=8 STRIDE=4 (50% 重叠，现有做法)"),
    ("B_noOverlap", 8, "WIN=8 STRIDE=8 (0% 重叠)"),
    ("C_gap", 16, "WIN=8 STRIDE=16 (间隔 8 bin)"),
]


def pca_proj(z):
    """返回逐 bin 的 PCA 投影 (nb, PROJ_DIM)。"""
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


def windows_with_stride(proj, stride, max_win=600):
    nb = proj.shape[0]
    starts = list(range(0, nb - WIN + 1, stride))
    if len(starts) < 60:
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
    return float(1.0 - ((Yte - (Xte_ @ W + ym)) ** 2).sum() /
                 max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def r2_curve(x):
    out = {}
    for k in KS:
        X, Y = make_pairs(x, k)
        if X is None:
            continue
        cut = int(len(X) * 0.7)
        if cut < 60 or len(X) - cut < 30:
            continue
        out[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
    return out


def fit_tau(prof):
    from scipy.optimize import curve_fit
    if len(prof) < 4:
        return None
    ks = np.array(sorted(prof), float)
    r2 = np.array([prof[k] for k in sorted(prof)], float)

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.02), 2.0, min(r2[-1], 0.0)]
        popt, _ = curve_fit(f, ks, r2, p0=p0,
                            bounds=([0, 0.05, -2.0], [5, 200, 1.0]), maxfev=12000)
        return float(popt[1])
    except Exception:
        return None


def to_bins(sp, bin_ms):
    ratio = max(1, int(round(bin_ms / 50)))
    nb = sp.shape[1] // ratio
    return sp[:, :nb * ratio].reshape(sp.shape[0], nb, ratio).sum(axis=2)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp16")
    out.mkdir(parents=True, exist_ok=True)

    stein = root / "steinmetz"
    units = pd.read_parquet(stein / "units.parquet")

    print("=" * 84)
    print("Phase 16 — 窗口重叠是不是 τ 的来源?")
    print("=" * 84)
    for n, s, d in CONFIGS:
        print(f"  {n:<14} {d}")
    print()

    rc = units.groupby("session")["region"].nunique()
    sessions = list(rc[rc >= 4].sort_values(ascending=False).index[:N_SESSIONS])
    print(f"  session: {sessions}\n")

    rows = []
    t0 = time.time()

    for sess in sessions:
        f = stein / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)
        u = units[units["session"] == sess].reset_index(drop=True)
        if sp.shape[0] != len(u):
            continue

        for bw in BIN_WIDTHS:
            x = to_bins(sp, bw)
            for region, sub in u.groupby("region"):
                idx = sub["unit_idx"].to_numpy()
                idx = idx[(idx >= 0) & (idx < x.shape[0])]
                if len(idx) < MIN_UNITS:
                    continue
                z = x[idx]
                mu = z.mean(axis=1, keepdims=True)
                sd = z.std(axis=1, keepdims=True) + 1e-6
                proj = pca_proj((z - mu) / sd)
                if proj is None:
                    continue

                for name, stride, _ in CONFIGS:
                    w = windows_with_stride(proj, stride)
                    if w is None:
                        continue
                    prof = r2_curve(w)
                    if len(prof) < 4:
                        continue
                    tau = fit_tau(prof)
                    # 打乱对照
                    ws = np.stack([np.random.default_rng(SEED).permutation(w[:, j])
                                   for j in range(w.shape[1])], axis=1)
                    prof_s = r2_curve(ws)
                    floor = float(np.median(list(prof_s.values()))) if prof_s else float("nan")

                    rows.append({
                        "session": int(sess), "region": region, "bin_ms": bw,
                        "config": name, "n_units": int(len(idx)),
                        "n_win": int(len(w)),
                        "r2_1": round(prof.get(1, float("nan")), 4),
                        "r2_16": round(prof.get(16, float("nan")), 4),
                        "floor": round(floor, 4),
                        "net_1": round(prof.get(1, float("nan")) - floor, 4),
                        "tau": round(tau, 4) if tau else None,
                    })
        print(f"  sess {sess:02d} 完成，累计 {len(rows)} 条")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase16_overlap.csv", index=False)

    print("\n" + "=" * 84)
    print("结果 1: R²(k=1) 与净信号（按窗口构造 × bin 宽度）")
    print("=" * 84)
    print(f"{'构造':<14} {'bin':>5} {'n':>5} {'R²(1)':>8} {'净R²(1)':>9} {'R²(16)':>8} {'τ':>7}")
    for (cfg, bw), g in df.groupby(["config", "bin_ms"]):
        print(f"{cfg:<14} {bw:>5} {len(g):>5} {g['r2_1'].median():>8.4f} "
              f"{g['net_1'].median():>9.4f} {g['r2_16'].median():>8.4f} "
              f"{g['tau'].median():>7.3f}")

    print("\n" + "=" * 84)
    print("结果 2: τ 的缩放比值（真实 ms 常数应为 2.0，固定 bins 则为 1.0）")
    print("=" * 84)
    summary = {}
    for cfg in [c[0] for c in CONFIGS]:
        s = df[df.config == cfg]
        a = s[s.bin_ms == 50].set_index(["session", "region"])["tau"]
        b = s[s.bin_ms == 100].set_index(["session", "region"])["tau"]
        m = pd.concat([a.rename("t50"), b.rename("t100")], axis=1).dropna()
        if len(m) < 5:
            continue
        ratio = float((m.t50 / m.t100).median())
        summary[cfg] = {"n": int(len(m)), "ratio": round(ratio, 3),
                        "tau50": round(float(m.t50.median()), 3),
                        "tau100": round(float(m.t100.median()), 3)}
        verdict = ("✅ 固定 ms（真实时间常数）" if 1.7 <= ratio <= 2.3
                   else "❌ 固定 bins（伪影）" if ratio < 1.15 else "⚠️ 中间")
        print(f"  {cfg:<14} n={len(m):>4}  τ50={m.t50.median():>6.3f}  "
              f"τ100={m.t100.median():>6.3f}  比值={ratio:>5.3f}  {verdict}")

    print("\n" + "=" * 84)
    print("结果 3: 重叠的贡献（A 与 B 的差）")
    print("=" * 84)
    for bw in BIN_WIDTHS:
        a = df[(df.config == "A_overlap50") & (df.bin_ms == bw)]["r2_1"].median()
        b = df[(df.config == "B_noOverlap") & (df.bin_ms == bw)]["r2_1"].median()
        na = df[(df.config == "A_overlap50") & (df.bin_ms == bw)]["net_1"].median()
        nb_ = df[(df.config == "B_noOverlap") & (df.bin_ms == bw)]["net_1"].median()
        print(f"  {bw:>3}ms: R²(1) 重叠 {a:.4f} → 不重叠 {b:.4f} "
              f"(降 {100*(a-b)/max(a,1e-9):.0f}%) | "
              f"净 {na:.4f} → {nb_:.4f}")

    # 最终判读
    print("\n" + "=" * 84)
    print("判读")
    print("=" * 84)
    ra = summary.get("A_overlap50", {}).get("ratio")
    rb = summary.get("B_noOverlap", {}).get("ratio")
    net_a = df[df.config == "A_overlap50"]["net_1"].median()
    net_b = df[df.config == "B_noOverlap"]["net_1"].median()

    if net_b < 0.02:
        v = ("❌ 去掉窗口重叠后，净可预测性接近 0 —— "
             "前序实验中'可预测性'的主体是窗口重叠造成的机械相关，不是神经动力学")
    elif rb and 1.7 <= rb <= 2.3:
        v = "✅ 去掉重叠后 τ 变成真实时间常数（比值≈2），之前的 bin 缩放是重叠导致的"
    elif rb and rb < 1.15:
        v = "❌ 即使去掉重叠，τ 仍固定于 bin 数 —— 还有别的方法学原因"
    else:
        v = f"⚠️ 不重叠后比值 {rb}，介于两者之间"
    print(f"  重叠对 R²(1) 的贡献: {net_a:.4f} → 去重叠 {net_b:.4f}")
    print(f"\n  {v}")

    (out / "phase16_report.json").write_text(json.dumps({
        "schema_version": "1.0", "configs": {c[0]: c[2] for c in CONFIGS},
        "bin_widths": BIN_WIDTHS, "ks": KS, "win": WIN,
        "by_config": {f"{c}|{bw}": {"n": int(len(g)),
                                    "r2_1": round(float(g["r2_1"].median()), 4),
                                    "net_1": round(float(g["net_1"].median()), 4),
                                    "tau": (round(float(g["tau"].median()), 3)
                                            if g["tau"].notna().any() else None)}
                      for (c, bw), g in df.groupby(["config", "bin_ms"])},
        "tau_scaling": summary,
        "overlap_effect": {str(bw): {
            "r2_1_overlap": round(float(df[(df.config == "A_overlap50") & (df.bin_ms == bw)]["r2_1"].median()), 4),
            "r2_1_noOverlap": round(float(df[(df.config == "B_noOverlap") & (df.bin_ms == bw)]["r2_1"].median()), 4),
        } for bw in BIN_WIDTHS},
        "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
