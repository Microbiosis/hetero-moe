"""Phase 21: 降维后的 VAR(1) —— 让拟合在所有分辨率下都可靠。

## exp20 的教训

exp20 用 64 维 PCA 拟合 VAR(1)：64×64 = **4096 个参数**。
但 100ms 分箱下每个片段只有约 2000 个样本 —— **参数比样本多**。

结果（真实数据）：
    bin    τ_ms    VAR R²
    50ms   852.3   +0.221   ← 可用
   100ms   898.2   -1.462   ← 崩溃
   200ms   956.5   -6.357   ← 崩溃

正控（合成数据，真实最慢 τ=300ms）在同样条件下也从 0.595 掉到 0.057，
说明这是**样本量/维度比**的问题，不是数据本身的问题。

## 本实验

扫描 D_PCA ∈ {4, 8, 16, 32}，对每个维度：
  1. 跑正控（已知 τ），确认能恢复
  2. 跑真实数据，记录 VAR R²
  3. 只有 VAR R² > 0.05 的结果才可信

**判据**：找到使三种分辨率下 VAR R² 都 > 0.05 的最大维度，
         在那个维度上看 τ 是否分辨率一致。

这是"先保证测量有效，再读数值"的正规做法。
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
BIN_WIDTHS = [50, 100, 200]
DIMS = [4, 8, 16, 32]
N_BLOCKS = 4
RIDGE = 1e-2
TAU_TRUE_MS = 200.0        # 合成数据的中位 τ（最慢模态为其 1.5 倍 = 300ms）
MIN_BINS_PER_BLOCK = 300
TARGET_AREA = "M1_PMd"
R2_MIN_TRUST = 0.05        # VAR R² 低于此视为不可信


def fit_var1(X, ridge=RIDGE):
    x0, x1 = X[:-1], X[1:]
    if len(x0) < 2 * X.shape[1]:
        return None, None
    x0c = x0 - x0.mean(0, keepdims=True)
    x1c = x1 - x1.mean(0, keepdims=True)
    G = x0c.T @ x0c
    tr = np.trace(G) / G.shape[0]
    G.flat[:: G.shape[0] + 1] += ridge * max(tr, 1e-9)
    try:
        A = np.linalg.solve(G, x0c.T @ x1c)
    except np.linalg.LinAlgError:
        return None, None
    return A, np.linalg.eigvals(A)


def var_r2(X, A):
    x0 = X[:-1] - X[:-1].mean(0, keepdims=True)
    x1 = X[1:] - X[1:].mean(0, keepdims=True)
    pred = x0 @ A.T
    return float(1.0 - ((x1 - pred) ** 2).sum() / max((x1 ** 2).sum(), 1e-12))


def tau_of(ev, bin_ms):
    mag = np.abs(ev)
    mag = mag[(mag > 1e-4) & (mag < 1.0)]
    if len(mag) == 0:
        return None, None
    lam = float(np.sort(mag)[::-1][0])
    tb = -1.0 / np.log(lam)
    return tb, tb * bin_ms


def analyze(X, bin_ms):
    blk = len(X) // N_BLOCKS
    if blk < MIN_BINS_PER_BLOCK:
        return None
    taus, r2s, rads = [], [], []
    for b in range(N_BLOCKS):
        seg = X[b * blk:(b + 1) * blk]
        A, ev = fit_var1(seg)
        if A is None:
            continue
        tb, tm = tau_of(ev, bin_ms)
        if tb is not None and 0.05 < tb < 500:
            taus.append(tm)
        r2s.append(var_r2(seg, A))
        m = np.abs(ev); m = m[m < 10]
        if len(m):
            rads.append(float(m.max()))
    if not taus:
        return None
    return {"tau_ms": float(np.median(taus)),
            "var_r2": float(np.median(r2s)),
            "radius": float(np.median(rads)) if rads else None,
            "n_blocks": len(taus)}


def pca_bins(x_counts, d):
    z = x_counts.astype(np.float32)
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True) + 1e-6
    z = (z - mu) / sd
    nb = z.shape[1]
    k = min(d, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    w_, U = np.linalg.eigh(zc @ zc.T)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < d:
        proj = np.pad(proj, ((0, 0), (0, d - proj.shape[1])))
    return proj.astype(np.float32)


def to_bins(sp, bin_ms, src_ms):
    r = max(1, int(round(bin_ms / src_ms)))
    nb = sp.shape[1] // r
    return sp[:, :nb * r].reshape(sp.shape[0], nb, r).sum(axis=2)


def synth(d, n_bins_50, rng):
    """合成 AR(1)，最慢模态 τ = 1.5 × 200 = 300ms。"""
    scale = np.linspace(0.5, 1.5, d)
    a = np.exp(-50.0 / (TAU_TRUE_MS * scale))
    x = np.zeros((n_bins_50, d))
    noise = rng.standard_normal((n_bins_50, d))
    for t in range(1, n_bins_50):
        x[t] = a * x[t - 1] + noise[t]
    return x


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp21")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 80)
    print("Phase 21 — 降维让 VAR(1) 在所有分辨率下可靠")
    print("=" * 80)
    print(f"  维度 {DIMS} | bin {BIN_WIDTHS} | 可信阈值 VAR R² > {R2_MIN_TRUST}")
    print()

    # ============ 正控扫描
    print("=" * 80)
    print("正控：合成 AR(1)，最慢模态 τ = 300 ms")
    print("=" * 80)
    print(f"{'d':>3} {'bin':>5} {'τ_ms':>9} {'VAR R²':>9} {'谱半径':>8} {'可信':>6}")
    ctrl = {}
    for d in DIMS:
        x = synth(d, 12000, rng)
        for bw in BIN_WIDTHS:
            r_ = bw // 50
            xs = x[::r_] if r_ > 1 else x
            r = analyze(xs, bw)
            if r is None:
                continue
            trust = r["var_r2"] > R2_MIN_TRUST
            ctrl[f"{d}|{bw}"] = {**r, "trusted": bool(trust)}
            print(f"{d:>3} {bw:>5} {r['tau_ms']:>9.1f} {r['var_r2']:>9.3f} "
                  f"{r['radius']:>8.3f} {'✅' if trust else '❌':>6}")

    # 正控一致性（每个 d 只看可信的）
    print("\n  各维度下可信的分辨率数:")
    ctrl_ok = {}
    for d in DIMS:
        ns = [bw for bw in BIN_WIDTHS
              if ctrl.get(f"{d}|{bw}", {}).get("trusted")]
        ctrl_ok[d] = ns
        print(f"    d={d:>2}: {len(ns)}/3 {ns if ns else ''}")

    # ============ 真实数据
    print("\n" + "=" * 80)
    print(f"真实数据：猕猴 {TARGET_AREA}")
    print("=" * 80)

    mac = root / "primate" / "macaque_nlb"
    seg = mac / "segments.parquet"
    if not seg.exists():
        print("❌ 缺少 segments.parquet")
        return
    sdf = pd.read_parquet(seg)
    meta = sdf[sdf["area"] == TARGET_AREA]
    print(f"  片段 {len(meta)}")

    # 预载
    loaded = []
    for i, row in meta.iterrows():
        f = mac / row["file"]
        if f.exists():
            loaded.append((str(row["file"]), np.load(f)["spikes"].astype(np.float32)))
    print(f"  已载入 {len(loaded)}")

    rows = []
    t0 = time.time()
    for d in DIMS:
        n_done = 0
        for fname, sp in loaded:
            for bw in BIN_WIDTHS:
                x = to_bins(sp, bw, 20.0)
                if x.shape[1] < MIN_BINS_PER_BLOCK * N_BLOCKS:
                    continue
                proj = pca_bins(x, d)
                if proj is None:
                    continue
                r = analyze(proj, bw)
                if r is None:
                    continue
                rows.append({"d": d, "file": fname, "bin_ms": bw,
                             "n_bins": int(x.shape[1]),
                             **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items()}})
                n_done += 1
        print(f"  d={d:>2}: {n_done} 条 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase21.csv", index=False)

    print("\n" + "=" * 80)
    print("真实数据结果（按维度 × 分辨率）")
    print("=" * 80)
    print(f"{'d':>3} {'bin':>5} {'n':>4} {'τ_ms':>9} {'VAR R²':>9} {'谱半径':>8} {'可信':>6}")
    for (d, bw), g in df.groupby(["d", "bin_ms"]):
        trust = g["var_r2"].median() > R2_MIN_TRUST
        print(f"{d:>3} {bw:>5} {len(g):>4} {g['tau_ms'].median():>9.1f} "
              f"{g['var_r2'].median():>9.3f} {g['radius'].median():>8.3f} "
              f"{'✅' if trust else '❌':>6}")

    # ============ 找最佳维度
    print("\n" + "=" * 80)
    print("判定：哪个维度能让三种分辨率都可信？")
    print("=" * 80)
    best_d, best_info = None, None
    for d in DIMS:
        s = df[df.d == d]
        meds = s.groupby("bin_ms")["var_r2"].median()
        if len(meds) < len(BIN_WIDTHS):
            print(f"  d={d:>2}: 分辨率不全 ({len(meds)}/3)")
            continue
        all_ok = bool((meds > R2_MIN_TRUST).all())
        taus = s.groupby("bin_ms")["tau_ms"].median()
        ratios = {bw: round(float(taus[bw] / taus[100]), 3) for bw in BIN_WIDTHS if 100 in taus.index}
        consistent = all(0.75 <= r <= 1.35 for r in ratios.values()) if ratios else False
        print(f"  d={d:>2}: VAR R² 全可信={'✅' if all_ok else '❌'} "
              f"({', '.join(f'{bw}ms={v:.3f}' for bw, v in meds.items())}) | "
              f"τ 一致={'✅' if consistent else '❌'} ({ratios})")
        if all_ok and consistent and best_d is None:
            best_d, best_info = d, {"var_r2": meds.to_dict(), "ratios": ratios,
                                     "taus": taus.to_dict()}

    print("\n" + "=" * 80)
    print("最终判读")
    print("=" * 80)
    if best_d:
        tau_val = best_info["taus"].get(100, best_info["taus"].get(50))
        verdict = (f"✅ 在 d={best_d} 时，三种分辨率下 VAR R² 均可信且 τ 一致 "
                   f"→ **猕猴 {TARGET_AREA} 最慢线性模态 τ ≈ {tau_val:.0f} ms**")
    else:
        # 退一步：只报可信的分辨率
        trusted = [(d, bw, g) for (d, bw), g in df.groupby(["d", "bin_ms"])
                   if g["var_r2"].median() > R2_MIN_TRUST]
        if trusted:
            d0, bw0, g0 = max(trusted, key=lambda t: t[2]["var_r2"].median())
            verdict = (f"⚠️ 无维度能让三种分辨率同时可信。"
                       f"最可信的配置是 d={d0}, {bw0}ms（VAR R²={g0['var_r2'].median():.3f}），"
                       f"该配置下 τ ≈ {g0['tau_ms'].median():.0f} ms，"
                       f"但**无法验证分辨率不变性**，只能作为暂定值。")
        else:
            verdict = "❌ 所有配置下 VAR R² 均不可信，无法估计 τ。"
    print(f"  {verdict}")

    (out / "phase21_report.json").write_text(json.dumps({
        "schema_version": "1.0", "dims": DIMS, "bin_widths": BIN_WIDTHS,
        "r2_min_trust": R2_MIN_TRUST, "area": TARGET_AREA,
        "control": ctrl,
        "control_trusted_per_dim": {str(k): v for k, v in ctrl_ok.items()},
        "real": {f"{d}|{bw}": {"n": int(len(g)),
                               "tau_ms": round(float(g["tau_ms"].median()), 1),
                               "var_r2": round(float(g["var_r2"].median()), 4),
                               "radius": round(float(g["radius"].median()), 4)}
                 for (d, bw), g in df.groupby(["d", "bin_ms"])},
        "best_dim": best_d,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
