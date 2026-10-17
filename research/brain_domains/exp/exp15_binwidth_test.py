"""Phase 15: 改进版 —— 双分辨率判定 τ_fast、放宽截尾、50 次置换。

## 三个改进

### 改进 1（最关键）: 双分辨率判定 τ_fast 是伪影还是真实

原始数据是 **50ms** 分箱（Steinmetz），因此可把分辨率提高一倍
（OmniBrain 之前统一降到 100ms）。

判定逻辑：
    若 τ_fast 恒定 = 0.37 **bins**：
        100ms bin → 37 ms
         50ms bin → 18.5 ms        ← 随 bin 缩放 → 分辨率伪影
    若 τ_fast 恒定 = 37 **ms**：
        100ms bin → 0.37 bins
         50ms bin → 0.74 bins       ← 真实时间常数

即：**看 τ 在 bin 单位下是否不变**。配对比值 ≈1 → 伪影，≈2 → 真实。

### 改进 2: 拟合上界 60 → 200 bins
exp14 有 27%（82/300）的 τ_slow 恰好等于上界 60.0。

### 改进 3: 置换 20 → 50 次
20 次的最小可达 p 是 0.048，无法给强证据。50 次为 0.020。
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
KS_FULL = [1, 2, 3, 4, 6, 8, 12, 16]
KS_FAST = [1, 2, 4, 8]
WIN_LEN = 800
WIN, STRIDE = 8, 4
PROJ_DIM = 64
ALPHA = 1000.0
MIN_UNITS = 10
TAU_MAX = 200.0
N_PERM = 50
N_SESSIONS = 12
BIN_WIDTHS = [50, 100]


def pca_windows(z, max_win=WIN_LEN):
    nb = z.shape[1]
    k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None
    zc = z - z.mean(axis=1, keepdims=True)
    w_, U = np.linalg.eigh(zc @ zc.T)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < PROJ_DIM:
        proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
    starts = list(range(0, nb - WIN + 1, STRIDE))
    if len(starts) < 100:
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


def r2_curve(x, ks):
    out = {}
    for k in ks:
        X, Y = make_pairs(x, k)
        if X is None:
            continue
        cut = int(len(X) * 0.7)
        if cut < 100 or len(X) - cut < 60:
            continue
        out[k] = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
    return out


def fit_single(ks, r2, tau_max=TAU_MAX):
    from scipy.optimize import curve_fit
    ks = np.asarray(ks, float)
    r2 = np.asarray(r2, float)
    ok = np.isfinite(r2)
    if ok.sum() < 4:
        return None
    ks, r2 = ks[ok], r2[ok]

    def f(k, a, tau, c):
        return a * np.exp(-k / np.maximum(tau, 1e-3)) + c

    try:
        p0 = [max(r2[0] - r2[-1], 0.05), 3.0, min(r2[-1], 0.0)]
        bounds = ([0, 0.05, -2.0], [5, tau_max, 1.0])
        popt, _ = curve_fit(f, ks, r2, p0=p0, bounds=bounds, maxfev=12000)
        rss = float(((r2 - f(ks, *popt)) ** 2).sum())
        return {"tau": float(popt[1]), "rss": rss,
                "censored": bool(popt[1] > tau_max * 0.98)}
    except Exception:
        return None


def fit_double(ks, r2, tau_max=TAU_MAX):
    from scipy.optimize import curve_fit
    ks = np.asarray(ks, float)
    r2 = np.asarray(r2, float)
    ok = np.isfinite(r2)
    if ok.sum() < 5:
        return None
    ks, r2 = ks[ok], r2[ok]

    def f(k, a1, t1, a2, t2, c):
        return a1 * np.exp(-k / max(t1, 1e-3)) + a2 * np.exp(-k / max(t2, 1e-3)) + c

    best = None
    for a1_0, t1_0, a2_0, t2_0 in [(0.4, 0.8, 0.15, 6.0), (0.5, 1.0, 0.2, 15.0),
                                   (0.3, 0.5, 0.25, 5.0), (0.6, 1.5, 0.1, 30.0)]:
        try:
            p0 = [a1_0, t1_0, a2_0, t2_0, min(r2[-1], 0.0)]
            bounds = ([0, 0.05, 0, 1.0, -2.0], [5, tau_max, 5, tau_max, 1.0])
            popt, _ = curve_fit(f, ks, r2, p0=p0, bounds=bounds, maxfev=30000)
            rss = float(((r2 - f(ks, *popt)) ** 2).sum())
            if best is None or rss < best["rss"]:
                a1, t1, a2, t2, c = popt
                if t1 > t2:
                    a1, a2, t1, t2 = a2, a1, t2, t1
                best = {"a_fast": float(a1), "tau_fast": float(t1),
                        "a_slow": float(a2), "tau_slow": float(t2),
                        "c": float(c), "rss": rss,
                        "censored_fast": bool(t1 < 0.06),
                        "censored_slow": bool(t2 > tau_max * 0.98)}
        except Exception:
            continue
    if best is None:
        return None
    tot = best["a_fast"] + best["a_slow"]
    best["slow_frac"] = float(best["a_slow"] / tot) if abs(tot) > 1e-9 else np.nan
    return best


def tau_simple(z, ks=KS_FAST):
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True) + 1e-6
    w = pca_windows((z - mu) / sd)
    if w is None:
        return None
    prof = r2_curve(w, ks)
    if len(prof) < 3:
        return None
    ks_ok = sorted(prof)
    f = fit_single(ks_ok, [prof[k] for k in ks_ok])
    return f["tau"] if f else None


def load_session(stein: Path, sess: int, units: pd.DataFrame):
    f = stein / f"spikes_session_{sess:03d}.npz"
    if not f.exists():
        return None
    sp = np.load(f)["spikes"].astype(np.float32)
    u = units[units["session"] == sess].reset_index(drop=True)
    if sp.shape[0] != len(u):
        return None
    return sp, u


def to_bins(sp, bin_ms):
    ratio = max(1, int(round(bin_ms / 50)))
    nb = sp.shape[1] // ratio
    return sp[:, :nb * ratio].reshape(sp.shape[0], nb, ratio).sum(axis=2)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp15")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    stein = root / "steinmetz"
    units = pd.read_parquet(stein / "units.parquet")

    print("=" * 82)
    print("Phase 15 — 改进版: 双分辨率 + 放宽截尾 + 50 次置换")
    print(f"  bin {BIN_WIDTHS} ms | τ 上界 {TAU_MAX} | 置换 {N_PERM}")
    print("=" * 82)

    # ============ Part A
    print("\n" + "=" * 82)
    print("Part A — τ_fast 随 bin 缩放（伪影）还是固定在 ms（真实）？")
    print("=" * 82)

    rows = []
    t0 = time.time()

    for sess in sorted(units["session"].unique()):
        got = load_session(stein, sess, units)
        if got is None:
            continue
        sp, u = got

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
                w = pca_windows((z - mu) / sd)
                if w is None:
                    continue
                prof = r2_curve(w, KS_FULL)
                if len(prof) < 5:
                    continue
                ks_ok = sorted(prof)
                y = [prof[k] for k in ks_ok]
                s1 = fit_single(ks_ok, y)
                s2 = fit_double(ks_ok, y)
                if s1 is None or s2 is None:
                    continue
                n = len(ks_ok)
                aic1 = n * np.log(max(s1["rss"], 1e-12) / n) + 6
                aic2 = n * np.log(max(s2["rss"], 1e-12) / n) + 10
                rows.append({
                    "session": int(sess), "region": region, "bin_ms": bw,
                    "n_units": int(len(idx)), "n_win": int(len(w)),
                    "r2_1": round(prof.get(1, float("nan")), 4),
                    "tau_single": round(s1["tau"], 4),
                    "censored_single": bool(s1["censored"]),
                    "tau_fast": round(s2["tau_fast"], 4),
                    "tau_slow": round(s2["tau_slow"], 4),
                    "slow_frac": (round(s2["slow_frac"], 4)
                                  if np.isfinite(s2["slow_frac"]) else None),
                    "censored_fast": bool(s2["censored_fast"]),
                    "censored_slow": bool(s2["censored_slow"]),
                    "delta_aic": round(aic1 - aic2, 3),
                })
        print(f"  sess {sess:02d}: 累计 {len(rows)} 条")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase15_binwidth.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    print("\n" + "=" * 82)
    print("按分辨率汇总")
    print("=" * 82)
    both = df.groupby("bin_ms").agg(
        n=("tau_fast", "size"),
        tau_fast_med=("tau_fast", "median"),
        fast_cens=("censored_fast", "mean"),
        tau_single_med=("tau_single", "median"),
        single_cens=("censored_single", "mean"),
        tau_slow_med=("tau_slow", "median"),
        slow_cens=("censored_slow", "mean"),
        daic_med=("delta_aic", "median"),
        daic_win=("delta_aic", lambda s: float((s > 0).mean())),
    )
    print(both.round(3).to_string())

    ratio_bins = None
    vfast = "无配对"
    try:
        p = df.pivot_table(index=["session", "region"], columns="bin_ms",
                           values=["tau_fast", "tau_single"])
        m = pd.DataFrame({"tf50": p[("tau_fast", 50)],
                          "tf100": p[("tau_fast", 100)]}).dropna()
        print(f"\n  配对样本 (τ_fast): {len(m)}")
        if len(m) > 10:
            ratio_bins = float((m["tf50"] / m["tf100"]).median())
            print(f"  τ_fast(50ms)/τ_fast(100ms) 中位 = {ratio_bins:.3f}")
            print("    · τ 固定为某个 ms 值 → 比值 ≈ 2.00")
            print("    · τ 固定为某个 bin 数 → 比值 ≈ 1.00")
            if 1.7 <= ratio_bins <= 2.3:
                vfast = "✅ τ_fast 固定在 ms 上 → 真实时间常数（约 37ms）"
            elif 0.85 <= ratio_bins <= 1.15:
                vfast = "❌ τ_fast 固定在 bin 数上 → 分辨率伪影"
            else:
                vfast = f"⚠️ 比值 {ratio_bins:.2f} 介于两者之间，需更多分辨率点"
            print(f"  判定: {vfast}")

        # τ_single 也做同样检验（它应该接近真实，因为量级 >1 bin）
        m2 = pd.DataFrame({"t50": p[("tau_single", 50)],
                           "t100": p[("tau_single", 100)]}).dropna()
        if len(m2) > 10:
            rb2 = float((m2["t50"] / m2["t100"]).median())
            print(f"\n  对照 τ_single 比值 = {rb2:.3f} "
                  f"({'固定在 ms ✅' if 1.7 <= rb2 <= 2.3 else '固定在 bins（同 τ_fast 的伪影模式）' if rb2 < 1.15 else '中间'})")
    except Exception as e:
        print(f"  配对失败: {e}")

    print("\n" + "=" * 82)
    print("改进 2 验证: 上界 200 后截尾比例（exp14 时为 27.3%，上界 60）")
    print("=" * 82)
    for bw in BIN_WIDTHS:
        s = df[df.bin_ms == bw]
        print(f"  {bw:3d}ms: τ_single {s['censored_single'].mean()*100:5.1f}% | "
              f"τ_slow {s['censored_slow'].mean()*100:5.1f}% | "
              f"τ_fast 撞下界 {s['censored_fast'].mean()*100:5.1f}%")

    # ============ Part B
    print("\n" + "=" * 82)
    print(f"Part B — {N_PERM} 次置换的伪区域检验（100ms）")
    print("=" * 82)

    rc = units.groupby("session")["region"].nunique()
    sessions = list(rc[rc >= 4].sort_values(ascending=False).index[:N_SESSIONS])

    perm_rows = []
    for sess in sessions:
        got = load_session(stein, sess, units)
        if got is None:
            continue
        sp, u = got
        x = to_bins(sp, 100)

        groups = []
        for region, sub in u.groupby("region"):
            idx = sub["unit_idx"].to_numpy()
            idx = idx[(idx >= 0) & (idx < x.shape[0])]
            if len(idx) >= MIN_UNITS:
                groups.append((region, idx))
        if len(groups) < 3:
            continue

        obs = [t for t in (tau_simple(x[idx]) for _, idx in groups) if t is not None]
        if len(obs) < 3:
            continue
        obs_sd = float(np.std(obs))

        sizes = [len(idx) for _, idx in groups]
        all_idx = np.arange(x.shape[0])
        null_sds = []
        for _ in range(N_PERM):
            perm = rng.permutation(all_idx)
            nt = []
            pos = 0
            for sz in sizes:
                t = tau_simple(x[perm[pos:pos + sz]])
                pos += sz
                if t is not None:
                    nt.append(t)
            if len(nt) >= 3:
                null_sds.append(float(np.std(nt)))

        if null_sds:
            nm = float(np.median(null_sds))
            pval = float((np.array(null_sds) >= obs_sd).mean())
            perm_rows.append({"session": int(sess), "n_regions": len(obs),
                              "obs_sd": round(obs_sd, 4), "null_sd": round(nm, 4),
                              "ratio": round(obs_sd / max(nm, 1e-9), 3), "p": pval})
            print(f"  sess {sess:02d}: {len(obs)} 区域 | SD {obs_sd:.3f} vs {nm:.3f} "
                  f"| 比值 {obs_sd/max(nm,1e-9):.2f} | p={pval:.3f}")

    pdf = pd.DataFrame(perm_rows)
    pdf.to_csv(out / "phase15_perm.csv", index=False)

    med_ratio, n_sig, n_sig2, v_reg = None, None, None, "无结果"
    if len(pdf):
        med_ratio = float(pdf["ratio"].median())
        n_sig = int((pdf["p"] < 0.05).sum())
        n_sig2 = int((pdf["p"] < 0.02).sum())
        print("\n" + "-" * 82)
        print(f"  比值中位 {med_ratio:.3f} | p<0.05: {n_sig}/{len(pdf)} | "
              f"p<0.02: {n_sig2}/{len(pdf)}")
        if med_ratio > 1.4 and n_sig >= len(pdf) * 0.5:
            v_reg = "✅ 区域结构清晰（多数 session 显著）"
        elif med_ratio > 1.3:
            v_reg = f"⚠️ 弱-中等（比值 {med_ratio:.2f}，{n_sig}/{len(pdf)} 显著）"
        else:
            v_reg = "❌ 区域标签不优于随机分组"
        print(f"  判定: {v_reg}")

    (out / "phase15_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "bin_widths": BIN_WIDTHS, "tau_max": TAU_MAX, "n_perm": N_PERM,
        "by_binwidth": {str(int(k)): {c: (None if pd.isna(v) else round(float(v), 4))
                                      for c, v in row.items()}
                        for k, row in both.iterrows()},
        "tau_fast_scaling_ratio": (round(ratio_bins, 3) if ratio_bins else None),
        "tau_fast_verdict": vfast,
        "partB": {"median_ratio": (round(med_ratio, 3) if med_ratio else None),
                  "n_sig_005": n_sig, "n_sig_002": n_sig2,
                  "n_sessions": int(len(pdf)), "verdict": v_reg},
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
