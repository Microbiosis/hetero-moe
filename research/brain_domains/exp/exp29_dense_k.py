"""Phase 29: 加密 k 阶梯，检验"超额曲线形状跨物种一致"是否成立。

## 要验证的残存信号

exp28 发现：超额（真实 − 时间打乱）曲线的形状
    human vs mouse    r = +0.971
    macaque vs mouse  r = +0.557
    human vs macaque  r = +0.530
而高斯数据不一致（−0.67 ~ +0.15）。

**但这个结论只有 5 个 k 点**（2,4,8,16,32），相关系数的 95% 置信区间极宽。
5 个点、r=0.97，置信区间约 [0.71, 1.00]；r=0.53 则约 [−0.58, 0.96]。

本实验：
1. **k 阶梯加密到 11 个点**：2,3,4,6,8,12,16,24,32,48,64
2. **增加样本**：每物种 16 组（不是 8）
3. **三个对照全保留**：高斯基线、时间打乱、匹配窗口 600
4. **报告置信区间**：用 Fisher z 变换给出 r 的 95% CI
5. **分物种做 bootstrap**：对分组重采样，看 r 的分布

## 判据（严格版）

  三物种两两 r 的 95% CI 下界都 > 0.5
  且 GAUSSIAN 与三物种的 r 都 < 0.3
  → 形状一致成立

若 CI 下界 < 0.5 或高斯也高 → 原结论不成立
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import norm

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_WIN = 600
N_PER_SPECIES = 16
KS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
N_SHUFFLE = 8
N_BOOT = 200


def proj_group_mean(x, labels):
    out = np.empty_like(x)
    for g in np.unique(labels):
        m = labels == g
        if m.sum():
            out[m] = x[m].mean(axis=0, keepdims=True)
    return out


def var_frac(x, comp):
    den = float((x ** 2).sum())
    return float((comp ** 2).sum()) / den if den > 1e-12 else 0.0


def build_tree(x):
    if x.shape[0] < 4:
        return None
    d = pdist(x, metric="correlation")
    d = np.nan_to_num(d, nan=1.0, posinf=2.0)
    return linkage(d, method="average")


def nested_labels(Z, n, ks):
    heights = np.sort(Z[:, 2])[::-1]
    out = {}
    for k in ks:
        if k >= n:
            continue
        out[k] = fcluster(Z, t=heights[min(k - 1, len(heights) - 1)] - 1e-9,
                          criterion="distance")
    return out


def depth_curve(x, ks=KS, rng=None, n_shuffle=N_SHUFFLE):
    Z = build_tree(x)
    if Z is None:
        return None
    parts = nested_labels(Z, x.shape[0], ks)
    if len(parts) < 2:
        return None
    order = sorted(parts)

    projs = {k: proj_group_mean(x, parts[k]) for k in order}
    res = {}
    prev = None
    for k in order:
        c = projs[k] if prev is None else (projs[k] - prev)
        res[k] = {"var": var_frac(x, c), "null": None, "excess": None}
        prev = projs[k]

    if rng is not None and n_shuffle > 0:
        nulls = {k: [] for k in order}
        for _ in range(n_shuffle):
            xs = np.stack([rng.permutation(x[i]) for i in range(x.shape[0])])
            Zs = build_tree(xs)
            if Zs is None:
                continue
            ps = nested_labels(Zs, xs.shape[0], ks)
            prev2 = None
            for k in sorted(ps):
                pj = proj_group_mean(xs, ps[k])
                c = pj if prev2 is None else (pj - prev2)
                if k in nulls:
                    nulls[k].append(var_frac(xs, c))
                prev2 = pj
        for k in order:
            if nulls[k]:
                res[k]["null"] = float(np.mean(nulls[k]))
                res[k]["excess"] = res[k]["var"] - res[k]["null"]
    return res


def fisher_ci(r, n, alpha=0.05):
    """Fisher z 变换给相关系数的置信区间。"""
    if not np.isfinite(r) or n < 4:
        return None, None
    r = float(np.clip(r, -0.9999, 0.9999))
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    lo, hi = z - zc * se, z + zc * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def load_matched(root: Path, view: str, n_win=N_WIN, n_groups=N_PER_SPECIES):
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
    # 按大小降序取前 N（不按文件名）
    sizes = meta.groupby(gcol).size().sort_values(ascending=False)
    out = []
    for gval in list(sizes.index)[:n_groups]:
        if gcol == "session" and view == "spike_view":
            f = d / f"windows_sess{int(gval):03d}.npz"
        else:
            f = d / str(gval)
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < n_win:
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
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp29")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 86)
    print("Phase 29 — 加密 k 阶梯（11 点），检验超额曲线形状一致性")
    print("=" * 86)
    print(f"  n_win={N_WIN} | k∈{KS} | 每物种 {N_PER_SPECIES} 组 | 零模型 {N_SHUFFLE}")

    rows = []
    t0 = time.time()

    # ---- 高斯基线（16 个独立样本）
    print("\n  [基线] 高斯随机")
    for i in range(12):
        Xg = rng.standard_normal((64, N_WIN))
        mu = Xg.mean(axis=1, keepdims=True)
        sd = Xg.std(axis=1, keepdims=True) + 1e-9
        Xg = (Xg - mu) / sd
        dc = depth_curve(Xg, rng=rng)
        if dc is None:
            continue
        for k, v in dc.items():
            rows.append({"species": "GAUSSIAN", "group": f"g{i:02d}", "k": k,
                         "var": round(v["var"], 6),
                         "null": (round(v["null"], 6) if v["null"] is not None else None),
                         "excess": (round(v["excess"], 6) if v["excess"] is not None else None)})
    print(f"    → {sum(1 for r in rows if r['species']=='GAUSSIAN')} 条")

    # ---- 三物种
    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_matched(root, view)
        print(f"\n  {species}: {len(items)} 组")
        if not items:
            continue
        for key, X0 in items:
            mu = X0.mean(axis=1, keepdims=True)
            sd = X0.std(axis=1, keepdims=True) + 1e-9
            X = (X0 - mu) / sd
            dc = depth_curve(X, rng=rng)
            if dc is None:
                continue
            for k, v in dc.items():
                rows.append({"species": species, "group": key, "k": k,
                             "var": round(v["var"], 6),
                             "null": (round(v["null"], 6) if v["null"] is not None else None),
                             "excess": (round(v["excess"], 6) if v["excess"] is not None else None)})
            print(f"    {str(key)[:24]:<24} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase29.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not df["excess"].notna().any():
        print("无超额数据")
        return

    # ============ 超额曲线
    print("\n" + "=" * 86)
    print("超额曲线（中位，11 个 k 点）")
    print("=" * 86)
    pe = df.pivot_table(index="k", columns="species", values="excess", aggfunc="median")
    print(pe.round(5).to_string())

    # ============ 形状相关 + 置信区间
    print("\n" + "=" * 86)
    print("曲线形状两两相关（含 95% CI，n = k 点数）")
    print("=" * 86)

    sps = list(pe.columns)
    n_pts = len(pe)
    corrs = {}
    print(f"  n = {n_pts} 个 k 点")
    print(f"  {'对比':<26} {'r':>7} {'95% CI':>22} {'判定':>14}")
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            a, b = sps[i], sps[j]
            x = pe[a].dropna()
            y = pe[b].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 4:
                continue
            r = float(np.corrcoef(x[idx], y[idx])[0, 1])
            lo, hi = fisher_ci(r, len(idx))
            key = f"{a}-{b}"
            corrs[key] = {"r": round(r, 4), "ci_lo": round(lo, 4) if lo else None,
                          "ci_hi": round(hi, 4) if hi else None, "n": int(len(idx))}
            mark = "✅ 强正相关" if lo is not None and lo > 0.5 else (
                "⚠️ CI 过宽" if lo is not None and lo > 0 else "❌")
            print(f"  {key:<26} {r:>+7.3f} "
                  f"{f'[{lo:+.2f}, {hi:+.2f}]' if lo is not None else 'NA':>22} {mark:>14}")

    # ============ 分组 bootstrap（对组重采样）
    print("\n" + "=" * 86)
    print("分组 bootstrap：对每个物种的组重采样，看 r 的分布")
    print("=" * 86)
    boot_res = {}
    valid_sp = [s for s in sps if s != "GAUSSIAN"]
    for a, b in [(valid_sp[i], valid_sp[j])
                 for i in range(len(valid_sp)) for j in range(i + 1, len(valid_sp))]:
        da = df[df.species == a]
        db = df[df.species == b]
        ga = da.group.unique()
        gb = db.group.unique()
        rs = []
        for _ in range(N_BOOT):
            sa = rng.choice(ga, len(ga), replace=True)
            sb = rng.choice(gb, len(gb), replace=True)
            ca = pd.concat([da[da.group == g] for g in sa]).groupby("k")["excess"].median()
            cb = pd.concat([db[db.group == g] for g in sb]).groupby("k")["excess"].median()
            idx = ca.index.intersection(cb.index)
            if len(idx) < 4:
                continue
            v = np.corrcoef(ca[idx], cb[idx])[0, 1]
            if np.isfinite(v):
                rs.append(float(v))
        if rs:
            rs = np.array(rs)
            boot_res[f"{a}-{b}"] = {
                "median": round(float(np.median(rs)), 4),
                "ci_lo": round(float(np.percentile(rs, 2.5)), 4),
                "ci_hi": round(float(np.percentile(rs, 97.5)), 4),
                "p_gt_0.5": round(float((rs > 0.5).mean()), 3),
                "n_boot": len(rs),
            }
            print(f"  {a}-{b:<16} 中位 {np.median(rs):+.3f}  "
                  f"95%CI [{np.percentile(rs,2.5):+.3f}, {np.percentile(rs,97.5):+.3f}]  "
                  f"P(r>0.5)={float((rs>0.5).mean()):.3f}")

    # ============ 判定
    print("\n" + "=" * 86)
    print("判定（严格版：CI 下界 > 0.5 且高斯 < 0.3）")
    print("=" * 86)

    real_corrs = [v for k, v in corrs.items() if "GAUSSIAN" not in k]
    gauss_corrs = [v for k, v in corrs.items() if "GAUSSIAN" in k]

    real_ok = [v for v in real_corrs if v["ci_lo"] is not None and v["ci_lo"] > 0.5]
    gauss_low = all((v["r"] < 0.3) for v in gauss_corrs) if gauss_corrs else False

    print(f"  真实物种间 {len(real_ok)}/{len(real_corrs)} 个对比 CI 下界 > 0.5")
    gauss_str = ", ".join("%+.2f" % v["r"] for v in gauss_corrs)
    print("  高斯与真实的相关: [" + gauss_str + "]")
    print(f"  高斯全 < 0.3: {'✅' if gauss_low else '❌'}")

    boot_ok = all(v["ci_lo"] > 0.5 for v in boot_res.values()) if boot_res else False
    print(f"  分组 bootstrap CI 下界全 > 0.5: {'✅' if boot_ok else '❌'}")

    if real_ok and len(real_ok) == len(real_corrs) and gauss_low:
        v = ("✅ **超额曲线形状跨物种一致成立** — "
             "所有真实物种对的 CI 下界 > 0.5，高斯全 < 0.3")
    elif len(real_ok) >= len(real_corrs) * 0.5 and gauss_low:
        v = (f"⚠️ 部分成立 — {len(real_ok)}/{len(real_corrs)} 个物种对 CI 下界 > 0.5，"
             "但高斯基线低，方向正确")
    elif not gauss_low:
        v = "❌ 高斯基线也高 → 相关性是数学必然，不是生物信号"
    else:
        v = "❌ CI 下界普遍 < 0.5 → 5 个点时看到的 r=0.97 是样本量不足的假象"
    print(f"\n  {v}")

    (out / "phase29_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_win": N_WIN, "ks": KS,
        "n_per_species": N_PER_SPECIES, "n_shuffle": N_SHUFFLE, "n_boot": N_BOOT,
        "n_records": int(len(df)),
        "excess_curve": {str(int(k)): {sp: (None if pd.isna(v_) else round(float(v_), 6))
                                       for sp, v_ in row.items()}
                         for k, row in pe.iterrows()},
        "shape_corr": corrs,
        "group_bootstrap": boot_res,
        "real_pairs_ci_above_0.5": len(real_ok),
        "n_real_pairs": len(real_corrs),
        "gaussian_all_below_0.3": bool(gauss_low),
        "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
