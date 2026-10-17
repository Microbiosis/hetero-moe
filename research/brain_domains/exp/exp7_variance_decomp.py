"""Phase 7: 方差分解 —— 物种真的是主导因子吗?

## 要回答的两个问题

### Q1. 组内方差 vs 组间方差
exp6c 发现"形状统计量守恒、尺度统计量不同"。但还有个更基本的问题:

    同一个物种内不同 session 的差异，是否大于物种之间的差异?

若**组内 > 组间**，则"物种"不是有意义的分类维度 —— 个体/记录条件的影响更大。
这是很多跨物种研究的隐含陷阱: 只比较了物种均值, 没看组内离散度。

### Q2. 尺度统计量的差异能否由 recording 规模解释
exp6c 显示 eff_dim / ac1 / spec_slope 物种间差异大。
但两者 unit 数零重叠 (小鼠 205-1240, 猕猴 40-192)。

检验: 用**回归**把统计量对 log(n_units) 做校正, 再看残差的物种差异。
  若校正后差异大幅缩小 → 规模解释
  若校正后仍大       → 物种解释

## 方法

Q1: 方差分解
    对每个统计量, 计算:
      SS_between = Σ n_g (mean_g - mean_all)²
      SS_within  = Σ Σ (x_ig - mean_g)²
      ω² = SS_between / (SS_between + SS_within)    ← 组间方差占比
    再与"随机打乱物种标签"的 ω² 比较（置换检验, 1000 次）

Q2: 回归校正
    对每位点: stat ~ a + b·log10(n_units)
    取残差, 再算物种间 ε²

## 判据

  ω² < 0.1  → 物种几乎不解释方差, 组内变异主导
  ω² > 0.3  → 物种是主导因子

  Q2 校正后 ε² 降幅 > 60% → 原是规模伪影
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_PERM = 1000


def omega_sq(df, stat, group_col="species"):
    """组间方差占比 ω²。"""
    s = df[[group_col, stat]].dropna()
    if len(s) < 5:
        return None
    grand = s[stat].mean()
    ss_b = (s.groupby(group_col)[stat].agg(["mean", "size"])
            .assign(ss=lambda d: d["size"] * (d["mean"] - grand) ** 2)["ss"].sum())
    ss_w = ((s[stat] - s.groupby(group_col)[stat].transform("mean")) ** 2).sum()
    tot = ss_b + ss_w
    return float(ss_b / tot) if tot > 0 else 0.0


def perm_test(df, stat, group_col="species", n_perm=N_PERM, seed=SEED):
    """置换检验: 打乱物种标签后 ω² 的分布。"""
    rng = np.random.default_rng(seed)
    s = df[[group_col, stat]].dropna().reset_index(drop=True)
    obs = omega_sq(s, stat, group_col)
    if obs is None:
        return None, None
    null = []
    vals = s[stat].to_numpy()
    for _ in range(n_perm):
        tmp = s.copy()
        tmp[stat] = rng.permutation(vals)
        v = omega_sq(tmp, stat, group_col)
        if v is not None:
            null.append(v)
    null = np.array(null)
    p = float((null >= obs).mean())
    return obs, p


def residualize(df, stat, cov="n_units"):
    """对 log10(cov) 做线性回归, 返回残差（保留原索引）。"""
    s = df[[stat, cov]].copy()
    mask = s[[stat, cov]].notna().all(axis=1) & (s[cov] > 0)
    out = pd.Series(np.nan, index=df.index)
    if mask.sum() < 5:
        return out
    x = np.log10(s.loc[mask, cov].to_numpy(dtype=float))
    y = s.loc[mask, stat].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return out
    out.loc[mask] = y - X @ beta
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp7")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase 7 — 方差分解: 物种是主导因子吗?")
    print("=" * 78)

    # 复用 exp6 的逐分组统计（已存盘）
    f6 = Path("/out/exp6/phase6_per_group.csv")
    if f6.exists():
        long_df = pd.read_csv(f6)
        print(f"  载入 exp6 结果 {len(long_df)} 行")
    else:
        print("  ❌ 缺少 exp6 输出，无法继续")
        return

    # 转宽表: 每行一个分组，列是各统计量
    #
    # ⚠️ 不能把 "file" 放进 index —— 小鼠的 file 是 NaN（exp6 里只有 macaque 有 file），
    #    pivot_table 会直接丢掉这些行，导致只剩猕猴。改为先合成一个稳定 key。
    long_df = long_df.copy()
    long_df["gk"] = (long_df["species"].astype(str) + "|" +
                     long_df["group"].astype(str) + "|" +
                     long_df["file"].fillna("-").astype(str))
    wide = long_df.pivot_table(index=["species", "view", "gk"],
                               columns="stat", values="median").reset_index()
    print(f"  转宽表后: {len(wide)} 行")
    print(wide.groupby("species").size().to_string())

    # 附上 n_units
    # n_units: 小鼠按 session 数单元, 猕猴按 file 取
    wide["n_units"] = np.nan
    stein_units = Path("/data/steinmetz/units.parquet")
    if stein_units.exists():
        u = pd.read_parquet(stein_units)
        cnt = u.groupby("session").size().to_dict()
        is_mouse = wide["species"] == "mouse"
        # gk 形如 "mouse|12|-" → 取中间那段做 session
        sess = wide.loc[is_mouse, "gk"].str.split("|").str[1].astype(int)
        wide.loc[is_mouse, "n_units"] = sess.map(lambda s_: cnt.get(s_, np.nan)).values
    mac_meta = root / "spike_view_macaque" / "meta.parquet"
    if mac_meta.exists():
        mm = pd.read_parquet(mac_meta)
        mcnt = mm.groupby("file")["n_units"].first().to_dict()
        is_mac = wide["species"] == "macaque"
        fname = wide.loc[is_mac, "gk"].str.split("|").str[2]
        wide.loc[is_mac, "n_units"] = fname.map(lambda f_: mcnt.get(f_, np.nan)).values
    print(f"  n_units 覆盖: {wide['n_units'].notna().sum()}/{len(wide)}")

    ST = ["ac1", "ac_ratio", "spec_slope", "hf_ratio", "eff_dim", "kurtosis"]
    print(f"\n  分组总数: {len(wide)}")
    print(wide.groupby("species").size().to_string())

    # ============ Q1: 方差分解（仅 spike，控制模态）============
    spike = wide[wide["species"].isin(["mouse", "macaque"])]
    print("\n" + "=" * 78)
    print("Q1. 组内方差 vs 组间方差（仅 spike: 小鼠 vs 猕猴）")
    print("=" * 78)
    print(f"{'统计量':<14} {'ω²':>8} {'置换p':>9} {'组内SD':>10} {'组间差':>10} {'判定':>16}")

    q1 = {}
    for st in ST:
        if st not in spike.columns:
            continue
        w2, p = perm_test(spike, st)
        if w2 is None:
            continue
        # 组内 SD（合并）与组间均值差
        g = spike[[ "species", st]].dropna()
        sd_in = g.groupby("species")[st].std().mean()
        means = g.groupby("species")[st].mean()
        gap = float(means.max() - means.min()) if len(means) >= 2 else float("nan")
        ratio = gap / sd_in if sd_in > 1e-9 else float("inf")
        if w2 < 0.10:
            v = "组内主导"
        elif w2 > 0.30:
            v = "✅ 物种主导"
        else:
            v = "混合"
        q1[st] = {"omega2": round(w2, 4), "perm_p": p,
                  "within_sd": round(float(sd_in), 4),
                  "between_gap": round(gap, 4),
                  "gap_over_sd": round(float(ratio), 2), "verdict": v}
        print(f"{st:<14} {w2:>8.4f} {p:>9.3f} {sd_in:>10.4f} {gap:>10.4f} {v:>16}")

    # ============ Q2: 规模校正 ============
    print("\n" + "=" * 78)
    print("Q2. 用 log10(n_units) 校正后，物种差异还剩多少?")
    print("=" * 78)
    print(f"{'统计量':<14} {'原始ε²':>9} {'校正ε²':>9} {'降幅':>8} {'判定':>20}")

    q2 = {}
    for st in ST:
        if st not in spike.columns:
            continue
        s = spike[["species", st, "n_units"]].dropna()
        if len(s) < 10 or s["n_units"].nunique() < 3:
            continue
        # 原始
        g = [s[s.species == sp][st].to_numpy() for sp in s.species.unique()]
        g = [x for x in g if len(x) >= 3]
        if len(g) < 2:
            continue
        H0, p0 = kruskal(*g)
        n = sum(len(x) for x in g); k = len(g)
        e0 = max(0.0, (H0 - k + 1) / (n - k))
        # 校正
        res = residualize(spike, st, "n_units")
        s2 = spike[[ "species"]].copy()
        s2["r"] = res
        s2 = s2.dropna()
        g2 = [s2[s2.species == sp]["r"].to_numpy() for sp in s2.species.unique()]
        g2 = [x for x in g2 if len(x) >= 3]
        if len(g2) < 2:
            continue
        H1, p1 = kruskal(*g2)
        n2 = sum(len(x) for x in g2); k2 = len(g2)
        e1 = max(0.0, (H1 - k2 + 1) / (n2 - k2))
        drop = (e0 - e1) / e0 * 100 if e0 > 1e-9 else 0.0
        if drop > 60:
            v = "✅ 规模伪影为主"
        elif drop < 20:
            v = "物种效应为主"
        else:
            v = "混合"
        q2[st] = {"eps2_raw": round(e0, 4), "eps2_adjusted": round(e1, 4),
                  "drop_pct": round(float(drop), 1), "verdict": v}
        print(f"{st:<14} {e0:>9.4f} {e1:>9.4f} {drop:>7.1f}% {v:>20}")

    # ============ 综合判读 ============
    print("\n" + "=" * 78)
    print("综合判读")
    print("=" * 78)
    conserved = [s for s, v in q1.items() if v["omega2"] < 0.10]
    dominated = [s for s, v in q1.items() if v["omega2"] > 0.30]
    scale_artifact = [s for s, v in q2.items() if v["drop_pct"] > 60]
    species_real = [s for s, v in q2.items() if v["drop_pct"] < 20 and v["eps2_raw"] > 0.14]

    print(f"  组内主导 (ω²<0.10): {conserved or '无'}")
    print(f"  物种主导 (ω²>0.30): {dominated or '无'}")
    print(f"  规模伪影 (校正降幅>60%): {scale_artifact or '无'}")
    print(f"  物种效应真实 (校正后仍>0.14): {species_real or '无'}")

    (out / "phase7_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_perm": N_PERM,
        "n_groups": int(len(wide)),
        "q1_variance_decomposition": q1,
        "q2_scale_adjustment": q2,
        "summary": {"within_dominated": conserved, "species_dominated": dominated,
                    "scale_artifacts": scale_artifact, "true_species_effects": species_real},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  结果 → {out}/phase7_report.json")


if __name__ == "__main__":
    main()
