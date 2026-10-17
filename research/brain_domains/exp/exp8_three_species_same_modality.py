"""Phase 8: 三物种同模态对照 —— 解决 Phase 6 的模态混淆。

## 这是什么

Phase 6 的三物种比较混入了模态差异（人类是头皮 EEG）。
Phase 8 用**人类侵入式**记录（h1/h2）替换头皮 EEG，使三物种全部为 spike。

此时物种差异才可解读。

## 关键优势: 单元数部分重叠

  猕猴        40 – 192 units
  人类 h1/h2 176 – 192 units    ← 重叠
  小鼠       205 – 1240 units   ← 需取低端子集

因此:
  - 猕猴 vs 人类：**直接可比**（规模相当）
  - 小鼠 vs 人类：取小鼠 205–250 units 的 session 做匹配子集

## 三个层次的分析

### L1. 全量比较（所有分组）
三物种，七统计量，Kruskal-Wallis + ε² + 置换检验。

### L2. 规模匹配子集
只用单元数落在 [170, 210] 的分组：
  猕猴：部分片段
  人类：全部
  小鼠：低端子集
这才是真正同规模对比。

### L3. 方差分解
组内 vs 组间，判断"物种"是否是有意义的分类维度。

## 判据

  若 L2 与 L1 结论一致 → 规模不是主导因素，物种效应可信
  若 L2 中差异大幅缩小 → 原差异是规模伪影
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
N_PERM = 500
MAX_WIN_PER_GROUP = 600
STATS = ["ac1", "ac_ratio", "spec_slope", "hf_ratio", "eff_dim", "kurtosis"]


# ------------------------------------------------- 统计量（与 exp6 一致）
def shape_stats(w):
    x = w.astype(np.float64)
    n, C, T = x.shape
    a1, b1 = x[:, :, :-1], x[:, :, 1:]
    a2, b2 = x[:, :, :-2], x[:, :, 2:]

    def ac(a, b):
        num = (a * b).sum(axis=2)
        den = np.sqrt((a ** 2).sum(axis=2) * (b ** 2).sum(axis=2)) + 1e-12
        return np.median(num / den, axis=1)

    r1, r2 = ac(a1, b1), ac(a2, b2)

    spec = np.abs(np.fft.rfft(x, axis=2)) ** 2
    F = spec.shape[2]
    freq = np.fft.rfftfreq(T, d=1.0)[1:]
    sp = spec[:, :, 1:] + 1e-12
    lf = np.log(freq)[None, None, :]
    lp = np.log(sp)
    lfc = lf - lf.mean(axis=2, keepdims=True)
    slope = (lfc * (lp - lp.mean(axis=2, keepdims=True))).sum(axis=2) / (lfc ** 2).sum(axis=2)
    hi = spec[:, :, F // 2:].sum(axis=2)

    xc = x - x.mean(axis=2, keepdims=True)
    try:
        s = np.linalg.svd(xc, compute_uv=False)
        lam = s ** 2
        ed = (lam.sum(axis=1) ** 2) / ((lam ** 2).sum(axis=1) + 1e-12)
    except np.linalg.LinAlgError:
        ed = np.full(n, np.nan)

    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-12
    kurt = np.median((((x - mu) / sd) ** 4).mean(axis=2) - 3.0, axis=1)

    return {
        "ac1": float(np.nanmedian(r1)),
        "ac_ratio": float(np.nanmedian(r2 / np.maximum(np.abs(r1), 0.05))),
        "spec_slope": float(np.nanmedian(slope)),
        "hf_ratio": float(np.nanmedian(hi / (spec.sum(axis=2) + 1e-12))),
        "eff_dim": float(np.nanmedian(ed)),
        "kurtosis": float(np.nanmedian(kurt)),
    }


def collect(root: Path):
    """三物种（全部 spike）逐分组统计。"""
    rng = np.random.default_rng(SEED)
    rows = []

    specs = [
        ("mouse", root / "spike_view", "session", None),
        ("macaque", root / "spike_view_macaque", "file", "n_units"),
        ("human", root / "spike_view_human", "file", "n_units"),
    ]

    for species, d, gcol, ucol in specs:
        if not d.exists():
            print(f"  [skip] {species}: {d} 不存在")
            continue
        meta = pd.read_parquet(d / "meta.parquet")
        print(f"  {species}: {len(meta):,} 窗, 分组 {meta[gcol].nunique()}")

        for gval, sub in meta.groupby(gcol):
            if ucol and ucol in sub.columns:
                n_u = int(sub[ucol].iloc[0])
                fname = gval if gcol == "file" else None
            else:
                # 小鼠: 从 units.parquet 取
                fname = None
                n_u = None

            if gcol == "session":
                f = d / f"windows_sess{int(gval):03d}.npz"
                if not f.exists():
                    continue
                w = np.load(f)["windows"]
            else:
                f = d / str(gval)
                if not f.exists():
                    continue
                w = np.load(f)["windows"]

            if len(w) < 100:
                continue
            if len(w) > MAX_WIN_PER_GROUP:
                idx = rng.choice(len(w), MAX_WIN_PER_GROUP, replace=False)
                w = w[np.sort(idx)]

            st = shape_stats(w)
            rows.append({"species": species, "group": f"{species}|{gval}",
                         "n_units": n_u, **st})
        print(f"    → {sum(1 for r in rows if r['species']==species)} 组")

    df = pd.DataFrame(rows)

    # 补小鼠的 n_units
    stein = Path("/data/steinmetz/units.parquet")
    if stein.exists():
        u = pd.read_parquet(stein)
        cnt = u.groupby("session").size().to_dict()
        is_m = df["species"] == "mouse"
        sess = df.loc[is_m, "group"].str.split("|").str[1].astype(int)
        df.loc[is_m, "n_units"] = sess.map(lambda s_: cnt.get(s_, np.nan)).values

    return df


def eps2_and_perm(df, stat, n_perm=N_PERM, seed=SEED):
    s = df[["species", stat]].dropna()
    groups = [s[s.species == sp][stat].to_numpy() for sp in s.species.unique()]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return None, None, None
    H, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    e2 = max(0.0, (H - k + 1) / (n - k))

    rng = np.random.default_rng(seed)
    vals = s[stat].to_numpy()
    null = []
    for _ in range(n_perm):
        tmp = s.copy()
        tmp[stat] = rng.permutation(vals)
        g2 = [tmp[tmp.species == sp][stat].to_numpy() for sp in tmp.species.unique()]
        g2 = [g for g in g2 if len(g) >= 3]
        if len(g2) < 2:
            continue
        H2, _ = kruskal(*g2)
        n2 = sum(len(g) for g in g2)
        null.append(max(0.0, (H2 - len(g2) + 1) / (n2 - len(g2))))
    p_perm = float((np.array(null) >= e2).mean()) if null else None
    return float(e2), float(p), p_perm


def omega_sq(df, stat):
    s = df[["species", stat]].dropna()
    if len(s) < 5:
        return None
    grand = s[stat].mean()
    ss_b = (s.groupby("species")[stat].agg(["mean", "size"])
            .assign(ss=lambda d: d["size"] * (d["mean"] - grand) ** 2)["ss"].sum())
    ss_w = ((s[stat] - s.groupby("species")[stat].transform("mean")) ** 2).sum()
    tot = ss_b + ss_w
    return float(ss_b / tot) if tot > 0 else 0.0


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp8")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 8 — 三物种同模态对照（全部侵入式 spike）")
    print("=" * 80)

    t0 = time.time()
    df = collect(root)
    df.to_csv(out / "phase8_groups.csv", index=False)

    print(f"\n  分组 {len(df)}")
    print(df.groupby("species").agg(
        n=("ac1", "size"), units_med=("n_units", "median"),
        units_min=("n_units", "min"), units_max=("n_units", "max")).to_string())

    # ===== L1: 全量比较
    print("\n" + "=" * 80)
    print("L1. 全量比较（三物种，全部 spike）")
    print("=" * 80)
    print(f"{'统计量':<13} {'ε²':>8} {'置换p':>9} {'m(mouse)':>10} {'m(mac)':>9} {'m(human)':>10} {'ω²':>7}")

    L1 = {}
    for st in STATS:
        e2, p, pp = eps2_and_perm(df, st)
        if e2 is None:
            continue
        med = df.groupby("species")[st].median().to_dict()
        w2 = omega_sq(df, st)
        L1[st] = {"eps2": round(e2, 4), "p": p, "perm_p": pp,
                  "median": {k_: round(float(v_), 4) for k_, v_ in med.items()},
                  "omega2": round(w2, 4) if w2 is not None else None}
        pp_s = f"{pp:.3f}" if pp is not None else "NA"
        print(f"{st:<13} {e2:>8.4f} {pp_s:>9} "
              f"{med.get('mouse', float('nan')):>10.4f} "
              f"{med.get('macaque', float('nan')):>9.4f} "
              f"{med.get('human', float('nan')):>10.4f} "
              f"{(w2 if w2 is not None else float('nan')):>7.3f}")

    # ===== L2: 规模匹配子集
    print("\n" + "=" * 80)
    print("L2. 规模匹配子集（单元数 170–210）")
    print("=" * 80)
    sub = df[(df["n_units"] >= 170) & (df["n_units"] <= 210)]
    print(f"  匹配后分组数: {len(sub)}")
    print(sub.groupby("species").size().to_string())

    L2 = {}
    if len(sub) > 0 and sub["species"].nunique() >= 2:
        print(f"\n{'统计量':<13} {'ε²':>8} {'置换p':>9} {'mouse':>9} {'macaque':>9} {'human':>9}")
        for st in STATS:
            e2, p, pp = eps2_and_perm(sub, st)
            if e2 is None:
                continue
            med = sub.groupby("species")[st].median().to_dict()
            L2[st] = {"eps2": round(e2, 4), "p": p, "perm_p": pp,
                      "median": {k_: round(float(v_), 4) for k_, v_ in med.items()}}
            print(f"{st:<13} {e2:>8.4f} {(pp if pp is not None else float('nan')):>9.3f} "
                  f"{med.get('mouse', float('nan')):>9.4f} "
                  f"{med.get('macaque', float('nan')):>9.4f} "
                  f"{med.get('human', float('nan')):>9.4f}")
    else:
        print("  ⚠️ 匹配子集物种不足")

    # ===== L3: 比较 L1 vs L2
    print("\n" + "=" * 80)
    print("L3. L1 vs L2 对比：规模是否主导？")
    print("=" * 80)
    if L2:
        print(f"{'统计量':<13} {'ε²(L1)':>9} {'ε²(L2)':>9} {'降幅':>8} {'判定':>20}")
        comp = {}
        for st in STATS:
            if st not in L1 or st not in L2:
                continue
            a, b = L1[st]["eps2"], L2[st]["eps2"]
            drop = (a - b) / a * 100 if a > 1e-9 else 0.0
            if drop > 50:
                v = "规模主导"
            elif drop < 20:
                v = "✅ 物种效应稳健"
            else:
                v = "混合"
            comp[st] = {"eps2_full": a, "eps2_matched": b,
                        "drop_pct": round(float(drop), 1), "verdict": v}
            print(f"{st:<13} {a:>9.4f} {b:>9.4f} {drop:>7.1f}% {v:>20}")
    else:
        comp = {}

    # ===== 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    if L1:
        robust = [s for s, v in L1.items()
                  if v["eps2"] < 0.14 and v["perm_p"] is not None and v["perm_p"] > 0.05]
        strong = [s for s, v in L1.items() if v["eps2"] > 0.3]
        print(f"  跨三物种守恒 (ε²<0.14 且置换不显著): {robust or '无'}")
        print(f"  物种差异大 (ε²>0.30): {strong or '无'}")
    if comp:
        stable = [s for s, v in comp.items() if v["verdict"].startswith("✅")]
        print(f"  规模校正后仍稳健: {stable or '无'}")

    (out / "phase8_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "note": "三物种全部为侵入式 spike，模态混淆已消除",
        "n_groups": int(len(df)),
        "n_perm": N_PERM,
        "L1_full": L1,
        "L2_matched": L2,
        "L3_comparison": comp,
        "caveat": ("人类数据仅来自 3 名患者（h1=HumanPitt, h2=T5，均为癫痫监测植入），"
                   "不代表健康人群。"),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
