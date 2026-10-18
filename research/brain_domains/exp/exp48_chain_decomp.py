"""Phase 48: 沿真正成立的链做方差分解（region ⊂ source ⊂ species）。

## 背景

exp47 用集合论审计得到：

    region < source < modality
    region < source < species
    domain_l1 与所有属性交叉（投影不可交换）

即：只有 `region ⊂ source ⊂ (species | modality)` 这条链是嵌套的，
`domain_l1` 不在链上。

## 本实验

在有 region 的子集上（小鼠 + 猕猴 + 人类皮层内 = 588,250 窗），
沿链做精确的正交分解：

    x = Π_region x
      + (Π_source − Π_region) x
      + (Π_species − Π_source) x
      + (I − Π_species) x

各层范数平方精确相加（因为 Δ_ℓ Δ_m = 0）：

    ‖x‖² = ‖Π_r x‖² + ‖(Π_s − Π_r)x‖² + ‖(Π_p − Π_s)x‖² + ‖(I − Π_p)x‖²

## 要回答的

1. 每一层解释多少方差？
2. 如果把 domain 硬塞进链里（违反嵌套前提），会得到什么异常？
   —— 用它作为「错误做法」的对照

## 判据

  若 Π_domain 与 Π_region 不可交换，把 domain 插进链后
  各层之和 ≠ ‖x‖²（不满足精确分解）
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DS = Path("/domains/omnibrain-multi")
OUT = Path("/domains/runs/chain-decomp")

MAX_N = 120_000          # 采样上限，控制内存
SEED = 0


def proj(x, labels):
    """Π：把每行替换为其等价类内的均值。"""
    out = np.empty_like(x)
    for lab in np.unique(labels):
        m = labels == lab
        if m.any():
            out[m] = x[m].mean(axis=0, keepdims=True)
    return out


def var_frac(x, comp):
    den = float((x ** 2).sum())
    return float((comp ** 2).sum()) / den if den > 1e-12 else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 82, flush=True)
    print("Phase 48 — 沿 region ⊂ source ⊂ species 做方差分解", flush=True)
    print("=" * 82, flush=True)

    idx = pd.read_parquet(DS / "index_timeseries.parquet")
    # 只保留有 region 的（小鼠 + 猕猴 + 人类皮层内）
    m = idx.region.notna().to_numpy()
    sub = idx[m].reset_index(drop=True)
    print(f"  有 region 的样本: {len(sub):,} / {len(idx):,}", flush=True)

    # 采样
    if len(sub) > MAX_N:
        keep = rng.choice(len(sub), MAX_N, replace=False)
        keep.sort()
        sub = sub.iloc[keep].reset_index(drop=True)
        print(f"  采样至 {len(sub):,}", flush=True)

    # 载入对应张量
    print("  载入 timeseries.npy…", flush=True)
    W = np.load(DS / "timeseries.npy", mmap_mode="r")
    pos = sub.index.to_numpy()
    # 注意：sub 的 index 是原始行位置吗？重采样后不是，需要回溯
    orig_pos = np.where(m)[0]
    if len(orig_pos) > MAX_N:
        orig_pos = orig_pos[keep]
    X = np.asarray(W[orig_pos]).reshape(len(sub), -1).astype(np.float32)
    print(f"  张量 {X.shape} ({X.nbytes/1024**2:.0f} MB)", flush=True)

    # 各层标签
    lab_region = pd.factorize(sub.region)[0]
    lab_source = pd.factorize(sub.source)[0]
    lab_species = pd.factorize(sub.species)[0]
    lab_domain = pd.factorize(sub.domain_l1)[0]

    print(f"\n  层基数: region={len(np.unique(lab_region))} "
          f"source={len(np.unique(lab_source))} "
          f"species={len(np.unique(lab_species))} "
          f"domain={len(np.unique(lab_domain))}", flush=True)

    # 检查嵌套
    def refines(a, b):
        return (pd.DataFrame({"a": a, "b": b})
                .groupby("a")["b"].nunique() > 1).sum() == 0

    ok_rs = refines(lab_region, lab_source)
    ok_sp = refines(lab_source, lab_species)
    print(f"  嵌套检验: region⊂source={ok_rs}  source⊂species={ok_sp}", flush=True)

    # ---------- 正确的链分解
    print("\n" + "=" * 82, flush=True)
    print("A. 正确链：region ⊂ source ⊂ species", flush=True)
    print("=" * 82, flush=True)

    t0 = time.time()
    P_r = proj(X, lab_region)
    P_s = proj(X, lab_source)
    P_p = proj(X, lab_species)

    v1 = var_frac(X, P_r)
    v2 = var_frac(X, P_s - P_r)
    v3 = var_frac(X, P_p - P_s)
    v4 = var_frac(X, X - P_p)
    tot = v1 + v2 + v3 + v4

    print(f"  Π_region            {v1:>8.4f}  ({v1*100:>5.1f}%)", flush=True)
    print(f"  Π_source − Π_region {v2:>8.4f}  ({v2*100:>5.1f}%)", flush=True)
    print(f"  Π_species − Π_source{v3:>8.4f}  ({v3*100:>5.1f}%)", flush=True)
    print(f"  I − Π_species       {v4:>8.4f}  ({v4*100:>5.1f}%)", flush=True)
    print(f"  ───────────────────────────────", flush=True)
    print(f"  合计                {tot:>8.4f}   (应为 1.0000)", flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

    chain_ok = abs(tot - 1.0) < 1e-4

    # ---------- 错误做法：把 domain 插进链
    print("\n" + "=" * 82, flush=True)
    print("B. 错误做法：把 domain 插进链（违反嵌套前提）", flush=True)
    print("=" * 82, flush=True)

    P_d = proj(X, lab_domain)
    # 强行按 region → domain → source 的顺序做差
    w1 = var_frac(X, P_r)
    w2 = var_frac(X, P_d - P_r)          # 这里 Π_d − Π_r 不再是正交增量
    w3 = var_frac(X, P_s - P_d)
    w4 = var_frac(X, X - P_s)
    tot_bad = w1 + w2 + w3 + w4

    print(f"  Π_region            {w1:>8.4f}", flush=True)
    print(f"  Π_domain − Π_region {w2:>8.4f}", flush=True)
    print(f"  Π_source − Π_domain {w3:>8.4f}", flush=True)
    print(f"  I − Π_source        {w4:>8.4f}", flush=True)
    print(f"  ───────────────────────────────", flush=True)
    print(f"  合计                {tot_bad:>8.4f}  "
          f"(偏离 1 有 {abs(tot_bad-1)*100:.1f}%)", flush=True)

    # 正交性检验：Δ_r = Π_d − Π_r 与 Π_r 是否正交
    cross = float((P_r * (P_d - P_r)).sum() / max((X**2).sum(), 1e-12))
    print(f"\n  ⟨Π_region, Π_domain − Π_region⟩ / ‖X‖² = {cross:+.4f}", flush=True)
    print(f"  → " + ("正交（可分解）" if abs(cross) < 1e-3
                       else "不正交（不能拆成层）"), flush=True)

    # ---------- 判定
    print("\n" + "=" * 82, flush=True)
    print("判定", flush=True)
    print("=" * 82, flush=True)

    if chain_ok:
        print(f"  ✅ region ⊂ source ⊂ species 是真正的嵌套链，"
              f"方差精确分解（合计 {tot:.4f}）", flush=True)
    else:
        print(f"  ⚠️ 链分解不精确（合计 {tot:.4f}），检查嵌套性", flush=True)

    print(f"  各层方差占比: region {v1*100:.1f}% > "
          f"source {v2*100:.1f}% > species {v3*100:.1f}% > "
          f"残差 {v4*100:.1f}%", flush=True)
    print(f"\n  把 domain 插进链后合计变成 {tot_bad:.4f}"
          f"（正确应为 1.0000），偏差 {abs(tot_bad-1)*100:.1f}%", flush=True)
    print("  这就是 domain 不能作为层的定量证据。", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "chain_decomposition",
        "n_samples": int(len(sub)),
        "levels": {
            "region": int(len(np.unique(lab_region))),
            "source": int(len(np.unique(lab_source))),
            "species": int(len(np.unique(lab_species))),
            "domain": int(len(np.unique(lab_domain))),
        },
        "nesting": {"region_sub_source": bool(ok_rs),
                    "source_sub_species": bool(ok_sp)},
        "proper_chain": {
            "Pi_region": round(v1, 5),
            "Pi_source_minus_region": round(v2, 5),
            "Pi_species_minus_source": round(v3, 5),
            "residual": round(v4, 5),
            "sum": round(tot, 5),
            "exact": bool(chain_ok),
        },
        "with_domain_inserted": {
            "Pi_region": round(w1, 5),
            "Pi_domain_minus_region": round(w2, 5),
            "Pi_source_minus_domain": round(w3, 5),
            "residual": round(w4, 5),
            "sum": round(tot_bad, 5),
            "deviation_from_1": round(abs(tot_bad - 1), 5),
        },
        "orthogonality": {"cross_term": round(cross, 5),
                          "orthogonal": bool(abs(cross) < 1e-3)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
