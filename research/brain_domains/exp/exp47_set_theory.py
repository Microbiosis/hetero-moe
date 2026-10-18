"""Phase 47: 用集合论审计 omnibrain-multi 的分层结构。

## 为什么要用集合论

我把数据集称为"五级分层"。但从集合论看，"层"有严格定义：

    每个属性 α 定义一个等价关系 ~_α，其商集 S/~_α 是一个划分。
    α 比 β 细（α refines β），当且仅当 α 的每个等价类都完整落在
    β 的某个等价类内。

只有当属性构成**链**（A refines B refines C …）时，才能称为"分层"。
若两个属性互不细化，它们就是**交叉**的，不能排成层。

## 三个要检验的问题

### Q1 属性之间的细化关系

对 (source, species, modality, region, domain) 五个属性，
两两判定是否满足细化关系，得到细化矩阵。

若某对属性互不细化 → 它们不能构成层级关系。

### Q2 若属性不嵌套，投影链的塔性质还成立吗

嵌套投影链 Δ_ℓ = Π_ℓ − Π_{ℓ−1} 的前提是 Π 的 Loewner 序。
若划分不嵌套，Π 之间没有序，Δ_ℓ 也不能定义。

检验：在 domain 上分组、在 region 上分组，两个投影算子是否可交换
（Π_dom Π_reg = Π_reg Π_dom）。若不可交换，说明它们描述的是
两个正交的信息维度，不能合并成一层。

### Q3 笛卡儿积的空洞

标签空间的笛卡儿积 theoretical = |S| × |M| × |R| × |D|。
实际存在的组合数远小于此。空洞的位置本身就说明数据的边界。

## 输出

- 细化矩阵（5×5）
- 最长链
- 不可交换的投影对
- 空洞统计
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

INDEX = Path("/domains/omnibrain-multi/index_timeseries.parquet")
OUT = Path("/domains/runs/set-theory")

ATTRS = ["source", "species", "modality", "region", "domain_l1"]


# ------------------------------------------------------------ 核心算子
def quotient_size(s: pd.Series) -> int:
    """商集的势 = 等价类个数（缺失值单独算一类）。"""
    return int(s.fillna("__NA__").nunique())


def refines(a: pd.Series, b: pd.Series) -> tuple[bool, int]:
    """a 是否比 b 细。

    判据：a 的每个等价类，在 b 上只取一个值（含缺失）。
    返回 (是否细化, 违反的 a 类个数)。
    """
    df = pd.DataFrame({"a": a.fillna("__NA__"), "b": b.fillna("__NA__")})
    g = df.groupby("a")["b"].nunique()
    viol = int((g > 1).sum())
    return viol == 0, viol


def proj_operator(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Π_α：把每行替换为其等价类内的均值。"""
    out = np.empty_like(x)
    for lab in np.unique(labels):
        m = labels == lab
        if m.any():
            out[m] = x[m].mean(axis=0, keepdims=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 82, flush=True)
    print("Phase 47 — 用集合论审计分层结构", flush=True)
    print("=" * 82, flush=True)

    df = pd.read_parquet(INDEX)
    print(f"  {len(df):,} 行 × {len(df.columns)} 列", flush=True)

    # ============ Q1: 细化矩阵
    print("\n" + "=" * 82, flush=True)
    print("Q1  细化矩阵（行比列细？）", flush=True)
    print("=" * 82, flush=True)

    n = len(ATTRS)
    mat = np.zeros((n, n), dtype=int)     # 1=细化, 0=不
    viol_mat = np.zeros((n, n), dtype=int)

    for i, a in enumerate(ATTRS):
        for j, b in enumerate(ATTRS):
            if i == j:
                mat[i, j] = 1
                continue
            ok, viol = refines(df[a], df[b])
            mat[i, j] = int(ok)
            viol_mat[i, j] = viol

    # 打印
    short = {"source": "src", "species": "spc", "modality": "mod",
             "region": "reg", "domain_l1": "dom"}
    header = "".join(f"{short[b]:>7}" for b in ATTRS)
    print(f"{'':>10}{header}", flush=True)
    for i, a in enumerate(ATTRS):
        row = "".join(f"{'YES' if mat[i,j] else '.':>7}" for j in range(n))
        print(f"{short[a]:>10}{row}", flush=True)

    print("\n  商集的势:", flush=True)
    for a in ATTRS:
        print(f"    {a:<12} {quotient_size(df[a]):>6}", flush=True)

    # ============ 最长链
    print("\n" + "=" * 82, flush=True)
    print("Q1b  最长细化链", flush=True)
    print("=" * 82, flush=True)

    # 按势从小到大排序，找出能串成链的最长序列
    sizes = {a: quotient_size(df[a]) for a in ATTRS}
    order = sorted(ATTRS, key=lambda a: sizes[a])

    def is_chain(seq):
        """seq 从细到粗，每对相邻必须满足 refines。"""
        for i in range(len(seq) - 1):
            ai = ATTRS.index(seq[i])
            bi = ATTRS.index(seq[i + 1])
            if not mat[ai, bi]:
                return False
        return True

    from itertools import combinations
    best = []
    for r in range(len(order), 1, -1):
        for comb in combinations(order, r):
            if is_chain(list(comb)):
                best = list(comb)
                break
        if best:
            break

    print(f"  最长链: {' ⊂ '.join(best) if best else '（无）'}", flush=True)
    print(f"  长度: {len(best)}", flush=True)

    # 两两关系摘要
    print("\n  两两关系:", flush=True)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ATTRS[i], ATTRS[j]
            if mat[i, j] and not mat[j, i]:
                r = f"{a} ⊂ {b}"
            elif mat[j, i] and not mat[i, j]:
                r = f"{b} ⊂ {a}"
            elif mat[i, j] and mat[j, i]:
                r = f"{a} ≡ {b}（互为细化）"
            else:
                r = f"{a} × {b}（交叉，不可比）"
            print(f"    {r}", flush=True)

    # ============ Q2: 投影算子可交换性
    print("\n" + "=" * 82, flush=True)
    print("Q2  投影算子是否可交换（Πa Πb == Πb Πa？）", flush=True)
    print("=" * 82, flush=True)

    # 抽样以控制计算量
    rng = np.random.default_rng(0)
    sub = df.sample(n=min(20000, len(df)), random_state=0).reset_index(drop=True)
    # 用 index 位置造一个假的"数据"（真实张量太大，用索引值代替以检验算子代数）
    # 注意：可交换性与数据无关，只与划分有关，用任意向量即可
    x = rng.standard_normal((len(sub), 8))

    com = {}
    for i in range(n):
        for j in range(i + 1, n):
            la = pd.factorize(sub[ATTRS[i]].fillna("__NA__"))[0]
            lb = pd.factorize(sub[ATTRS[j]].fillna("__NA__"))[0]
            Pab = proj_operator(proj_operator(x, la), lb)
            Pba = proj_operator(proj_operator(x, lb), la)
            diff = float(np.abs(Pab - Pba).max())
            commutes = diff < 1e-6
            com[f"{ATTRS[i]}|{ATTRS[j]}"] = {
                "max_diff": round(diff, 6), "commutes": bool(commutes)}
            mark = "可交换" if commutes else "不可交换"
            print(f"    {ATTRS[i]:<11} × {ATTRS[j]:<11} max|diff|={diff:.2e}  {mark}",
                  flush=True)

    # ============ Q3: 笛卡儿积的空洞
    print("\n" + "=" * 82, flush=True)
    print("Q3  标签空间的空洞", flush=True)
    print("=" * 82, flush=True)

    n_src = df.source.nunique()
    n_mod = df.modality.nunique()
    n_reg = df.region.fillna("__NA__").nunique()
    n_dom = df.domain_l1.fillna("__NA__").nunique()
    theoretical = n_src * n_mod * n_reg * n_dom

    obs = df.groupby(["source", "modality",
                      df.region.fillna("__NA__"), df.domain_l1.fillna("__NA__")],
                     observed=True).size()
    actual = len(obs)
    print(f"  理论组合（source × modality × region × domain）= "
          f"{n_src} × {n_mod} × {n_reg} × {n_dom} = {theoretical}", flush=True)
    print(f"  实际存在 = {actual}", flush=True)
    print(f"  空洞 = {theoretical - actual} "
          f"({(theoretical-actual)/theoretical*100:.1f}%)", flush=True)

    # 各 source 的标签空间
    print("\n  各 source 的标签覆盖:", flush=True)
    for src, g in df.groupby("source"):
        print(f"    {src:<22} domain={g.domain_l1.nunique():>2}  "
              f"region={g.region.fillna('__NA__').nunique():>3}  "
              f"modality={g.modality.nunique()}", flush=True)

    # ============ 判定
    print("\n" + "=" * 82, flush=True)
    print("判定", flush=True)
    print("=" * 82, flush=True)

    if len(best) >= 4:
        v = f"✅ 存在长度 {len(best)} 的细化链：{' ⊂ '.join(best)}"
    elif len(best) == 3:
        v = f"⚠️ 最长链仅长度 3：{' ⊂ '.join(best)}"
    else:
        v = f"❌ 几乎不构成链，最长仅 {len(best)}"

    print(f"  {v}", flush=True)

    n_cross = sum(1 for i in range(n) for j in range(i + 1, n)
                  if not mat[i, j] and not mat[j, i])
    print(f"  交叉（不可比）的属性对: {n_cross} / {n*(n-1)//2}", flush=True)
    n_noncomm = sum(1 for vv in com.values() if not vv["commutes"])
    print(f"  投影算子不可交换的对: {n_noncomm} / {len(com)}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "set_theory_audit",
        "n_samples": int(len(df)),
        "attributes": ATTRS,
        "quotient_sizes": {a: quotient_size(df[a]) for a in ATTRS},
        "refinement_matrix": {a: {b: int(mat[i, j])
                                  for j, b in enumerate(ATTRS)}
                              for i, a in enumerate(ATTRS)},
        "violations": {f"{a}|{b}": int(viol_mat[i, j])
                       for i, a in enumerate(ATTRS)
                       for j, b in enumerate(ATTRS) if i != j},
        "longest_chain": best,
        "commutativity": com,
        "cartesian": {
            "theoretical": int(theoretical),
            "actual": int(actual),
            "holes": int(theoretical - actual),
            "holes_pct": round((theoretical - actual) / theoretical * 100, 1),
        },
        "verdict": v,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
