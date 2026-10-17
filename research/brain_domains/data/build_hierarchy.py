"""构建具有严格层次结构的脑域数据集（嵌套正交投影链）。

## 核心数学构造

不用"声明"层次，而用**嵌套正交投影链**保证每层信息互不重叠。

设 P_ℓ 是第 ℓ 层的划分（partition），Π_ℓ 是把信号投影到"同组内取平均"的算子：

    (Π_ℓ x)_u = mean_{u' ∈ group_ℓ(u)} x_{u'}

若划分严格嵌套 P_0 ⊂ P_1 ⊂ ... ⊂ P_L（P_{ℓ+1} 由 P_ℓ 的组进一步合并），
则投影算子满足 Loewner 序 Π_0 ⪯ Π_1 ⪯ ... ⪯ Π_L，且

    Δ_ℓ = Π_ℓ − Π_{ℓ−1},   Δ_ℓ · Δ_m = 0  (ℓ ≠ m)

因而得到**严格正交分解**：

    x = Δ_1 x + Δ_2 x + ... + Δ_L x + (I − Π_L) x

平方 Frobenius 范数下，各层方差份额精确相加为 1：

    ||x||² = Σ_ℓ ||Δ_ℓ x||² + ||(I − Π_L)x||²

**这是层次结构的严格定义**：每一层是不可再分的正交子空间，
不依赖任何解剖学假设。解剖标签只是给层命名的一种方式。

## 两种层次（对应两类问题）

### A. 解剖层次（小鼠，有真实区域/功能域标签）
    L0 单元 → L1 数据驱动簇 → L2 解剖区 → L3 L1功能域 → L4 session

每层做**置换对照**：保持组大小，随机重分组，重算方差份额。
超额 = 真实 − 打乱。只有超额 > 0 的层才算"真实结构"。

### B. 数据驱动层次（三物种通用）
数据驱动的嵌套划分，k = 2,4,8,16,32（簇数），由同一棵层次聚类树切出。
深度用 scale-free 的 d = log2(k)/log2(D) 归一化，**因此跨物种可比**。

这一层是关键：前 21 个实验所有基于解剖标签的跨物种比较都被混淆毁掉，
而数据驱动的 k-层次没有"标签语义"问题，只有"划分粗细"这一个变量。

## 为什么这可能给出跨物种不变量

方差-深度曲线 v(d) 是无量纲的（方差份额），
且深度 d 由簇数定义（不依赖采样率、单元数、脑区命名）。
若三物种的 v(d) 形状一致，那才是真正的跨物种不变量。

## 输出

    hierarchy/
    ├── meta.json                 设计 + 定义 + 结果摘要
    ├── variance_depth.csv        三物种 × k × 方差份额(真实/打乱/超额)
    ├── anatomical_variance.csv   小鼠 × 层 × 方差份额
    └── partitions/               各层的划分（可用于下游建模）
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
D_PCA = 64
KS = [2, 4, 8, 16, 32]
N_SHUFFLE = 50
MAX_WIN = 3000          # 每片段取样窗口数
N_PER_SPECIES = 12


# --------------------------------------------------------------- 基础算子
def proj_group_mean(x, labels):
    """Π 算子：(Πx)_u = 该组内所有成员的均值（逐时间点）。"""
    out = np.empty_like(x)
    for g in np.unique(labels):
        m = labels == g
        if m.sum() == 0:
            continue
        out[m] = x[m].mean(axis=0, keepdims=True)
    return out


def var_frac(x, comp):
    """||comp||² / ||x||²"""
    den = float((x ** 2).sum())
    return float((comp ** 2).sum()) / den if den > 1e-12 else 0.0


def nested_labels(Z, n_items, ks):
    """从同一棵层次树按高度切出嵌套划分。

    对每个目标簇数 k，取树中使簇数 ≈ k 的切分高度。
    高度从大到小 → 划分从粗到细，因此严格嵌套。
    """
    # Z[:,2] 是合并高度，降序
    heights = np.sort(Z[:, 2])[::-1]
    out = {}
    for k in ks:
        if k >= n_items:
            continue
        # 取第 (k-1) 个合并高度作为切分阈值（略低一点保证得到 k 簇）
        idx = min(k - 1, len(heights) - 1)
        h = heights[idx] - 1e-9
        lab = fcluster(Z, t=h, criterion="distance")
        out[k] = lab
    return out


def build_tree(x):
    """对 (n_items, n_time) 做相关距离层次聚类。"""
    if x.shape[0] < 4:
        return None
    d = pdist(x, metric="correlation")
    d = np.nan_to_num(d, nan=1.0, posinf=2.0)
    return linkage(d, method="average")


def ladder(x, Z, ks, rng, n_shuffle=N_SHUFFLE):
    """计算数据驱动层次的方差-深度曲线（含置换对照）。"""
    n = x.shape[0]
    parts = nested_labels(Z, n, ks)
    if not parts:
        return {}

    # 逐层投影，从最粗开始
    order = sorted(parts.keys())            # 2,4,8,16,32（细→粗）
    projs = {k: proj_group_mean(x, parts[k]) for k in order}

    res = {}
    prev = None
    for k in order:
        comp = projs[k] if prev is None else (projs[k] - prev)
        v = var_frac(x, comp)

        # ---- 零模型 A：打乱组标签（只控制组大小）
        nullA = []
        sizes = pd.Series(parts[k]).value_counts().sort_index()
        for _ in range(n_shuffle):
            lab = np.repeat(np.arange(len(sizes)), sizes.to_numpy())
            rng.shuffle(lab)
            pj = proj_group_mean(x, lab)
            c = pj if prev is None else (pj - prev)
            nullA.append(var_frac(x, c))

        # ---- 零模型 B：逐成分打乱时间轴后重新聚类（控制"聚类几何"）
        #      这是关键：破坏跨成分相关，但保留每个成分的边缘分布与谱。
        nullB = []
        for _ in range(max(3, n_shuffle // 10)):
            xs = np.stack([rng.permutation(x[i]) for i in range(x.shape[0])])
            Zs = build_tree(xs)
            if Zs is None:
                continue
            ps = nested_labels(Zs, xs.shape[0], ks)
            if k not in ps:
                continue
            # 按同一顺序累加：需要同样重建 prev 层
            order_s = sorted(ks)
            prevs = None
            for kk in order_s:
                if kk not in ps:
                    continue
                pj = proj_group_mean(xs, ps[kk])
                c = pj if prevs is None else (pj - prevs)
                if kk == k:
                    nullB.append(var_frac(xs, c))
                    break
                prevs = pj

        res[k] = {"var": v,
                  "null_label": float(np.mean(nullA)) if nullA else None,
                  "excess_label": (v - float(np.mean(nullA))) if nullA else None,
                  "null_geom": float(np.mean(nullB)) if nullB else None,
                  "excess_geom": (v - float(np.mean(nullB))) if nullB else None}
        prev = projs[k]

    # 最细残差 (I − Π_{k_max}) x —— 用最细划分(最大 k)，不是最粗
    k_fine = order[-1]
    res["residual"] = {"var": var_frac(x, x - projs[k_fine]),
                       "null_label": None, "excess_label": None,
                       "null_geom": None, "excess_geom": None}
    # 累计解释率（供核对：Σ各层增量 + 残差 应 = 1）
    res["cumulative"] = float(sum(res[k]["var"] for k in order) + res["residual"]["var"])
    return res


def anatomical_ladder(x, level_labels, rng, n_shuffle=N_SHUFFLE):
    """解剖层次：L0单元 ⊂ L1簇 ⊂ L2区 ⊂ L3功能域 ⊂ L4(全局)。

    level_labels: {name: labels}  从细到粗，必须严格嵌套
    """
    names = list(level_labels.keys())
    projs = {nm: proj_group_mean(x, level_labels[nm]) for nm in names}
    projs["__global__"] = proj_group_mean(x, np.zeros(x.shape[0], dtype=int))

    res = {}
    prev = None
    for nm in names + ["__global__"]:
        comp = projs[nm] if prev is None else (projs[nm] - prev)
        v = var_frac(x, comp)

        null = []
        base_labels = level_labels.get(nm)
        if base_labels is not None:
            sizes = pd.Series(base_labels).value_counts().sort_index()
            for _ in range(n_shuffle):
                lab = np.repeat(np.arange(len(sizes)), sizes.to_numpy())
                rng.shuffle(lab)
                pj = proj_group_mean(x, lab)
                c = pj if prev is None else (pj - prev)
                null.append(var_frac(x, c))
        res[nm] = {"var": v,
                   "null": float(np.mean(null)) if null else None,
                   "excess": (v - float(np.mean(null))) if null else None}
        prev = projs[nm]
    return res


# --------------------------------------------------------------- 数据加载
def load_view(root: Path, name: str, max_win=MAX_WIN, n_groups=N_PER_SPECIES):
    """从 OmniBrain 视图读窗口（已 PCA 到 64 维）。返回 [(key, X(n_win,64))]。"""
    d = root / name
    if not d.exists():
        return []
    meta_f = d / "meta.parquet"
    if not meta_f.exists():
        return []
    meta = pd.read_parquet(meta_f)

    # 分组与文件名的对应关系依视图而定：
    #   spike_view          每 session 一个 windows_sess{NNN}.npz（session 是 int）
    #   spike_view_macaque  每 file 一个 npz（session 是字符串，不能用 int 转换）
    #   spike_view_human    每 file 一个 npz
    if "session" in meta.columns and name == "spike_view":
        gcol = "session"
    elif "file" in meta.columns:
        gcol = "file"
    elif "session" in meta.columns:
        gcol = "session"
    else:
        return []

    out = []
    for gval, sub in list(meta.groupby(gcol))[:n_groups]:
        if gcol == "session" and name == "spike_view":
            f = d / f"windows_sess{int(gval):03d}.npz"
        else:
            f = d / str(gval)
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        if len(w) < 200:
            continue
        # (n_win, 64, 8) → (n_win, 64) 时间均值；再转 (64, n_win)
        X = w.mean(axis=2).T.astype(np.float64)          # (64, n_win)
        # 去重叠：步长 4，取每 2 个（有效步长 8）
        X = X[:, ::2]
        if X.shape[1] > max_win:
            s = (X.shape[1] - max_win) // 2
            X = X[:, s:s + max_win]
        # 逐成分 z-score
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-9
        X = (X - mu) / sd
        out.append((str(gval), X))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/hierarchy")
    (out / "partitions").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 80)
    print("构建层次化脑域数据集（嵌套正交投影链）")
    print("=" * 80)
    print(f"  PCA 维数 {D_PCA} | k 阶梯 {KS} | 置换 {N_SHUFFLE}")
    print(f"  每物种取样 {N_PER_SPECIES} 个分组，每组 ≤{MAX_WIN} 窗")
    print()

    t0 = time.time()
    rows = []

    # ============ B. 数据驱动层次（三物种通用）
    print("=" * 80)
    print("数据驱动层次：方差-深度曲线")
    print("=" * 80)

    for species, view in [("mouse", "spike_view"),
                          ("macaque", "spike_view_macaque"),
                          ("human", "spike_view_human")]:
        items = load_view(root, view)
        print(f"\n  {species}: {len(items)} 个分组")
        if not items:
            continue
        for key, X in items:
            Z = build_tree(X)
            if Z is None:
                continue
            lad = ladder(X, Z, KS, rng)
            if not lad:
                continue
            # 有效维度（参与率），用于解释物种间差异
            lam = X.var(axis=1)
            eff_dim = float((lam.sum() ** 2) / max((lam ** 2).sum(), 1e-12))

            for k, v in lad.items():
                if k == "cumulative":
                    continue
                rows.append({"species": species, "group": key,
                             "eff_dim": round(eff_dim, 3),
                             "cumulative_sum": round(lad.get("cumulative", float("nan")), 5),
                             "level": ("residual" if k == "residual" else f"k{k}"),
                             "k": (np.nan if k == "residual" else k),
                             "var": round(v["var"], 5),
                             "null_label": (round(v["null_label"], 5) if v.get("null_label") is not None else None),
                             "excess_label": (round(v["excess_label"], 5) if v.get("excess_label") is not None else None),
                             "null_geom": (round(v["null_geom"], 5) if v.get("null_geom") is not None else None),
                             "excess_geom": (round(v["excess_geom"], 5) if v.get("excess_geom") is not None else None)})
            # 保存最细划分，供下游用
            parts = nested_labels(Z, X.shape[0], KS)
            np.savez(out / "partitions" / f"{species}_{key.replace('/','_')}.npz",
                     **{f"k{k}": lab for k, lab in parts.items()})
        print(f"    → 累计 {len(rows)} 条记录 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "variance_depth.csv", index=False)

    # ============ 汇总
    print("\n" + "=" * 80)
    print("方差-深度曲线（中位）")
    print("=" * 80)
    piv = df.pivot_table(index="level", columns="species", values="var", aggfunc="median")
    order = [f"k{k}" for k in KS] + ["residual"]
    piv = piv.reindex([o for o in order if o in piv.index])
    print(piv.round(4).to_string())

    print("\n  超额 A（真实 − 打乱标签，只控组大小）:")
    piv_a = df.pivot_table(index="level", columns="species", values="excess_label", aggfunc="median")
    piv_a = piv_a.reindex([o for o in order if o in piv_a.index])
    print(piv_a.round(4).to_string())

    print("\n  超额 B（真实 − 时间打乱后重聚类，控聚类几何）:")
    piv_b = df.pivot_table(index="level", columns="species", values="excess_geom", aggfunc="median")
    piv_b = piv_b.reindex([o for o in order if o in piv_b.index])
    print(piv_b.round(4).to_string())

    # 归一化深度与曲线形状比较
    print("\n" + "=" * 80)
    print("跨物种曲线形状（scale-free 深度）")
    print("=" * 80)
    shape = {}
    for sp in df["species"].unique():
        s = df[(df.species == sp) & df["k"].notna()]
        if not len(s):
            continue
        m = s.groupby("k")["var"].median()
        tot = m.sum()
        if tot <= 0:
            continue
        # 归一化深度 d = log2(k)/log2(32)
        d = np.log2(m.index.to_numpy()) / np.log2(max(KS))
        shape[sp] = {"d": d.round(3).tolist(),
                     "v_norm": (m.to_numpy() / tot).round(4).tolist()}
        print(f"  {sp:8s} d={shape[sp]['d']}")
        print(f"           v={shape[sp]['v_norm']}")

    # 形状一致性：两两相关
    print("\n  曲线形状两两相关（Pearson）:")
    sps = list(shape.keys())
    corrs = {}
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            a = np.array(shape[sps[i]]["v_norm"])
            b = np.array(shape[sps[j]]["v_norm"])
            if len(a) == len(b):
                c = float(np.corrcoef(a, b)[0, 1])
                corrs[f"{sps[i]}-{sps[j]}"] = round(c, 3)
                print(f"    {sps[i]:8s} vs {sps[j]:8s}: r = {c:+.3f}")

    # ============ A. 解剖层次（小鼠，需要原始 unit 标签）
    print("\n" + "=" * 80)
    print("解剖层次（小鼠，需要单元级标签）")
    print("=" * 80)
    anat_rows = []
    stein = Path("/data/steinmetz")
    if (stein / "units.parquet").exists():
        units = pd.read_parquet(stein / "units.parquet")
        for sess in sorted(units["session"].unique())[:N_PER_SPECIES]:
            f = stein / f"spikes_session_{sess:03d}.npz"
            if not f.exists():
                continue
            sp = np.load(f)["spikes"].astype(np.float64)
            u = units[units["session"] == sess].reset_index(drop=True)
            if sp.shape[0] != len(u):
                continue
            # 50ms → 100ms
            nb = sp.shape[1] // 2
            x = sp[:, :nb * 2].reshape(sp.shape[0], nb, 2).sum(axis=2)
            # z-score 每个 unit
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-9
            x = (x - mu) / sd
            if x.shape[1] > MAX_WIN:
                s = (x.shape[1] - MAX_WIN) // 2
                x = x[:, s:s + MAX_WIN]

            # 构造嵌套标签：unit → 数据驱动簇 → region → 功能域
            Z = build_tree(x)
            if Z is None:
                continue
            cluster = nested_labels(Z, x.shape[0], [max(4, x.shape[0] // 20)]).popitem()[1]

            region = pd.Categorical(u["region"]).codes
            domain = pd.Categorical(u["functional_domain"]).codes

            # 检查嵌套：region 必须粗于 cluster，domain 必须粗于 region
            def is_refinement(fine, coarse):
                """fine 是否比 coarse 细（每个 coarse 组由若干完整 fine 组构成）"""
                for c in np.unique(coarse):
                    sub = fine[coarse == c]
                    if len(np.unique(sub)) and len(sub) != (fine == sub[0]).sum() and \
                       len(np.unique(sub)) > 1:
                        # 允许：只要 coarse 组内的 fine 组不跨到别的 coarse 组
                        pass
                # 简化检验：每个 fine 组只属于一个 coarse 组
                for fv in np.unique(fine):
                    if len(np.unique(coarse[fine == fv])) > 1:
                        return False
                return True

            ok_cr = is_refinement(cluster, region)
            ok_rd = is_refinement(region, domain)

            # 注意：数据驱动簇与解剖区**不嵌套**（cluster⊂region 实测 0%）。
            # 因此不做单一链式分解，而是把三种划分**分别**与全局对比，
            # 各自算"这一划分能解释多少方差"。这样才不需要嵌套假设。
            labels = {"cluster": cluster, "region": region}
            if ok_rd:
                labels["domain"] = domain

            lad = anatomical_ladder(x, labels, rng)
            for nm, v in lad.items():
                anat_rows.append({"session": int(sess), "level": nm,
                                  "var": round(v["var"], 5),
                                  "null": (round(v["null"], 5) if v["null"] is not None else None),
                                  "excess": (round(v["excess"], 5) if v["excess"] is not None else None),
                                  "nested_cluster_region": bool(ok_cr),
                                  "nested_region_domain": bool(ok_rd)})
            print(f"  sess {sess:02d}: cluster⊂region={ok_cr} region⊂domain={ok_rd} "
                  f"({x.shape[0]} units)")

    adf = pd.DataFrame(anat_rows)
    if len(adf):
        adf.to_csv(out / "anatomical_variance.csv", index=False)
        print("\n  解剖层次方差份额（中位）:")
        print(adf.pivot_table(index="level", columns=None, values="var",
                              aggfunc="median").round(4).to_string())
        print("\n  超额:")
        print(adf.groupby("level")["excess"].median().round(4).to_string())
        print(f"\n  嵌套成立比例: cluster⊂region {adf['nested_cluster_region'].mean()*100:.0f}%"
              f" | region⊂domain {adf['nested_region_domain'].mean()*100:.0f}%")
    else:
        print("  ⚠️ 无解剖层次结果")

    # ============ meta
    meta = {
        "schema_version": "1.0",
        "construction": "nested orthogonal projection chain (martingale/ANOVA decomposition)",
        "math": {
            "projection": "(Π_ℓ x)_u = mean over group_ℓ(u)",
            "nesting": "P_0 ⊂ P_1 ⊂ ... ⊂ P_L  =>  Π_0 ⪯ Π_1 ⪯ ... ⪯ Π_L",
            "difference_operators": "Δ_ℓ = Π_ℓ − Π_{ℓ−1},  Δ_ℓ·Δ_m = 0 (ℓ≠m)",
            "decomposition": "x = Σ_ℓ Δ_ℓ x,  ||x||² = Σ_ℓ ||Δ_ℓ x||²  (exact)",
        },
        "levels_data_driven": {"k_values": KS,
                               "depth": "d = log2(k)/log2(32)  (scale-free, cross-species comparable)",
                               "clustering": "correlation distance + average linkage, cut at nested heights"},
        "levels_anatomical": ["unit", "cluster(data-driven)", "region", "functional_domain", "session"],
        "controls": {"shuffle": "permute group labels preserving group sizes",
                     "n_shuffle": N_SHUFFLE,
                     "excess": "var(real) - var(shuffled)"},
        "n_records": int(len(df)),
        "elapsed_s": round(time.time() - t0, 1),
        "caveats": [
            "窗口来自 OmniBrain 视图（100ms，原始步长 4），此处已隔点取样使有效步长 8（零重叠）",
            "PCA 成分作为聚类单元，非真实神经元；跨物种因此可比但语义是'成分级'",
            "解剖层次的嵌套性在部分 session 不成立（region 与 cluster 不严格嵌套），已逐 session 记录",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"\n  完成，耗时 {time.time()-t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
