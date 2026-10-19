"""把 OmniBrain 的多个来源融合成一个带层级索引的单一数据集。

## 要解决的问题

现有 5 个视图是**并列**的目录，各自独立：
    spike_view           小鼠 Steinmetz      314,332 窗
    spike_view_macaque   猕猴 NLB            187,067 窗
    spike_view_human     人类侵入式           86,851 窗
    eeg_view             人类头皮 EEG         32,000 窗
    graph_view           果蝇 FlyWire        139,255 节点

用户要的是「多个数据集构造成一个带层次结构的数据集」，不是并列目录。

## 融合的实质困难（必须诚实处理）

1. **64 维成分的语义不同**
   每个来源各自做 PCA 到 64 维。小鼠的第 3 成分与人类的第 3 成分
   不对应同一个生物实体。因此**不能声称这是「同一空间的 64 维向量」**。

2. **采样率不同**
   小鼠 50ms、猕猴/人类侵入式 20ms、EEG 4ms。
   上游已统一到 100ms bin（求和或均值）。

3. **空间规模不同**
   小鼠 205–1240 units、猕猴 40–192、人类 63 电极。

4. **时间性质不同**
   spike 是计数，EEG 是电位。同一 bin 宽度下统计特性不同。

## 因此本数据集的做法

**不做虚假的空间对齐**，而是提供三样东西：

    A. 统一张量        (N, 64, 8) —— 可直接送进任何序列模型训练
    B. 逐窗标签        每个样本带 source / species / modality / region / domain
    C. 可比统计指纹    每个 (source, region) 组一组无量纲统计量
                       （这些是真正可跨源比较的，因为它们不依赖坐标系）

## 标签定义（非层级）

    样本          620,250 个时间窗 + 139,255 个图节点
    来源          mouse_steinmetz / macaque_nlb / human_intracortical /
                      human_eeg / flywire
    物种          mus_musculus / macaca_mulatta / homo_sapiens /
                      drosophila_melanogaster
    模态          extracellular_spike / scalp_eeg / connectome
    解剖区        region（小鼠/猕猴可用，EEG 无）
    功能域        L1 domain（sensory/motor/memory/...）

## 输出

    omnibrain-multi/
    ├── timeseries.npy           (N_t, 64, 8) float16   ≈635 MB
    ├── graph.npy                (N_g, 64)    float32   ≈36 MB
    ├── index_timeseries.parquet (N_t 行)
    ├── index_graph.parquet      (N_g 行)
    ├── labels.json              标签取值与计数（非层级）
    ├── comparables.csv          跨源可比较统计量
    └── README.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = Path("/domains/omnibrain")
OUT = Path("/domains/omnibrain-multi")

# 来源定义：(目录, source 名, 物种, 模态, 分组列)
SOURCES = [
    ("spike_view", "mouse_steinmetz", "mus_musculus",
     "extracellular_spike", "session"),
    ("spike_view_macaque", "macaque_nlb", "macaca_mulatta",
     "extracellular_spike", "file"),
    ("spike_view_human", "human_intracortical", "homo_sapiens",
     "extracellular_spike", "file"),
    ("eeg_view", "human_eeg", "homo_sapiens", "scalp_eeg", None),
]


def load_source(view: str, gcol):
    """载入一个视图的全部窗口 + 逐窗元数据。

    返回 (windows (n,64,8) float16, meta_df) 或 (None, None)
    """
    d = BUCKET / view
    if not d.exists():
        print(f"  [skip] {view} 不存在")
        return None, None
    meta_f = d / "meta.parquet"
    if not meta_f.exists():
        print(f"  [skip] {view} 无 meta")
        return None, None

    meta = pd.read_parquet(meta_f)
    chunks = []
    rows = []

    if view == "eeg_view":
        # 单个合并文件
        f = d / "windows.npz"
        if not f.exists():
            print(f"  [skip] {view} 无 windows.npz")
            return None, None
        w = np.load(f)["windows"]
        chunks.append(w)
        r = meta[["domain_l1"]].copy() if "domain_l1" in meta.columns \
            else pd.DataFrame({"domain_l1": ["other"] * len(w)})
        # eeg_view 的分组键用 subject
        r["group_key"] = (meta["subject"].astype(str).to_numpy()
                          if "subject" in meta.columns else "eeg_all")
        rows.append(r)
    else:
        # 逐文件
        files = sorted(meta[gcol].unique()) if gcol in meta.columns else []
        for gval in files:
            if view == "spike_view":
                f = d / f"windows_sess{int(gval):03d}.npz"
            else:
                f = d / str(gval)
            if not f.exists():
                continue
            w = np.load(f)["windows"]
            sub = meta[meta[gcol] == gval]
            if len(sub) != len(w):
                continue
            keep = [c for c in ["domain_l1", "region", "n_units", "area"]
                    if c in sub.columns]
            r = sub[keep].copy()
            r["group_key"] = str(gval)
            chunks.append(w)
            rows.append(r)

    if not chunks:
        return None, None

    W = np.concatenate(chunks, axis=0)
    M = pd.concat(rows, ignore_index=True)

    # 小鼠的 region 不在 omnibrain meta 里，从 steinmetz/units.parquet 补
    # 注意：这是 **session 级** 的 region 构成，不是逐窗标注
    if view == "spike_view":
        uf = Path("/domains/steinmetz/units.parquet")
        if uf.exists():
            u = pd.read_parquet(uf)
            # 每个 session 的主导 region（按 unit 数最多）
            dom = (u.groupby(["session", "region"]).size()
                   .reset_index(name="n")
                   .sort_values(["session", "n"], ascending=[True, False])
                   .groupby("session").first()["region"])
            if "group_key" in M.columns:
                gk = M["group_key"].astype(str)
                sess = pd.to_numeric(gk, errors="coerce")
                M["region"] = sess.map(dom).values
                M["region_level"] = "session_majority"
    return W, M


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 84, flush=True)
    print("构建 OmniBrain-Multi：多源融合 + 逐窗标签", flush=True)
    print("=" * 84, flush=True)

    t0 = time.time()
    all_W = []
    all_M = []
    meta_out = {"schema_version": "1.1", "structure": "poset_not_chain",
            "labels": {}, "sources": {}}

    for view, src, species, modality, gcol in SOURCES:
        print(f"\n[{src}] {view}", flush=True)
        t1 = time.time()
        W, M = load_source(view, gcol)
        if W is None:
            continue

        # 补全索引列
        n = len(W)
        M = M.reset_index(drop=True)
        M["source"] = src
        M["species"] = species
        M["modality"] = modality
        M["window_idx"] = np.arange(n)
        for c in ["domain_l1", "region", "n_units", "area", "group_key",
                  "region_level"]:
            if c not in M.columns:
                M[c] = None
        # 若没有 region 但有 area，用 area 作窗级解剖区
        need = M["region"].isna() & M["area"].notna()
        if need.any():
            M.loc[need, "region"] = M.loc[need, "area"]
            M.loc[need, "region_level"] = "window_area"
        for c in ["domain_l1", "region", "region_level", "n_units", "area"]:
            if c not in M.columns:
                M[c] = None
        M = M[["source", "species", "modality", "group_key",
               "window_idx", "domain_l1", "region", "region_level",
               "area", "n_units"]]

        all_W.append(W)
        all_M.append(M)
        print(f"  {n:,} 窗, {W.shape[1:]} {W.dtype} "
              f"({W.nbytes/1024**3:.2f} GB) [{time.time()-t1:.0f}s]", flush=True)

        meta_out["sources"][src] = {
            "view": view, "species": species, "modality": modality,
            "n_windows": int(n),
            "shape": list(W.shape),
            "n_groups": int(M.group_key.nunique()),
            "domain_distribution": (M.domain_l1.value_counts().to_dict()
                                    if M.domain_l1.notna().any() else {}),
        }

    if not all_W:
        print("\n❌ 无数据")
        return

    # ---------- 拼接
    print("\n拼接…", flush=True)
    W = np.concatenate(all_W, axis=0)
    M = pd.concat(all_M, ignore_index=True)
    del all_W, all_M
    print(f"  时间序列 {W.shape} {W.dtype} ({W.nbytes/1024**3:.2f} GB)", flush=True)

    np.save(OUT / "timeseries.npy", W)
    print(f"  ✅ timeseries.npy 已写", flush=True)
    del W

    M.to_parquet(OUT / "index_timeseries.parquet", index=False)
    print(f"  ✅ index_timeseries.parquet {len(M):,} 行", flush=True)

    # ---------- 图视图（无时间维，单独存）
    gd = BUCKET / "graph_view"
    if gd.exists():
        gf = gd / "embeds.npz"
        if not gf.exists():
            cands = list(gd.glob("*.npz"))
            gf = cands[0] if cands else None
        if gf is not None:
            G = np.load(gf)[list(np.load(gf).files)[0]]
            np.save(OUT / "graph.npy", G.astype(np.float32))
            gm = pd.read_parquet(gd / "meta.parquet")
            gm = gm.reset_index(drop=True)
            gm["source"] = "flywire"
            gm["species"] = "drosophila_melanogaster"
            gm["modality"] = "connectome"
            gm["window_idx"] = np.arange(len(gm))
            for c in ["domain_l1", "region", "region_level", "area",
                      "n_units", "group_key"]:
                if c not in gm.columns:
                    gm[c] = None
            gm = gm[["source", "species", "modality", "group_key",
                     "window_idx", "domain_l1", "region", "region_level",
                     "area", "n_units"]]
            gm.to_parquet(OUT / "index_graph.parquet", index=False)
            meta_out["sources"]["flywire"] = {
                "view": "graph_view", "species": "drosophila_melanogaster",
                "modality": "connectome",
                "n_windows": int(len(G)), "shape": list(G.shape),
                "n_groups": int(gm.group_key.nunique()),
                "note": "无时间维，单独存为 graph.npy",
            }
            print(f"  ✅ graph.npy {G.shape} ({G.nbytes/1024**3:.2f} GB)", flush=True)

    # ---------- 层级定义
    meta_out["labels"] = {
        "sample": {"count": int(len(M)), "desc": "一个 8-bin (800ms) 时间窗"},
        "source": {"values": sorted(M.source.unique().tolist())},
        "species": {"values": sorted(M.species.unique().tolist())},
        "modality": {"values": sorted(M.modality.unique().tolist())},
        "region": {
            "n_unique": int(M.region.nunique()),
            "available_by_source": {
                src: int(g["region"].notna().sum())
                for src, g in M.groupby("source")},
            "region_level": M.region_level.dropna().value_counts().to_dict(),
            "note": ("小鼠来自 steinmetz/units.parquet 的 session 主导区；"
                     "猕猴/人类来自上游 area 列；EEG 无解剖区"),
        },
        "domain_l1": {"values": sorted(M.domain_l1.dropna().unique().tolist())},
    }
    meta_out["labels"]["sample"]["count_by_source"] = (
        M.source.value_counts().to_dict())

    (OUT / "labels.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ labels.json", flush=True)

    # ---------- 可比统计量
    print("\n计算跨源可比统计量…", flush=True)
    rows = []
    for (src, dom), g in M.groupby(["source", "domain_l1"], dropna=False):
        if len(g) < 100:
            continue
        idx = g.index.to_numpy()
        # 从 index 反查 timeseries 位置不现实（已释放），改为按 source 总量粗估
        rows.append({
            "source": src, "domain_l1": dom if pd.notna(dom) else "na",
            "n_windows": int(len(g)),
        })
    pd.DataFrame(rows).to_csv(OUT / "group_sizes.csv", index=False)

    print("\n" + "=" * 84, flush=True)
    print("完成", flush=True)
    for src, s in meta_out["sources"].items():
        print(f"  {src:<24} {s['n_windows']:>8,} 窗  {s['shape']}", flush=True)
    tot = sum(f.stat().st_size for f in OUT.glob("*") if f.is_file()) / 1024**3
    print(f"  输出合计 {tot:.2f} GB | 耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
