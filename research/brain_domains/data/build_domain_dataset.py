"""构建脑域数据集 v1 — 从桶内原始数据生成可训练/可评测的域标注数据集。

## 产出两个子数据集

### A. steinmetz (主数据集, 推荐先用)
来源: 桶内 spikes/steinmetz-spikes/
  39 个 session, 每 session (n_units, 54053) uint8 尖峰计数 (50ms 分箱)
  metadata.json 提供逐 unit 的脑区标签 + train/val/test 时间切分边界

产出:
  steinmetz/spikes_session_XXX.npz   每个 session 的尖峰矩阵 (float32, 已归一化)
  steinmetz/units.parquet            逐 unit: session, unit_idx, region, functional_domain, domain_id
  steinmetz/splits.json              逐 session 的时间切分 (train_end/val_end 来自元数据)

任务支持:
  - 域条件群体预测: 用 t 时刻全体 unit 的活动预测 t+1, 模型额外接收 unit 的 domain_id
  - 跨域泛化: 按 functional_domain 做 leave-one-domain-out
  - 域分类 sanity check: 由活动模式预测所属域

### B. flywire (对照, 需谨慎)
来源: 桶内 connectome/flywire-connectome/
  nodes.parquet        139,255 神经元元数据 (cell_class → 脑区前缀)
  connections.parquet  15,091,983 条突触 (pre, post, syn_count, 6 种神经递质分数)

产出:
  flywire/node_features.parquet   逐神经元: 连接组统计特征 + region + domain_id
  flywire/feature_sets.json       特征分组 (safe / leaky) — 见泄漏警告

⚠️ 泄漏警告 (README 陷阱 #2):
  连接组特征与脑区身份高度耦合 —— 神经元在特定脑区, 其输入输出伙伴分布本就不同。
  因此「用连接组特征预测脑区」接近同义反复。本数据集把特征分成两组:
    safe:  仅总出入度与突触总数 (粗粒度, 泄漏较弱)
    leaky: 加上按神经递质加权的分布 (编码了伙伴类型 = 脑区身份)
  任何「域结构有用」的结论必须用 safe 组 + 随机标签对照得出。

## 共同约束
  - 排除未标注区域 (Steinmetz 的 "root", FlyWire 的 cell_class 缺失)
  - 区域按功能归并 (functional_domain) 供小规模实验; 原始 region 保留供细粒度分析
  - 每个域至少 MIN_UNITS 个样本, 否则并入 other
  - 全部输出随附 MANIFEST.json: 形状/计数/域分布/校验和

用法:
    python build_domain_dataset.py <数据目录> <输出目录>

数据目录需包含: stein_meta.json, session_000.npy..session_038.npy,
               fw_nodes.parquet, fw_conn.parquet (可选, 缺失则跳过 flywire)
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_UNITS = 40          # 域最小样本数, 低于则归入 other
SPIKE_SCALE = 1.0 / 8.0  # uint8 计数 → float32 的缩放 (最大 35 → ~4.4, 避免激活过大)

# Steinmetz 小鼠脑区 → 功能域 (Allen CCF 缩写前缀匹配)
STEINMETZ_FUNCTIONAL_PREFIXES = [
    ("VIS", "visual"), ("AUD", "auditory"), ("SSp", "somatosensory"),
    ("SSs", "somatosensory"), ("MOp", "motor"), ("MOs", "motor"),
    ("ACA", "prefrontal"), ("PL", "prefrontal"), ("ILA", "prefrontal"),
    ("ORB", "prefrontal"), ("FRP", "prefrontal"),
    ("CA1", "hippocampus"), ("CA2", "hippocampus"), ("CA3", "hippocampus"),
    ("DG", "hippocampus"), ("SUB", "hippocampus"), ("ProS", "hippocampus"),
    ("LGd", "thalamus"), ("LP", "thalamus"), ("LD", "thalamus"),
    ("MD", "thalamus"), ("PO", "thalamus"), ("VPM", "thalamus"),
    ("VPL", "thalamus"), ("MG", "thalamus"), ("VAL", "thalamus"),
    ("CP", "striatum"), ("ACB", "striatum"), ("STR", "striatum"),
    ("LS", "septal"), ("TH", "hypothalamus"), ("ZI", "hypothalamus"),
    ("MRN", "midbrain"), ("MB", "midbrain"), ("SNr", "midbrain"),
    ("VTA", "midbrain"), ("SC", "midbrain"), ("IC", "midbrain"),
    ("MY", "brainstem"), ("PAG", "brainstem"), ("RN", "brainstem"),
]

# FlyWire 脑区前缀 → 功能域
FLYWIRE_FUNCTIONAL = {
    "ME": "visual", "LO": "visual", "LOP": "visual", "LA": "olfactory",
    "Kenyon_Cell": "memory", "mushroom_body": "memory", "CX": "central_complex",
    "mechanosensory": "somatosensory", "gustatory": "somatosensory",
    "AN": "ascending", "DAN": "descending", "visual": "visual",
    "olfactory": "olfactory", "ALPN": "olfactory", "ALLN": "olfactory",
}


def classify_steinmetz(region: str) -> str:
    r = str(region)
    for prefix, dom in STEINMETZ_FUNCTIONAL_PREFIXES:
        if r.startswith(prefix):
            return dom
    return "other"


def classify_flywire(region: str) -> str:
    return FLYWIRE_FUNCTIONAL.get(str(region), "other")


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- Steinmetz
def build_steinmetz(data_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "stein_meta.json").read_text(encoding="utf-8"))

    rows = []
    splits = {}
    stats = []

    for s in meta["sessions"]:
        idx = s["index"]
        npy = data_dir / f"session_{idx:03d}.npy"
        if not npy.exists():
            print(f"  [skip] session {idx}: 缺少 {npy.name}")
            continue

        regions = s["brain_regions"]
        arr = np.load(npy)
        assert arr.shape[0] == len(regions), \
            f"session {idx}: 尖峰 {arr.shape[0]} units != 标签 {len(regions)}"

        # 排除未标注
        keep = [i for i, r in enumerate(regions) if str(r).strip().lower() != "root"]
        kept_regions = [str(regions[i]) for i in keep]

        x = arr[keep].astype(np.float32) * SPIKE_SCALE
        np.savez_compressed(out / f"spikes_session_{idx:03d}.npz", spikes=x)

        for j, r in enumerate(kept_regions):
            rows.append({
                "session": idx,
                "unit_idx": j,
                "orig_unit_idx": keep[j],
                "region": r,
                "functional_domain": classify_steinmetz(r),
            })

        sb = s.get("split_boundaries", {})
        splits[str(idx)] = {
            "train_end": int(sb.get("train_end", arr.shape[1] * 2 // 3)),
            "val_end": int(sb.get("val_end", arr.shape[1] * 5 // 6)),
            "n_bins": int(arr.shape[1]),
        }
        stats.append({
            "session": idx, "n_units_raw": int(arr.shape[0]),
            "n_units_kept": int(len(keep)), "n_bins": int(arr.shape[1]),
            "spike_max_raw": int(arr.max()),
            "duration_s": float(s.get("duration_s", 0)),
        })
        print(f"  session {idx:02d}: {len(keep)}/{arr.shape[0]} units, "
              f"{arr.shape[1]} bins, 保留 {x.shape[1]} bins")

    units = pd.DataFrame(rows)

    # 低频域归入 other, 再按 MIN_UNITS 过滤域
    dom_counts = units["functional_domain"].value_counts()
    small = [d for d, n in dom_counts.items() if n < MIN_UNITS and d != "other"]
    if small:
        print(f"  低样本功能域并入 other: {small}")
        units.loc[units["functional_domain"].isin(small), "functional_domain"] = "other"

    # domain_id 按功能域分配
    doms = sorted(units["functional_domain"].unique())
    dom2id = {d: i for i, d in enumerate(doms)}
    units["domain_id"] = units["functional_domain"].map(dom2id)

    units.to_parquet(out / "units.parquet", index=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    dom_dist = units.groupby("functional_domain").agg(
        n_units=("unit_idx", "size"), domain_id=("domain_id", "first")).reset_index()

    print(f"\n[Steinmetz] {len(units):,} units, {units['session'].nunique()} sessions")
    print(dom_dist.to_string(index=False))

    return {
        "name": "steinmetz_domain_v1",
        "n_sessions": int(units["session"].nunique()),
        "n_units": int(len(units)),
        "n_domains": int(len(doms)),
        "domains": doms,
        "domain_distribution": dom_dist.to_dict("records"),
        "sessions": stats,
        "files": sorted(p.name for p in out.glob("*")),
        "checksums": {p.name: sha256_of(p) for p in out.glob("*") if p.suffix in (".npz", ".parquet")},
        "task": "域条件群体预测 / 跨域泛化 / 域分类 sanity check",
        "splits_source": "metadata.json 的 split_boundaries (时间切分)",
    }


# ------------------------------------------------------------------ FlyWire
def build_flywire(data_dir: Path, out: Path) -> dict | None:
    nodes_path = data_dir / "fw_nodes.parquet"
    conn_path = data_dir / "fw_conn.parquet"
    if not nodes_path.exists():
        print("[FlyWire] 缺少 nodes.parquet, 跳过")
        return None
    out.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(nodes_path)
    cc = nodes["cell_class"].astype("string")
    nodes["region"] = cc.str.split(">").str[0].str.split(".").str[0]
    nodes = nodes[nodes["region"].notna() & (nodes["region"] != "")].copy()
    print(f"[FlyWire] 有 region 的神经元: {len(nodes):,} / 139,255")

    ok_regions = [r for r, n in nodes["region"].value_counts().items() if n >= MIN_UNITS]
    nodes = nodes[nodes["region"].isin(ok_regions)].copy()
    print(f"[FlyWire] 区域数 {len(ok_regions)}, 保留神经元 {len(nodes):,}")

    feats = nodes[["root_id", "region"]].copy()

    if conn_path.exists():
        print(f"[FlyWire] 聚合连接表 {conn_path.stat().st_size / 1e6:.0f} MB ...")
        conn = pd.read_parquet(conn_path)
        conn = conn[conn["syn_count"] > 0]

        for side, key in (("in", "post"), ("out", "pre")):
            g = conn.groupby(key)
            agg = g.agg(degree=("syn_count", "size"), synapses=("syn_count", "sum"))
            # 逐递质加权: 用向量化累加 (groupby.apply 在 1500 万行上太慢)
            conn["_w"] = conn["syn_count"].astype("float64")
            nt_agg = {}
            for nt in ("gaba", "ach", "glut", "oct", "ser", "da"):
                conn["_v"] = conn["_w"] * conn[nt].astype("float64")
                nt_agg[nt] = conn.groupby(key)["_v"].sum()

            block = agg.join(pd.DataFrame(nt_agg))
            block.columns = [f"{side}_degree", f"{side}_synapses"] + \
                            [f"{side}_{c}_syn" for c in ("gaba", "ach", "glut", "oct", "ser", "da")]
            feats = feats.merge(block, left_on="root_id", right_index=True, how="left")

        for c in feats.columns:
            if c.startswith(("in_", "out_")):
                feats[c] = feats[c].fillna(0)
        feats["in_syn_per_edge"] = feats["in_synapses"] / feats["in_degree"].clip(lower=1)
        feats["out_syn_per_edge"] = feats["out_synapses"] / feats["out_degree"].clip(lower=1)
        del conn
    else:
        print("[FlyWire] 缺少 connections.parquet, 仅输出元数据 (无连接组特征)")
        # 无连接表时补零列, 保证下游 schema 一致
        for side in ("in", "out"):
            feats[f"{side}_degree"] = 0
            feats[f"{side}_synapses"] = 0
            feats[f"{side}_syn_per_edge"] = 0.0
            for nt in ("gaba", "ach", "glut", "oct", "ser", "da"):
                feats[f"{side}_{nt}_syn"] = 0.0

    feats["functional_domain"] = feats["region"].map(classify_flywire)
    doms = sorted(feats["functional_domain"].unique())
    d2i = {d: i for i, d in enumerate(doms)}
    feats["domain_id"] = feats["functional_domain"].map(d2i)
    feats.to_parquet(out / "node_features.parquet", index=False)

    safe_cols = ["in_degree", "out_degree", "in_synapses", "out_synapses",
                 "in_syn_per_edge", "out_syn_per_edge"]
    leaky_cols = [c for c in feats.columns
                  if any(f"_{nt}_syn" in c for nt in ("gaba", "ach", "glut", "oct", "ser", "da"))]
    (out / "feature_sets.json").write_text(json.dumps({
        "safe": safe_cols,
        "leaky": leaky_cols,
        "warning": ("连接组的递质加权分布编码了输入输出伙伴的类型, 而伙伴类型与脑区身份高度耦合 "
                    "(README 陷阱 #2)。任何「域结构有用」的结论必须用 safe 组 + 随机标签对照得出。"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    dist = feats.groupby("functional_domain").agg(
        n_neurons=("root_id", "size"), domain_id=("domain_id", "first")).reset_index()
    print(f"\n[FlyWire] {len(feats):,} 神经元")
    print(dist.to_string(index=False))

    return {
        "name": "flywire_domain_v1",
        "n_neurons": int(len(feats)),
        "n_regions": int(feats["region"].nunique()),
        "n_domains": int(len(doms)),
        "domains": doms,
        "domain_distribution": dist.to_dict("records"),
        "feature_sets": {"safe": safe_cols, "leaky": leaky_cols},
        "files": sorted(p.name for p in out.glob("*")),
        "checksums": {p.name: sha256_of(p) for p in out.glob("*") if p.suffix == ".parquet"},
        "leakage_note": "见 feature_sets.json 的 warning 字段",
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    data_dir, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("脑域数据集构建 v1")
    print("=" * 70)

    manifest = {"schema_version": "1.0", "built_from": str(data_dir),
                "min_units_per_domain": MIN_UNITS,
                "spike_scale": SPIKE_SCALE, "datasets": {}}

    print("\n--- A. Steinmetz (小鼠, 尖峰) ---")
    manifest["datasets"]["steinmetz"] = build_steinmetz(data_dir, out_root / "steinmetz")

    print("\n--- B. FlyWire (果蝇, 连接组) ---")
    fw = build_flywire(data_dir, out_root / "flywire")
    if fw:
        manifest["datasets"]["flywire"] = fw

    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"完成 → {out_root}")
    for name, d in manifest["datasets"].items():
        print(f"  {name}: {d['n_domains']} domains, files={len(d['files'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
