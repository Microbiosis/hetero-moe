"""从桶内已有数据生成脑域索引表.

输入(均已在 illusion-brain/training-data 桶内):
  connectome/flywire-connectome/nodes.parquet   果蝇, 139,255 神经元
  spikes/steinmetz-spikes/metadata.json         小鼠, 27,212 神经元-脑区记录

输出:
  flywire_regions.csv    果蝇脑区 -> 神经元数 / 类别编码
  steinmetz_regions.csv  小鼠脑区 -> 神经元数 / 层级归并
  domain_crosswalk.csv   两者的功能域对照(跨物种)

用途: 为域条件 SSM 提供 n_domains 与 domain_id, 并支持跨物种域结构比较。
本脚本只读桶数据, 不做任何模型训练。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# 果蝇脑区前缀归并到功能大域(参考细胞类型缩写与已知功能系统)
FLY_FUNCTIONAL = {
    "ME": "visual_processing",
    "LO": "visual_processing",
    "LOP": "visual_processing",
    "LA": "olfactory",
    "olfactory": "olfactory",
    "Kenyon_Cell": "olfactory",
    "ALPN": "olfactory",
    "ALLN": "olfactory",
    "mushroom_body": "memory",
    "CX": "central_complex",
    "mechanosensory": "sensory",
    "gustatory": "sensory",
    "AN": "ascending",
    "DAN": "descending",
    "visual": "visual_system",
}

# 小鼠脑区 → 功能大域(Allen CCF 缩写)
STEINMETZ_FUNCTIONAL = {
    "VISp": "visual", "VISl": "visual", "VISal": "visual",
    "VISam": "visual", "VISpm": "visual", "VISrl": "visual", "VISa": "visual",
    "MOp": "motor", "MOs": "motor", "MY": "brainstem",
    "CA1": "hippocampus", "CA2": "hippocampus", "CA3": "hippocampus",
    "DG": "hippocampus", "SUB": "hippocampus",
    "PL": "prefrontal", "IL": "prefrontal", "ACA": "prefrontal",
    "ORB": "prefrontal", "ORBm": "prefrontal",
    "AUD": "auditory", "COA": "auditory", "BLA": "auditory",
    "LGd": "thalamus", "LD": "thalamus", "LP": "thalamus",
    "MD": "thalamus", "VPL": "thalamus", "VPM": "thalamus",
    "MG": "thalamus", "PO": "thalamus", "VAL": "thalamus",
    "SSp": "somatosensory", "SSs": "somatosensory", "SI": "somatosensory",
    "LS": "septal", "LSc": "septal", "LSr": "septal", "CL": "septal",
    "CP": "striatum", "ACB": "striatum", "PALP": "basal_ganglia",
    "TH": "hypothalamus", "MB": "midbrain",
}


def build_flywire(nodes_path: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(nodes_path)

    # 方案 C: 从 cell_class 抽脑区前缀, 丢掉缺失(23%) 与极端长尾(<10 神经元)
    cc = df["cell_class"].dropna().astype(str)
    df = df.loc[cc.index].copy()
    df["region"] = cc.map(lambda s: s.split(">")[0].split(".")[0])

    counts = df["region"].value_counts()
    keep = counts[counts >= 1000].index
    sub = df[df["region"].isin(keep)]

    rows = []
    for i, (region, n) in enumerate(counts[counts >= 1000].items()):
        rows.append({
            "domain_id": i,
            "region": region,
            "n_neurons": int(n),
            "functional_domain": FLY_FUNCTIONAL.get(region, "other"),
            "species": "drosophila",
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "flywire_regions.csv", index=False)
    print(f"[FlyWire] 保留 {len(out)} 个脑区, 覆盖 {out['n_neurons'].sum():,} 神经元 "
          f"({100 * out['n_neurons'].sum() / len(df):.1f}%)")
    print(out.groupby("functional_domain")["n_neurons"].agg(["count", "sum"]).to_string())
    return out


def build_steinmetz(meta_path: Path, out_dir: Path) -> pd.DataFrame:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    counter = Counter()
    for s in meta["sessions"]:
        for r in s.get("brain_regions", []):
            counter[str(r)] += 1

    rows = []
    for i, (region, n) in enumerate(sorted(counter.items(), key=lambda x: -x[1])):
        if n < 10:  # 丢弃极端长尾
            continue
        rows.append({
            "domain_id": len(rows),
            "region": region,
            "n_neurons": int(n),
            "functional_domain": STEINMETZ_FUNCTIONAL.get(region, "other"),
            "species": "mus_musculus",
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "steinmetz_regions.csv", index=False)
    print(f"\n[Steinmetz] 保留 {len(out)} 个脑区, 覆盖 {out['n_neurons'].sum():,} 神经元-脑区记录")
    print(out.groupby("functional_domain")["n_neurons"].agg(["count", "sum"]).to_string())
    return out


# 跨物种功能域归一: 两套命名统一到同一批语义标签, 否则对照表全是 0/0 无法用。
# 果蝇的视觉分三块(视觉通路/视觉系统/中枢复合体), 小鼠统一叫 visual;
# 果蝇嗅觉对应小鼠的嗅觉皮层, 虽不在 Steinmetz 采样区, 仍保留以示物种差异。
CROSSWALK_CANON = {
    "visual_processing": "visual",
    "visual_system": "visual",
    "olfactory": "olfactory",
    "central_complex": "central_complex",
    "sensory": "somatosensory",
    "ascending": "thalamus",
    "auditory": "auditory",
    "hippocampus": "hippocampus",
    "prefrontal": "prefrontal",
    "motor": "motor",
    "thalamus": "thalamus",
    "somatosensory": "somatosensory",
    "midbrain": "brainstem",
    "hypothalamus": "hypothalamus",
    "septal": "septal",
    "striatum": "striatum",
    "memory": "hippocampus",
    "basal_ganglia": "striatum",
    "brainstem": "brainstem",
    "descending": "brainstem",
    "other": "other",
}


def build_crosswalk(fly: pd.DataFrame, stein: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """按归一后的功能语义建跨物种对照, 供跨物种域结构比较."""
    fly_dom = fly.groupby("functional_domain")["n_neurons"].sum()
    ste_dom = stein.groupby("functional_domain")["n_neurons"].sum()

    f = {CROSSWALK_CANON.get(k, k): v for k, v in fly_dom.items()}
    s = {CROSSWALK_CANON.get(k, k): v for k, v in ste_dom.items()}

    rows = []
    for name in sorted(set(f) | set(s)):
        rows.append({
            "functional_domain": name,
            "flywire_neurons": int(f.get(name, 0)),
            "steinmetz_neurons": int(s.get(name, 0)),
            "in_both": name in f and name in s,
        })

    out = pd.DataFrame(rows).sort_values(
        ["in_both", "flywire_neurons", "steinmetz_neurons"], ascending=False)
    out.to_csv(out_dir / "domain_crosswalk.csv", index=False)
    print("\n[跨物种对照] 功能语义归一后的映射表")
    print(out.to_string(index=False))
    shared = out[out["in_both"]]
    print(f"\n两物种共有功能域: {len(shared)} 个 — {', '.join(shared['functional_domain'])}")
    return out


if __name__ == "__main__":
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir = work / "domain_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    fly = build_flywire(work / "fw_nodes.parquet", out_dir)
    stein = build_steinmetz(work / "stein_meta.json", out_dir)
    build_crosswalk(fly, stein, out_dir)
    print(f"\n输出目录: {out_dir}")
