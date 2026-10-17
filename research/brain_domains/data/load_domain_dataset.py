"""脑域数据集加载器 + 自检。

用法:
    python load_domain_dataset.py <数据集目录>          # 自检
    from load_domain_dataset import DomainDataset       # 编程使用

自检项 (任何一项失败即报错, 不静默通过):
  1. units 表与每个 session 的尖峰矩阵对齐 (units 数 == 矩阵行数)
  2. domain_id 与 functional_domain 一一对应
  3. 每个 session 的切分边界在合法范围 (0 < train_end < val_end < n_bins)
  4. 尖峰值域有限、非负、无 NaN
  5. FlyWire 特征表无 NaN, safe/leaky 分组完整
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DomainDataset:
    """Steinmetz 域标注尖峰数据集。

    属性
    ----
    units : DataFrame [session, unit_idx, orig_unit_idx, region, functional_domain, domain_id]
    splits : dict session -> {train_end, val_end, n_bins}
    domains : list[str]  按 domain_id 排序的功能域名
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        # 允许传数据集总根 (含 steinmetz/ 子目录) 或 steinmetz 目录本身
        if not (self.root / "units.parquet").exists() and (self.root / "steinmetz" / "units.parquet").exists():
            self.root = self.root / "steinmetz"
        self.units = pd.read_parquet(self.root / "units.parquet")
        self.splits = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))
        dom_map = self.units.drop_duplicates("domain_id").sort_values("domain_id")
        self.domains = dom_map["functional_domain"].tolist()

    def sessions(self):
        return sorted(self.units["session"].unique().tolist())

    def load_session(self, idx: int):
        """返回 (spikes float32 [n_units, n_bins], unit_rows DataFrame, split dict)。"""
        path = self.root / f"spikes_session_{idx:03d}.npz"
        spikes = np.load(path)["spikes"]
        rows = self.units[self.units["session"] == idx].reset_index(drop=True)
        return spikes, rows, self.splits[str(idx)]

    def iter_sessions(self):
        for idx in self.sessions():
            yield idx, *self.load_session(idx)

    def domain_counts(self):
        return (self.units.groupby(["domain_id", "functional_domain"])
                .size().rename("n_units").reset_index().sort_values("domain_id"))


def sanity_check(root: str | Path, verbose=True) -> bool:
    root = Path(root)
    ds = DomainDataset(root)
    data_root = ds.root.parent if ds.root.name == "steinmetz" else ds.root
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  ❌ {msg}")
        elif verbose:
            print(f"  ✅ {msg}")
        return cond

    print("=== 1. 尖峰矩阵与 units 对齐 ===")
    total_units = 0
    for idx, spikes, rows, sp in ds.iter_sessions():
        chk(spikes.shape[0] == len(rows),
            f"session {idx}: {spikes.shape[0]} rows == {len(rows)} units")
        chk(spikes.shape[1] == sp["n_bins"],
            f"session {idx}: {spikes.shape[1]} bins == metadata {sp['n_bins']}")
        total_units += len(rows)
    chk(total_units == len(ds.units), f"总 unit 数 {total_units} == units 表 {len(ds.units)}")

    print("\n=== 2. domain_id 映射 ===")
    m = ds.units.groupby("functional_domain")["domain_id"].nunique()
    chk((m == 1).all(), "每个 functional_domain 唯一对应一个 domain_id")
    chk(ds.units["domain_id"].isin(range(len(ds.domains))).all(),
        f"domain_id ∈ [0, {len(ds.domains)-1}]")

    print("\n=== 3. 切分边界 ===")
    for idx, sp in ds.splits.items():
        chk(0 < sp["train_end"] < sp["val_end"] < sp["n_bins"],
            f"session {idx}: 0 < {sp['train_end']} < {sp['val_end']} < {sp['n_bins']}")

    print("\n=== 4. 尖峰值域 (抽查 3 个 session) ===")
    for idx in ds.sessions()[:3]:
        spikes, _, _ = ds.load_session(idx)
        chk(np.isfinite(spikes).all(), f"session {idx}: 无 NaN/Inf")
        chk(spikes.min() >= 0, f"session {idx}: 非负 (min={spikes.min():.4f})")

    print("\n=== 5. FlyWire 特征表 ===")
    fw_dir = data_root / "flywire"
    if (fw_dir / "node_features.parquet").exists():
        feats = pd.read_parquet(fw_dir / "node_features.parquet")
        fs = json.loads((fw_dir / "feature_sets.json").read_text(encoding="utf-8"))
        num_cols = [c for c in feats.columns if c.startswith(("in_", "out_"))]
        chk(not feats[num_cols].isna().any().any(), f"数值列无 NaN ({len(num_cols)} 列)")
        chk(all(c in feats.columns for c in fs["safe"]), "safe 特征列全部存在")
        chk(all(c in feats.columns for c in fs["leaky"]), "leaky 特征列全部存在")
        chk(len(feats) > 0, f"神经元 {len(feats):,} 条")
    else:
        print("  (跳过, 无 flywire 目录)")

    print("\n" + "=" * 50)
    print("自检结果:", "✅ 全部通过" if ok else "❌ 有失败项")
    return ok


def describe(root: str | Path):
    """打印数据集概要, 用于快速决策能否支撑某类实验。"""
    ds = DomainDataset(root)
    print(f"数据集: {root}")
    print(f"  sessions: {len(ds.sessions())}")
    print(f"  units:    {len(ds.units):,}")
    print(f"  domains:  {len(ds.domains)}")
    print()
    print("域分布:")
    print(ds.domain_counts().to_string(index=False))

    print("\n每 session 可用域数分布:")
    per = ds.units.groupby("session")["domain_id"].nunique()
    print(f"  min={per.min()} median={per.median():.0f} max={per.max()}")

    print("\n跨域实验可行性 (leave-one-domain-out 需要的样本量):")
    cnt = ds.units.groupby("functional_domain").size()
    usable = cnt[cnt >= 100].sort_values(ascending=False)
    print(f"  ≥100 units 的域: {len(usable)} / {len(cnt)}")
    for d, n in usable.items():
        print(f"    {d:16s} {n:6,d}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    describe(root)
    print()
    sys.exit(0 if sanity_check(root) else 1)
