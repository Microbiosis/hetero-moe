"""灵长类脑域数据集加载器 + 自检。

用法:
    python load_primate.py <primate 目录>

自检项:
  1. human_eeg: 每个被试的 npz 形状 (n_trials, 63, 250) 与 trials 表行数一致
  2. human_eeg: 通道表 63 行, 5 域, domain_id 与 region 一一对应
  3. human_eeg: 数值有限, 量级合理 (EEG 白化后 |x| 通常 < 50)
  4. human_eeg: session 分布与切分可行性 (是否有 session 3 作测试)
  5. macaque_nlb: 每个片段 npz 的 unit 数与 segments 表一致
  6. macaque_nlb: units 表按 (dataset, session, unit_idx) 唯一
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EEG_T = 250
EEG_CH = 63


class HumanEEGDataset:
    """THINGS-EEG 域标注数据集。

    属性
    ----
    channels : DataFrame [channel_idx, channel, region, domain_id]
    trials   : DataFrame [subject, trial_idx, src_index, session, label, text]
    subjects : list[str]
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.channels = pd.read_csv(self.root / "channels.csv")
        self.trials = pd.read_parquet(self.root / "trials.parquet")
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.subjects = sorted(self.trials["subject"].unique().tolist())

    def load_split(self, sub: str, split: str):
        """返回 (eeg float16 [n, 63, 250], 该被试该切分的 trials 子表)。"""
        d = np.load(self.root / f"eeg_{sub}_{split}.npz")
        t = self.trials[(self.trials["subject"] == sub) &
                        (self.trials["split"] == split)].reset_index(drop=True)
        return d["eeg"], t

    def splits(self):
        return sorted(self.trials["split"].unique().tolist())

    def iter_subject_splits(self):
        for s in self.subjects:
            for sp in self.splits():
                arr, t = self.load_split(s, sp)
                yield s, sp, arr, t

    def get(self, split: str):
        """按切分取全部被试的数据 (拼接)。"""
        arrs, ts = [], []
        for s in self.subjects:
            a, t = self.load_split(s, split)
            arrs.append(a); ts.append(t)
        return np.concatenate(arrs, axis=0), pd.concat(ts, ignore_index=True)

    def domain_of_channel(self):
        return self.channels["domain_id"].to_numpy()


class MacaqueDataset:
    """NLB 猕猴数据集（域标签为数据集级，见 meta 的 caveat）。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.units = pd.read_parquet(self.root / "units.parquet")
        self.segments = pd.read_parquet(self.root / "segments.parquet")
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))

    def load_segment(self, filename: str):
        d = np.load(self.root / filename)
        return d["spikes"]

    def iter_segments(self):
        for _, r in self.segments.iterrows():
            yield r, self.load_segment(r["file"])


def sanity_check(root: str | Path) -> bool:
    root = Path(root)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  ❌ {msg}")
        else:
            print(f"  ✅ {msg}")

    # ---------------- human_eeg
    he = root / "human_eeg"
    if he.exists():
        print("=== 1. human_eeg 对齐 ===")
        ds = HumanEEGDataset(he)
        bad = 0
        for sub, sp, eeg, tr in ds.iter_subject_splits():
            if eeg.shape[0] != len(tr) or eeg.shape[1:] != (EEG_CH, EEG_T):
                bad += 1
                print(f"      {sub}/{sp}: eeg {eeg.shape} vs trials {len(tr)}")
        n_files = len(list(he.glob("eeg_*_*.npz")))
        chk(bad == 0, f"{n_files} 个 (被试×切分) 文件的 eeg 与 trials 表逐一对齐")

        print("\n=== 2. 通道与域映射 ===")
        chk(len(ds.channels) == EEG_CH, f"通道数 {len(ds.channels)} == {EEG_CH}")
        m = ds.channels.groupby("region")["domain_id"].nunique()
        chk((m == 1).all(), "每个 region 唯一对应一个 domain_id")
        nd = ds.channels["domain_id"].nunique()
        chk(nd == 5, f"域数 {nd} == 5")

        print("\n=== 3. 数值检查 (全部被试) ===")
        worst = 0.0
        for sub, sp, eeg, _ in ds.iter_subject_splits():
            worst = max(worst, float(np.abs(eeg).max()))
            if not np.isfinite(eeg).all():
                chk(False, f"{sub}/{sp}: 含 NaN/Inf")
        chk(worst < 1e3, f"全部有限, 最大 |x| = {worst:.1f} (< 1e3)")

        print("\n=== 4. 三切分可行性 ===")
        sps = ds.splits()
        chk(sps == ["test", "train", "val"], f"切分 = {sps}")
        for sp in sps:
            a, t = ds.get(sp)
            chk(len(a) == len(t) and len(t) > 0, f"{sp}: {len(t)} 条, 形状 {a.shape[1:]}")

    # ---------------- macaque
    mc = root / "macaque_nlb"
    if mc.exists():
        print("\n=== 5. macaque_nlb 片段对齐 ===")
        md = MacaqueDataset(mc)
        bad = 0
        for r, sp in md.iter_segments():
            if sp.shape != (r["n_units"], r["n_bins"]):
                bad += 1
        chk(bad == 0, f"{len(md.segments)} 个片段的 npz 形状与 segments 表一致")

        print("\n=== 6. units 唯一性 ===")
        dup = md.units.duplicated(subset=["dataset", "session", "unit_idx"]).sum()
        chk(dup == 0, f"(dataset, session, unit_idx) 无重复 (dup={dup})")
        chk(md.units["domain_id"].nunique() == 3, "3 个域 (DMFC / M1_PMd / S1_M1)")
        print(f"      唯一 units {len(md.units)}, 片段 {len(md.segments)}, "
              f"sessions {md.units['session'].nunique()}")

    print("\n" + "=" * 52)
    print("自检结果:", "✅ 全部通过" if ok else "❌ 有失败项")
    return ok


def describe(root: str | Path):
    root = Path(root)
    print(f"灵长类数据集: {root}")
    he = root / "human_eeg"
    if he.exists():
        ds = HumanEEGDataset(he)
        print(f"\n[human_eeg] {len(ds.subjects)} 被试, {len(ds.trials):,} 试次, "
              f"{len(ds.channels)} 通道, {ds.channels['domain_id'].nunique()} 域")
        print(ds.channels.groupby(["domain_id", "region"]).size()
              .rename("n_channels").reset_index().to_string(index=False))
        print("\n每被试试次:")
        print(ds.trials.groupby("subject").size().rename("n_trials").to_string())
    mc = root / "macaque_nlb"
    if mc.exists():
        md = MacaqueDataset(mc)
        print(f"\n[macaque_nlb] {len(md.segments)} 片段, {len(md.units)} 唯一 units, "
              f"{md.units['session'].nunique()} sessions, {md.units['domain_id'].nunique()} 域")
        print(md.units.groupby(["domain_id", "area"]).size()
              .rename("n_units").reset_index().to_string(index=False))
        print("\n每数据集片段数:")
        print(md.segments.groupby("dataset").agg(
            n_segments=("file", "size"), n_sessions=("session", "nunique"),
            bins=("n_bins", "sum")).to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    describe(root)
    print()
    sys.exit(0 if sanity_check(root) else 1)
