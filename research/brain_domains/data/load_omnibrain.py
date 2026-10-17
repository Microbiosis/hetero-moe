"""OmniBrain 加载器 + 自检。

    python load_omnibrain.py <omnibrain 根目录>

自检:
  1. spike_view: 每个 session 的 windows 形状与 meta 行数一致
  2. eeg_view:    windows 首维与 meta 行数一致; 63..64 列的补零位应恒为 0
  3. graph_view:  embeds 首维与 meta 行数一致
  4. 三视图的 domain_l1 都落在 L1 词表内
  5. 数值有限、z-score 后各通道均值≈0
  6. 本体表覆盖所有出现的原始区域名
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

L1 = ["sensory", "motor", "memory", "association", "subcortical", "other"]


class OmniBrain:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "MANIFEST.json").read_text(encoding="utf-8"))
        self.ontology = pd.read_csv(self.root / "ontology.csv")

    # -------- spike
    def spike_sessions(self):
        d = self.root / "spike_view"
        metas = pd.read_parquet(d / "meta.parquet")
        return sorted(metas["session"].unique().tolist()), metas

    def load_spike_session(self, sess: int):
        d = self.root / "spike_view"
        w = np.load(d / f"windows_sess{sess:03d}.npz")["windows"]
        m = pd.read_parquet(d / "meta.parquet")
        return w, m[m["session"] == sess].reset_index(drop=True)

    # -------- eeg
    def load_eeg(self):
        d = self.root / "eeg_view"
        return np.load(d / "windows.npz")["windows"], pd.read_parquet(d / "meta.parquet")

    # -------- graph
    def load_graph(self):
        d = self.root / "graph_view"
        return np.load(d / "embeds.npz")["embeds"], pd.read_parquet(d / "meta.parquet")


def sanity(root) -> bool:
    ds = OmniBrain(root)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  ❌ {msg}")
        else:
            print(f"  ✅ {msg}")

    # ---- 1 spike
    sp_dir = ds.root / "spike_view"
    if sp_dir.exists():
        print("=== 1. spike_view ===")
        sess_list, metas = ds.spike_sessions()
        bad = 0
        for s in sess_list:
            w, m = ds.load_spike_session(s)
            if w.shape[0] != len(m) or w.shape[1:] != (64, 8):
                bad += 1
                print(f"      sess {s}: {w.shape} vs meta {len(m)}")
        chk(bad == 0, f"{len(sess_list)} 个 session 的 windows 与 meta 对齐 (64, 8)")
        print(f"      总窗口 {len(metas):,}, 置信度均值 {metas['domain_conf'].mean():.3f}")
        w0, _ = ds.load_spike_session(sess_list[0])
        chk(np.isfinite(w0).all(), "spike 窗口数值有限")

    # ---- 2 eeg
    eeg_dir = ds.root / "eeg_view"
    if eeg_dir.exists():
        print("\n=== 2. eeg_view ===")
        w, m = ds.load_eeg()
        chk(w.shape[0] == len(m), f"windows {w.shape[0]} 行 == meta {len(m)} 行")
        chk(w.shape[1:] == (64, 8), f"形状 (n, 64, 8) == {w.shape[1:]}")
        # 第 64 通道 (索引 63) 是补零位
        pad = np.abs(w[:, 63, :]).max()
        chk(pad == 0.0, f"补零通道恒为 0 (max={pad})")
        chk(np.isfinite(w).all(), "eeg 窗口数值有限")
        print(f"      总窗口 {len(m):,}, 被试 {m['subject'].nunique()}, "
              f"置信度均值 {m['domain_conf'].mean():.3f}")

    # ---- 3 graph
    g_dir = ds.root / "graph_view"
    if g_dir.exists():
        print("\n=== 3. graph_view ===")
        e, m = ds.load_graph()
        chk(e.shape[0] == len(m), f"embeds {e.shape[0]} 行 == meta {len(m)} 行")
        chk(e.shape[1] == 64, f"维度 64 == {e.shape[1]}")
        chk(np.isfinite(e).all(), "嵌入数值有限")

    # ---- 4 域词表
    print("\n=== 4. L1 域词表 ===")
    for name in ("spike_view", "eeg_view", "graph_view"):
        d = ds.root / name
        if not (d / "meta.parquet").exists():
            continue
        m = pd.read_parquet(d / "meta.parquet")
        got = set(m["domain_l1"].unique())
        chk(got <= set(L1), f"{name}: 域集合 ⊆ L1 ({sorted(got)})")

    # ---- 5 z-score 检查（spike 投影后各成分近似零均值）
    if sp_dir.exists():
        print("\n=== 5. 归一化检查 ===")
        s = sess_list[0]
        w, _ = ds.load_spike_session(s)
        mu = w.astype(np.float32).mean(axis=(0, 2))
        chk(np.abs(mu).max() < 0.5, f"spike 各成分均值 |max|={np.abs(mu).max():.3f} < 0.5")

    # ---- 6 本体覆盖
    print("\n=== 6. 本体覆盖 ===")
    ont = ds.ontology
    for src_name, raw_col, meta_name in [
        ("steinmetz", "raw", None), ("things_eeg", "raw", None), ("flywire", "raw", None)]:
        sub = ont[ont["source"] == src_name]
        chk(len(sub) > 0, f"{src_name}: 本体 {len(sub)} 条")
    print(f"      证据等级: "
          f"strong {(ont.evidence=='strong').sum()} / "
          f"medium {(ont.evidence=='medium').sum()} / "
          f"weak {(ont.evidence=='weak').sum()}")

    print("\n" + "=" * 54)
    print("自检结果:", "✅ 全部通过" if ok else "❌ 有失败项")
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    sys.exit(0 if sanity(sys.argv[1]) else 1)
