"""把猕猴 NLB 数据接入 OmniBrain 的时序视图（spike_view）。

## 为什么之前被排除

OmniBrain 第一版排除猕猴，理由是"上游 bin 宽未核实"。
现在已核实：NLB 官方 README 明确 **"spike counts in 20 ms time bins"**。

同时更正一处此前的错误标注：
  area2-bump  ≠  "S1_M1 混合"
  area2-bump  =  Brodmann 区 2（躯体感觉皮层）—— NLB 官方文档

## 处理流程（与 spike_view 一致）

  1. 读 macaque_nlb/spikes_<dataset>_<NN>.npz  (n_units, n_bins) @20ms
  2. 20ms → 100ms: 5 个 bin 求和（spike counts 可加）
  3. 逐 unit z-score（沿时间轴）
  4. 逐片段 Gram 矩阵 PCA → 64 维（与小鼠同法）
  5. 分窗 8 bins / stride 4
  6. 域标签：该片段的记录区域（数据集级，非逐单元）

## ⚠️ 与小鼠数据的可比性限制（写入 meta）

小鼠的域标签是**逐单元解剖区**（CA1/VISp/...），经窗内活动投票成"主导域"。
猕猴的域标签是**整个片段一个区域**（Area2/DMFC/M1_PMd），因为上游
不提供逐单元标注。

因此：
  - 小鼠: 域是窗级变量（随时变化）
  - 猕猴: 域是片段级常量（整段不变）

这意味着**猕猴数据的域条件建模必然无信息增量**——同一片段内域恒为常量，
模型只需记住"我在哪个片段"即可。这不是数据缺陷，是标签粒度的本质差异。

本数据集定位为**跨物种动力学对照**（不同区域的时间统计特性），
不作为域条件建模的训练材料。

## 输出

  omnibrain/spike_view_macaque/
  ├── windows_<dataset>_<NN>.npz     (n_win, 64, 8) float16
  ├── meta.parquet                   逐窗: dataset, session, t0_bin, area, domain_l1
  └── meta.json                      汇总 + 可比性限制说明
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 必须在 import numpy 之前设置（BLAS 多线程，见 build_omnibrain 的教训）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

BIN_MS = 100.0
SRC_BIN_MS = 20.0          # NLB 官方
WIN_BINS = 8
STRIDE_BINS = 4
PROJ_DIM = 64
SEED = 0
EPS = 1e-6

# NLB 数据集 → 区域与 L1 功能类（按官方文档更正）
NLB_META = {
    "area2-bump": {"area": "Area2", "l1": "sensory",
                   "desc": "Brodmann 区 2（躯体感觉皮层）"},
    "dmfc-rsg": {"area": "DMFC", "l1": "association",
                 "desc": "背内侧额叶皮层（ready-set-go 计时任务）"},
    "churchland": {"area": "M1_PMd", "l1": "motor",
                   "desc": "初级运动 / 背侧前运动皮层"},
}


def atomic_save_np(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def span_windows(n_bins, win=WIN_BINS, stride=STRIDE_BINS):
    return [(s, s + win) for s in range(0, n_bins - win + 1, stride)]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/primate/macaque_nlb")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/omnibrain/spike_view_macaque")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("猕猴 NLB → OmniBrain 时序视图")
    print(f"  源 {src}")
    print(f"  输出 {out}")
    print(f"  bin {SRC_BIN_MS}ms → {BIN_MS}ms | 窗 {WIN_BINS} | 维 {PROJ_DIM}")
    print("=" * 74)

    seg_f = src / "segments.parquet"
    if not seg_f.exists():
        raise FileNotFoundError(f"缺少 {seg_f}")
    segs = pd.read_parquet(seg_f)
    print(f"  片段 {len(segs)}, 数据集 {segs['dataset'].nunique()}")

    ratio = int(BIN_MS / SRC_BIN_MS)          # 5
    rows = []
    stats = []
    t0 = time.time()

    for _, seg in segs.iterrows():
        f = src / seg["file"]
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)      # (n_units, n_bins_20ms)
        ds = seg["dataset"]
        meta_ds = NLB_META.get(ds, {"area": "unknown", "l1": "other", "desc": ""})

        # 1) 20ms → 100ms: 5 bin 求和
        nb = sp.shape[1] // ratio
        if nb < WIN_BINS:
            continue
        x = sp[:, :nb * ratio].reshape(sp.shape[0], nb, ratio).sum(axis=2)

        # 2) 逐 unit z-score
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + EPS
        z = (x - mu) / sd

        # 3) Gram 矩阵 PCA → 64
        k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
        if k < 2:
            continue
        zc = z - z.mean(axis=1, keepdims=True)
        C = zc @ zc.T
        w, U = np.linalg.eigh(C)
        order = np.argsort(-w)[:k]
        U = U[:, order]
        proj = zc.T @ U                                   # (nb, k)
        if proj.shape[1] < PROJ_DIM:
            proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
        else:
            proj = proj[:, :PROJ_DIM]
        ev = float(np.clip(w[order], 0, None).sum() / max(w.sum(), EPS))

        # 4) 分窗
        wins = span_windows(nb)
        if not wins:
            continue
        arr = np.stack([proj[s:e].T for s, e in wins]).astype(np.float16)

        fn = f"windows_{ds}_{int(seg.name):02d}.npz"
        atomic_save_np(out / fn, windows=arr)

        for (s, _e) in wins:
            rows.append({
                "file": fn, "dataset": ds, "session": str(seg["session"]),
                "t0_bin": int(s), "area": meta_ds["area"],
                "domain_l1": meta_ds["l1"],
                "n_units": int(sp.shape[0]),
            })
        stats.append({"dataset": ds, "file": fn, "area": meta_ds["area"],
                      "n_units": int(sp.shape[0]), "n_bins_20ms": int(sp.shape[1]),
                      "n_bins_100ms": int(nb), "n_windows": int(len(wins)),
                      "pca_k": int(k), "explained_var": round(ev, 4)})
        print(f"  {ds:12s} {seg['file']:32s} {sp.shape[0]:4d}u × {nb:6d}b "
              f"→ {len(wins):5d} 窗 (k={k}, EV={ev:.3f})")

    meta_df = pd.DataFrame(rows)
    meta_df.to_parquet(out / "meta.parquet", index=False)

    dist = meta_df.groupby(["dataset", "area", "domain_l1"]).size().rename("n_windows").reset_index()
    # 唯一 unit 数：按 (dataset, file) 取一次，不能对逐窗口的 n_units 求和
    n_units_total = int(
        meta_df.drop_duplicates("file")["n_units"].sum()) if len(meta_df) else 0

    summary = {
        "schema_version": "1.0",
        "source": str(src),
        "src_bin_ms": SRC_BIN_MS,
        "bin_ms": BIN_MS, "win_bins": WIN_BINS, "proj_dim": PROJ_DIM,
        "n_segments": len(stats), "n_windows": int(len(meta_df)),
        "n_units_total": n_units_total,
        "shape": f"(n_win, {PROJ_DIM}, {WIN_BINS}) float16",
        "domain_distribution": dist.to_dict("records") if len(dist) else [],
        "bin_width_evidence": ('NLB 官方 README: "spike counts in 20 ms time bins" '
                               '(github.com/eminorhan/neural-pile-primate)'),
        "area_correction": ("area2-bump 此前被误标为 'S1_M1 混合'; 实为 Brodmann 区 2 "
                            "(躯体感觉皮层), 依 NLB 官方 datasets 页更正。"),
        "comparability_caveat": (
            "小鼠 spike_view 的域是**逐单元解剖区**经活动投票得到的**窗级**变量; "
            "猕猴此处是**片段级**常量（上游无逐单元标注）。因此在猕猴数据上做"
            "域条件建模必然无信息增量——同一片段内域不变。本视图定位为"
            "**跨物种动力学对照**, 不作域条件训练材料。"),
        "segments": stats,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  片段 {len(stats)} | 窗口 {len(meta_df):,} | units {summary['n_units_total']:,}")
    if len(dist):
        print(dist.to_string(index=False))
    print(f"  耗时 {summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
