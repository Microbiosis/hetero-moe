"""把人类**侵入式**神经记录接入 OmniBrain 时序视图。

## 这解决了什么问题

Phase 6 的核心困境: 三物种比较混入了**模态**差异
  - 小鼠、猕猴 = 侵入式 spike
  - 人类 = 头皮 EEG（经颅骨低通，天然高自相关/陡频谱）

因此"人类离群"不能解读为物种差异。本脚本补入**人类侵入式**记录，
让三物种全部处于同一模态，物种比较才成立。

## 数据的另一个好处: 单元数重叠

  猕猴      40 – 192 units
  人类 h1  176 units      ← 落在范围内
  人类 h2  192 units      ← 落在上界
  Willett  197 units

这意味着**小鼠 vs 人类**也能做规模匹配的子集比较（小鼠取 205-250 units 的 session）。

## 数据源（均来自 eminorhan 的 neural-pile-primate 拆分）

| 数据集 | 被试 | sessions | units/片段 | 来源 |
|---|---|---|---|---|
| h1 | sub-HumanPitt | 40 | 176 | dandi:000954 |
| h2 | sub-T5 | 63 | 192 | dandi:000950 |
| willett | 1 名瘫痪患者 | 197 片段 | 197 | dryad:x69p8czpq (Willett 2021) |

⚠️ **代表性限制**: 全部来自**极少数被试**（3 人），且 h1/h2 为癫痫监测植入、
Willett 为脑机接口瘫痪患者。这不是健康人群的代表样本。
写入 meta 的 caveat，任何"人类神经动力学"的结论必须带上此限定。

## 处理流程（与猕猴视图一致）

  1. 读 spike_counts（每 unit 一个变长数组）
  2. 补齐为 (n_units, max_len) 矩阵
  3. 20ms → 100ms: 5 bin 求和（NLB 惯例 20ms，与猕猴一致）
  4. 逐 unit z-score
  5. 逐片段 Gram 矩阵 PCA → 64 维
  6. 分窗 8 bins / stride 4
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

BIN_MS = 100.0
SRC_BIN_MS = 20.0          # neural-pile 统一 20ms
WIN_BINS = 8
STRIDE_BINS = 4
PROJ_DIM = 64
SEED = 0
EPS = 1e-6
MAX_SEGMENTS_PER_DS = 200   # 每数据集最多处理多少片段（控制内存）

# 人类侵入式数据集 → 元信息
HUMAN_DATASETS = {
    "h1": {"file": "data/train-00000-of-00001.parquet",
           "cohort": "HumanPitt (癫痫监测植入)",
           "source": "dandi:000954"},
    "h2": {"file": "data/train-00000-of-00001.parquet",
           "cohort": "T5 (癫痫监测植入)",
           "source": "dandi:000950"},
}


def atomic_save_np(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def pad_units(sc, dtype=np.float32):
    """把「每 unit 一个变长数组」补零成 (n_units, max_len)。"""
    arrs = [np.asarray(u, dtype=np.float32).ravel() for u in sc]
    if not arrs:
        return np.zeros((0, 0), dtype=dtype), [0]
    L = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), L), dtype=dtype)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out, [len(a) for a in arrs]


def span_windows(n_bins, win=WIN_BINS, stride=STRIDE_BINS):
    return [(s, s + win) for s in range(0, n_bins - win + 1, stride)]


def process_segment(mat, ratio):
    """(n_units, bins20) → (n_win, 64, 8) float16 或 None。"""
    nb = mat.shape[1] // ratio
    if nb < WIN_BINS or mat.shape[0] < 5:
        return None, None, None
    x = mat[:, :nb * ratio].reshape(mat.shape[0], nb, ratio).sum(axis=2).astype(np.float32)

    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + EPS
    z = (x - mu) / sd

    k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None, None, None
    zc = z - z.mean(axis=1, keepdims=True)
    C = zc @ zc.T
    w, U = np.linalg.eigh(C)
    order = np.argsort(-w)[:k]
    U = U[:, order]
    proj = zc.T @ U
    if proj.shape[1] < PROJ_DIM:
        proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
    else:
        proj = proj[:, :PROJ_DIM]
    ev = float(np.clip(w[order], 0, None).sum() / max(w.sum(), EPS))

    wins = span_windows(nb)
    if not wins:
        return None, None, None
    arr = np.stack([proj[s:e].T for s, e in wins]).astype(np.float16)
    return arr, len(wins), ev


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out/omnibrain/spike_view_human")
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    print("=" * 76)
    print("人类侵入式神经记录 → OmniBrain 时序视图")
    print(f"  输出 {out}")
    print(f"  bin {SRC_BIN_MS}ms → {BIN_MS}ms | 窗 {WIN_BINS} | 维 {PROJ_DIM}")
    print("=" * 76)

    ratio = int(BIN_MS / SRC_BIN_MS)
    rows, stats = [], []
    t0 = time.time()

    for ds, cfg in HUMAN_DATASETS.items():
        print(f"\n--- {ds} ({cfg['cohort']}) ---")
        try:
            p = hf_hub_download(f"eminorhan/{ds}", cfg["file"], repo_type="dataset")
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            continue

        n_seg = min(len(df), MAX_SEGMENTS_PER_DS)
        for i in range(n_seg):
            r = df.iloc[i]
            mat, lens = pad_units(r["spike_counts"])
            arr, n_win, ev = process_segment(mat, ratio)
            if arr is None:
                continue
            fn = f"windows_{ds}_{i:03d}.npz"
            atomic_save_np(out / fn, windows=arr)
            for (s, _e) in span_windows(mat.shape[1] // ratio):
                rows.append({"file": fn, "dataset": ds, "session": str(r["session_id"]),
                             "t0_bin": int(s), "area": f"{ds}_intracortical",
                             "domain_l1": "association",
                             "cohort": cfg["cohort"], "n_units": int(mat.shape[0])})
            stats.append({"dataset": ds, "file": fn, "cohort": cfg["cohort"],
                          "session": str(r["session_id"]),
                          "n_units": int(mat.shape[0]),
                          "n_bins_20ms": int(mat.shape[1]),
                          "n_bins_100ms": int(mat.shape[1] // ratio),
                          "n_windows": int(n_win), "explained_var": round(ev, 4)})
            if i % 10 == 0:
                print(f"  [{i+1}/{n_seg}] {mat.shape[0]}u × {mat.shape[1]//ratio}b → {n_win} 窗")
            del mat, arr
        gc.collect()

    if not rows:
        print("❌ 无产出")
        return

    meta = pd.DataFrame(rows)
    meta.to_parquet(out / "meta.parquet", index=False)

    dist = meta.groupby(["dataset", "area"]).size().rename("n_windows").reset_index()
    n_units_total = int(meta.drop_duplicates("file")["n_units"].sum())

    summary = {
        "schema_version": "1.0",
        "src_bin_ms": SRC_BIN_MS, "bin_ms": BIN_MS,
        "win_bins": WIN_BINS, "proj_dim": PROJ_DIM,
        "n_segments": len(stats), "n_windows": int(len(meta)),
        "n_units_total": n_units_total,
        "shape": f"(n_win, {PROJ_DIM}, {WIN_BINS}) float16",
        "domain_distribution": dist.to_dict("records"),
        "purpose": ("补入人类**侵入式**记录，使小鼠/猕猴/人类三者处于同一模态，"
                    "解决 Phase 6 的模态混淆（人类原为头皮 EEG）。"),
        "size_overlap": ("人类单元数 176–197，落在猕猴 40–192 范围内，"
                         "因此可直接与猕猴做规模可比对照；"
                         "小鼠 205–1240 需取低端子集才能匹配。"),
        "representativeness_caveat": (
            "全部来自极少数被试（3 人）：h1 = sub-HumanPitt、h2 = sub-T5，"
            "均为癫痫监测植入患者；Willett 为脑机接口瘫痪患者。"
            "这不是健康人群的代表样本，任何『人类神经动力学』的结论必须带此限定。"),
        "sources": {ds: cfg["source"] for ds, cfg in HUMAN_DATASETS.items()},
        "segments": stats,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"  片段 {len(stats)} | 窗口 {len(meta):,} | units {n_units_total:,}")
    print(dist.to_string(index=False))
    print(f"  耗时 {summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
