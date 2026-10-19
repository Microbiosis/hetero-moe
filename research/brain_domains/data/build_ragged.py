#!/usr/bin/env python3
"""v4 — 不补齐，按原生长度存。

v3 的 counts.npy 是 (5483479, 192) uint8，把每个片段补零到 192 列。
这暗示了一个不存在的共同维度：第 j 列在不同片段里是不同单元，
而且很多片段本来就只有 40–65 个单元。

从集合论看，数据集是索引族 D = ⊔_i X_i，每个 X_i 有自己的形状
(n_bins_i, n_units_i)。不需要把它们塞进同一个矩阵。

改存：

    counts_ragged.npy      1-D uint8，Σ n_bins_i × n_units_i 个元素
    counts_index.parquet   加 offset 列（元素偏移），保留 n_bins / n_units

读取：

    a = C[off : off + n_bins*n_units].reshape(n_bins, n_units)

这样：
  - 没有补零，体积等于原生大小
  - 没有"第 j 列"的假象
  - 每条的 bin 宽度、单元数都是这条自己的属性（写进 index）
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/domains/counts-v3")
OUT = Path("/domains/counts-v4")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    ci = pd.read_parquet(SRC / "counts_index.parquet")
    print(f"segments {len(ci)}", flush=True)

    native = int((ci.n_bins_20ms * ci.n_units).sum())
    padded = int((ci.n_bins_20ms * 192).sum())
    print(f"padded {padded:,} ({padded/1e6:.0f} MB) | "
          f"native {native:,} ({native/1e6:.0f} MB) | "
          f"waste {100*(padded-native)/padded:.1f}%", flush=True)

    W = np.load(SRC / "counts.npy", mmap_mode="r")
    print(f"source {W.shape} {W.dtype}", flush=True)

    out = OUT / "counts_ragged.npy"
    if out.exists():
        out.unlink()
    R = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(native,))

    offs = np.zeros(len(ci), dtype=np.int64)
    k = 0
    for i, (_, r) in enumerate(ci.iterrows()):
        nb, nu = int(r.n_bins_20ms), int(r.n_units)
        m = nb * nu
        blk = np.asarray(W[r.bin_start:r.bin_end, :nu]).reshape(-1)
        R[k:k + m] = blk
        offs[i] = k
        k += m
        if i % 40 == 0:
            print(f"  [{i+1}/{len(ci)}] {r.file} {nb}x{nu} -> {k:,} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    assert k == native, (k, native)
    R.flush()
    del R, W

    ci = ci.copy()
    ci["offset"] = offs
    ci["n_elements"] = ci.n_bins_20ms * ci.n_units
    # 每条记录自己的时间分辨率（目前全是 20 ms，但作为列写出来，
    # 以后加入不同 bin 宽的成员时不需要改结构）
    ci["bin_ms"] = 20.0
    ci.to_parquet(OUT / "counts_index.parquet", index=False)

    meta = {
        "schema_version": "2.0",
        "layout": "ragged",
        "members": int(len(ci)),
        "total_elements": native,
        "padded_equivalent": padded,
        "waste_avoided_pct": round(100 * (padded - native) / padded, 2),
        "shapes": sorted({(int(r.n_bins_20ms), int(r.n_units))
                          for _, r in ci.iterrows()}, key=lambda t: -t[0])[:5],
        "n_distinct_widths": int(ci.n_units.nunique()),
        "note": ("每个片段是独立的集合 (n_bins_i, n_units_i)。"
                 "不存在跨片段的共同单元维度。"),
    }
    (OUT / "layout.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print()
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print(f"\nwrote {out} {out.stat().st_size/1e6:.0f} MB", flush=True)
    print(ci.groupby(["source", "region"]).agg(
        segs=("file", "size"), bins=("n_bins_20ms", "sum"),
        elems=("n_elements", "sum")).to_string(), flush=True)

    # 自检：随机抽三段，圆整后与原始一致
    rng = np.random.default_rng(0)
    Rr = np.load(out, mmap_mode="r")
    W2 = np.load(SRC / "counts.npy", mmap_mode="r")
    for i in rng.choice(len(ci), 3, replace=False):
        r = ci.iloc[i]
        a = np.asarray(Rr[r.offset:r.offset + r.n_elements]).reshape(
            int(r.n_bins_20ms), int(r.n_units))
        b = np.asarray(W2[r.bin_start:r.bin_end, :int(r.n_units)])
        print(f"  check {r.file}: equal={np.array_equal(a, b)} shape={a.shape}",
              flush=True)


if __name__ == "__main__":
    main()
