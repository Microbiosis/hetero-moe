#!/usr/bin/env python3
"""exp85_split_membership.py — 判定 `omnibrain4d/eeg/` 的 200 张图**属于哪个 split**。

## 为什么必须查

§十九 断言「基线表的指标 B 泄漏」，理由是：
`eeg_train/sub-XX.npz`（1654×80）与 `eeg/sub-XX.npz`（200×80）是「同一批图的子集」。
⚠️ 但这个「子集」关系**从未被验证过**。

而一个实测结果让这个断言站不住：
- **泄漏协议**（拟合 = eeg_train 的 1654 张，解码 = eeg/ 的 200 张）：630-d 得 **2.2×**（0.0111）
- **干净协议**（拟合 = train split 的 2000 张，解码 = eeg/ 的 200 张）：630-d 得 **2.3×**（0.0111）
⇒ 两个数几乎一样 ⇒ 若 200 张真在 1654 张里，泄漏本该让数字**大幅**虚高。

⇒ 三种可能，本脚本逐一排除：
(a) 200 张 ⊂ test split(1654) ⇒ §十九 成立（泄漏）；
(b) 200 张 ⊂ **train** split(16540) ⇒ §十九 **不成立**（无泄漏），
    而且「干净协议」反而泄漏了（解码集 ⊂ 拟合集）；
(c) 两边都不含（另有来源）⇒ 两个数都不能用。

## 判别方法（数值级、最硬的证据）

`eeg/sub-01.npz` 的第 0 张图第 0 次呈现，是一个 (63, 250) 的**具体波形**。
⇒ 在 `eeg_train/sub-01.npz`（1654×80）里找**数值完全相等**的试次 ⇒ 命中即属于 test split；
⇒ 在 `train.pt`（16540×4）里同样找 ⇒ 命中即属于 train split。
（同一张图的同一次呈现 ⇒ 数值应逐位相等；不同图 ⇒ 不可能相等。）

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp85_split_membership.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 60m -d
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
TRAIN_4D = Path("/domains/omnibrain4d/eeg_train")


def log(*a):
    print(*a, flush=True)


def find_match(hay, needle):
    """在 (n, ...) 的数组里找与 needle (C, T) 数值完全相等的行。返回索引或 None。

    ⚠️ 逐行比较 16540×4 行会慢 ⇒ 先用一行（第一个时间点 + 一个通道）的
    字节做**快速筛选**，再对候选行做全量比较。
    """
    import torch
    n = hay.shape[0]
    # 快速筛选：取 needle 的前 8 个元素拼成一个 key
    key = np.ascontiguousarray(needle.reshape(-1)[:8]).tobytes()
    hits = []
    for i in range(n):
        row = np.ascontiguousarray(np.asarray(hay[i]).reshape(-1)[:8]).tobytes()
        if row == key:
            hits.append(i)
    for i in hits:
        if np.array_equal(np.asarray(hay[i]), needle):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n-probe", type=int, default=1,
                    help="取几张图做探针（每张都在两个 split 里找）")
    args = ap.parse_args()

    import torch

    sid = args.subject
    eeg_test4d = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
    eeg_train4d = np.load(TRAIN_4D / f"sub-{sid:02d}.npz")["eeg"]
    log(f"omnibrain4d/eeg/sub-{sid:02d}.npz      : {eeg_test4d.shape}")
    log(f"omnibrain4d/eeg_train/sub-{sid:02d}.npz: {eeg_train4d.shape}")

    try:
        obj = torch.load(TRAIN_PT / f"sub-{sid:02d}" / "train.pt",
                         map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(TRAIN_PT / f"sub-{sid:02d}" / "train.pt",
                         map_location="cpu", weights_only=False)
    tr = obj["eeg"]
    log(f"train.pt (真 train split)              : {tuple(tr.shape)}")

    verdict = {"in_test_split": 0, "in_train_split": 0, "not_found": 0}
    for g in range(args.n_probe):
        needle = np.asarray(eeg_test4d[g, 0]).astype(np.float32)
        log(f"\n探针 {g}：eeg/ 的第 {g} 张图第 0 次呈现 "
            f"（{needle.shape}，前 3 值 {needle.reshape(-1)[:3]}）")

        i_tr = find_match(eeg_train4d, needle)
        log(f"  在 eeg_train（1654×80）里找 ⇒ {i_tr}")
        i_te = find_match(tr, needle)
        log(f"  在 train.pt（16540×4）里找 ⇒ {i_te}")

        if i_tr is not None:
            verdict["in_test_split"] += 1
        if i_te is not None:
            verdict["in_train_split"] += 1
        if i_tr is None and i_te is None:
            verdict["not_found"] += 1

    log("\n" + "=" * 70)
    log(f"判定：{verdict}")
    if verdict["in_test_split"] and not verdict["in_train_split"]:
        log("⇒ eeg/ 的 200 张属于 **test split**（= eeg_train 那 1654 张的子集）")
        log("   ⇒ §十九 的「泄漏」判断**成立**；")
        log("   ⇒ exp84 的「干净协议」正确（拟合 train split、解码 test split）")
    elif verdict["in_train_split"] and not verdict["in_test_split"]:
        log("⇒ eeg/ 的 200 张属于 **train split**（不在 eeg_train 那 1654 张里）")
        log("   ⇒ §十九 的「泄漏」判断**不成立**（拟合/解码不重叠）")
        log("   ⇒ 而 exp84 的「干净协议」**反而泄漏**（解码集 ⊂ 拟合集）")
    else:
        log("⇒ 两边都（或都没）找到 ⇒ 需要更细的溯源，别急着下结论")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
