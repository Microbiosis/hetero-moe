#!/usr/bin/env python3
"""冻结 v54 的预登记协议 → research/aligner/v54_evaluator_swap/protocol.lock.json

为什么要有锁
------------
"预登记" 只有在**能被证明早于结果**时才有意义。没有锁的话, 任何人可以在看到
结果之后改 STEPS / SEEDS / D_shared / 判据, 而产物里看不出差别 —— 这正是
v53 的 n_classes 事故同一类问题 (产物里两个来源, 事后无法分辨)。

锁里固定两样东西:
  1. **网格**   d_shareds / arms / evaluators / seeds / steps / tokens / n_classes / batch
  2. **判据**   PREDICTIONS_V54 的 sha256 (事后改动判据 → 哈希对不上)

runner 开跑前对拍; 不一致直接拒绝 (除非显式 V54_ALLOW_DEVIATION=1, 且偏离写进产物)。

用法:
    python tools/freeze_protocol.py            # 生成/覆盖锁 (全网格开跑前必须做一次)
    python tools/freeze_protocol.py --check    # 只核对, 不写
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PKG = os.path.join(ROOT, "research/aligner/v54_evaluator_swap")
LOCK = os.path.join(PKG, "protocol.lock.json")


def build(grid: dict, predictions: dict, note: str) -> dict:
    pred_sha = hashlib.sha256(
        json.dumps(predictions, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    body = {"grid": grid, "predictions": predictions, "predictions_sha": pred_sha,
            "frozen_note": note}
    sha = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    return {**body, "sha256": sha}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只核对, 不写")
    a = ap.parse_args()

    from research.aligner.v54_evaluator_swap.audit import (
        ARMS, EVALUATORS, N_CLASSES, PREDICTIONS_V54,
    )
    grid = {"d_shareds": [96, 252, 384], "arms": list(ARMS), "evaluators": list(EVALUATORS),
            "seeds": 5, "seed_base": 2000, "steps": 200, "n_per_class": 16,
            "tokens": 8, "n_classes": N_CLASSES, "batch": 288}
    note = ("全网格开跑前冻结。d_shared=252 而非 256: 块状 one-hot 要求 D_shared 被 "
            "n_classes=6 整除 (256/6 不整, v52 的 256 仅适用于 8 类)。"
            "网格 180 格 = 3×4×3×5。")

    fresh = build(grid, PREDICTIONS_V54, note)
    if a.check:
        if not os.path.exists(LOCK):
            print("✗ 无锁"); return 1
        cur = json.load(open(LOCK, encoding="utf-8"))
        ok = cur.get("sha256") == fresh["sha256"]
        print(("✓ 锁与当前代码一致" if ok else "✗ 锁与当前代码不一致"))
        if not ok:
            for k in ("grid", "predictions_sha"):
                if cur.get(k) != fresh.get(k):
                    print(f"   {k}: 锁={cur.get(k)} 现在={fresh.get(k)}")
        return 0 if ok else 1

    json.dump(fresh, open(LOCK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"锁已写入 {os.path.relpath(LOCK, ROOT)}  sha256={fresh['sha256']}")
    print(f"  网格: {grid['d_shareds']} × {len(grid['arms'])} 臂 × "
          f"{len(grid['evaluators'])} 评价器 × {grid['seeds']} seeds = "
          f"{len(grid['d_shareds']) * len(grid['arms']) * len(grid['evaluators']) * grid['seeds']} 格")
    print(f"  判据: {list(PREDICTIONS_V54)}  sha={fresh['predictions_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
