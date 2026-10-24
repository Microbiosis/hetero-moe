#!/usr/bin/env python3
"""exp89b_perchannel.py — exp89 的**逐通道语义对齐曲线**（补头皮图用，零训练）。

## 为什么单独一个脚本

exp89 的 JSON 里已经存了逐通道曲线，但两个 Job 写的是**同一个**
`exp89/exp89_posaxis.json` ⇒ **后写的覆盖先写的** ⇒ 下载下来只有 5 个被试。
而组级数字从日志重建是全的（10 个被试）⇒ 只缺曲线。

⇒ 这个脚本**只做一件事**：把 10 个被试的逐通道曲线算出来并打印成一行，
再写自己的 JSON（不覆盖 exp89 的）。

## 它不做什么

- 不算指标 B（贵，与曲线无关）
- 不做打乱/随机组对照（exp89 已经算过，日志里有）
- 不训练任何东西（零训练，与 exp89 同协议）

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp89b_perchannel.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 60m -d
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp89_posaxis as E89   # noqa: E402

OUT_DIR = Path("/domains/runs/illusionbrain-align/exp89b")


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    import exp73_decode200 as E73

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fte = torch.load(E89.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"图像特征：test {IMG_test.shape}")

    curves, names0 = [], None
    # 通道名先取（train split 的 ch_names；它与 test split 的顺序一致性
    # 由 exp89 的 (b) 逐通道 ERP 相关验过）
    try:
        names0, _e = E89.load_train_chunk(1, 1)
        log("CHLIST " + ",".join(str(c) for c in names0))
    except Exception as e:
        log(f"CHLIST 取不到：{type(e).__name__}: {e}"[:200])
        names0 = None
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        try:
            eeg, _keys = E89.load_test_eeg(sid)
            n_img, n_rep, n_ch, T = eeg.shape
            feats = E73.per_rep_targets(eeg, chunk=args.chunk)
            cr = E89.per_channel_rsa(feats, IMG_test, n_ch, args.n_perm,
                                     np.random.default_rng(E89.SEED + sid))
            curves.append(cr.tolist())
            log(f"CURVE2 sub-{sid:02d} " + ",".join(f"{v:+.4f}" for v in cr))
            del eeg, feats
        except Exception as e:
            log(f"CURVE2 sub-{sid:02d} ERROR {type(e).__name__}: {e}"[:200])
    if not curves:
        log("没有算到任何曲线")
        return 1

    med = np.median(np.array(curves), axis=0)
    log("CURVE2_MEDIAN " + ",".join(f"{v:+.4f}" for v in med))

    (out_dir / "per_channel_curve.json").write_text(
        json.dumps({"config": vars(args), "per_subject_curves": curves,
                    "median_curve": med.tolist(), "ch_names": names0},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"结果写入 {out_dir}/per_channel_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
