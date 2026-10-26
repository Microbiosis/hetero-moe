#!/usr/bin/env python3
"""exp84_clean_metricb.py — 基线表**指标 B 的干净协议**重跑（可选执行）。

## 为什么需要这个（以及它最后**证明了**什么）

起因：基线表（§十六·六）与阶段 D 变体表的**指标 B** 用
`/domains/omnibrain4d/eeg_train/sub-XX.npz` 当拟合集。
我曾断言那个目录装的是 **THINGS-EEG 的 test split**
（实测 2,084,040,260 B ÷ (63×250×2) = 132,320 = 1654×80）
⇒ 而解码集 `eeg/` 的 200 张是其子集 ⇒ **拟合/解码图像重叠 ⇒ 泄漏**。

⚠️ **那个断言是错的**（FINDINGS §十九）：2,084,040,260 ÷ (63×250×2) = **66,160 = 16540×4**，
我少除了 float16 的一个 2。exp85 用**数值级溯源**复核：
`eeg_train/` 装的就是 **train split (16540, 4)**，且 `eeg/` 的 200 张**不在**它里面
⇒ **指标 B 从来没有泄漏** ⇒ 基线表与变体表都**不需要重做**。

⇒ 于是本脚本的作用从「重做」变成**交叉验证**：
用**另一个容器**（`train.pt`）里的 train split 当拟合集重算一遍，
看数字是否与基线表原协议一致。
实测 630-d 那一行：**0.0111（2.3× 随机）** vs 基线表原协议 **0.0111（2.2×）**
⇒ 两个协议本来就都是干净的、结果一致 ⇒ 这一点互为佐证。

**干净协议**（本脚本的拟合集）：**train split**
（`/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-XX/train.pt`，
实测 16540 图 × 4 试次）+ `ViT-H-14_features_train.pt`（16540），
**解码集**仍是 test split 的 200 图。

## ⚠️ 必须先验证**图像配对**（否则会得到「比随机还差」的假象）

`train.pt` 的图像顺序**未必**与 `ViT-H-14_features_train.pt` 的行序一致
（两者来自不同流水线）⇒ 若错配，ridge 拟合的就是乱配对 ⇒ 结果无意义。

⇒ 本脚本用**可证伪的判据**做验证：**630-d 原始特征那一行**在正确配对下
必须**远高于随机**（基线表里它有 2.2×；exp73 的同类协议有 6.2×），
错配时则落在随机水平或以下。

| 模式 | 做什么 |
|---|---|
| `--verify-only` | 只算 630-d 那一行（1 个被试，~3 分钟）⇒ 判定配对是否正确 |
| 全量 | 7 个基模 + 630-d，10 被试 × 2 趟（指标 B 干净、指标 A 不变） |

## 用法

    # 第一步：验证配对（不跑全量）
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp84_clean_metricb.py --verify-only --subjects 1 ...

    # 第二步：配对通过后再跑全量
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp84_clean_metricb.py --subjects 1,2,3,4,5,6,7,8,9,10 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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

import base_specs as BS            # noqa: E402
import eval_brain_alignment as EVAL   # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
MODELS_DIR = Path("/training/models")
OUT_DIR = Path("/domains/runs/illusionbrain-align/clean-metricb")
SEED = 0


def log(*a):
    print(*a, flush=True)


def load_train_split(subject, n_images):
    """train split 的前 n_images 张（float32）。

    ⚠️ 桶里是**小写** `things-eeg` ⇒ 写成 `things-EEG` 会静默取到空目录。
    """
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = obj["eeg"]
    log(f"  train split：{tuple(eeg.shape)}（keys {list(obj.keys())[:8]}）")
    return np.asarray(eeg[:n_images], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1")
    ap.add_argument("--train-images", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify-only", action="store_true",
                    help="只算 630-d 那一行（配对正确性验证）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    import registry as REG

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"图像特征：train {IMG_train.shape} / test {IMG_test.shape}")

    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    rng = np.random.default_rng(SEED)
    per_subject = []
    for sid in sids:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        rec = {"subject": f"sub-{sid:02d}", "rows": {}}
        try:
            eeg_tr = load_train_split(sid, args.train_images)
            eeg_te = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
            IMG_tr = IMG_train[:args.train_images]

            # --- 630-d 基线行（配对正确性的判据）---
            X630_tr = EVAL.baseline_630_features(eeg_tr)
            X630_te = EVAL.baseline_630_features(eeg_te)
            mu, sd = EVAL.zscore_fit(X630_tr)
            rb = EVAL.metric_b_decode(EVAL.zscore_apply(X630_tr, mu, sd),
                                       EVAL.zscore_apply(X630_te, mu, sd),
                                       IMG_tr, IMG_test, rng,
                                       n_null=args.n_null)
            rec["rows"]["630-d 原始特征"] = {
                "acc1": rb["acc1_cos"],
                "null_acc1": float(np.median(rb["null_acc1_cos"])),
                "alpha": rb.get("alpha"),
            }
            ratio = rb["acc1_cos"] / max(float(np.median(rb["null_acc1_cos"])), 1e-9)
            log(f"  [630-d 原始特征] B acc1={rb['acc1_cos']:.4f} "
                f"(零中位 {np.median(rb['null_acc1_cos']):.4f}) = {ratio:.1f}×  "
                f"α={rb.get('alpha')}")
            if ratio < 1.5:
                log("  ⚠️ 干净协议下 630-d 仍接近随机 ⇒ **图像配对可能错位**，"
                    "不要继续跑全量（先查 train.pt 的图像顺序）")
            del X630_tr, X630_te

            if not args.verify_only:
                names = BS.models_for_version("stable")
                handles = {}
                import braindecode.models as bdmodels
                for nm in names:
                    handles[nm] = REG.load_base_model(
                        nm, models_dir=MODELS_DIR, braindecode_models=bdmodels,
                        spec=BS.BASE_SPECS.get(nm), verbose=False)
                for nm, h in handles.items():
                    if not h.ok:
                        rec["rows"][nm] = {"error": h.error[:200]}
                        continue
                    try:
                        fte_ = EVAL.model_trial_features(
                            h, eeg_te, BS.THINGS_EEG_CHANS, args.batch_size,
                            args.device, "mean_all")
                        ftr_ = EVAL.model_trial_features(
                            h, eeg_tr, BS.THINGS_EEG_CHANS, args.batch_size,
                            args.device, "mean_all")
                        mu, sd = EVAL.zscore_fit(ftr_["features"])
                        rb = EVAL.metric_b_decode(
                            EVAL.zscore_apply(ftr_["features"], mu, sd),
                            EVAL.zscore_apply(fte_["features"], mu, sd),
                            IMG_tr, IMG_test, rng, n_null=args.n_null)
                        rec["rows"][nm] = {
                            "acc1": rb["acc1_cos"],
                            "null_acc1": float(np.median(rb["null_acc1_cos"])),
                            "alpha": rb.get("alpha")}
                        log(f"  [{nm}] B acc1={rb['acc1_cos']:.4f} "
                            f"(零中位 {np.median(rb['null_acc1_cos']):.4f})")
                    except Exception as e:
                        rec["rows"][nm] = {"error": f"{type(e).__name__}: {e}"[:200]}
                        log(f"  [{nm}] ✗ {rec['rows'][nm]['error']}")
            del eeg_tr, eeg_te
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["trace"] = traceback.format_exc()[-600:]
            log(f"  ✗ {rec['error']}")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        per_subject.append(rec)

    (out_dir / "clean_metricb.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/clean_metricb.json")


if __name__ == "__main__":
    main()
