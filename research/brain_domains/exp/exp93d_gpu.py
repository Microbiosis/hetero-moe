#!/usr/bin/env python3
"""exp93d_gpu.py — exp93d 的 **GPU 版**（模型层读出 @ n_train=16540，跑满 10 个被试）。

## 为什么要有 GPU 版

exp93d 要把同一批模型读出在 **全量 16540 张训练图**下重测（检验 A18 是不是「训练图数不够」）。
但它的成本结构是：

- **编码** 16540×4 试次 × 2 个模型 = **66 万次前向**（CPU 上一个被试 ≈ 60 分钟）
- **ridge** 66160×630 的 Gram ≈ 2.6e10 flops（CPU 上 ≈ 10 秒）

⇒ 瓶颈 100% 在编码，而编码的模型只有 **160 万参数** ⇒ **该上 GPU**。
（A10G 上一个被试的编码 ≈ 5 秒 ⇒ 10 个被试几分钟。）

⇒ 口径与 exp92-C 的 `branch level` **逐字一致**（同一个 `EVAL.metric_b_decode`、
同一套 z-score、同一个 α 选择），只把**前向**搬到 GPU ⇒ 数字可与 exp92-C / exp93a 对拍。

## 用法

    python -u research/brain_domains/exp/exp93d_gpu.py --self-test

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93d_gpu.py --subjects 1,2,3,4,5,6,7,8,9,10 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor a10g-large --timeout 120m -d --name t93d-gpu
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

import eval_brain_alignment as EVAL   # noqa: E402
import exp73_decode200 as E73         # noqa: E402
import exp81_tokenizer_eval as E81    # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93d-gpu")

CKPT_S1A = "/domains/runs/illusionbrain-align/exp80/k1024/tokenizer_sub-01.pt"
CKPT_S1B = "/domains/runs/illusionbrain-align/exp86v2/ffn-plain/tokenizer_assembled.pt"
SEED = 0
READOUTS = ("recon", "onehot_mean")


def log(*a):
    print(*a, flush=True)


def load_train_subject(subject, n_images=None):
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = np.asarray(obj["eeg"], dtype=np.float16)
    if n_images is not None:
        eeg = eeg[:n_images]
    return eeg


def encode_on_device(model, rnd_cb, eeg, device, batch_size):
    """(n_img, n_rep, C, T) → {readout: (n_img, n_rep, D)}（**前向在 device 上**）。"""
    import torch
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    acc = {k: [] for k in READOUTS}
    model = model.to(device)
    rnd = rnd_cb.to(device)
    with torch.no_grad():
        for s in range(0, flat.shape[0], batch_size):
            x = torch.from_numpy(np.ascontiguousarray(flat[s:s + batch_size])).to(device).float()
            o = E81.readouts_for(model, x, rnd)
            for k in READOUTS:
                acc[k].append(o[k].cpu().numpy())
    model = model.to("cpu")
    return {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc.items()}


def self_test():
    log("=== 自测：GPU 编码通道 ===")
    import torch
    rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  设备 {dev}")
    from exp80_eeg_tokenizer_train import EEGTokenizer
    m = EEGTokenizer(n_codes=64)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(64, m.vq.dim, generator=g) * 0.5
    eeg = rng.normal(size=(3, 2, 63, 250)).astype(np.float16)
    f = encode_on_device(m, rnd, eeg, dev, 64)
    for k in READOUTS:
        assert f[k].shape[0] == 3 and f[k].shape[1] == 2, (k, f[k].shape)
    log(f"  ✓ 读出形状 " + ", ".join(f"{k}={tuple(f[k].shape)}" for k in READOUTS))
    # CPU/GPU 一致性（同模型同输入 ⇒ 输出应几乎相同）
    if dev == "cuda":
        m2 = EEGTokenizer(n_codes=64)
        m2.load_state_dict(m.state_dict())
        g2 = torch.Generator().manual_seed(0)
        rnd2 = torch.randn(64, m2.vq.dim, generator=g2) * 0.5
        f2 = encode_on_device(m2, rnd2, eeg, "cpu", 64)
        for k in READOUTS:
            dev_ = float(np.abs(f[k] - f2[k]).max())
            log(f"  · {k}: GPU vs CPU 最大差 {dev_:.2e}")
            assert dev_ < 1e-3, f"{k} 的 GPU/CPU 结果不一致"
        log("  ✓ GPU 与 CPU 输出一致")
    log("=== 自测通过 ===")


def run_subject(subject, IMG_train, IMG_test, args, device, rng):
    import torch
    t0 = time.time()
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_tr = load_train_subject(subject, args.train_images)
    eeg_te = np.load(TEST_4D / f"sub-{subject:02d}.npz")["eeg"]
    rec["n_train_images"] = int(len(eeg_tr))

    out = {}

    # ---- 手工 630-d（与 exp92-C 同口径）----
    ftr = E73.per_rep_targets(eeg_tr)
    fte = E73.per_rep_targets(eeg_te)
    mu, sd = EVAL.zscore_fit(ftr)
    rb = EVAL.metric_b_decode(EVAL.zscore_apply(ftr, mu, sd),
                              EVAL.zscore_apply(fte, mu, sd),
                              IMG_train[:args.train_images], IMG_test, rng,
                              n_null=args.n_null)
    out["raw630"] = {"acc1_cos": float(rb["acc1_cos"]),
                     "null_acc1_cos": float(np.median(rb["null_acc1_cos"]))}
    log(f"  LEVEL raw630          acc1={rb['acc1_cos']:.4f} "
        f"(零中位 {out['raw630']['null_acc1_cos']:.4f})")
    del ftr, fte

    # ---- 模型读出（前向在 GPU）----
    for tag, ckpt, want in (("s1a", CKPT_S1A, READOUTS),
                            ("s1b", CKPT_S1B, READOUTS)):
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt}")
            continue
        model, _ck, rnd_cb = E81.load_model(ckpt)
        ftr = encode_on_device(model, rnd_cb, eeg_tr, device, args.batch_size)
        fte = encode_on_device(model, rnd_cb, eeg_te, device, args.batch_size)
        for ro in want:
            mu, sd = EVAL.zscore_fit(ftr[ro])
            rb = EVAL.metric_b_decode(EVAL.zscore_apply(ftr[ro], mu, sd),
                                      EVAL.zscore_apply(fte[ro], mu, sd),
                                      IMG_train[:args.train_images], IMG_test, rng,
                                      n_null=args.n_null)
            out[f"{tag}_{ro}"] = {"acc1_cos": float(rb["acc1_cos"]),
                                 "null_acc1_cos": float(np.median(rb["null_acc1_cos"]))}
            log(f"  LEVEL {tag}_{ro:<12} acc1={rb['acc1_cos']:.4f} "
                f"(零中位 {out[f'{tag}_{ro}']['null_acc1_cos']:.4f})")
        del model, ftr, fte
        torch.cuda.empty_cache() if device == "cuda" else None
    rec["level"] = out
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=16540)
    ap.add_argument("--n-null", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备 {device}｜torch {torch.__version__}")
    if device != "cuda":
        log("⚠️ 没有 GPU ⇒ 会很慢（CPU 上一个被试 ≈ 60 分钟）")

    IMG_test = torch.load(FEAT_TEST, map_location="cpu",
                          weights_only=False)["img_features"].numpy().astype(np.float64)
    IMG_train = torch.load(FEAT_TRAIN, map_location="cpu",
                           weights_only=False)["img_features"].numpy().astype(np.float64)
    log(f"图像特征 test {IMG_test.shape} | train {IMG_train.shape}")

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = run_subject(sid, IMG_train, IMG_test, args, device,
                              np.random.default_rng(SEED + sid))
            per_subject.append(rec)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)
        log(f"  用时 {time.time()-t0:.0f}s")

    n_err = sum(1 for r in per_subject if "error" in r)
    log(f"\n完成：{len(per_subject)-n_err}/{len(per_subject)} 成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))
    if per_subject and n_err == len(per_subject):
        log("✗✗ 全部被试都报错 ⇒ 结果不可用")

    (out_dir / "exp93d_gpu.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93d_gpu.json")


if __name__ == "__main__":
    main()
