#!/usr/bin/env python3
"""exp93d_level.py — exp93d：**可切设备**的档位脚本（模型层读出 @ n_train=16540）。

## 这个脚本存在的理由：把「GPU 有没有用」变成**可测**的问题

2026-10-25 的第一版叫 `exp93d_gpu.py`，跑出「CPU 上一个被试 60 分钟 → 换档位后 130 秒」
⇒ 我据此写成「GPU 加速 28×」。**那个归因是错的**（混淆了三件事）：

| 混淆项 | 旧 CPU 跑 | 新档位跑 |
|---|---|---|
| flavor | `cpu-upgrade`（**8 vCPU / 32 GB**） | `a10g-large`（**12 vCPU / 46 GB**） |
| 争用 | **同时还有 2 个 exp93e Job 在抢核** | 独占 |
| 前向 | CPU | GPU（但模型只有 160 万参数） |

⇒ 而且**项目自己早就测过**（`FINDINGS_TIER0_RSA.md` §十六·五，5 档位实测吞吐）：
**a100-large（最快）只比 cpu-upgrade 快 1.45×，a10g-small 反而比 CPU 慢**
——「小模型 GPU 利用率极低，**搬运/启动开销主导**」。

⇒ 所以本脚本**不做性能宣称**，只做一件事：
**在同一 flavor、同争用条件下，`--device cuda` 与 `--device cpu` 的差 = GPU 的真实贡献**，
并且**分阶段计时**（前向 vs 后处理）⇒ 让人看见时间到底花在哪。

## 口径

与 exp92-C 的 `branch level` **逐字一致**（同一个 `EVAL.metric_b_decode`、同一套 z-score、
同一个 α 选择）；只有**前向**的设备可变。

## 用法

    # A/B 对照：同一 flavor、同时跑、只差 device
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93d_level.py --device cuda --subjects 1 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor a10g-large --timeout 60m -d --name t93d-bench-cuda

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93d_level.py --device cpu --subjects 1 \
      ... --flavor a10g-large --name t93d-bench-cpu-same
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
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93d-level")

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
    """(n_img, n_rep, C, T) → ({readout: 特征}, 编码秒数)。

    ⚠️ **返回编码耗时**：不把「前向」与「后处理」分开记，
    「换了设备变快」就会被误读成「GPU 加速」。
    """
    import torch
    t0 = time.time()
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    acc = {k: [] for k in READOUTS}
    model = model.to(device)
    rnd = rnd_cb.to(device)
    with torch.no_grad():
        for s in range(0, flat.shape[0], batch_size):
            x = torch.from_numpy(
                np.ascontiguousarray(flat[s:s + batch_size])).to(device).float()
            o = E81.readouts_for(model, x, rnd)
            for k in READOUTS:
                acc[k].append(o[k].cpu().numpy())
    model = model.to("cpu")
    feats = {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc.items()}
    return feats, time.time() - t0


def self_test():
    log("=== 自测：编码通道 + 计时 ===")
    import torch
    rng = np.random.default_rng(0)
    from exp80_eeg_tokenizer_train import EEGTokenizer
    m = EEGTokenizer(n_codes=64)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(64, m.vq.dim, generator=g) * 0.5
    eeg = rng.normal(size=(3, 2, 63, 250)).astype(np.float16)
    for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        f, enc = encode_on_device(m, rnd, eeg, dev, 64)
        for k in READOUTS:
            assert f[k].shape[0] == 3 and f[k].shape[1] == 2, (k, f[k].shape)
        log(f"  ✓ {dev}: 形状 " + ", ".join(f"{k}={tuple(f[k].shape)}" for k in READOUTS)
            + f"｜编码 {enc:.3f}s")
    log("=== 自测通过 ===")


def run_subject(subject, IMG_train, IMG_test, args, device, rng):
    import torch
    t_start = time.time()
    rec = {"subject": f"sub-{subject:02d}", "device": device}
    eeg_tr = load_train_subject(subject, args.train_images)
    eeg_te = np.load(TEST_4D / f"sub-{subject:02d}.npz")["eeg"]
    rec["n_train_images"] = int(len(eeg_tr))

    out = {}
    enc_s = dec_s = 0.0

    # ---- 手工 630-d（与 exp92-C 同口径）----
    t0 = time.time()
    ftr = E73.per_rep_targets(eeg_tr)
    fte = E73.per_rep_targets(eeg_te)
    mu, sd = EVAL.zscore_fit(ftr)
    rb = EVAL.metric_b_decode(EVAL.zscore_apply(ftr, mu, sd),
                              EVAL.zscore_apply(fte, mu, sd),
                              IMG_train[:args.train_images], IMG_test, rng,
                              n_null=args.n_null)
    dec_s += time.time() - t0
    out["raw630"] = {"acc1_cos": float(rb["acc1_cos"]),
                     "null_acc1_cos": float(np.median(rb["null_acc1_cos"]))}
    log(f"  LEVEL raw630          acc1={rb['acc1_cos']:.4f} "
        f"(零中位 {out['raw630']['null_acc1_cos']:.4f})")
    del ftr, fte

    # ---- 模型读出（前向在 device 上）----
    for tag, ckpt, want in (("s1a", CKPT_S1A, READOUTS),
                            ("s1b", CKPT_S1B, READOUTS)):
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt}")
            continue
        model, _ck, rnd_cb = E81.load_model(ckpt)
        ftr, t_enc = encode_on_device(model, rnd_cb, eeg_tr, device, args.batch_size)
        fte, _ = encode_on_device(model, rnd_cb, eeg_te, device, args.batch_size)
        enc_s += t_enc
        for ro in want:
            t0 = time.time()
            mu, sd = EVAL.zscore_fit(ftr[ro])
            rb = EVAL.metric_b_decode(EVAL.zscore_apply(ftr[ro], mu, sd),
                                      EVAL.zscore_apply(fte[ro], mu, sd),
                                      IMG_train[:args.train_images], IMG_test, rng,
                                      n_null=args.n_null)
            dec_s += time.time() - t0
            out[f"{tag}_{ro}"] = {"acc1_cos": float(rb["acc1_cos"]),
                                 "null_acc1_cos": float(np.median(rb["null_acc1_cos"]))}
            log(f"  LEVEL {tag}_{ro:<12} acc1={rb['acc1_cos']:.4f} "
                f"(零中位 {out[f'{tag}_{ro}']['null_acc1_cos']:.4f})")
        del model, ftr, fte
        if device == "cuda":
            torch.cuda.empty_cache()

    rec["level"] = out
    rec["timing"] = {"encode_s": round(enc_s, 1), "decode_s": round(dec_s, 1),
                     "total_s": round(time.time() - t_start, 1)}
    log(f"  ⏱ 编码(前向) {enc_s:.0f}s｜后处理(ridge/CV) {dec_s:.0f}s"
        f"｜合计 {time.time()-t_start:.0f}s｜设备 {device}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=16540)
    ap.add_argument("--n-null", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="**对照用**：同一 flavor 下 cuda vs cpu 的差 = GPU 的真实贡献")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"设备 {device}（指定 {args.device}）｜torch {torch.__version__}"
        f"｜cuda 可用 {torch.cuda.is_available()}")
    if device == "cuda" and not torch.cuda.is_available():
        log("✗ 指定了 cuda 但不可用")
        raise SystemExit(2)

    IMG_test = torch.load(FEAT_TEST, map_location="cpu",
                          weights_only=False)["img_features"].numpy().astype(np.float64)
    IMG_train = torch.load(FEAT_TRAIN, map_location="cpu",
                           weights_only=False)["img_features"].numpy().astype(np.float64)
    log(f"图像特征 test {IMG_test.shape} | train {IMG_train.shape}")

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
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

    n_err = sum(1 for r in per_subject if "error" in r)
    log(f"\n完成：{len(per_subject)-n_err}/{len(per_subject)} 成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))
    if per_subject and n_err == len(per_subject):
        log("✗✗ 全部被试都报错 ⇒ 结果不可用")

    (out_dir / "exp93d_level.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93d_level.json")


if __name__ == "__main__":
    main()
