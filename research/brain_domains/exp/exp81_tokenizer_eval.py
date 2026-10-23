#!/usr/bin/env python3
"""exp81_tokenizer_eval.py — S1：自训 EEG 离散 tokenizer 的评测（**干净协议**）。

## 口径纪律（这一版和基线表不同，差别是必须的）

基线表（§十六·六）与阶段 D 变体表的**指标 B** 用
`/domains/omnibrain4d/eeg_train/sub-XX.npz` 当训练集 —— 但那个目录装的是
**THINGS-EEG 的 test split**（实测 2,084,040,260 ÷ (63×250×2) = 132,320 = 1654×80），
而测试集 `/domains/omnibrain4d/eeg/sub-XX.npz`（200 图）是**同一批图**的子集
⇒ **训练/测试图像重叠 ⇒ 指标 B 泄漏**。

本脚本用**干净协议**：

| | 拟合 | 解码 |
|---|---|---|
| 指标 A（RSA） | —（不训练） | test split 200 图 × 80 试次（与基线表同口径） |
| 指标 B（ridge 200 选 1） | **train split**（`train.pt` 前 N 图 × 4 试次，16540 张里取） | test split 200 图 |

⇒ 指标 B 的数字**不能**和基线表直接比（那边是泄漏的）；指标 A 可以。

## 评测对象与读出

自训 tokenizer 产出**每时间片一个 token**（31 个），与 braintokenizer 的
「19 位置 × 512 时间步」布局不同 ⇒ 读出必须重新定义：

| 读出 | 构造 | 维度 |
|---|---|---|
| `emb_mean` | 码本向量按 token 平均 | d=64（**主读出**） |
| `onehot_mean` | 每个 token 的 one-hot 码按 token 平均 | K=1024（纯离散） |
| `code_ids` | 31 个码号本身 | 31（码号顺序是任意的 ⇒ 预期无意义，作负对照） |
| `continuous` | **不量化**，同一 encoder 的特征按 token 平均 | 64（**对照 A**） |
| `random_codebook` | 同一 encoder + **随机码本**量化 | 64（**对照 B**） |

## 判据（预先写死）

1. `emb_mean` 的指标 A 显著（p<0.05），且不低于 braintokenizer 的 +0.0155；
2. 码本利用率 > 50%；
3. 指标 B（干净协议）高于随机；
4. `emb_mean` > `continuous`（量化有用）与 `emb_mean` > `random_codebook`（要靠训练）。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp81_tokenizer_eval.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 90m -d \
      --ckpt /domains/runs/illusionbrain-align/exp80/tokenizer_sub-01.pt
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

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp81")
SEED = 0

READOUTS = ("emb_mean", "onehot_mean", "code_ids", "continuous", "random_codebook")


def log(*a):
    print(*a, flush=True)


def load_model(ckpt_path, seed=SEED):
    """加载自训 tokenizer + 造一个随机码本（对照 B）。"""
    import torch
    from exp80_eeg_tokenizer_train import EEGTokenizer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EEGTokenizer(n_codes=int(ck["n_codes"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randn(int(ck["n_codes"]), model.vq.dim, generator=g) * 0.5
    return model, ck, rnd


def readouts_for(model, x, rnd_codebook):
    """一批 EEG → 各读出的池化特征。"""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        z = model.encode(x)                                  # (B, T', D)
        zq, _loss, idx = model.vq(z)                         # zq: (B,T',D)
        codes = idx                                          # (B, T')
        K = model.vq.n_codes
        # 对照 B：随机码本
        flat = z.reshape(-1, z.shape[-1])
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ rnd_codebook.t()
             + rnd_codebook.pow(2).sum(1))
        r_idx = d.argmin(1).view(codes.shape)
        # 随机码本对应的「嵌入」就是随机码本向量本身
        r_emb = rnd_codebook[r_idx]                          # (B,T',D)
        emb = model.vq.lookup(codes)                         # (B,T',D)
        onehot = F.one_hot(codes, K).float()                 # (B,T',K)
        out = {
            "emb_mean": emb.mean(dim=1),                     # (B, D)
            "onehot_mean": onehot.mean(dim=1),               # (B, K)
            "code_ids": codes.float(),                       # (B, T')
            "continuous": z.mean(dim=1),                     # (B, D)
            "random_codebook": r_emb.mean(dim=1),            # (B, D)
        }
    return out


def extract(model, eeg, rnd_cb, batch_size=64, device="cpu"):
    """(n_img, n_rep, C, T) → {readout: (n_img, n_rep, D)}"""
    import torch
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    acc = {k: [] for k in READOUTS}
    for s in range(0, flat.shape[0], batch_size):
        sub = flat[s:s + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(sub)).float().to(device)
        o = readouts_for(model, x, rnd_cb)
        for k in READOUTS:
            acc[k].append(o[k].cpu().numpy())
    return {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc.items()}


def load_train_split(subject, n_images):
    """train split 的前 n_images 张（每张 n_rep 次）—— 指标 B 的**拟合集**。"""
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = obj["eeg"]
    log(f"  train split：{tuple(eeg.shape)}")
    arr = eeg[:n_images]
    return arr.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=2000,
                    help="指标 B 的拟合集用多少张 train split 的图")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"图像特征：train {IMG_train.shape} / test {IMG_test.shape}")

    model, ck, rnd_cb = load_model(args.ckpt)
    log(f"✓ 自训 tokenizer：{args.ckpt} | 码本 K={model.vq.n_codes} "
        f"d={model.vq.dim} | token 数 {ck.get('n_tokens')} | "
        f"训练步数 {ck.get('args', {}).get('steps')}")

    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    rng = np.random.default_rng(SEED)
    per_subject = []
    for sid in sids:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        rec = {"subject": f"sub-{sid:02d}", "rows": {}}
        try:
            eeg_te = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
            eeg_tr = load_train_split(sid, args.train_images)
            F_te = extract(model, eeg_te, rnd_cb, args.batch_size, args.device)
            F_tr = extract(model, eeg_tr, rnd_cb, args.batch_size, args.device)
            for k in READOUTS:
                Xte, Xtr = F_te[k], F_tr[k]
                a = EVAL.metric_a_rsa(Xte, IMG_test, args.n_perm, rng)
                mu, sd = EVAL.zscore_fit(Xtr)
                rb = EVAL.metric_b_decode(
                    EVAL.zscore_apply(Xtr, mu, sd), EVAL.zscore_apply(Xte, mu, sd),
                    IMG_train[:args.train_images], IMG_test, rng,
                    n_null=args.n_null)
                rec["rows"][k] = {"dim": int(Xte.shape[-1]),
                                  "r_img": a["r_img"], "p_img": a["p_img"],
                                  "acc1": rb["acc1_cos"],
                                  "null_acc1": float(np.median(rb["null_acc1_cos"]))}
                log(f"  {k:<18} dim={Xte.shape[-1]:<5} A r={a['r_img']:+.4f} "
                    f"p={a['p_img']:.3f} | B acc1={rb['acc1_cos']:.4f} "
                    f"(零中位 {np.median(rb['null_acc1_cos']):.4f})")
            # 码本利用率（训练后）
            with torch.no_grad():
                z = model.encode(torch.from_numpy(
                    eeg_te[:2].reshape(-1, eeg_te.shape[2], eeg_te.shape[3])
                ).float())
                _zq, _l, idx = model.vq(z)
                u = model.vq.usage(idx)
            rec["codebook"] = u
            log(f"  码本利用率：{u['used']}/{u['total']} = {u['utilization']:.1%} "
                f"| 熵 {u['entropy_nats']:.2f} nats")
            del eeg_te, eeg_tr, F_te, F_tr
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["trace"] = traceback.format_exc()[-600:]
            log(f"  ✗ {rec['error']}")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        per_subject.append(rec)

    # ---- 汇总（中位 + 显著被试数 + 配对对照）----
    summary = {}
    for k in READOUTS:
        rs = [r["rows"][k]["r_img"] for r in per_subject if k in r.get("rows", {})]
        ps = [r["rows"][k]["p_img"] for r in per_subject if k in r.get("rows", {})]
        ac = [r["rows"][k]["acc1"] for r in per_subject if k in r.get("rows", {})]
        nu = [r["rows"][k]["null_acc1"] for r in per_subject if k in r.get("rows", {})]
        if not rs:
            continue
        rs_s = sorted(rs)
        summary[k] = {
            "n": len(rs), "r_median": rs_s[len(rs_s) // 2],
            "p_median": sorted(ps)[len(ps) // 2],
            "n_sig": sum(1 for p in ps if p < 0.05),
            "acc1_median": sorted(ac)[len(ac) // 2],
            "null_median": sorted(nu)[len(nu) // 2],
        }
        log(f"  {k:<18} 中位 A r={summary[k]['r_median']:+.4f} "
            f"({summary[k]['n_sig']}/{summary[k]['n']} 显著) | "
            f"B acc1={summary[k]['acc1_median']:.4f} "
            f"(零中位 {summary[k]['null_median']:.4f})")

    (out_dir / "tokenizer_eval.json").write_text(
        json.dumps({"ckpt": args.ckpt, "config": vars(args),
                    "per_subject": per_subject, "summary": summary},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/tokenizer_eval.json")


if __name__ == "__main__":
    main()
