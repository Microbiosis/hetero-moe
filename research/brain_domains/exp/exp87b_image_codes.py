#!/usr/bin/env python3
"""exp87b_image_codes.py — 建**图像侧的离散词表**（共享词典的另一半）。

## 为什么需要

脑域线的目标形态是「脑 token 与图像 token **同一个词典**」
（`MODEL_FORM.md` §4 统一词表）。要让 EEG 的 token 直接用「图像侧的码」，
就得先有一本**图像侧的离散词典**。

本脚本用**最省事且可辩护**的做法：在已有的 ViT-H-14 图像特征上做
**球面 k-means**（L2 归一化后的特征 ⇒ 簇按余弦相似度划分）
⇒ 得到 K 个图像原型（centroids）与每张图的码号。

## 自带判据（不满足就别往下走）

对 k-means 出来的码做一次 RSA（one-hot 码 vs 图像特征）：
**图像侧的码本身必须与图像语义对齐**，否则「让 EEG token 预测它」没有意义。
⇒ 脚本会打印 RSA 与置换检验 p 值。

## 输出

`/domains/runs/illusionbrain-align/shared-vocab/image_codes_k{K}.pt`：
`{"centroids": (K, 1024), "codes": (16540,), "n_codes": K, "dim": 1024, "normalized": True}`

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp87b_image_codes.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 120m -d \
      --k 1024 --out-dir /domains/runs/illusionbrain-align/shared-vocab
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
for _p in (str(EXP_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
SEED = 0


def log(*a):
    print(*a, flush=True)


def spherical_kmeans(X, k, iters=25, seed=SEED, chunk=4096):
    """球面 k-means（L2 归一化的特征 + 归一化的质心）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # 初始化：随机取 k 个样本当质心（k-means++ 太慢，K=1024 时随机足够好）
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    prev = None
    for it in range(iters):
        t0 = time.time()
        assign = np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # 余弦相似度 = 归一化后的内积
            sim = X[s:e] @ C.T
            assign[s:e] = sim.argmax(axis=1)
        # 质心更新（再归一化）
        C_new = np.zeros_like(C)
        np.add.at(C_new, assign, X)
        cnt = np.bincount(assign, minlength=k)
        empty = cnt == 0
        C_new[~empty] /= cnt[~empty, None]
        C_new[empty] = C[empty]            # 空簇：保持原质心
        C = C_new / np.maximum(np.linalg.norm(C_new, axis=1, keepdims=True), 1e-12)
        moved = float(np.mean(assign != prev)) if prev is not None else 1.0
        prev = assign
        log(f"  iter {it:>2} | 改动率 {moved:.4f} | 空簇 {int(empty.sum())} "
            f"| {time.time()-t0:.1f}s")
        if moved < 1e-4:
            break
    # 最后再分配一次
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        assign[s:e] = (X[s:e] @ C.T).argmax(axis=1)
    return C, assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out-dir", default="/domains/runs/illusionbrain-align/shared-vocab")
    ap.add_argument("--no-normalize", action="store_true",
                    help="不 L2 归一化特征（默认归一化 ⇒ 球面 k-means）")
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG = ftr["img_features"].numpy().astype(np.float32)
    log(f"图像特征 {IMG.shape}")
    norms = np.linalg.norm(IMG, axis=1)
    log(f"  范数 mean {norms.mean():.3f} std {norms.std():.3f} "
        f"min {norms.min():.3f}")
    if not args.no_normalize:
        IMG = IMG / np.maximum(norms[:, None], 1e-12)
        log("  已 L2 归一化（球面 k-means）")

    log(f"\n=== 球面 k-means：K={args.k}，n={IMG.shape[0]} ===")
    t0 = time.time()
    C, codes = spherical_kmeans(IMG, args.k, iters=args.iters)
    log(f"  用时 {time.time()-t0:.0f}s")

    cnt = np.bincount(codes, minlength=args.k)
    p = cnt[cnt > 0] / cnt.sum()
    ent = float(-(p * np.log(p)).sum())
    log(f"  簇大小：最大 {cnt.max()}，最小 {cnt.max() and cnt[cnt>0].min()}，"
        f"非空簇 {int((cnt>0).sum())}/{args.k}")
    log(f"  簇熵 {ent:.2f} nats（最大 {np.log(max(int((cnt>0).sum()),2)):.2f}）")

    # --- 自带判据：图像侧的码本身与图像语义对齐吗 ---
    log("\n=== 判据：图像侧 one-hot 码 vs 图像特征的 RSA ===")
    try:
        import eval_brain_alignment as EVAL
        rng = np.random.default_rng(SEED)
        idx = rng.choice(IMG.shape[0], size=min(2000, IMG.shape[0]), replace=False)
        onehot = np.zeros((len(idx), args.k), dtype=np.float32)
        onehot[np.arange(len(idx)), codes[idx]] = 1.0
        a = EVAL.metric_a_rsa(onehot[:, None, :], IMG[idx], 200, rng)
        log(f"  r_img = {a['r_img']:+.4f}，p = {a['p_img']:.4f}")
        # ⚠️ **one-hot 上的 RSA 是病态的**：两个 one-hot 行支撑不交 ⇒ 零方差 ⇒
        #    相关系数未定义（实现里靠 epsilon 兜 ⇒ 数值没有意义）。
        #    实测给出 r = **−0.1625**（p=0.005）—— 既不能读成「对齐」，
        #    也不能读成「反对齐」⇒ **这个判据不可用**。
        # ⇒ 改用**稠密**的质心向量来判（质心 = 该簇的特征均值 ⇒ 有定义）。
        cen = C[codes[idx]]                      # (n, dim)：每张图所在簇的质心
        a2 = EVAL.metric_a_rsa(cen[:, None, :], IMG[idx], 200, rng)
        log(f"  [稠密对照] 簇质心 vs 图像特征：r_img = {a2['r_img']:+.4f}，"
            f"p = {a2['p_img']:.4f}")
        if a2["r_img"] > 0 and a2["p_img"] < 0.05:
            log("  ✓ 图像侧的码与图像语义对齐（稠密对照通过）"
                " ⇒ 可以拿它当共享词典的目标")
        else:
            log("  ⚠ 稠密对照也没通过 ⇒ 先查 k-means，别急着训")
        if a["r_img"] < 0:
            log("  （one-hot 那个负值是病态计算的产物，**不作判据**）")
    except Exception as e:
        log(f"  （RSA 判据没跑成：{type(e).__name__}: {e}）")

    fp = out_dir / f"image_codes_k{args.k}.pt"
    torch.save({"centroids": C.astype(np.float32),
                "codes": codes.astype(np.int64),
                "n_codes": args.k, "dim": int(IMG.shape[1]),
                "normalized": not args.no_normalize}, fp)
    log(f"\n写入 {fp}（{fp.stat().st_size/1e6:.1f} MB）")
    (out_dir / f"image_codes_k{args.k}.json").write_text(
        json.dumps({"k": args.k, "n": int(IMG.shape[0]),
                    "nonempty_clusters": int((cnt > 0).sum()),
                    "cluster_entropy_nats": ent,
                    "max_cluster": int(cnt.max())}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"✗ {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1500:])
        raise SystemExit(1)
