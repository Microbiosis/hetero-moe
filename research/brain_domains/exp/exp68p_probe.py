#!/usr/bin/env python3
"""exp68p — 探查 train split 的确切形状与对应关系（不跑重计算）。

要回答四件事：
  1. ViT-H-14_features_train.pt 里 img_features / text_features 各多少行？
  2. 原始 train.pt 的 eeg / label / text / img 各是什么形状？
  3. label 的取值个数 —— 是 16540（图像级）还是 1654（概念级）？
     THINGS-EEG2 的 train split = 1654 概念 × 10 图 = 16540 图像。
  4. ViT img_features 的行序是否与 EEG 的图像序一致？
     检验：同一概念的 10 张图在 ViT 特征空间里应当比随机两张图更相似。
     置换零 = 打乱概念分组后这个差距应消失。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

FEAT_TRAIN_PT = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
TRAIN_PT = "/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/train.pt"
OUT = Path("/domains/runs/tier0-encoding")


def log(*a):
    print(*a, flush=True)


def main():
    import torch
    t0 = time.time()
    rep = {}

    log("=" * 80)
    log("exp68p — train split 形状与对应关系探查")
    log("=" * 80)

    # --- 1) ViT 特征 ---
    f = torch.load(FEAT_TRAIN_PT, map_location="cpu", weights_only=False)
    log(f"\n[ViT train 特征] {FEAT_TRAIN_PT}")
    for k in f.keys():
        v = f[k]
        log(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        rep[f"feat_{k}_shape"] = list(tuple(v.shape))
    IMG = f["img_features"].numpy().astype(np.float32)
    TXT = f["text_features"].numpy().astype(np.float32)
    del f

    # --- 2) 原始 train.pt ---
    log(f"\n[原始 train.pt] {TRAIN_PT}")
    d = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)
    for k in d.keys():
        v = d[k]
        if hasattr(v, "shape"):
            log(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v,'dtype','')}")
            rep[f"raw_{k}_shape"] = list(tuple(v.shape))
    eeg = d["eeg"]
    lab = np.asarray(d["label"])
    txt = np.asarray(d["text"])
    imgp = np.asarray(d["img"])

    n_img_eeg = eeg.shape[0]
    n_rep = eeg.shape[1]
    log(f"  eeg {tuple(eeg.shape)} | n_images={n_img_eeg} | n_reps={n_rep}")

    # --- 3) label 语义 ---
    lab0 = lab[:, 0]
    uq = np.unique(lab0)
    log(f"\n  label 唯一值个数 = {len(uq)}  (min {uq.min()}, max {uq.max()})")
    log(f"  label 是否沿 rep 恒定: {bool(np.all(lab == lab[:, :1]))}")
    counts = np.bincount(lab0)
    log(f"  每个 label 的图像数: min={counts[counts>0].min()} max={counts[counts>0].max()}")
    rep["n_label_unique"] = int(len(uq))
    rep["label_constant_over_rep"] = bool(np.all(lab == lab[:, :1]))
    rep["images_per_label_min"] = int(counts[counts > 0].min())
    rep["images_per_label_max"] = int(counts[counts > 0].max())

    # label 是否按 10 张一组连续排布
    order_ok = bool(np.all(np.diff(lab0) >= 0)) and bool(
        len(set(lab0[:10].tolist())) == 1)
    log(f"  label 是否连续分组（每概念 10 张连排）: {order_ok}")
    log(f"  label[:12] = {lab0[:12].tolist()}")
    log(f"  label[0:20] 每 10 个一折看: {[int(lab0[i]) for i in range(0, 20)]}")
    rep["label_contiguous_groups"] = order_ok

    # text 是否同一 label 内恒定
    same_text = all(len(set(map(str, txt[lab0 == c]))) == 1
                    for c in uq[:50])
    log(f"  同一 label 内 text 是否恒定（前 50 个 label）: {same_text}")
    rep["text_constant_within_label"] = bool(same_text)
    log(f"  text 样例: {str(txt[0,0])!r} | img 样例: {str(imgp[0,0])!r}")
    log(f"  img 样例(第 1234 行): {str(imgp[1234,0])!r}")

    del eeg, d

    # --- 4) ViT img_features 的行序 vs EEG 的图像序 ---
    log("\n  [对齐检验] 同一概念的 10 张图在 ViT 特征空间里应更相似")
    if len(IMG) == n_img_eeg:
        V = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-12)
        # 同一概念内的图像对（取前若干个概念的全部对）
        pairs = []
        for c in uq[:300]:
            idx = np.where(lab0 == c)[0]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    pairs.append((idx[i], idx[j]))
        pairs = np.array(pairs)
        same = np.einsum("ij,ij->i", V[pairs[:, 0]], V[pairs[:, 1]]).mean()
        rng = np.random.default_rng(0)
        rnd = []
        for _ in range(200):
            p = rng.choice(n_img_eeg, size=len(pairs), replace=False)
            rnd.append(np.einsum("ij,ij->i", V[p[:, 0]], V[p[:, 1]]).mean())
        rnd = np.array(rnd)
        log(f"    同一概念的平均余弦相似 = {same:+.4f}")
        log(f"    随机配对的零分布        = {rnd.mean():+.4f} ± {rnd.std():.4f}"
            f"  [95%={np.percentile(rnd,95):+.4f}]")
        gap = float(same - rnd.mean())
        log(f"    差距 = {gap:+.4f}  (置换零 p = "
            f"{float((np.sum(rnd >= same) + 1) / (len(rnd) + 1)):.4f})")
        rep["align_same_concept_cos"] = float(same)
        rep["align_null_cos_mean"] = float(rnd.mean())
        rep["align_null_cos_sd"] = float(rnd.std())
        rep["align_gap"] = gap
        rep["align_p"] = float((np.sum(rnd >= same) + 1) / (len(rnd) + 1))
        log(f"    → {'✅ ViT img_features 行序与 EEG 图像序一致' if gap > 3 * rnd.std() else '❌ 对齐存疑'}")
    else:
        log(f"    ⚠️ ViT img_features 行数 {len(IMG)} != EEG 图像数 {n_img_eeg}"
            "  → 需要概念级聚合后再比对")
        rep["align_note"] = "row count mismatch"

    rep["elapsed_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp68p_probe.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
