"""Phase 45: 同图像 vs 同概念 —— 模型学到的是"图像"还是"概念"？

## 为什么这个对比重要

之前所有解码实验（exp34、Space 的 run_decode）都在 test 集上做，
划分是「同图像的前 40 次重复训练 / 后 40 次重复测试」。

隐患：同一张图像重复呈现，低层视觉特征（亮度、对比度、边缘分布）
几乎完全相同。分类器可能只是在识别「这张图像的特定指纹」，
而不是「这是哪个概念」。

200 类解码 15-50x 随机，可能被这个因素高估。

## train 集提供判别机会

THINGS-EEG train = 1654 概念 x 每概念 10 张不同图像 x 4 次重复。

两种划分：

    within-image    图像 A 的 rep 0-1  ->  图像 A 的 rep 2-3   测"记住了这张图"
    within-concept  图像 A..H（8 张）   ->  图像 I,J（另 2 张）  测"泛化到新图像"

若 within-image >> within-concept，之前的高准确率部分是图像指纹效应。

## 判据

    A ≈ B  -> 解码反映概念级表征，旧结论成立
    A >> B -> 旧结论被图像指纹效应高估，需重新表述
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FS = 250.0
DS = Path("/domains/omnibrain4d/eeg_train")
OUT = Path("/domains/runs/image-vs-concept")

N_CONCEPT = 100
N_TRAIN_IMG = 8
N_TEST_IMG = 2
WIN_MS = (50, 500)
N_TIME_BINS = 10
N_SUBJECTS = 5
SEED = 0


def _feat(x, seg):
    """x: (n, n_rep, ch, T) -> (n, ch*seg) 时间降采样后 rep 平均"""
    n, r, ch, T = x.shape
    y = x[:, :, :, :seg * N_TIME_BINS].reshape(n, r, ch, N_TIME_BINS, seg)
    y = y.mean(axis=4).reshape(n, r, ch * N_TIME_BINS)
    return y.mean(axis=1)


def ridge_fit_predict(Xtr, ytr, Xte, n_class, lam=1.0):
    n, d = Xtr.shape
    Y = np.zeros((n, n_class), dtype=np.float32)
    Y[np.arange(n), ytr] = 1.0
    A = Xtr.T @ Xtr
    A.flat[:: d + 1] += lam * np.trace(A) / d
    try:
        W = np.linalg.solve(A, Xtr.T @ Y)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, Xtr.T @ Y, rcond=None)[0]
    return (Xte @ W).argmax(axis=1)


def run_subject(eeg, meta_s, rng):
    a = int(WIN_MS[0] / 1000 * FS)
    b = int(WIN_MS[1] / 1000 * FS)
    seg = (b - a) // N_TIME_BINS

    meta0 = meta_s[meta_s.rep == 0].reset_index(drop=True)
    concepts = meta0.text.to_numpy()

    uniq = np.unique(concepts)
    pick = rng.choice(uniq, size=min(N_CONCEPT, len(uniq)), replace=False)

    sel_idx, sel_lab = [], []
    for lab, c in enumerate(pick):
        idx = np.where(concepts == c)[0][:10]
        if len(idx) < 10:
            continue
        sel_idx.extend(idx.tolist())
        sel_lab.extend([lab] * len(idx))

    if not sel_lab:
        return None

    sel_idx = np.array(sel_idx)
    sel_lab = np.array(sel_lab)
    n_cls = len(np.unique(sel_lab))

    sub = eeg[sel_idx][:, :, :, a:b].astype(np.float32)   # (n_img, 4, ch, T)

    # ---- A: within-image（rep 0-1 训练，rep 2-3 测试）
    Xa = _feat(sub[:, :2], seg)
    Xb = _feat(sub[:, 2:], seg)
    mu, sd = Xa.mean(0, keepdims=True), Xa.std(0, keepdims=True) + 1e-6
    predA = ridge_fit_predict((Xa - mu) / sd, sel_lab, (Xb - mu) / sd, n_cls)
    accA = float((predA == sel_lab).mean())

    # ---- B: within-concept（前 8 图训练，后 2 图测试）
    Xall = _feat(sub, seg)
    order = np.concatenate([np.arange(10) for _ in range(n_cls)])
    tr = order < N_TRAIN_IMG
    te = ~tr
    Xtr, Xte = Xall[tr], Xall[te]
    ytr, yte = sel_lab[tr], sel_lab[te]
    mu2, sd2 = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    predB = ridge_fit_predict((Xtr - mu2) / sd2, ytr, (Xte - mu2) / sd2, n_cls)
    accB = float((predB == yte).mean())

    # ---- C: shuffled（打乱训练标签）
    predC = ridge_fit_predict((Xtr - mu2) / sd2, rng.permutation(ytr),
                              (Xte - mu2) / sd2, n_cls)
    accC = float((predC == yte).mean())

    return {
        "A_within_image": round(accA, 5),
        "B_within_concept": round(accB, 5),
        "C_shuffled": round(accC, 5),
        "n_classes": int(n_cls),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "chance": round(1.0 / n_cls, 5),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 84, flush=True)
    print("Phase 45 — 同图像 vs 同概念：解码反映的是什么", flush=True)
    print("=" * 84, flush=True)
    print(f"  {N_CONCEPT} 概念 x 10 图 | 训练 {N_TRAIN_IMG} / 测试 {N_TEST_IMG}",
          flush=True)
    print(f"  期望随机 {1/N_CONCEPT*100:.1f}%", flush=True)

    meta = pd.read_parquet(DS / "meta.parquet")
    rows = []
    t0 = time.time()

    for sid in range(1, N_SUBJECTS + 1):
        name = f"sub-{sid:02d}"
        f = DS / f"{name}.npz"
        if not f.exists():
            continue
        print(f"\n  --- {name} ---", flush=True)
        t1 = time.time()
        eeg = np.load(f)["eeg"]
        r = run_subject(eeg, meta[meta.subject == name], rng)
        del eeg
        if r is None:
            print("    失败", flush=True)
            continue
        r["subject"] = name
        rows.append(r)
        ch = r["chance"]
        print(f"    A within_image    {r['A_within_image']:.4f} "
              f"({r['A_within_image']/ch:.1f}x)", flush=True)
        print(f"    B within_concept  {r['B_within_concept']:.4f} "
              f"({r['B_within_concept']/ch:.1f}x)", flush=True)
        print(f"    C shuffled        {r['C_shuffled']:.4f} "
              f"({r['C_shuffled']/ch:.1f}x)  [{time.time()-t1:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)
    if not len(df):
        return

    print("\n" + "=" * 84, flush=True)
    print("汇总（中位）", flush=True)
    print("=" * 84, flush=True)
    med = df[["A_within_image", "B_within_concept", "C_shuffled"]].median()
    ch = float(df.chance.median())
    for k, v in med.items():
        print(f"  {k:<18} {v:.4f}  ({v/ch:.1f}x 随机)", flush=True)

    ratio = float(med["A_within_image"] / max(med["B_within_concept"], 1e-9))
    print(f"\n  A/B 比 = {ratio:.2f}", flush=True)
    print(f"  A-C   = {med['A_within_image']-med['C_shuffled']:+.4f}", flush=True)
    print(f"  B-C   = {med['B_within_concept']-med['C_shuffled']:+.4f}", flush=True)

    print("\n" + "=" * 84, flush=True)
    print("判定", flush=True)
    print("=" * 84, flush=True)

    if med["B_within_concept"] <= med["C_shuffled"] * 1.5:
        verdict = (f"❌ 概念泛化失败：B ({med['B_within_concept']:.4f}) 接近"
                   f"打乱基线 C ({med['C_shuffled']:.4f})。"
                   f"之前的解码准确率主要来自同图像重复，不是概念。")
    elif ratio > 2.0:
        verdict = (f"⚠️ 图像指纹效应占主导：A/B = {ratio:.1f}。"
                   f"同图像 {med['A_within_image']:.4f} vs "
                   f"同概念 {med['B_within_concept']:.4f}。"
                   f"旧结论（test 集同图像划分）被高估约 {ratio:.1f} 倍。")
    elif ratio < 1.3:
        verdict = (f"✅ 解码反映概念级表征：A/B = {ratio:.2f}，"
                   f"同图像与同概念接近（{med['A_within_image']:.4f} vs "
                   f"{med['B_within_concept']:.4f}）。旧结论成立。")
    else:
        verdict = (f"⚠️ 部分图像效应：A/B = {ratio:.2f}，"
                   f"A={med['A_within_image']:.4f} B={med['B_within_concept']:.4f}")

    print(f"  {verdict}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "image_vs_concept",
        "dataset": "THINGS-EEG train",
        "n_concept": N_CONCEPT, "n_train_img": N_TRAIN_IMG,
        "n_test_img": N_TEST_IMG, "n_subjects": len(df),
        "window_ms": list(WIN_MS), "n_time_bins": N_TIME_BINS,
        "medians": {k: round(float(v), 4) for k, v in med.items()},
        "chance": round(ch, 5), "A_over_B": round(ratio, 3),
        "per_subject": rows, "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
