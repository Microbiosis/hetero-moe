"""Phase 46: 控制样本量的 within-image vs within-concept 对比。

## 修正 exp45 的设计缺陷

exp45 得到 B(7.0x) > A(2.9x)，我最初读成"概念泛化更好"，
实际是**训练样本量不同**：

    A within-image    训练 = 1 张图 x rep0-1 平均 -> **1 样本/类**
    B within-concept  训练 = 8 张图 x 4 rep 平均  -> **8 样本/类**

样本量差 8 倍，不能比。

## 正确设计：训练样本数固定为 8/类

    A_ctrl  train = 图1..8 的 rep0   ->  test = 同 8 张图的 rep1
            每类训练 8 样本，测试是**同图像**的新重复

    B_ctrl  train = 图1..8 的 rep0   ->  test = 图9,10 的 rep0
            每类训练 8 样本，测试是**同概念的新图像**

两者训练集完全相同（8 张图的 rep0），唯一差别是**测试集**：
- A 测同一批图像的新重复（低层视觉特征相同）
- B 测没见过的图像（只有概念相同）

## 判据

    A >> B  -> 解码依赖图像特定特征，跨图像泛化弱
    A ≈ B   -> 解码反映概念级表征
    B > A   -> 异常（可能 B 的测试更容易）

## 附加：多测试重复以提高统计稳定性

B 的测试只有 2 张图 × 4 rep = 8 样本/类，A 有 8 张图 × 4 rep = 32。
为公平，报告时同时给出每类样本数，并用 balanced accuracy。
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
OUT = Path("/domains/runs/controlled-vs")

N_CONCEPT = 100
WIN_MS = (50, 500)
N_TIME_BINS = 10
N_SUBJECTS = 5
SEED = 0


def feat(x, seg):
    """x: (n, r, ch, T) -> (n, r, ch*seg) 不做 rep 平均，保留试次维"""
    n, r, ch, T = x.shape
    y = x[:, :, :, :seg * N_TIME_BINS].reshape(n, r, ch, N_TIME_BINS, seg)
    return y.mean(axis=4).reshape(n, r, ch * N_TIME_BINS)


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


def balanced_acc(y_true, y_pred, n_cls):
    accs = []
    for c in range(n_cls):
        m = y_true == c
        if m.sum() > 0:
            accs.append((y_pred[m] == c).mean())
    return float(np.mean(accs)) if accs else 0.0


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

    # (n_img, 4, ch, T) -> (n_img, 4, feat)
    sub = eeg[sel_idx][:, :, :, a:b].astype(np.float32)
    F = feat(sub, seg)

    order = np.concatenate([np.arange(10) for _ in range(n_cls)])
    tr_img = order < 8          # 前 8 张图
    te_img = order >= 8         # 后 2 张图

    # ---------- 训练集：前 8 张图的 rep0（8 样本/类）
    Xtr = F[tr_img, 0]
    ytr = sel_lab[tr_img]

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd

    res = {"n_classes": n_cls, "chance": round(1.0 / n_cls, 5),
           "n_train_per_class": 8}

    # ---------- A: 同图像（前 8 张图的 rep1）
    Xa = (F[tr_img, 1] - mu) / sd
    ya = sel_lab[tr_img]
    predA = ridge_fit_predict(Xtr_n, ytr, Xa, n_cls)
    res["A_same_image"] = round(balanced_acc(ya, predA, n_cls), 5)
    res["A_n_test_per_class"] = 8

    # ---------- B: 同概念新图像（后 2 张图的全部 4 个 rep）
    Xb = []
    yb = []
    for li in range(n_cls):
        m = (sel_lab == li) & te_img
        Xb.append(F[m])          # (2, 4, feat)
        yb.extend([li] * 8)
    Xb = np.concatenate(Xb, axis=0).reshape(-1, F.shape[-1])
    yb = np.array(yb)
    Xb_n = (Xb - mu) / sd
    predB = ridge_fit_predict(Xtr_n, ytr, Xb_n, n_cls)
    res["B_new_image"] = round(balanced_acc(yb, predB, n_cls), 5)
    res["B_n_test_per_class"] = 8

    # ---------- C: 打乱标签
    predC = ridge_fit_predict(Xtr_n, rng.permutation(ytr), Xb_n, n_cls)
    res["C_shuffled"] = round(balanced_acc(yb, predC, n_cls), 5)

    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 84, flush=True)
    print("Phase 46 — 控制样本量的 within-image vs within-concept", flush=True)
    print("=" * 84, flush=True)
    print(f"  {N_CONCEPT} 概念 | 训练=前8图的 rep0 (8 样本/类)", flush=True)
    print(f"  A 测=同8图的 rep1 (8/类) | B 测=后2图的全部 rep (8/类)", flush=True)
    print(f"  随机 {1/N_CONCEPT*100:.1f}% | balanced accuracy", flush=True)

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
            continue
        r["subject"] = name
        rows.append(r)
        print(f"    A same_image   {r['A_same_image']:.4f} "
              f"({r['A_same_image']/r['chance']:.1f}x)", flush=True)
        print(f"    B new_image    {r['B_new_image']:.4f} "
              f"({r['B_new_image']/r['chance']:.1f}x)", flush=True)
        print(f"    C shuffled     {r['C_shuffled']:.4f}  [{time.time()-t1:.0f}s]",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    if not len(df):
        return

    print("\n" + "=" * 84, flush=True)
    print("汇总（中位，balanced accuracy）", flush=True)
    print("=" * 84, flush=True)
    med = df[["A_same_image", "B_new_image", "C_shuffled"]].median()
    ch = float(df.chance.median())
    for k, v in med.items():
        print(f"  {k:<16} {v:.4f}  ({v/ch:.1f}x)", flush=True)

    ratio = med["A_same_image"] / max(med["B_new_image"], 1e-9)
    print(f"\n  A/B = {ratio:.2f}", flush=True)
    print(f"  B-C = {med['B_new_image']-med['C_shuffled']:+.4f}", flush=True)

    print("\n" + "=" * 84, flush=True)
    print("判定", flush=True)
    print("=" * 84, flush=True)
    print("  注：两者训练集完全相同（8 张图 x rep0），只有测试集不同。", flush=True)

    if med["B_new_image"] <= med["C_shuffled"] * 2:
        v = (f"❌ 跨图像泛化失败：B={med['B_new_image']:.4f} 接近打乱 "
             f"{med['C_shuffled']:.4f}")
    elif ratio > 2.0:
        v = (f"⚠️ 图像特定特征占主导：A/B={ratio:.1f}"
             f"（同图像 {med['A_same_image']:.4f} vs 新图像 {med['B_new_image']:.4f}）")
    elif ratio > 1.3:
        v = (f"⚠️ 部分图像依赖：A/B={ratio:.2f}，"
             f"新图像仍达 {med['B_new_image']/ch:.1f}x 随机")
    else:
        v = (f"✅ 解码反映概念级表征：A/B={ratio:.2f}，"
             f"新图像 {med['B_new_image']:.4f} ({med['B_new_image']/ch:.1f}x) ≈ "
             f"同图像 {med['A_same_image']:.4f} ({med['A_same_image']/ch:.1f}x)")
    print(f"  {v}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "controlled_within_image_vs_concept",
        "fix_note": ("exp45 的 A 每类仅 1 个训练样本（1 图 x 2rep 平均），"
                     "B 有 8 个，样本量差 8 倍不可比。本实验固定两者训练集"
                     "完全相同（8 张图的 rep0），只改测试集。"),
        "dataset": "THINGS-EEG train",
        "n_concept": N_CONCEPT, "n_subjects": len(df),
        "window_ms": list(WIN_MS), "n_time_bins": N_TIME_BINS,
        "medians": {k: round(float(v_), 4) for k, v_ in med.items()},
        "chance": round(ch, 5), "A_over_B": round(float(ratio), 3),
        "per_subject": rows, "verdict": v,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
