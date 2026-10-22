#!/usr/bin/env python3
"""exp71 — train split（16540 图 × 4 重复）在**先验指定窗口**上的编码显著性。

## 要回答的问题

exp70 的标度律说：**图像数 ↑ ⇒ 零分布地板下降**（可检出性 ↑），
**重复数 ↑ ⇒ 信号天花板抬高**（效应强度 ↑）。两个因子各管各的。

但还有一格没测：**图像数 16540（极多 ⇒ 地板极低）× 重复数 4（极少 ⇒ 天花板极低）**，
也就是 train split 本身。exp69 在这一格测到 R² ≈ +0.0003 ~ +0.0028，
但用的是**固定窗零分布**（乐观），正确口径下（sub-01）p = 0.438 —— 不可分辨。

**这一格决定了"到底该用哪个 split"。**

- 若在 16540 图 × 4 重复下仍不可分辨 ⇒ **test split（200 图 × 80 重复）才是最优底座**，
  train split 不是更好的训练集，只是更噪的。
- 若可分辨 ⇒ **图像多的优势压过了重复少的劣势**，train split 值得作为大规模底座，
  而且单试次解码也有希望。

## 为什么这一格没有窗口选择偏置

exp70 的"观测取 max_over_windows 而零分布固定在一个窗"是不一致口径（失误 6）。
本脚本**直接用先验指定的窗口**：

    200–400 ms  ← 由 exp64/65/66/67 的 RSA 结果**独立**确定（效应一直落在
                   100–400 ms，跨 10 个被试、跨两个 split 复现）

观测统计量 = 固定窗的 R²，零分布 = 同一固定窗的置换分布。
**观测与零分布形式完全一致 ⇒ p 值诚实，无选择偏置。**

（附带算一个全窗剖面，仅用于展示 16540 图下效应的时间形状；那个 max 只作参考，
不作显著性判据。）

## 判据

    观测 R²（200–400 ms）> 零分布 95 分位   ⇒ 16540 图这一格可检出
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

FS = 250.0
WIN = 25
BASELINE = 12
N_FOLDS = 5
N_SUBJECTS = 5          # 单被试约 10 分钟，5 个被试足够
N_NULL = 10
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
PRIOR_WIN = (2, 4)      # 200–400 ms（先验指定，0-based bin 索引）
SEED = 0

EegTRAIN = Path("/domains/omnibrain4d/eeg_train")
OUT_DIR = Path("/domains/runs/tier0-encoding")


def log(*a):
    print(*a, flush=True)


# ================================================================ 工具
def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, te), te) for te in np.array_split(idx, k)]


def build_image_targets(eeg, chunk=400):
    """(n_img, n_rep, n_ch, T) → (n_img, n_ch*n_bin) 每张图的试次平均（窗主序）。"""
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)
        e = e.transpose(0, 1, 3, 2)                  # (c, rep, bin, ch) 窗主序
        out[s:s + chunk] = e.mean(axis=1).reshape(e.shape[0], n_ch * n_bin)
        del e
    return out


def ridge_window(X, Ycols, folds, alphas, n_null, rng, n_ch=63):
    """primal ridge，只在**固定窗**的列上做。

    返回 (best_alpha, 观测 R², 零分布数组, 逐窗剖面(可选))
    观测与零分布形式一致（同一固定窗、同一 α 选择规则）⇒ p 值诚实。
    """
    import scipy.linalg as sla
    n, d = X.shape
    n_t = Ycols.shape[1]
    folds = list(folds)
    real = {a: None for a in alphas}
    profile = {a: None for a in alphas} if Ycols.shape[1] == 630 else None

    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        XtX = (Xtr.T @ Xtr).astype(np.float64)
        ybar = Ycols[tr].mean(axis=0, keepdims=True)
        Ytr = Ycols[tr] - ybar
        Yte = Ycols[te]
        ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
        scale = float(np.trace(XtX)) / d
        XtY = Xtr.T @ Ytr
        for a in alphas:
            A = XtX.copy()
            A.flat[:: d + 1] += a * scale
            W = sla.solve(A, XtY, assume_a="pos", overwrite_a=True, check_finite=False)
            pred = Xte @ W + ybar
            r2 = 1.0 - ((Yte - pred) ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-12)
            real[a] = r2 if real[a] is None else real[a] + r2
            if profile is not None:
                pr = r2.reshape(10, n_ch) if n_t == 630 else None
                profile[a] = pr if profile[a] is None else profile[a] + pr

    nf = len(folds)
    best_a, best_r2 = None, None
    for a in alphas:
        v = float(np.median(real[a] / nf))
        if best_a is None or v > best_r2:
            best_a, best_r2 = a, v
    prof = None
    if profile is not None:
        pw = np.median((profile[best_a] / nf).reshape(10, n_ch), axis=1)
        prof = [float(x) for x in pw]

    # 零分布：同一固定窗、同一 α 选择规则
    nulls = np.zeros(n_null)
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        XtX = (Xtr.T @ Xtr).astype(np.float64)
        ybar = Ycols[tr].mean(axis=0, keepdims=True)
        Ytr = Ycols[tr] - ybar
        Yte = Ycols[te]
        ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
        scale = float(np.trace(XtX)) / d
        for pi in range(n_null):
            Yp = Ytr[rng.permutation(len(tr))]
            A = XtX.copy()
            A.flat[:: d + 1] += best_a * scale
            W = sla.solve(A, Xtr.T @ Yp, assume_a="pos", overwrite_a=True,
                          check_finite=False)
            pred = Xte @ W + ybar
            r2 = 1.0 - ((Yte - pred) ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-12)
            nulls[pi] += float(np.median(r2))
    nulls /= nf
    return best_a, best_r2, nulls, prof


# ================================================================ 主
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    import torch
    f = torch.load("/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt",
                   map_location="cpu", weights_only=False)
    IMG = f["img_features"].numpy().astype(np.float64)
    log(f"  ViT img_features {IMG.shape}")

    # 载入各被试的图像级试次平均（只取先验窗口的列，省内存）
    lo, hi = PRIOR_WIN
    cols = np.arange(lo * 63, hi * 63)
    Yw = {}
    for sid in range(1, N_SUBJECTS + 1):
        fp = EegTRAIN / f"sub-{sid:02d}.npz"
        if not fp.exists():
            continue
        ts = time.time()
        eeg = np.load(fp)["eeg"]
        Y = build_image_targets(eeg)
        del eeg
        Yw[sid] = Y
        log(f"  sub-{sid:02d}: 图像级目标 {Y.shape}  [{time.time()-ts:.0f}s]")

    if not Yw:
        log("没有可用被试")
        return

    rows = []
    rows_done = False
    for sid, Y in Yw.items():
        ts = time.time()
        folds = make_folds(Y.shape[0], N_FOLDS, rng)
        a, r2, nulls, prof = ridge_window(IMG, Y[:, cols], folds, ALPHAS, N_NULL, rng)
        p = float((np.sum(nulls >= r2) + 1) / (N_NULL + 1))

        # BBB：其余被试对同一图像的平均响应 → 本被试
        others = [v for k, v in Yw.items() if k != sid]
        bbb = None
        if others:
            Yb = np.mean(np.stack(others, axis=0), axis=0)
            _, r2b, _, _ = ridge_window(Yb, Y[:, cols], folds, ALPHAS, 0, rng)
            bbb = float(r2b)

        rec = {"subject": f"sub-{sid:02d}", "n_img": int(Y.shape[0]),
               "window_ms": [lo * 100, hi * 100],
               "alpha": a, "r2_prior_window": r2,
               "null_median": float(np.median(nulls)),
               "null_p95": float(np.percentile(nulls, 95)),
               "null_max": float(nulls.max()),
               "p": p, "r2_bbb_same_window": bbb,
               "per_window_profile": prof,
               "elapsed_s": round(time.time() - ts, 1)}

        # 只对第一个被试补算全窗剖面（展示 16540 图下效应的时间形状）
        if prof is None and not rows_done:
            log(f"  sub-{sid:02d}: 补算全窗剖面 …")
            _, _, _, full_prof = ridge_window(IMG, Y, folds, ALPHAS, 0, rng)
            rec["per_window_profile"] = full_prof
            log(f"    逐窗 R²: " + ", ".join(f"{i*100}-{i*100+100}:{v:+.5f}"
                                           for i, v in enumerate(full_prof)))
        rows_done = True

        rows.append(rec)
        log(f"  sub-{sid:02d}: 先验窗 {lo*100}-{hi*100}ms  R²={r2:+.5f}  "
            f"零 中位={rec['null_median']:+.5f} 95%={rec['null_p95']:+.5f}  "
            f"p={p:.3f} | BBB {bbb if bbb is None else round(bbb,5)}  "
            f"[{rec['elapsed_s']}s]")
        (OUT_DIR / "exp71_train_prior_window.json").write_text(
            json.dumps({"per_subject": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ---------------- 汇总 ----------------
    log("\n" + "=" * 80)
    log(f"train split 16540 图 × 4 重复，先验窗 {lo*100}-{hi*100} ms")
    log("=" * 80)
    r2s = [r["r2_prior_window"] for r in rows]
    ns = [r["null_median"] for r in rows]
    ps = [r["p"] for r in rows]
    sig = sum(1 for p in ps if p <= 1.0 / (N_NULL + 1) + 1e-9)
    summary = {
        "n_subjects": len(rows),
        "r2_median": float(np.median(r2s)),
        "null_median": float(np.median(ns)),
        "p_median": float(np.median(ps)),
        "n_sig": sig,
        "detectable": bool(sig >= max(2, len(rows) // 2)),
        "r2_bbb_median": (float(np.median([r["r2_bbb_same_window"] for r in rows
                                          if r["r2_bbb_same_window"] is not None]))
                          if any(r["r2_bbb_same_window"] is not None for r in rows) else None),
    }
    log(f"  R² 中位 {summary['r2_median']:+.5f} | 零中位 {summary['null_median']:+.5f} | "
        f"p 中位 {summary['p_median']:.3f} | 达最小 p 的被试 {sig}/{len(rows)}")
    log(f"  → {'✅ 这一格可检出：图像多的优势压过了重复少的劣势' if summary['detectable'] else '❌ 不可分辨：test split（200 图 × 80 重复）才是最优底座'}")

    report = {"schema_version": "1.0", "script": "exp71_train_prior_window",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "design": {"n_images": 16540, "n_reps": 4,
                         "window_ms": [lo * 100, hi * 100],
                         "window_source": "先验指定（exp64/65/66/67 的 RSA 独立结果）",
                         "cv": "5-fold over IMAGES", "ridge": "primal (n>>d)",
                         "null": "permuting training-fold Y rows, 同一固定窗（口径一致）"},
              "summary": summary, "per_subject": rows,
              "elapsed_s": round(time.time() - t0, 1)}
    (OUT_DIR / "exp71_train_prior_window_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
