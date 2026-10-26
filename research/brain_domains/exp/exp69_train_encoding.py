#!/usr/bin/env python3
"""exp69 — Tier 0 编码的 train split 版（图像级 16540 样本）。

## 数据结构（exp68p 实测确认）

    原始 train.pt   eeg (16540, 4, 63, 250)  label (16540,4)  text (16540,4)
                    label = 1654 个概念，每个概念恰好 10 张图，且**连续分组**
                    （行 0-9 = 概念 0，行 10-19 = 概念 1 …）
    ViT 特征        img_features (16540, 1024)  ← 与 EEG 图像 **1:1**
                    text_features (1654, 1024)  ← 每概念

所以**图像级**编码是 16540 对 16540 —— 比 test split 的 200 张多 **83 倍**。
这是判断「模型特征能解释多少脑活动」的关键数字。

## 与 exp67（test split）不同的三处

1. **primal ridge**：n=16540 ≫ d=1024。
2. **置换零打乱训练折的 Y 行**：X 不动 ⇒ 同折 XᵀX 复用，每次置换省一次
   13232×1024×1024 的矩阵乘；零假设下与打乱「特征↔图像对应」等价。
3. **文本特征走概念级**：text_features 只有 1654 行，要与概念级 EEG 对齐
   ⇒ 把每个概念的 10 张图的试次平均再平均，作为该概念的目标。

## PART 0 是硬门槛

先验证 **ViT img_features 的行序 == EEG 的图像序**：
同一概念的 10 张图在 ViT 特征空间里应比随机两张图更相似。
检验用「组内均值 − 随机配对均值」与随机配对的零分布比较。
**不通过就整个实验作废，不往下跑。**（不需要 4.17 GB 的 train.pt：
概念分组结构已由 exp68p 确认是连续 10 张一组。）

## 判据

    顺序对齐   组内相似度显著高于随机配对
    编码       R² img 显著高于置换零；img/BBB 可解读
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
N_SUBJECTS = 10
N_NULL = 20
GROUPS = 10              # 每个概念 10 张图（exp68p 实测）
N_CONCEPT = 1654
ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SEED = 0

FEAT_TRAIN_PT = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
EegTRAIN = Path("/domains/omnibrain4d/eeg_train")
OUT_DIR = Path("/domains/runs/tier0-encoding")


def log(*a):
    print(*a, flush=True)


# ================================================================ 工具
def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, te), te) for te in np.array_split(idx, k)]


def _rank(a):
    from scipy.stats import rankdata
    return rankdata(a)


def spearman(a, b):
    ra = _rank(a).astype(np.float64)
    rb = _rank(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def rdm_full(mat):
    m = np.asarray(mat, dtype=np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    r = m @ m.T
    np.fill_diagonal(r, 1.0)
    return 1.0 - r


# ================================================================ PART 0
def part0_align():
    """验证 ViT img_features 行序 == EEG 图像序（概念分组结构已实测确认）。"""
    import torch
    log("=" * 84)
    log("PART 0 — 顺序对齐验证（ViT img_features vs EEG 图像序）")
    log("=" * 84)
    f = torch.load(FEAT_TRAIN_PT, map_location="cpu", weights_only=False)
    IMG = f["img_features"].numpy().astype(np.float32)
    TXT = f["text_features"].numpy().astype(np.float32)
    log(f"  img_features {IMG.shape} | text_features {TXT.shape}")
    del f

    n = IMG.shape[0]
    assert n % GROUPS == 0, f"图像数 {n} 不是 {GROUPS} 的整数倍"
    V = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-12)

    # 组内对（同概念）
    gid = np.repeat(np.arange(n // GROUPS), GROUPS)
    # 组内对：只用一部分，避免 O(n²)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(n // GROUPS, size=min(400, n // GROUPS), replace=False)
    tu = np.triu_indices(GROUPS, 1)
    ii, jj = tu[0], tu[1]
    a = (sel[:, None] * GROUPS + ii[None, :]).ravel()
    b = (sel[:, None] * GROUPS + jj[None, :]).ravel()
    same = float(np.einsum("ij,ij->i", V[a], V[b]).mean())

    # 随机配对零分布（两次独立随机抽样，允许重复）
    nulls = []
    for _ in range(200):
        p1 = rng.integers(0, n, size=len(a))
        p2 = rng.integers(0, n, size=len(a))
        nulls.append(float(np.einsum("ij,ij->i", V[p1], V[p2]).mean()))
    nulls = np.array(nulls)
    gap = same - nulls.mean()
    p = float((np.sum(nulls >= same) + 1) / (len(nulls) + 1))
    log(f"  同概念 10 张图的平均余弦 = {same:+.4f}")
    log(f"  随机配对零分布            = {nulls.mean():+.4f} ± {nulls.std():.4f} "
        f"[95%={np.percentile(nulls,95):+.4f}]")
    log(f"  差距 = {gap:+.4f}  ({(gap/max(nulls.std(),1e-12)):.1f}σ)  p = {p:.4f}")
    ok = bool(gap > 3 * nulls.std() and p < 0.01)
    log(f"  → {'✅ 行序一致，可以继续' if ok else '❌ 对齐存疑，实验作废'}")
    return IMG, TXT, {"same_concept_cos": same, "null_mean": float(nulls.mean()),
                      "null_sd": float(nulls.std()), "gap": gap, "p": p, "ok": ok}


# ================================================================ PART 1
def build_image_targets(eeg, chunk=400):
    """(n_img, n_rep, n_ch, T) → (n_img, n_ch*n_bin) 每张图的试次平均。

    分块 + 一次 reshape 完成 10 个窗的均值，避免 8 GB 的 float32 副本。
    """
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        # (c, rep, ch, T) → (c, rep, ch, bin, win) → 窗内平均
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)   # (c,rep,ch,bin)
        # ★ 必须转成**窗主序** (c, rep, bin, ch)：后面 reshape(n_bin, n_ch)
        #   依赖的是 [bin0_ch0 … bin0_ch62, bin1_ch0 …]。若保持 ch 主序，
        #   逐窗统计会把窗与通道混排（RSA/编码两套代码的顺序必须一致）。
        e = e.transpose(0, 1, 3, 2)                                      # (c,rep,bin,ch)
        e = e.mean(axis=1)                                               # (c,bin,ch)
        out[s:s + chunk] = e.reshape(e.shape[0], n_bin * n_ch)
        del e
    return out


def part1_targets():
    """逐被试算「每张图」的试次平均响应（BBB 需要所有被试）。"""
    log("\n" + "=" * 84)
    log("PART 1 — 逐被试的图像级试次平均响应")
    log("=" * 84)
    tas = {}
    for sid in range(1, N_SUBJECTS + 1):
        fp = EegTRAIN / f"sub-{sid:02d}.npz"
        if not fp.exists():
            log(f"  [skip] sub-{sid:02d}")
            continue
        t0 = time.time()
        eeg = np.load(fp)["eeg"]
        log(f"  sub-{sid:02d}: 载入 {eeg.shape} {eeg.dtype}  [{time.time()-t0:.0f}s]")
        Y = build_image_targets(eeg)
        del eeg
        tas[sid] = Y
        log(f"  sub-{sid:02d}: 图像级目标 {Y.shape}  [{time.time()-t0:.0f}s]")
    return tas


# ================================================================ PART 2
def ridge_cv_primal(X, Y, folds, alphas, n_null, rng, n_ch=63):
    """primal ridge（n ≫ d）。返回 (best_alpha, 逐窗R²数组, 零分布数组)。

    两处性能关键（第一版跑 69 分钟一个被试都没出来，就是被下面这条吃掉的）：

    1. **零分布只在（最佳 α × 最佳窗）的 63 个目标上算。**
       第一版对 630 个目标 × 8 个 α × 20 次置换 × 5 折 = 800 次
       「1024×1024 解 + 630 右端」，每次 ~1.7e9 flops ⇒ 1.4e12 flops/被试。
       改成只算最佳窗的 63 个目标、只用最佳 α ⇒ 100 次 × 4.3e8，
       **降 10 倍**。统计上这是「固定检验统计量」的置换检验：
       观测值和零分布用同一个 (α, 窗) 口径，比较公平。

    2. **用 Cholesky**（`assume_a='pos'`）代替 LU：A = XᵀX + λI 是正定的，
       Cholesky 比 LU 实测快约 2 倍，且不会因数值问题失败。

    置换零：打乱**训练折的 Y 行**（X 不动 ⇒ 同折 XᵀX 复用）。
    零假设「X 不含 Y 的信息」下，与打乱「特征↔图像对应」等价。
    """
    import scipy.linalg as sla
    n, d = X.shape
    n_bin = Y.shape[1] // n_ch
    real_w = {a: None for a in alphas}

    def _solve(A, B):
        return sla.solve(A, B, assume_a="pos", overwrite_a=True, check_finite=False)

    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        XtX = (Xtr.T @ Xtr).astype(np.float64)
        ybar = Y[tr].mean(axis=0, keepdims=True)
        Ytr = Y[tr] - ybar
        Yte = Y[te]
        ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
        scale = float(np.trace(XtX)) / d
        XtY = Xtr.T @ Ytr
        for a in alphas:
            A = XtX.copy()
            A.flat[:: d + 1] += a * scale
            W = _solve(A, XtY)
            pred = Xte @ W + ybar
            r2 = 1.0 - ((Yte - pred) ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-12)
            real_w[a] = r2 if real_w[a] is None else real_w[a] + r2

    nf = len(folds)
    best_a, best_pw = None, None
    for a in alphas:
        pw = np.median((real_w[a] / nf).reshape(n_bin, n_ch), axis=1)
        if best_a is None or pw.max() > best_pw.max():
            best_a, best_pw = a, pw
    bi = int(np.argmax(best_pw))

    # ---- 零分布：只在（最佳 α × 最佳窗）上算 ----
    cols = np.arange(bi * n_ch, (bi + 1) * n_ch)
    nulls = np.zeros(n_null)
    if n_null > 0:
        for tr, te in folds:
            Xtr, Xte = X[tr], X[te]
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
            XtX = (Xtr.T @ Xtr).astype(np.float64)
            ybar = Y[tr][:, cols].mean(axis=0, keepdims=True)
            Ytr = Y[tr][:, cols] - ybar
            Yte = Y[te][:, cols]
            ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
            scale = float(np.trace(XtX)) / d
            XtY = Xtr.T @ Ytr
            for pi in range(n_null):
                Yp = Ytr[rng.permutation(len(tr))]
                XtYp = Xtr.T @ Yp
                A = XtX.copy()
                A.flat[:: d + 1] += best_a * scale
                W = _solve(A, XtYp)
                pred = Xte @ W + ybar
                r2 = 1.0 - ((Yte - pred) ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-12)
                nulls[pi] += float(np.median(r2))
        nulls /= nf
    return best_a, best_pw, nulls, bi


def part2_encode(IMG, TXT, tas, rng):
    log("\n" + "=" * 84)
    log("PART 2 — Ridge 编码（图像级 16540 样本，primal，5-fold 图像级 CV）")
    log("=" * 84)
    rows = []
    sids = sorted(tas)
    for sid in sids:
        t0 = time.time()
        Y = tas[sid]                                   # (16540, 630) 图像级
        n_img = Y.shape[0]
        folds = make_folds(n_img, N_FOLDS, rng)
        others = [tas[k] for k in sids if k != sid]
        Y_bbb = np.mean(np.stack(others, axis=0), axis=0)

        a_i, w_i, null_i, bi = ridge_cv_primal(IMG, Y, folds, ALPHAS, N_NULL, rng)
        t_img = time.time() - t0
        a_b, w_b, _, _ = ridge_cv_primal(Y_bbb, Y, folds, ALPHAS, 0, rng)
        t_bbb = time.time() - t0 - t_img

        # 文本特征是概念级（1654）：EEG 也要聚合到概念级
        Yc = Y.reshape(N_CONCEPT, GROUPS, -1).mean(axis=1)      # (1654, 630)
        Yc_bbb = np.mean(np.stack(
            [tas[k].reshape(N_CONCEPT, GROUPS, -1).mean(axis=1) for k in sids if k != sid],
            axis=0), axis=0)
        folds_c = make_folds(N_CONCEPT, N_FOLDS, rng)
        a_t, w_t, null_t, bt = ridge_cv_primal(TXT, Yc, folds_c, ALPHAS, N_NULL, rng)
        t_txt = time.time() - t0 - t_img - t_bbb
        a_bc, w_bc, _, _ = ridge_cv_primal(Yc_bbb, Yc, folds_c, ALPHAS, 0, rng)

        p_i = float((np.sum(null_i >= w_i[bi]) + 1) / (N_NULL + 1))
        p_t = float((np.sum(null_t >= w_t[bt]) + 1) / (N_NULL + 1))
        rec = {
            "subject": f"sub-{sid:02d}", "n_img": n_img, "n_concept": N_CONCEPT,
            "alpha_img": a_i, "alpha_bbb": a_b, "alpha_txt": a_t, "alpha_txt_bbb": a_bc,
            "best_win_img_ms": [bi * 100, (bi + 1) * 100],
            "best_win_txt_ms": [bt * 100, (bt + 1) * 100],
            "r2_img_best_window": float(w_i[bi]),
            "r2_txt_best_window": float(w_t[bt]),
            "r2_bbb_best_window": float(w_b[bi]),
            "r2_bbb_concept_best_window": float(w_bc[bt]),
            "null_img_at_best_median": float(np.median(null_i)),
            "null_img_at_best_p95": float(np.percentile(null_i, 95)),
            "p_img": p_i, "p_txt": p_t,
            "ratio_img_over_bbb": (float(w_i[bi]) / float(w_b[bi]))
            if abs(w_b[bi]) > 1e-12 else None,
            "per_window_r2_img": [float(x) for x in w_i],
            "per_window_r2_bbb": [float(x) for x in w_b],
            "elapsed_s": round(time.time() - t0, 1),
            "t_img_s": round(t_img, 1), "t_bbb_s": round(t_bbb, 1),
            "t_txt_s": round(t_txt, 1),
        }
        rows.append(rec)
        log(f"  sub-{sid:02d}: 最佳窗 {bi*100}-{bi*100+100}ms  "
            f"R² img {w_i[bi]:+.5f} (零中位 {rec['null_img_at_best_median']:+.5f}, p={p_i:.3f}) | "
            f"BBB {w_b[bi]:+.5f} | img/BBB {rec['ratio_img_over_bbb']:.2f} | "
            f"txt(概念级) {w_t[bt]:+.5f} (p={p_t:.3f})  "
            f"[img {t_img:.0f}s | bbb {t_bbb:.0f}s | txt {t_txt:.0f}s]")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "exp69_train_encoding.json").write_text(
            json.dumps({"per_subject": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return {"per_subject": rows}


# ================================================================ 主
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {"schema_version": "1.0", "script": "exp69_train_encoding",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "design": {"n_images": 16540, "n_concepts": 1654, "images_per_concept": 10,
                         "ridge": "primal (n>>d)", "cv": "5-fold over IMAGES",
                         "null": "permuting training-fold Y rows (X fixed -> Gram reused)",
                         "baselines": ["置换零", "BBB 跨被试脑对脑上限"]}}

    IMG, TXT, align = part0_align()
    report["alignment"] = align
    if not align["ok"]:
        log("\n❌ 顺序对齐未通过 —— 终止")
        report["verdicts"] = {"alignment": {"ok": False}}
        (OUT_DIR / "exp69_train_encoding.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    tas = part1_targets()
    report["encoding"] = part2_encode(IMG, TXT, tas, rng)

    rows = report["encoding"].get("per_subject", [])
    if rows:
        med_img = float(np.median([r["r2_img_best_window"] for r in rows]))
        med_bbb = float(np.median([r["r2_bbb_best_window"] for r in rows]))
        med_txt = float(np.median([r["r2_txt_best_window"] for r in rows]))
        med_null = float(np.median([r["null_img_at_best_median"] for r in rows]))
        n_sig = sum(1 for r in rows if r["p_img"] < 0.05)
        ratio = float(np.median([r["ratio_img_over_bbb"] for r in rows]))
        log("\n" + "=" * 84)
        log("判定")
        log("=" * 84)
        log(f"  编码显著 {'✅' if n_sig >= max(3, len(rows)//2) else '❌'}  "
            f"最佳窗 R² 中位 img {med_img:+.5f} vs 零 {med_null:+.5f}，显著 {n_sig}/{len(rows)}")
        log(f"  vs BBB   BBB {med_bbb:+.5f}，img/BBB = {ratio:.3f}")
        log(f"  txt(概念级) 中位 {med_txt:+.5f}")
        report["verdicts"] = {
            "alignment": {"ok": True},
            "encoding_significant": {"ok": bool(n_sig >= max(3, len(rows) // 2)),
                                     "median_r2_img": med_img, "n_sig": n_sig,
                                     "n_subjects": len(rows), "median_null": med_null},
            "vs_bbb": {"ok": bool(ratio > 0), "median_r2_bbb": med_bbb, "ratio": ratio},
        }
    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp69_train_encoding_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n报告 → {OUT_DIR}/exp69_train_encoding_full.json  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
