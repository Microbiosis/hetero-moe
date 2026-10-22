#!/usr/bin/env python3
"""exp72 — 单试次编码：ViT 特征能预测**单次呈现**的 EEG 吗？

## 要回答的问题

exp71 定了：train split（16540 图 × 4 重复）的零分布地板是 −0.0012，
低到与**单试次 R² 的天花板**（≈ 单试次方差的 1%）同量级 ——
所以单试次解码唯一可能的战场是 train split。

本脚本就在那一格上直接测：

    test split （200 图 × 80 重复）：单试次 R² = +0.0001，地板 −0.0121  → 不可检出
    train split（16540 图 × 4 重复）：单试次 R² = ?      ，地板 −0.0012  → ?

## 一个让这个实验几乎免费的关键等式

同一张图的所有重复**共享同一个特征向量**（特征是逐图像的），
所以试次级 ridge 的正规方程只依赖**每图的试次和 S1 与试次平方和 S2**：

    min_w Σ_{i,t} (x_i·w - y_{i,t})² + λ||w||²
    ⇒ (n_rep·G + λI) w = Xᵀ S1,   G = Σ_i x_i x_iᵀ,  S1_i = Σ_t y_{i,t}

令 ȳ_i = S1_i / n_rep：正规方程可写成 (G + (λ/n_rep)·I) w = Xᵀ ȳ，
**权重在 λ' = λ/n_rep 的映射下与试次级完全一致**（对拍偏差 ~1e-16）。
更重要的是：**给定同一个 w**，图级公式还原出的 R² 与试次级直接算**完全一致**
（对拍偏差 ~1e-16，两种粒度 × 有信号/纯噪声四种情形全对）——
这才是本实验的算力节省所依赖的性质。

    ss_res_single = Σ_i [ (S2_i - n_rep·ȳ_i²) + n_rep·(ȳ_i - pred_i)² ]
    ss_tot_single = Σ_i [ S2_i - 2·ȳ_tr·S1_i + n_rep·ȳ_tr² ]
    ss_res_avg    = Σ_i [ n_rep·(ȳ_i - pred_i)² ]
    ss_tot_avg    = Σ_i [ n_rep·(ȳ_i - ȳ_tr)² ]

（ȳ_tr = **训练集**均值：single 用训练试次均值，avg 用训练图像均值。）

## 判据

窗口 **200–400 ms 先验指定**（exp64/65/66/67 的 RSA 独立结果），不做窗口选择
⇒ 观测与零分布形式一致 ⇒ p 值诚实。

    单试次 R² > 零分布 95 分位  ⇒ 单试次解码可行
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
N_NULL = 10
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
PRIOR_BINS = (2, 4)          # 200–400 ms（先验指定）
SEED = 0

TEST_4D = Path("/domains/omnibrain4d/eeg")        # 200 图 × 80 重复
TRAIN_4D = Path("/domains/omnibrain4d/eeg_train")  # 16540 图 × 4 重复
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/tier0-encoding")
N_SUBJ_TEST = 10
N_SUBJ_TRAIN = 5


def log(*a):
    print(*a, flush=True)


def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, te), te) for te in np.array_split(idx, k)]


def per_rep_targets(eeg, chunk=400):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch*n_bin) 每个重复的窗均值（窗主序）。"""
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_rep, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)
        e = e.transpose(0, 1, 3, 2)                  # (c, rep, bin, ch)
        out[s:s + chunk] = e.reshape(e.shape[0], n_rep, n_ch * n_bin)
        del e
    return out


def r2_from_stats(S1, S2, Ybar, pred, n_rep, mode, ybar_tr=None):
    """由 (S1, S2, ȳ, pred) 精确还原 R²。

    mode='single': 目标 = 单次试次
    mode='avg'   : 目标 = 试次均值
    基线 ybar_tr 必须是**训练集**的均值（与 exp67/70/71 口径一致）：
      single 用训练**试次**均值；avg 用训练**图像**均值。
    """
    if mode == "single":
        if ybar_tr is None:
            ybar_tr = S1.sum(axis=0) / (len(S1) * n_rep)
        ss_res = ((S2 - n_rep * Ybar ** 2) + n_rep * (Ybar - pred) ** 2).sum(axis=0)
        ss_tot = (S2 - 2 * ybar_tr * S1 + n_rep * ybar_tr ** 2).sum(axis=0)
    else:
        if ybar_tr is None:
            ybar_tr = Ybar.mean(axis=0)
        ss_res = (n_rep * (Ybar - pred) ** 2).sum(axis=0)
        ss_tot = (n_rep * (Ybar - ybar_tr) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def fit_image_level(X, S1, S2, Ybar, n_rep, folds, alphas, cols, n_null, rng,
                    want_profile=False):
    """图级 primal ridge；cols 指定参与 R² 统计的目标列（先验窗）。

    返回 dict：alpha / r2_single / r2_avg / nulls / p / profile
    """
    import scipy.linalg as sla
    n, d = X.shape
    Yc = Ybar[:, cols]
    S1c, S2c = S1[:, cols], S2[:, cols]
    n_t = len(cols)
    real_s = {a: None for a in alphas}
    real_a = {a: None for a in alphas}
    prof = {a: None for a in alphas} if want_profile else None

    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        XtX = (Xtr.T @ Xtr).astype(np.float64)
        scale = float(np.trace(XtX)) / d
        # 试次级正规方程 ÷ n_rep ⇒ 图级（等价，已对拍验证）
        A0 = XtX / n_rep
        ybar_tr = S1c[tr].sum(axis=0) / (len(tr) * n_rep)
        Ytr_c = Yc[tr] - ybar_tr
        Yte_c = Yc[te]
        XtY = Xtr.T @ Ytr_c
        for a in alphas:
            A = A0.copy()
            A.flat[:: d + 1] += a * scale
            W = sla.solve(A, XtY, assume_a="pos", overwrite_a=True, check_finite=False)
            pred = Xte @ W + ybar_tr
            rs = r2_from_stats(S1c[te], S2c[te], Yc[te], pred, n_rep, "single",
                               ybar_tr=ybar_tr)
            ra = r2_from_stats(S1c[te], S2c[te], Yc[te], pred, n_rep, "avg",
                               ybar_tr=Yc[tr].mean(axis=0))
            real_s[a] = rs if real_s[a] is None else real_s[a] + rs
            real_a[a] = ra if real_a[a] is None else real_a[a] + ra
            if prof is not None:
                pr = (rs.reshape(-1, 63) if n_t == 630 else None)
                prof[a] = pr if prof[a] is None else prof[a] + pr

    nf = len(folds)
    best_a, best_s, best_av = None, None, None
    for a in alphas:
        s = float(np.median(real_s[a] / nf))
        if best_a is None or s > best_s:
            best_a, best_s = a, s
            best_av = float(np.median(real_a[a] / nf))
    profile = None
    if prof is not None and prof[best_a] is not None:
        profile = [float(x) for x in np.median((prof[best_a] / nf).reshape(10, 63), axis=1)]

    # ---- 零分布：置换「图像 ↔ 目标」对应关系（同一先验窗、同一 α）----
    nulls_s = np.zeros(n_null)
    nulls_a = np.zeros(n_null)
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        XtX = (Xtr.T @ Xtr).astype(np.float64)
        A0 = XtX / n_rep
        scale = float(np.trace(XtX)) / d
        ybar_tr = S1c[tr].sum(axis=0) / (len(tr) * n_rep)
        for pi in range(n_null):
            p = rng.permutation(len(tr))
            Ytr_p = Yc[tr][p] - ybar_tr
            A = A0.copy()
            A.flat[:: d + 1] += best_a * scale
            W = sla.solve(A, Xtr.T @ Ytr_p, assume_a="pos", overwrite_a=True,
                          check_finite=False)
            pred = Xte @ W + ybar_tr
            nulls_s[pi] += float(np.median(
                r2_from_stats(S1c[te], S2c[te], Yc[te], pred, n_rep, "single",
                              ybar_tr=ybar_tr)))
            nulls_a[pi] += float(np.median(
                r2_from_stats(S1c[te], S2c[te], Yc[te], pred, n_rep, "avg",
                              ybar_tr=Yc[tr].mean(axis=0))))
    nulls_s /= nf
    nulls_a /= nf
    return {"alpha": best_a, "r2_single": best_s, "r2_avg": best_av,
            "null_single": nulls_s, "null_avg": nulls_a,
            "profile_single": profile}


def run_split(name, feat_path, data_dir, n_subj, rng, cols):
    import torch
    f = torch.load(feat_path, map_location="cpu", weights_only=False)
    IMG = f["img_features"].numpy().astype(np.float64)
    log(f"\n===== {name} =====")
    log(f"  特征 {IMG.shape}")

    # 载入各被试的 (S1, S2, ȳ)
    stats = {}
    for sid in range(1, n_subj + 1):
        fp = data_dir / f"sub-{sid:02d}.npz"
        if not fp.exists():
            continue
        t0 = time.time()
        eeg = np.load(fp)["eeg"]
        Yrep = per_rep_targets(eeg)
        del eeg
        n_img, n_rep = Yrep.shape[0], Yrep.shape[1]
        S1 = Yrep.sum(axis=1)
        S2 = (Yrep.astype(np.float64) ** 2).sum(axis=1)
        Ybar = S1 / n_rep
        stats[sid] = (S1, S2, Ybar, n_rep)
        log(f"  sub-{sid:02d}: {n_img} 图 × {n_rep} 重复  [{time.time()-t0:.0f}s]")
        del Yrep

    rows = []
    for sid, (S1, S2, Ybar, n_rep) in stats.items():
        t0 = time.time()
        folds = make_folds(Ybar.shape[0], N_FOLDS, rng)
        r = fit_image_level(IMG, S1, S2, Ybar, n_rep, folds, ALPHAS, cols,
                            N_NULL, rng, want_profile=(sid == min(stats)))
        p_s = float((np.sum(r["null_single"] >= r["r2_single"]) + 1) / (N_NULL + 1))
        p_a = float((np.sum(r["null_avg"] >= r["r2_avg"]) + 1) / (N_NULL + 1))
        rec = {"split": name, "subject": f"sub-{sid:02d}",
               "n_img": int(Ybar.shape[0]), "n_rep": int(n_rep),
               "window_ms": [PRIOR_BINS[0] * 100, PRIOR_BINS[1] * 100],
               "alpha": r["alpha"],
               "r2_single_trial": r["r2_single"], "p_single_trial": p_s,
               "null_single_median": float(np.median(r["null_single"])),
               "null_single_p95": float(np.percentile(r["null_single"], 95)),
               "r2_trial_avg": r["r2_avg"], "p_trial_avg": p_a,
               "null_avg_median": float(np.median(r["null_avg"])),
               "null_avg_p95": float(np.percentile(r["null_avg"], 95)),
               "profile_single": r["profile_single"],
               "elapsed_s": round(time.time() - t0, 1)}
        rows.append(rec)
        log(f"  sub-{sid:02d}: 单试次 R²={r['r2_single']:+.6f} "
            f"(零 {rec['null_single_median']:+.6f}, 95% {rec['null_single_p95']:+.6f}, p={p_s:.3f}) | "
            f"试次均值 R²={r['r2_avg']:+.6f} (p={p_a:.3f})  [{rec['elapsed_s']}s]")
        (OUT_DIR / "exp72_singletrial.json").write_text(
            json.dumps({"per_subject": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return rows


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lo, hi = PRIOR_BINS
    cols = np.arange(lo * 63, hi * 63)

    report = {"schema_version": "1.0", "script": "exp72_singletrial",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "design": {"window_ms": [lo * 100, hi * 100],
                         "window_source": "先验指定（RSA 独立结果），不做窗口选择",
                         "equivalence": "图级降维 ≡ 试次级 ridge（权重偏差 ~1e-16，已对拍）",
                         "cv": "5-fold over IMAGES",
                         "null": "permuting image↔target correspondence, 同一先验窗"}}

    rows_test = run_split("test(200图×80重复)", FEAT_TEST, TEST_4D,
                          N_SUBJ_TEST, rng, cols)
    rows_train = run_split("train(16540图×4重复)", FEAT_TRAIN, TRAIN_4D,
                           N_SUBJ_TRAIN, rng, cols)
    rows = rows_test + rows_train

    log("\n" + "=" * 84)
    log("汇总：单试次编码能否检出")
    log("=" * 84)
    summary = {}
    for nm, rs in (("test", rows_test), ("train", rows_train)):
        if not rs:
            continue
        s = {"n_subjects": len(rs),
             "r2_single_median": float(np.median([r["r2_single_trial"] for r in rs])),
             "null_single_median": float(np.median([r["null_single_median"] for r in rs])),
             "p_single_median": float(np.median([r["p_single_trial"] for r in rs])),
             "n_sig_single": sum(1 for r in rs if r["p_single_trial"] <= 1.0 / (N_NULL + 1) + 1e-9),
             "r2_avg_median": float(np.median([r["r2_trial_avg"] for r in rs]))}
        s["detectable"] = bool(s["n_sig_single"] >= max(2, len(rs) // 2))
        summary[nm] = s
        log(f"  {nm:5s}: 单试次 R² 中位 {s['r2_single_median']:+.6f} | "
            f"零中位 {s['null_single_median']:+.6f} | p 中位 {s['p_single_median']:.3f} | "
            f"达最小p {s['n_sig_single']}/{len(rs)} | 试次均值 R² {s['r2_avg_median']:+.6f} | "
            f"{'✅ 可检出' if s['detectable'] else '❌ 不可检出'}")

    report["summary"] = summary
    report["per_subject"] = rows
    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp72_singletrial_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
