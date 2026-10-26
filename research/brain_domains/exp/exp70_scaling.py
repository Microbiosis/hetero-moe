#!/usr/bin/env python3
"""exp70 — 编码强度的二维标度律：图像数 × 重复数。

## 要解决的问题

exp67（test split，200 图 × 80 重复）给出 R² 中位 +0.0196；
exp69（train split，16540 图 × 4 重复）给出 R² ≈ +0.0019。
**两个 split 的重复次数不同，无法直接比较**，所以"更多数据是否提高编码强度"
这个问题至今没有答案。

exp69 提出的解释是：R² 的差距来自**重复次数**（4 vs 80 ⇒ 试次平均的噪声大 20 倍
⇒ 图像特异成分被稀释 20 倍）。验算：0.0091 ÷ 20 = 0.00046 ≈ 实测 0.00030。
**这个解释必须被检验，不能只靠验算。**

## 设计：两条一维曲线，把两个因子分开

全部在 **test split（200 图 × 80 重复）** 上做，子采样出各个点：

  臂 A（图像数效应）：重复数固定 80，图像数 ∈ {10, 20, 40, 80, 160, 200}
  臂 B（重复数效应）：图像数固定 200，重复数 ∈ {2, 4, 8, 16, 40, 80}

臂 B 的 n_reps=4 这一点**直接检验 exp69 的解释**：它应当落在 train split
（16540 图 × 4 重复）实测的 ~0.0019 附近。若是，则"重复次数"才是主导因子，
"图像数"在这个量级上几乎不起作用。

两臂共用同一套管线：试次平均 → 逐通道 z-score → 逐 100ms 窗 → 对偶 ridge
→ 5-fold **图像级** CV → 置换零 → BBB 上限。

## 为什么先算「每重复」的目标

只需要把每个重复各自的窗均值算出来存成 (n_img, n_rep, 630)，
任意「图像子集 × 重复子集」的试次平均就是在这两个轴上取均值 —— 曲线可以
几乎零成本地扫完，不用重复载入 504 MB 的 npz。

## 判据

  臂 A: R²(图像数) 单调上升且在某个 n 后饱和 ⇒ 图像数有用但有收益递减
  臂 B: R²(重复数) 单调上升并趋近饱和 ⇒ 重复数有用（去噪）
  两者比较 + 臂 B 的 4 重复点 vs train split 实测 ⇒ 判定哪个因子主导
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
N_NULL = 15
ALPHAS = [0.03, 0.3, 3.0, 30.0, 300.0, 3000.0]
IMG_COUNTS = [10, 20, 40, 80, 160, 200]
REP_COUNTS = [2, 4, 8, 16, 40, 80]
SEED = 0

FOURD = Path("/domains/omnibrain4d/eeg")
FEAT_PT = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
OUT_DIR = Path("/domains/runs/tier0-encoding")


def log(*a):
    print(*a, flush=True)


# ================================================================ 工具
def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, te), te) for te in np.array_split(idx, k)]


def rdm_full(mat):
    m = np.asarray(mat, dtype=np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    r = m @ m.T
    np.fill_diagonal(r, 1.0)
    return 1.0 - r


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


def rsa_null(R_a, R_b, n_perm, rng):
    a = R_a[np.triu_indices_from(R_a, 1)]
    r_obs = spearman(a, R_b[np.triu_indices_from(R_b, 1)])
    null = np.empty(n_perm)
    n = R_b.shape[0]
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearman(a, R_b[np.ix_(p, p)][np.triu_indices_from(R_b, 1)])
    return r_obs, null


# ================================================================ 管线
def build_rep_targets(eeg):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch*n_bin) 每个重复各自的窗均值。"""
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_rep, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, 100):
        e = eeg[s:s + 100].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        # (c, rep, ch, T) → (c, rep, ch, bin, win) → 窗内平均 → 转窗主序
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)
        e = e.transpose(0, 1, 3, 2)                 # (c, rep, bin, ch)
        out[s:s + 100] = e.reshape(e.shape[0], n_rep, n_ch * n_bin)
        del e
    return out


def ridge_cell(X, Y, rng, n_null=N_NULL, n_ch=63):
    """对偶 ridge，图像级 5-fold CV。返回 (best_alpha, 逐窗R², 零分布@最佳窗, bi)。

    零分布：打乱**训练折的 Y 行**（X 不动 ⇒ 同折 Gram 复用）。
    """
    n, d = X.shape
    n_bin = Y.shape[1] // n_ch
    folds = make_folds(n, N_FOLDS, rng)
    real_w = {a: None for a in ALPHAS}

    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        K = Xtr @ Xtr.T
        Kte = Xte @ Xtr.T
        scale = float((Xtr ** 2).sum()) / d
        ybar = Y[tr].mean(axis=0, keepdims=True)
        Ytr = Y[tr] - ybar
        Yte = Y[te]
        ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
        for a in ALPHAS:
            A = K + (scale * a) * np.eye(len(tr))
            W = np.linalg.solve(A, Ytr)
            pred = Kte @ W + ybar
            r2 = 1.0 - ((Yte - pred) ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-12)
            real_w[a] = r2 if real_w[a] is None else real_w[a] + r2

    nf = len(folds)
    best_a, best_pw = None, None
    for a in ALPHAS:
        pw = np.median((real_w[a] / nf).reshape(n_bin, n_ch), axis=1)
        if best_a is None or pw.max() > best_pw.max():
            best_a, best_pw = a, pw
    bi = int(np.argmax(best_pw))
    obs = float(best_pw.max())

    # ---- 零分布：每个置换也算「逐窗中位 R²」的整条曲线 ----
    # ★ 观测统计量是 max_over_windows，所以零分布**也必须**是 max_over_windows，
    #   否则窗口选择的向上偏置只作用在观测值上 ⇒ p 值乐观。
    #   （一次矩阵解本来就给出全部 630 个目标，取 max 不额外花钱）
    nulls = np.zeros((n_null, n_bin))
    if n_null > 0:
        for tr, te in folds:
            Xtr, Xte = X[tr], X[te]
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
            K = Xtr @ Xtr.T
            Kte = Xte @ Xtr.T
            scale = float((Xtr ** 2).sum()) / d
            ybar = Y[tr].mean(axis=0, keepdims=True)
            Ytr = Y[tr] - ybar
            Yte = Y[te]
            ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
            for pi in range(n_null):
                Yp = Ytr[rng.permutation(len(tr))]
                A = K + (scale * best_a) * np.eye(len(tr))
                W = np.linalg.solve(A, Yp)
                pred = Kte @ W + ybar
                r2 = 1.0 - ((Yte - pred) ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-12)
                nulls[pi] += np.median(r2.reshape(n_bin, n_ch), axis=1)
        nulls /= nf
    return best_a, best_pw, nulls, bi


def run_cell(X, Y, rng, label, n_null=N_NULL):
    """跑一个格子，返回观测值 + 两种零分布对照。

    p_max   : 正确的 p —— 观测是 max_over_windows，零分布也取 max_over_windows
    p_fixed : 旧做法（零分布固定在观测窗上），会偏乐观，仅作对照
    """
    a, pw, nulls, bi = ridge_cell(X, Y, rng, n_null)
    obs = float(pw.max())
    null_max = nulls.max(axis=1) if n_null > 0 else np.array([])
    p_max = (float((np.sum(null_max >= obs) + 1) / (n_null + 1)) if n_null > 0 else None)
    p_fixed = (float((np.sum(nulls[:, bi] >= obs) + 1) / (n_null + 1)) if n_null > 0 else None)
    return {"alpha": a, "r2": obs,
            "best_window_ms": [bi * 100, (bi + 1) * 100],
            "null_max_median": float(np.median(null_max)) if n_null > 0 else None,
            "null_max_p95": float(np.percentile(null_max, 95)) if n_null > 0 else None,
            "null_fixed_median": float(np.median(nulls[:, bi])) if n_null > 0 else None,
            "p": p_max, "p_fixed_window": p_fixed,
            "per_window": [float(x) for x in pw]}


# ================================================================ 主
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    import torch
    f = torch.load(FEAT_PT, map_location="cpu", weights_only=False)
    IMG = f["img_features"].numpy().astype(np.float64)
    TXT = f["text_features"].numpy().astype(np.float64)
    log(f"  ViT img {IMG.shape} txt {TXT.shape}")

    report = {"schema_version": "1.0", "script": "exp70_scaling",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "design": {"arm_A_images": IMG_COUNTS, "arm_B_reps": REP_COUNTS,
                         "fixed_reps_for_A": 80, "fixed_images_for_B": 200,
                         "cv": "5-fold over IMAGES", "ridge": "dual (n<d)",
                         "null": "permuting training-fold Y rows",
                         "note": "嵌套子集：臂 A 的 n=10 是 n=20 的子集，依此类推"}}

    rows = []
    for sid in range(1, N_SUBJECTS + 1):
        fp = FOURD / f"sub-{sid:02d}.npz"
        if not fp.exists():
            continue
        ts = time.time()
        eeg = np.load(fp)["eeg"]
        Yrep = build_rep_targets(eeg)          # (200, 80, 630)
        del eeg
        n_img_total = Yrep.shape[0]

        # 嵌套的图像子集（同一个随机排列的前 n 个 → 嵌套）
        perm = np.random.default_rng(SEED).permutation(n_img_total)

        rec = {"subject": f"sub-{sid:02d}", "arm_A": [], "arm_B": []}
        for n_img in IMG_COUNTS:
            sel = perm[:n_img]
            Y = Yrep[sel].mean(axis=1)          # 试次平均 (n_img, 630)
            c = run_cell(IMG[sel], Y, rng, f"A{n_img}", N_NULL)
            c["n_img"] = n_img
            c["n_reps"] = int(Yrep.shape[1])
            rec["arm_A"].append(c)
            log(f"  sub-{sid:02d} A: n_img={n_img:3d} n_reps=80  R²={c['r2']:+.5f} "
                f"(零max {c['null_max_median']:+.5f}, p={c['p']:.3f}) 窗 {c['best_window_ms']}")

        for n_rep in REP_COUNTS:
            Y = Yrep[:, :n_rep].mean(axis=1)     # (200, 630)
            c = run_cell(IMG, Y, rng, f"B{n_rep}", N_NULL)
            c["n_img"] = int(n_img_total)
            c["n_reps"] = n_rep
            rec["arm_B"].append(c)
            log(f"  sub-{sid:02d} B: n_img=200 n_reps={n_rep:3d}  R²={c['r2']:+.5f} "
                f"(零max {c['null_max_median']:+.5f}, p={c['p']:.3f}) 窗 {c['best_window_ms']}")

        rows.append(rec)
        log(f"  sub-{sid:02d} 完成 [{time.time()-ts:.0f}s]")
        (OUT_DIR / "exp70_scaling.json").write_text(
            json.dumps({"per_subject": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ---------------- 汇总 ----------------
    log("\n" + "=" * 80)
    log("标度律汇总（10 被试中位）")
    log("=" * 80)
    summary = {"arm_A": [], "arm_B": []}
    if rows:
        for arm, key in (("arm_A", "n_img"), ("arm_B", "n_reps")):
            keys = sorted({c[key] for r in rows for c in r[arm]})
            for k in keys:
                vals = [c["r2"] for r in rows for c in r[arm] if c[key] == k]
                nulls = [c["null_max_median"] for r in rows for c in r[arm] if c[key] == k]
                if not vals:
                    continue
                summary[arm].append({
                    key: k,
                    "r2_median": float(np.median(vals)),
                    "r2_iqr": [float(np.percentile(vals, 25)),
                               float(np.percentile(vals, 75))],
                    "null_max_median": float(np.median(nulls)),
                })
                log(f"  {arm} {key}={k:>4}: R² 中位 {np.median(vals):+.5f} "
                    f"(零 {np.median(nulls):+.5f})")
        report["summary"] = summary
    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp70_scaling_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(report.get("summary", {}), indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n报告 → {OUT_DIR}/exp70_scaling_full.json  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

