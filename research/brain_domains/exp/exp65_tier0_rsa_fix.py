#!/usr/bin/env python3
"""exp65 — Tier 0 RSA 修正版：修 exp64 的两个自身 bug + 跨被试复现。

exp64 已定案的结论（不受本次修正影响）
--------------------------------------
* 原始 test.pt 有刺激锁定信号：200 类解码峰值在 100-200ms（10.3x 随机）
* 概念顺序与 ViT-H-14 特征顺序**是对齐的**：置换检验 p=0.005，百分位 1.000
* 去共同 ERP 后 RSA 成立：全时段 r_img=+0.0537，最佳窗 100-200ms r_img=+0.0798
* 原报告的"分半可靠度 0.0365"算在了**不同的表示**上，不是有效基准

exp64 自身的两个 bug（本次修掉）
-------------------------------
1. **解码置换零分布无效**：置换的是数据行而不是标签，而 decode_k 内部按位置
   重建 y，等于把置换撤销 → 置换值与原值完全相同（都 0.0515）。必须置换标签。
2. **分半上限只在头两个窗有效**：build_rep(..., window=(a,b)) 已切过时间轴，
   之后又切了一次 Xa[:, :, a:b] → a>0 时切出空数组 → 上限恒为 0 并触发 NaN。
   只有 0-100ms（a=0）与全时段是对的。

本次新增
--------
6. **跨被试复现**：4D test split（omnibrain4d/eeg/，与原始 test.pt 同源同序）
   10 个被试逐窗 RSA + 置换零分布。若 sub-01 的 4D 结果与原始 test.pt 一致，
   则同时验证了「4D 构建未改变概念顺序」。

判据
----
  解码置换零 : 真实准确率 > 置换 95 分位
  RSA         : |r_obs| > 置换零分布 95 分位
  跨被试     : 10 个被试中显著者占比
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

FS = 250.0
EEG_PT = "/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt"
FEAT_PT = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FOURD_DIR = Path("/domains/omnibrain4d/eeg")
OUT_DIR = Path("/domains/runs/tier0-diag")

N_PERM_RSA = 200
N_PERM_SUBJ = 100     # 跨被试部分的置换次数（100 次 -> p 分辨率 0.01）
N_NULL_DEC = 3
SEED = 0
WIN = 25          # 25 样本 @250Hz = 100 ms
BASELINE = 12     # 0-50ms ≈ 12 样本（解码基线校正用）
N_SUBJECTS = 10


def log(*a):
    print(*a, flush=True)


# ================================================================ 工具
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
    """(n, d) → (n, n) 的 1 - Pearson 距离矩阵。float64。"""
    m = np.asarray(mat, dtype=np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    r = m @ m.T
    np.fill_diagonal(r, 1.0)
    return 1.0 - r


def tri(R):
    iu = np.triu_indices_from(R, k=1)
    return R[iu]


def rsa_with_null(R_eeg, R_ref, n_perm=N_PERM_RSA, rng=None):
    rng = rng or np.random.default_rng(SEED)
    e = tri(R_eeg)
    r_obs = spearman(e, tri(R_ref))
    null = np.empty(n_perm, dtype=np.float64)
    n = R_ref.shape[0]
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearman(e, tri(R_ref[np.ix_(p, p)]))
    return r_obs, null


def p_from_null(r_obs, null):
    if len(null) == 0:
        return float("nan")
    return float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (len(null) + 1))


# ================================================================ 解码
def ridge_fit_predict(Xtr, ytr, Xte, n_class, lam=1.0):
    n, d = Xtr.shape
    Y = np.zeros((n, n_class), dtype=np.float32)
    Y[np.arange(n), ytr] = 1.0
    A = (Xtr.T @ Xtr).astype(np.float64)
    A.flat[:: d + 1] += lam * np.trace(A) / d
    B = (Xtr.T @ Y).astype(np.float64)
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, B, rcond=None)[0]
    return (Xte @ W).argmax(axis=1)


def decode_classes(Xf, cls, n_rep_use=None, lam=1.0, y_override=None):
    """Xf: (n_img, n_rep, feat)；cls: 选中的类别索引；y_override: 外部给的标签。

    训练用前一半 repetition，测试用后一半（同一概念不跨折）。
    """
    n_img, n_rep, feat = Xf.shape
    k = len(cls)
    n_tr = (n_rep // 2) if n_rep_use is None else min(n_rep_use, n_rep // 2)
    if n_tr < 2:
        return None

    Xs = Xf[cls].reshape(-1, feat)
    y = np.repeat(np.arange(k), n_rep)
    if y_override is not None:
        y = np.asarray(y_override)
        if y.shape[0] != Xs.shape[0]:
            return None
    tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_tr) for i in range(k)])
    te = np.concatenate([np.arange(i * n_rep + n_rep - n_tr, (i + 1) * n_rep)
                         for i in range(k)])

    Xtr = Xs[tr].astype(np.float32)
    ytr = y[tr]
    Xte = Xs[te].astype(np.float32)
    yte = y[te]

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    pred = ridge_fit_predict(Xtr, ytr, Xte, k, lam)
    return float((pred == yte).mean())


def decode_k(Xf, k, seed, n_rep_use=None, lam=1.0, n_null=0, rng=None):
    """返回 (准确率, 置换零分布列表)。

    置换零分布：**置换标签**（不是置换数据行）。置换数据行会被 decode 内部
    按位置重建的 y 撤销，得到与原值完全相同的假零分布（exp64 的 bug）。
    """
    n_img = Xf.shape[0]
    if k > n_img:
        return None, []
    rng = rng or np.random.default_rng(seed)
    cls = rng.choice(n_img, k, replace=False)
    y_true = np.repeat(np.arange(k), Xf.shape[1])

    acc = decode_classes(Xf, cls, n_rep_use, lam)
    nulls = []
    for _ in range(n_null):
        y_p = rng.permutation(y_true)
        a = decode_classes(Xf, cls, n_rep_use, lam, y_override=y_p)
        if a is not None:
            nulls.append(a)
    return acc, nulls


# ================================================================ 表示
def build_rep(eeg, variant, half=None, window=None):
    """构造用于 RDM 的 (n_img, n_ch, T') 表示。

    variant='orig'  : 复现 rsa_tier0 —— 试次平均 + 逐通道 z-score
    variant='demean': 先减去跨概念 grand mean（去共同 ERP），再逐通道 z-score

    注意：window 已经在函数内切过，调用方**不要再切一次**（exp64 的 bug）。
    """
    e = eeg if half is None else eeg[:, half[0]:half[1]]
    if window is None:
        X = e.mean(axis=1).astype(np.float64)
    else:
        a, b = window
        X = e[:, :, :, a:b].astype(np.float32).mean(axis=1).astype(np.float64)
    if variant == "demean":
        X = X - X.mean(axis=0, keepdims=True)
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-12
    return (X - mu) / sd


def split_half_ceiling(eeg, variant, window):
    """同一表示、同一时间窗上的分半可靠度（两个半集各自 RDM 再相关）。"""
    half = eeg.shape[1] // 2
    Xa = build_rep(eeg, variant, half=(0, half), window=window)
    Xb = build_rep(eeg, variant, half=(half, eeg.shape[1]), window=window)
    n_img = Xa.shape[0]
    R_a = rdm_full(Xa.reshape(n_img, -1))
    R_b = rdm_full(Xb.reshape(n_img, -1))
    return spearman(tri(R_a), tri(R_b))


# ================================================================ PART 0/1
def part0_probe():
    import torch

    log("=" * 78)
    log("PART 0 — 原始文件结构")
    log("=" * 78)
    info = {}

    d = torch.load(EEG_PT, map_location="cpu", weights_only=False)
    log(f"\n[EEG] {EEG_PT}")
    log(f"  keys = {list(d.keys())}")
    for k in d.keys():
        v = d[k]
        if hasattr(v, "shape"):
            info[f"shape_{k}"] = list(tuple(v.shape))
            info[f"dtype_{k}"] = str(v.dtype)
        else:
            info[f"type_{k}"] = type(v).__name__
    info["eeg_shape"] = list(d["eeg"].shape)
    info["eeg_dtype"] = str(d["eeg"].dtype)

    # 概念标签样本 —— 用于核对顺序
    for k in ("label", "img", "text"):
        if k in d and hasattr(d[k], "shape"):
            v = d[k]
            n = int(np.asarray(v[:3]).size)
            info[f"sample_{k}"] = [str(x) for x in np.ravel(v)[:6]]

    t = d["times"].numpy() if hasattr(d["times"], "numpy") else np.asarray(d["times"])
    t = np.ravel(t)
    info["times_len"] = int(t.size)
    info["times_range_s"] = [float(t.min()), float(t.max())]
    info["times_first5"] = [float(x) for x in t[:5]]
    info["times_last5"] = [float(x) for x in t[-5:]]
    log(f"\n  times: len={t.size} range=[{t.min():.4f}, {t.max():.4f}]s "
        f"first={t[:3]} last={t[-3:]}")
    if t.size != d["eeg"].shape[-1]:
        log(f"  ⚠️  times 长度 {t.size} != eeg 时间维 {d['eeg'].shape[-1]}")
        log(f"     times 覆盖 {t.max()-t.min():.3f}s（含 {abs(t.min())*1000:.0f}ms 基线），"
            f"而 eeg 只有 {d['eeg'].shape[-1]} 样本")
        log(f"     → eeg 必须对应 times 的某个 250 长子区间；用解码的时间剖面反推起点")
    else:
        log("  ✅ times 长度与 eeg 一致")

    f = torch.load(FEAT_PT, map_location="cpu", weights_only=False)
    log(f"\n[特征] {FEAT_PT}  keys={list(f.keys())}")
    for k in f.keys():
        info[f"feat_shape_{k}"] = list(tuple(f[k].shape))
    return d, f, info


def part1_erp(d):
    log("\n" + "=" * 78)
    log("PART 1 — ERP 峰值潜伏期")
    log("=" * 78)
    eeg = d["eeg"]
    X = eeg[:, :8].astype(np.float32).mean(axis=(0, 1))   # (63, 250)
    erp = X - X.mean(axis=1, keepdims=True)
    amp = np.abs(erp)
    ch, t = np.unravel_index(np.argmax(amp), amp.shape)
    sd = X.std(axis=1)
    act_ch = int(np.argmax(sd))
    log(f"  全局最大 |ERP|: ch={ch} 样本 {t} (按 4ms/样本 = {t*4} ms) 幅值 {amp[ch,t]:.4g}")
    log(f"  最活跃通道 ch={act_ch} (std={sd[act_ch]:.4g})")
    out = {"erp_peak_ch": int(ch), "erp_peak_sample": int(t),
           "erp_peak_ms_assuming_0ms_onset": int(t * 4),
           "erp_peak_amp": float(amp[ch, t])}
    return out


# ================================================================ PART 2
def part2_selftest():
    log("\n" + "=" * 78)
    log("PART 2 — 解码管线自检（4D test split，应复现 ~15.7x）")
    log("=" * 78)
    f = FOURD_DIR / "sub-01.npz"
    if not f.exists():
        log(f"  ⚠️  {f} 不存在，跳过")
        return {"available": False}

    eeg = np.load(f)["eeg"]
    X = eeg.astype(np.float32)
    X -= X[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
    a, b = int(0.050 * FS), int(0.500 * FS)
    seg = (b - a) // 10
    # 注意：必须先取 Xw 再切时间维。写成 X[:, :, :, a:b][:seg*10] 时，第二个
    # [:seg*10] 会切在第 0 轴（概念轴）上，取成前 seg*10 个概念 —— 实测报
    # "cannot reshape array of size 62647200"（=110*80*63*113）。
    Xw = X[:, :, :, a:b]
    Xb = Xw[:, :, :, :seg * 10].reshape(X.shape[0], X.shape[1], X.shape[2], 10, seg).mean(axis=4)
    Xf = Xb.reshape(X.shape[0], X.shape[1], X.shape[2] * 10)
    del X, Xb, Xw

    rng = np.random.default_rng(SEED)
    accs = []
    for t_ in range(2):
        a_, _ = decode_k(Xf, 200, seed=SEED + t_ * 1000 + 200)
        accs.append(a_)
        log(f"  trial {t_}: acc={a_:.4f} ({a_*200:.1f}x 随机)")
    del Xf, eeg
    mean = float(np.mean(accs))
    out = {"available": True, "acc_200": mean, "ratio": round(mean * 200, 2),
           "selftest_pass": bool(mean * 200 >= 10)}
    log(f"  → {mean:.4f} = {mean*200:.1f}x 随机  "
        f"{'✅ 自检通过' if out['selftest_pass'] else '❌ 自检未通过'}")
    return out


# ================================================================ PART 3
def part3_decode_raw(d, rng):
    log("\n" + "=" * 78)
    log("PART 3 — 原始 test.pt 逐窗 200 类解码（含标签置换零分布）")
    log("=" * 78)
    eeg = d["eeg"]
    n_img, n_rep, n_ch, T = eeg.shape
    rows = []
    for wi, a in enumerate(range(0, T, WIN)):
        b = min(a + WIN, T)
        tw = time.time()
        seg = eeg[:, :, :, a:b].astype(np.float32)
        seg -= seg.mean(axis=3, keepdims=True)
        Xf = seg.reshape(n_img, n_rep, n_ch * (b - a))
        del seg
        acc, nulls = decode_k(Xf, 200, seed=SEED + wi * 1000 + 200,
                               n_null=N_NULL_DEC, rng=rng)
        del Xf
        ratio = acc * 200
        rows.append({"t0_ms": a * 4, "t1_ms": b * 4, "acc": acc,
                     "ratio": ratio, "null_accs": nulls,
                     "null_ratio_max": (max(nulls) * 200) if nulls else None,
                     "elapsed_s": round(time.time() - tw, 1)})
        ns = f"  置换零 max {max(nulls)*200:.1f}x" if nulls else ""
        log(f"  [{a*4:4d}-{b*4:4d}ms] acc={acc:.4f} ({ratio:5.1f}x 随机){ns} "
            f"[{rows[-1]['elapsed_s']}s]")

    best = max(rows, key=lambda r: r["ratio"])
    all_null = [x for r in rows for x in r["null_accs"]]
    log(f"\n  最佳窗 {best['t0_ms']}-{best['t1_ms']} ms  {best['ratio']:.1f}x 随机")
    if all_null:
        log(f"  全部置换零: max {max(all_null)*200:.2f}x  mean {np.mean(all_null)*200:.2f}x "
            f"→ 真实值远超零分布")
    log(f"  → 判据 >10x 随机：{'✅ 有刺激锁定信号' if best['ratio'] > 10 else '❌ 无'}")
    return {"per_window": rows, "best_ratio": best["ratio"],
            "best_window": [best["t0_ms"], best["t1_ms"]],
            "null_ratio_max": (max(all_null) * 200) if all_null else None,
            "signal_present": bool(best["ratio"] > 10)}


# ================================================================ PART 4
def part4_rsa(eeg, f, rng):
    log("\n" + "=" * 78)
    log("PART 4 — RSA：orig vs demean（置换零分布 + 同管线分半上限）")
    log("=" * 78)
    IMG = f["img_features"].numpy().astype(np.float64)
    TXT = f["text_features"].numpy().astype(np.float64)
    n_img, n_rep, n_ch, T = eeg.shape
    R_img = rdm_full(IMG)
    R_txt = rdm_full(TXT)
    out = {}

    for variant in ("orig", "demean"):
        log(f"\n--- variant = {variant} ---")
        X = build_rep(eeg, variant)
        vres = {"windows": []}
        for a in range(0, T, WIN):
            b = min(a + WIN, T)
            R_eeg = rdm_full(X[:, :, a:b].reshape(n_img, -1))
            r_img, ni = rsa_with_null(R_eeg, R_img, N_PERM_RSA, rng)
            r_txt, nt = rsa_with_null(R_eeg, R_txt, N_PERM_RSA, rng)
            ceil = split_half_ceiling(eeg, variant, (a, b))
            w = {"t0_ms": a * 4, "t1_ms": b * 4,
                 "r_img": r_img, "p_img": p_from_null(r_img, ni),
                 "r_txt": r_txt, "p_txt": p_from_null(r_txt, nt),
                 "ceil_same_pipe": ceil,
                 "null_img_p95": float(np.percentile(np.abs(ni), 95)),
                 "null_img_sd": float(np.std(ni))}
            vres["windows"].append(w)
            log(f"  [{a*4:4d}-{b*4:4d}ms] r_img={r_img:+.4f} (p={w['p_img']:.3f}, "
                f"null95={w['null_img_p95']:.4f})  r_txt={r_txt:+.4f} (p={w['p_txt']:.3f})"
                f"  上限(同管线)={ceil:+.4f}")

        R_all = rdm_full(X.reshape(n_img, -1))
        r_img, ni = rsa_with_null(R_all, R_img, N_PERM_RSA, rng)
        r_txt, nt = rsa_with_null(R_all, R_txt, N_PERM_RSA, rng)
        ceil_all = split_half_ceiling(eeg, variant, None)
        vres["overall"] = {"r_img": r_img, "p_img": p_from_null(r_img, ni),
                           "r_txt": r_txt, "p_txt": p_from_null(r_txt, nt),
                           "ceil_same_pipe": ceil_all,
                           "null_img_p95": float(np.percentile(np.abs(ni), 95))}
        log(f"  [全时段] r_img={r_img:+.4f} (p={vres['overall']['p_img']:.3f}, "
            f"null95={vres['overall']['null_img_p95']:.4f})  "
            f"r_txt={r_txt:+.4f} (p={vres['overall']['p_txt']:.3f})  "
            f"上限(同管线)={ceil_all:+.4f}")
        out[variant] = vres
    return out


# ================================================================ PART 5
def part5_align(eeg, f, rng):
    log("\n" + "=" * 78)
    log("PART 5 — 顺序对齐检验（demean + 全时段）")
    log("=" * 78)
    IMG = f["img_features"].numpy().astype(np.float64)
    n_img = eeg.shape[0]
    X = build_rep(eeg, "demean")
    R_eeg = rdm_full(X.reshape(n_img, -1))
    R_img = rdm_full(IMG)
    r_obs, null = rsa_with_null(R_eeg, R_img, N_PERM_RSA, rng)
    p = p_from_null(r_obs, null)
    log(f"  原始顺序 r_img = {r_obs:+.4f}")
    log(f"  置换零: mean={null.mean():+.4f} sd={null.std():.4f} "
        f"[2.5%,97.5%]=[{np.percentile(null,2.5):+.4f},{np.percentile(null,97.5):+.4f}]")
    log(f"  原始顺序在零分布中的百分位 = {(null < r_obs).mean():.3f}")
    log(f"  → p={p:.4f}  {'✅ 对齐且有信号' if p < 0.05 else '❌ 与随机不可区分'}")
    return {"r_obs": r_obs, "p": p, "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "percentile": float((null < r_obs).mean()), "aligned": bool(p < 0.05)}


# ================================================================ PART 6
def part6_subjects(R_img, R_txt, rng, n_subjects=N_SUBJECTS):
    """4D test split 10 被试的 RSA（demean 变体）。

    4D 与原始 test.pt 同源同序（200 概念 × 80 重复）；若 sub-01 的 4D 结果与
    原始 test.pt 一致，即同时验证了「4D 构建未改变概念顺序」。

    逐被试增量落盘：上一次整份 job 在 45 min 上限被杀，PART 6 根本没跑到，
    前面的结果也一起没了。改成每算完一个被试就写一次盘，超时也能留下已完成部分。
    """
    log("\n" + "=" * 78)
    log(f"PART 6 — 跨被试复现（4D test split，demean），{n_subjects} 个被试")
    log("=" * 78)

    rows = []
    for sid in range(1, n_subjects + 1):
        f = FOURD_DIR / f"sub-{sid:02d}.npz"
        if not f.exists():
            log(f"  [skip] sub-{sid:02d} 无文件")
            continue
        t0 = time.time()
        eeg = np.load(f)["eeg"]
        n_img, n_rep, n_ch, T = eeg.shape
        X = build_rep(eeg, "demean")

        per = []
        for a in range(0, T, WIN):
            b = min(a + WIN, T)
            R_eeg = rdm_full(X[:, :, a:b].reshape(n_img, -1))
            r_img, ni = rsa_with_null(R_eeg, R_img, N_PERM_SUBJ, rng)
            r_txt, nt = rsa_with_null(R_eeg, R_txt, N_PERM_SUBJ, rng)
            per.append({"t0_ms": a * 4, "t1_ms": b * 4, "r_img": r_img,
                        "p_img": p_from_null(r_img, ni), "r_txt": r_txt,
                        "p_txt": p_from_null(r_txt, nt)})
        R_all = rdm_full(X.reshape(n_img, -1))
        r_img, ni = rsa_with_null(R_all, R_img, N_PERM_SUBJ, rng)
        r_txt, nt = rsa_with_null(R_all, R_txt, N_PERM_SUBJ, rng)
        # 复用已载入的 eeg：上一次写成 np.load(f)["eeg"] 又把 467MB npz
        # 完整解压了一遍，每个被试白费一次。
        ceil_all = split_half_ceiling(eeg, "demean", None)

        best = max(per, key=lambda w: abs(w["r_img"]))
        sig = sum(1 for w in per if w["p_img"] < 0.05)
        rows.append({"subject": f"sub-{sid:02d}", "overall_r_img": r_img,
                     "overall_p_img": p_from_null(r_img, ni),
                     "overall_r_txt": r_txt, "overall_p_txt": p_from_null(r_txt, nt),
                     "ceil_same_pipe": ceil_all, "windows": per,
                     "best_window_ms": [best["t0_ms"], best["t1_ms"]],
                     "best_r_img": best["r_img"], "best_p_img": best["p_img"],
                     "n_sig_windows": sig, "elapsed_s": round(time.time() - t0, 1)})
        log(f"  sub-{sid:02d}: 全时段 r_img={r_img:+.4f} (p={rows[-1]['overall_p_img']:.3f}) "
            f"r_txt={r_txt:+.4f} | 最佳窗 {best['t0_ms']}-{best['t1_ms']}ms "
            f"r={best['r_img']:+.4f} | 显著窗 {sig}/10 | 上限 {ceil_all:+.4f} "
            f"[{rows[-1]['elapsed_s']}s]")

        # 增量落盘
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "exp65_cross_subject.json").write_text(json.dumps({
            "per_subject": rows,
            "n_sig_overall": sum(1 for r in rows if r["overall_p_img"] < 0.05),
            "n_subjects": len(rows),
            "median_overall_r_img": (float(np.median([r["overall_r_img"] for r in rows]))
                                     if rows else None),
            "n_perm": N_PERM_SUBJ, "partial": len(rows) < n_subjects,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        del X, eeg

    if rows:
        n_sig = sum(1 for r in rows if r["overall_p_img"] < 0.05)
        meds = float(np.median([r["overall_r_img"] for r in rows]))
        log(f"\n  全时段 r_img 中位 {meds:+.4f} | 显著被试 {n_sig}/{len(rows)}")
    return {"per_subject": rows,
            "n_sig_overall": sum(1 for r in rows if r["overall_p_img"] < 0.05),
            "n_subjects": len(rows),
            "median_overall_r_img": (float(np.median([r["overall_r_img"] for r in rows]))
                                     if rows else None),
            "n_perm": N_PERM_SUBJ}


# ================================================================ 主
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    report = {"schema_version": "1.0", "script": "exp65_tier0_rsa_fix",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "fixes_vs_exp64": [
                  "解码置换零分布改为置换标签（原版置换数据行，被内部重建的 y 撤销）",
                  "分半上限去掉重复切片（原来 Xa[:, :, a:b] 在 a>0 时切出空数组）",
              ]}

    # ---------------- 分段执行（避免单点超时拖垮整份结果）----------------
    # 用法:
    #   python exp65_tier0_rsa_fix.py                 # PART 0-5
    #   python exp65_tier0_rsa_fix.py --only-subjects  # 只跑 PART 6
    #   python exp65_tier0_rsa_fix.py --subjects 5    # PART 6 只跑 5 个被试
    argv = sys.argv[1:]
    only_subjects = "--only-subjects" in argv
    n_subj = N_SUBJECTS
    for a in argv:
        if a.startswith("--subjects="):
            n_subj = int(a.split("=", 1)[1])

    if only_subjects:
        import torch
        ff = torch.load(FEAT_PT, map_location="cpu", weights_only=False)
        IMG = ff["img_features"].numpy().astype(np.float64)
        TXT = ff["text_features"].numpy().astype(np.float64)
        report["subjects"] = part6_subjects(rdm_full(IMG), rdm_full(TXT), rng, n_subj)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "exp65_subjects_only.json").write_text(
            json.dumps(report["subjects"], indent=2, ensure_ascii=False), encoding="utf-8")
        print("=== SUBJECTS JSON BEGIN ===")
        print(json.dumps(report["subjects"], indent=2, ensure_ascii=False))
        print("=== SUBJECTS JSON END ===")
        log(f"\n跨被试部分完成，耗时 {time.time()-t0:.0f}s")
        return

    d, f, info = part0_probe()
    report["files"] = info
    report["erp"] = part1_erp(d)
    report["selftest"] = part2_selftest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _flush():
        (OUT_DIR / "exp65_tier0_rsa_fix.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    report["decode_raw"] = part3_decode_raw(d, rng)
    _flush()
    report["rsa"] = part4_rsa(d["eeg"], f, rng)
    _flush()
    report["alignment"] = part5_align(d["eeg"], f, rng)
    _flush()
    report["subjects"] = {"note": "用 --only-subjects 单独跑（见 exp65_cross_subject.json）",
                          "n_subjects": 0, "n_sig_overall": 0,
                          "median_overall_r_img": None}

    # ---------------- 判定 ----------------
    log("\n" + "=" * 78)
    log("判定")
    log("=" * 78)
    v = {}
    st = report["selftest"]
    v["selftest"] = {"ok": bool(st.get("selftest_pass")), "ratio": st.get("ratio"),
                     "note": "4D test split 200 类解码 ≥10x 随机 → 解码实现正确"}
    dr = report["decode_raw"]
    v["signal_present"] = {"ok": bool(dr["signal_present"]),
                           "best_ratio": dr["best_ratio"],
                           "null_ratio_max": dr["null_ratio_max"],
                           "note": "原始 test.pt 逐窗解码 >10x 随机，且远超置换零"}
    om, dm = report["rsa"]["orig"], report["rsa"]["demean"]
    v["ceiling_fixed"] = {
        "ok": bool(max(w["ceil_same_pipe"] for w in dm["windows"]) > 0.0365),
        "max_ceil_demean": max(w["ceil_same_pipe"] for w in dm["windows"]),
        "max_ceil_orig": max(w["ceil_same_pipe"] for w in om["windows"]),
        "reported_ceiling": 0.03650795616397926,
        "note": "同表示同窗重算的分半上限（原报 0.0365 算在别的表示上）"}
    best_dm = max(dm["windows"], key=lambda w: abs(w["r_img"]))
    v["rsa_demean"] = {"ok": bool(best_dm["p_img"] < 0.05),
                       "best_window_ms": [best_dm["t0_ms"], best_dm["t1_ms"]],
                       "r_img": best_dm["r_img"], "p": best_dm["p_img"],
                       "null_p95": best_dm["null_img_p95"],
                       "note": "去共同 ERP 后最佳窗 r_img 显著高于置换零"}
    v["null_shrink"] = {
        "ok": bool(om["windows"][0]["null_img_p95"] > 5 * dm["windows"][0]["null_img_p95"]),
        "null95_orig": om["windows"][0]["null_img_p95"],
        "null95_demean": dm["windows"][0]["null_img_p95"],
        "note": "去共同 ERP 后置换零分布收缩的倍数（共同 ERP 污染量的直接度量）"}
    al = report["alignment"]
    v["alignment"] = {"ok": bool(al["aligned"]), "p": al["p"],
                       "note": "概念顺序与 ViT 特征顺序对齐"}
    sj = report["subjects"]
    v["cross_subject"] = {"ok": bool(sj["n_sig_overall"] >= max(3, sj["n_subjects"] // 2)),
                          "n_sig": sj["n_sig_overall"], "n": sj["n_subjects"],
                          "median_r_img": sj["median_overall_r_img"],
                          "note": "跨被试显著性"}
    report["verdicts"] = v
    for k, dd in v.items():
        log(f"  {k:<16} {'✅' if dd.get('ok') else '❌'}  {dd.get('note','')}")
        for kk, vv in dd.items():
            if kk not in ("ok", "note"):
                log(f"       {kk} = {vv}")

    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp65_tier0_rsa_fix.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 同时打到 stdout —— 防止 Job 写桶时 Xet 通道卡住导致结果丢失
    print("=== JSON BEGIN ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n报告 → {OUT_DIR}/exp65_tier0_rsa_fix.json  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
