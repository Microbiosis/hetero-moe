#!/usr/bin/env python3
"""exp64 — Tier 0 RSA 诊断：信号存在性 + 同管线噪声上限 + 修正版 RSA + 对齐检验。

为什么需要这个脚本
------------------
2026-10-21 的 `scripts/rsa_tier0.py` 在 sub-01 test.pt 上做了 ViT-H-14 特征 RDM
与 EEG RDM 的 RSA：逐窗 r_img 最高 +0.096（0-100ms），但报出的"分半可靠度"只有
+0.0365，于是结论是「结果不可用，不能据此声称特征与 EEG 不对齐」。

审计源码后发现三处问题（都是实现问题，不是数据问题）：

1. **噪声上限算在了不同的表示上。**
   上限用的是「原始未 z-score 的 63x250 全时段」，r_img 用的是「z-score 后的
   63x25 单窗」。两者维数、标准化方式都不同，**不可比**。上限必须在同一表示、
   同一时间窗上重算。

2. **EEG RDM 未去除跨概念共同 ERP。**
   80 次重复平均后，200 个概念共享几乎相同的 ERP 形状。它主导了
   (ch x t) 展平后的 Pearson 相关，把条件特异成分压扁。

3. **概念顺序与 ViT 特征顺序的对齐从未验证。**
   顺序错了，RSA 恒等于 0，且与"没有信号"不可区分。

本脚本在**同一份原始 test.pt** 上一并回答：

  Q1 原始数据有没有刺激锁定信号？   → 200 类逐窗解码 + 置换零分布
  Q2 那个 0.0365 是不是算错了对象？ → 同表示同窗重算分半上限
  Q3 去共同 ERP 后 RSA 是否显著？   → 置换零分布（同时即对齐检验）
  Q4 时间轴与 ERP 峰值是否自洽？     → times 数组 + ERP 峰值潜伏期

并用自建 4D 数据做**解码管线自检**（应复现已知 ~15.7x 随机），证明本脚本的
解码实现是对的。

判据
----
  自检 : 4D 数据 200 类解码 >= 10x 随机   → 解码实现正确
  Q1   : 原始 test.pt 200 类解码 > 5%（10x 随机） → 数据有刺激锁定信号
  Q3   : |r_obs| > 置换零分布 95 分位          → 有效（且顺序对齐）
  Q2   : 同管线分半上限 > 0.0365               → 原上限算错对象

用法（在 HF Job / Space 内跑）
--------------------------
    python -u exp64_tier0_diag.py
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
FOURD = "/domains/omnibrain4d/eeg/sub-01.npz"
OUT_DIR = Path("/domains/runs/tier0-diag")

N_PERM_RSA = 200
N_PERM_DEC = 2
SEED = 0
WIN = 25          # 25 样本 @250Hz = 100 ms
BASELINE = 12     # 0-50ms = 12.5 样本，用于基线校正（解码用）


# ================================================================ 小工具
def log(*a):
    print(*a, flush=True)


def describe(obj, name="root", depth=0, max_depth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        log(f"{pad}{name}: dict[{len(obj)}] keys={list(obj.keys())[:12]}")
        if depth < max_depth:
            for k, v in list(obj.items())[:12]:
                describe(v, str(k), depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        log(f"{pad}{name}: {type(obj).__name__}[{len(obj)}]")
        if len(obj) and depth < max_depth:
            describe(obj[0], f"{name}[0]", depth + 1, max_depth)
    else:
        sh = getattr(obj, "shape", None)
        dt = getattr(obj, "dtype", None)
        log(f"{pad}{name}: {type(obj).__name__} shape={sh} dtype={dt}")


def _rank(a):
    from scipy.stats import rankdata
    return rankdata(a)


def spearman(a, b):
    """手写 Spearman（等价 scipy.stats.spearmanr，但可复用 rank，避免重复排序开销）。"""
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
    nrm = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / (nrm + 1e-12)
    r = m @ m.T
    np.fill_diagonal(r, 1.0)
    return 1.0 - r


def tri(R):
    """取上三角压缩向量（不含对角）。"""
    iu = np.triu_indices_from(R, k=1)
    return R[iu]


def rsa_with_null(R_eeg, R_ref, n_perm=N_PERM_RSA, rng=None):
    """r_obs + 置换零分布（置换概念标签 = 检验顺序对齐）。"""
    rng = rng or np.random.default_rng(SEED)
    e = tri(R_eeg)
    ref = tri(R_ref)
    r_obs = spearman(e, ref)
    null = np.empty(n_perm, dtype=np.float64)
    n = R_ref.shape[0]
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearman(e, tri(R_ref[np.ix_(p, p)]))
    return r_obs, null


def p_from_null(r_obs, null):
    """双尾 p 值（置换零分布以 0 为中心）。"""
    if len(null) == 0:
        return float("nan")
    return float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (len(null) + 1))


# ================================================================ 解码
def ridge_fit_predict(Xtr, ytr, Xte, n_class, lam=1.0):
    """闭式岭分类器（与 Space worker/train.py 同款）。"""
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


def decode_k(Xf, k, seed, n_rep_use=None, lam=1.0):
    """Xf: (n_img, n_rep, feat) → k 类解码准确率。

    训练用前一半 repetition，测试用后一半（同一概念不跨折）。
    """
    n_img, n_rep, feat = Xf.shape
    if k > n_img:
        return None
    rng = np.random.default_rng(seed)
    n_tr = (n_rep // 2) if n_rep_use is None else min(n_rep_use, n_rep // 2)
    if n_tr < 2:
        return None

    cls = rng.choice(n_img, k, replace=False)
    Xs = Xf[cls].reshape(-1, feat)
    y = np.repeat(np.arange(k), n_rep)
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


# ================================================================ PART 0/1
def part0_probe():
    import torch

    log("=" * 78)
    log("PART 0 — 原始文件结构")
    log("=" * 78)
    info = {}

    log(f"\n[EEG] {EEG_PT}")
    d = torch.load(EEG_PT, map_location="cpu", weights_only=False)
    describe(d, "test.pt", 0, 2)

    keys = list(d.keys()) if isinstance(d, dict) else []
    info["eeg_keys"] = keys
    for k in keys:
        v = d[k]
        sh = tuple(getattr(v, "shape", ())) if hasattr(v, "shape") else None
        info[f"shape_{k}"] = list(sh) if sh else None
        info[f"dtype_{k}"] = str(getattr(v, "dtype", type(v).__name__))
        if hasattr(v, "numel") and v.numel() <= 400:
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            info[f"head_{k}"] = [float(x) for x in np.ravel(arr)[:8]]
            info[f"tail_{k}"] = [float(x) for x in np.ravel(arr)[-8:]]
            if np.issubdtype(arr.dtype, np.floating):
                info[f"range_{k}"] = [float(arr.min()), float(arr.max())]

    eeg = d["eeg"]
    info["eeg_shape"] = list(eeg.shape)
    info["eeg_dtype"] = str(eeg.dtype)

    log(f"\n[特征] {FEAT_PT}")
    f = torch.load(FEAT_PT, map_location="cpu", weights_only=False)
    describe(f, "vith_test.pt", 0, 2)
    if isinstance(f, dict):
        info["feat_keys"] = list(f.keys())
        for k in f.keys():
            v = f[k]
            if hasattr(v, "shape"):
                info[f"feat_shape_{k}"] = list(tuple(v.shape))
                info[f"feat_dtype_{k}"] = str(v.dtype)

    # 时间轴一致性
    log("\n--- 时间轴 ---")
    tkey = None
    for cand in ("times", "time", "t", "timestamps"):
        if isinstance(d, dict) and cand in d:
            tkey = cand
            break
    if tkey:
        t = d[tkey]
        t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        t = np.ravel(t)
        info["times_key"] = tkey
        info["times_len"] = int(t.size)
        info["times_first"] = [float(x) for x in t[:5]]
        info["times_last"] = [float(x) for x in t[-5:]]
        info["times_min"] = float(t.min())
        info["times_max"] = float(t.max())
        log(f"  times: key={tkey} len={t.size} range=[{t.min():.4f}, {t.max():.4f}]s")
        log(f"  first {t[:5]}  last {t[-5:]}")
        if t.size == eeg.shape[-1]:
            dt = float(np.median(np.diff(t)))
            info["times_matches_eeg"] = True
            info["dt_s"] = dt
            log(f"  ✅ times 长度 == eeg 时间维 ({t.size})，dt = {dt*1000:.3f} ms")
        else:
            info["times_matches_eeg"] = False
            info["dt_s_assumed"] = 1.0 / FS
            log(f"  ⚠️  times 长度 {t.size} != eeg 时间维 {eeg.shape[-1]}")
            log(f"     → 时间轴不可直接映射，按 4ms/样本 换算，窗口位置可能有偏移")
    else:
        info["times_key"] = None
        info["times_matches_eeg"] = None
        log("  ⚠️  文件里没有 times 字段，按 4ms/样本 换算")

    return d, f, info


def part1_erp(d):
    """ERP 峰值潜伏期 —— 检验时间轴是否自洽。"""
    log("\n" + "=" * 78)
    log("PART 1 — ERP 峰值潜伏期（时间轴自洽性）")
    log("=" * 78)

    eeg = d["eeg"]
    X = eeg[:, :8].astype(np.float32).mean(axis=(0, 1))   # (63, 250) 前 8 概念
    erp = X - X.mean(axis=1, keepdims=True)
    amp = np.abs(erp)
    ch, t = np.unravel_index(np.argmax(amp), amp.shape)
    # 逐通道时间标准差，找最活跃通道
    sd = X.std(axis=1)
    act_ch = int(np.argmax(sd))
    act_t = int(np.argmax(sd[act_ch]))

    log(f"  全局最大 |ERP| : ch={ch} t={t} ({t*4} ms) 幅值 {amp[ch, t]:.4g}")
    log(f"  最活跃通道     : ch={act_ch} 峰值 t={act_t} ({act_t*4} ms) std={sd[act_ch]:.4g}")
    log(f"  0-50ms 基线 std : {X[:, :BASELINE].std(axis=1).max():.4g} (ch {act_ch})")

    out = {"erp_peak_ch": int(ch), "erp_peak_sample": int(t),
           "erp_peak_ms": int(t * 4), "erp_peak_amp": float(amp[ch, t]),
           "active_channel": act_ch, "active_channel_peak_ms": int(act_t * 4),
           "baseline_std": float(X[:, :BASELINE].std(axis=1).max())}
    return out


# ================================================================ PART 2
def part2_selftest():
    """解码管线自检：在自建 4D 数据上应复现已知 ~15.7x 随机。"""
    log("\n" + "=" * 78)
    log("PART 2 — 解码管线自检（4D 数据，应复现 ~15.7x）")
    log("=" * 78)

    if not Path(FOURD).exists():
        log(f"  ⚠️  {FOURD} 不存在，跳过自检")
        return {"available": False}

    eeg = np.load(FOURD)["eeg"]
    log(f"  4D shape {eeg.shape} {eeg.dtype}")

    # 与 train.py build_variants('full') 完全一致：0-50ms 基线 + 50-500ms 分 10 段
    # 内存：1GB 的 float32 拷贝就地减基线，避免同时存在两份
    X = eeg.astype(np.float32)
    base = X[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
    X -= base
    a, b = int(0.050 * FS), int(0.500 * FS)
    Xw = X[:, :, :, a:b]
    seg = (b - a) // 10
    Xb = Xw[:, :, :, :seg * 10].reshape(X.shape[0], X.shape[1], X.shape[2], 10, seg).mean(axis=4)
    Xf = Xb.reshape(X.shape[0], X.shape[1], X.shape[2] * 10)
    del X, Xw, Xb
    log(f"  特征 {Xf.shape}")

    accs = []
    for t in range(2):
        a_ = decode_k(Xf, 200, seed=SEED + t * 1000 + 200)
        if a_ is not None:
            accs.append(a_)
        log(f"  trial {t}: acc={accs[-1]:.4f} ({accs[-1]*200:.1f}x 随机)")
    del Xf, eeg

    mean = float(np.mean(accs))
    out = {"available": True, "acc_200": mean, "ratio": round(mean * 200, 2),
           "trials": len(accs)}
    log(f"  → 200 类解码 {mean:.4f} = {mean*200:.1f}x 随机")
    log(f"  → 自检 {'✅ 通过' if mean*200 >= 10 else '❌ 未通过'}（判据 ≥10x）")
    out["selftest_pass"] = bool(mean * 200 >= 10)
    return out


# ================================================================ PART 3
def part3_decode_raw(d, rng):
    """Q1：原始 test.pt 的 200 类逐窗解码。"""
    log("\n" + "=" * 78)
    log("PART 3 — 原始 test.pt 逐窗 200 类解码（Q1 信号存在性）")
    log("=" * 78)

    eeg = d["eeg"]
    n_img, n_rep, n_ch, T = eeg.shape
    rows = []

    for wi, a in enumerate(range(0, T, WIN)):
        b = min(a + WIN, T)
        seg = eeg[:, :, :, a:b].astype(np.float32)           # (200, 80, 63, b-a)
        # 逐试次逐通道去 DC（用整段均值，避免窗内基线重叠），就地减以省内存
        seg -= seg.mean(axis=3, keepdims=True)
        Xf = seg.reshape(n_img, n_rep, n_ch * (b - a))
        del seg

        acc = decode_k(Xf, 200, seed=SEED + wi * 1000 + 200)
        del Xf
        ratio = (acc * 200) if acc is not None else float("nan")
        rows.append({"t0_ms": a * 4, "t1_ms": b * 4, "acc": acc, "ratio": ratio})
        log(f"  [{a*4:4d}-{b*4:4d} ms]  200 类 acc={acc:.4f}  ({ratio:6.1f}x 随机)"
            f"  {'✅' if ratio > 10 else ''}")

    # 置换零分布（只在一个代表窗上算，chance 就是 chance）
    log("\n  置换零分布（打乱标签，窗 100-200ms）…")
    a, b = WIN, 2 * WIN
    seg = eeg[:, :, :, a:b].astype(np.float32)
    seg -= seg.mean(axis=3, keepdims=True)
    Xf = seg.reshape(n_img, n_rep, n_ch * (b - a))
    del seg
    nulls = []
    for p in range(N_PERM_DEC):
        Xp = Xf.copy()
        idx = rng.permutation(Xp.shape[0])
        Xp = Xp[idx]
        acc_n = decode_k(Xp, 200, seed=SEED + 9000 + p)
        if acc_n is not None:
            nulls.append(acc_n)
        log(f"    置换 {p}: acc={acc_n:.4f} ({acc_n*200:.1f}x)")
    del Xf

    best = max(rows, key=lambda r: (r["ratio"] if r["ratio"] == r["ratio"] else -1))
    log(f"\n  最佳窗: {best['t0_ms']}-{best['t1_ms']} ms  {best['ratio']:.1f}x 随机")
    log(f"  置换零: {nulls}  (max {max(nulls)*200 if nulls else float('nan'):.1f}x)")
    log(f"  → 判据 >10x 随机 = 有刺激锁定信号："
        f"{'✅ 是' if best['ratio'] > 10 else '❌ 否'}")

    return {"per_window": rows, "null_accs": nulls,
            "best_ratio": best["ratio"], "best_window": [best["t0_ms"], best["t1_ms"]],
            "signal_present": bool(best["ratio"] > 10)}


# ================================================================ PART 4
def build_rep(eeg, variant, half=None, window=None):
    """构造用于 RDM 的 (n_img, n_ch, T) 表示。

    variant='orig'  : 复现 rsa_tier0 —— 试次平均 + 逐通道 z-score
    variant='demean': 先减去跨概念 grand mean（去共同 ERP），再逐通道 z-score
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


def part4_rsa(eeg, f, rng):
    """Q2/Q3/Q4：RSA 复现 + 同管线上限 + 去 ERP + 置换零分布。"""
    log("\n" + "=" * 78)
    log("PART 4 — RSA：复现 / 同管线上限 / 去共同 ERP / 置换零分布")
    log("=" * 78)

    IMG = f["img_features"].numpy().astype(np.float64)
    TXT = f["text_features"].numpy().astype(np.float64)
    n_img, n_rep, n_ch, T = eeg.shape
    R_img = rdm_full(IMG)
    R_txt = rdm_full(TXT)
    log(f"  EEG {eeg.shape} | IMG {IMG.shape} | TXT {TXT.shape}")

    out = {"variants": {}}
    half = n_rep // 2

    for variant in ("orig", "demean"):
        log(f"\n--- variant = {variant} ---")
        X = build_rep(eeg, variant)
        vres = {"windows": []}

        for a in range(0, T, WIN):
            b = min(a + WIN, T)
            seg = X[:, :, a:b].reshape(n_img, -1)
            R_eeg = rdm_full(seg)

            r_img, null_img = rsa_with_null(R_eeg, R_img, N_PERM_RSA, rng)
            r_txt, null_txt = rsa_with_null(R_eeg, R_txt, N_PERM_RSA, rng)

            # 同一表示、同一时间窗上的分半上限
            Xa = build_rep(eeg, variant, half=(0, half), window=(a, b))
            Xb = build_rep(eeg, variant, half=(half, n_rep), window=(a, b))
            R_a = rdm_full(Xa[:, :, a:b].reshape(n_img, -1))
            R_b = rdm_full(Xb[:, :, a:b].reshape(n_img, -1))
            ceil = spearman(tri(R_a), tri(R_b))

            vres["windows"].append({
                "t0_ms": a * 4, "t1_ms": b * 4,
                "r_img": r_img, "p_img": p_from_null(r_img, null_img),
                "r_txt": r_txt, "p_txt": p_from_null(r_txt, null_txt),
                "ceil_same_pipe": ceil,
                "null_img_p95": float(np.percentile(np.abs(null_img), 95)),
                "null_img_std": float(np.std(null_img)),
            })
            w = vres["windows"][-1]
            log(f"  [{a*4:4d}-{b*4:4d}ms] r_img={r_img:+.4f} (p={w['p_img']:.3f}, "
                f"null95={w['null_img_p95']:.4f})  r_txt={r_txt:+.4f} "
                f"(p={w['p_txt']:.3f})  上限(同管线)={ceil:+.4f}")

        # 全时段
        R_all = rdm_full(X.reshape(n_img, -1))
        r_img_all, null_i = rsa_with_null(R_all, R_img, N_PERM_RSA, rng)
        r_txt_all, null_t = rsa_with_null(R_all, R_txt, N_PERM_RSA, rng)
        Xa = build_rep(eeg, variant, half=(0, half))
        Xb = build_rep(eeg, variant, half=(half, n_rep))
        ceil_all = spearman(tri(rdm_full(Xa.reshape(n_img, -1))),
                            tri(rdm_full(Xb.reshape(n_img, -1))))
        vres["overall"] = {"r_img": r_img_all, "p_img": p_from_null(r_img_all, null_i),
                           "r_txt": r_txt_all, "p_txt": p_from_null(r_txt_all, null_t),
                           "ceil_same_pipe": ceil_all}
        log(f"  [全时段 0-1000ms] r_img={r_img_all:+.4f} (p={vres['overall']['p_img']:.3f}) "
            f"r_txt={r_txt_all:+.4f} (p={vres['overall']['p_txt']:.3f}) "
            f"上限(同管线)={ceil_all:+.4f}")

        out["variants"][variant] = vres

    return out


def part5_align(eeg, f, rng):
    """Q4 对齐检验：概念顺序是否与 ViT 特征顺序一致。"""
    log("\n" + "=" * 78)
    log("PART 5 — 顺序对齐检验（demean + 全时段）")
    log("=" * 78)

    IMG = f["img_features"].numpy().astype(np.float64)
    n_img, n_rep, n_ch, T = eeg.shape
    half = n_rep // 2
    X = build_rep(eeg, "demean")
    R_eeg = rdm_full(X.reshape(n_img, -1))
    R_img = rdm_full(IMG)

    r_obs, null = rsa_with_null(R_eeg, R_img, N_PERM_RSA, rng)
    q = [float(np.percentile(null, q)) for q in (2.5, 50, 97.5)]
    log(f"  原始顺序 r_img = {r_obs:+.4f}")
    log(f"  置换零分布: mean={null.mean():+.4f} sd={null.std():.4f} "
        f"[2.5%, 97.5%] = [{q[0]:+.4f}, {q[2]:+.4f}]")
    log(f"  原始顺序在零分布中的百分位: "
        f"{float((null < r_obs).mean()):.3f}")
    p = p_from_null(r_obs, null)
    log(f"  → p = {p:.4f}  {'✅ 顺序对齐且有信号' if p < 0.05 else '❌ 与随机不可区分'}")

    return {"r_obs": r_obs, "p": p, "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "null_q2_5": q[0], "null_q97_5": q[2],
            "percentile": float((null < r_obs).mean()),
            "aligned": bool(p < 0.05)}


# ================================================================ 主
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    report = {"schema_version": "1.0", "script": "exp64_tier0_diag",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    d, f, info = part0_probe()
    report["files"] = info

    report["erp"] = part1_erp(d)
    report["selftest"] = part2_selftest()
    report["decode_raw"] = part3_decode_raw(d, rng)
    report["rsa"] = part4_rsa(d["eeg"], f, rng)
    report["alignment"] = part5_align(d["eeg"], f, rng)

    # ---------------- 判定 ----------------
    log("\n" + "=" * 78)
    log("判定")
    log("=" * 78)
    v = {}
    st = report["selftest"]
    v["selftest"] = {"ok": bool(st.get("selftest_pass")),
                     "ratio": st.get("ratio"),
                     "note": "4D 数据 200 类解码应 ≥10x 随机"}

    dr = report["decode_raw"]
    v["signal_present"] = {"ok": bool(dr["signal_present"]),
                           "best_ratio": dr["best_ratio"],
                           "note": "原始 test.pt 200 类逐窗解码应 >10x 随机"}

    om = report["rsa"]["variants"]["orig"]
    dm = report["rsa"]["variants"]["demean"]
    ceil_o = [w["ceil_same_pipe"] for w in om["windows"]]
    ceil_d = [w["ceil_same_pipe"] for w in dm["windows"]]
    v["ceiling_fixed"] = {
        "ok": bool(max(ceil_o) > 0.0365),
        "max_ceil_orig": max(ceil_o), "max_ceil_demean": max(ceil_d),
        "reported_ceiling": 0.03650795616397926,
        "note": "同表示同窗重算的分半上限，应显著高于原报的 0.0365"}

    best_dm = max(dm["windows"], key=lambda w: abs(w["r_img"]))
    v["rsa_demean"] = {"ok": bool(best_dm["p_img"] < 0.05),
                       "best_window_ms": [best_dm["t0_ms"], best_dm["t1_ms"]],
                       "r_img": best_dm["r_img"], "p": best_dm["p_img"],
                       "note": "去共同 ERP 后最佳窗 r_img 与置换 p"}

    al = report["alignment"]
    v["alignment"] = {"ok": bool(al["aligned"]), "p": al["p"],
                       "note": "概念顺序与 ViT 特征顺序是否对齐"}

    report["verdicts"] = v
    for k, d_ in v.items():
        log(f"  {k:<16} {'✅' if d_.get('ok') else '❌'}  {d_.get('note','')}")
        for kk, vv in d_.items():
            if kk != "ok" and kk != "note":
                log(f"       {kk} = {vv}")

    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp64_tier0_diag.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n报告 → {OUT_DIR}/exp64_tier0_diag.json  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
