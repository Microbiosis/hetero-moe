#!/usr/bin/env python3
"""exp93a_level_audit.py — 「档位」归因扫描（**零训练**）。

## 为什么要做

交叉验证（exp92）得到 **A18**：我们自训 tokenizer 的 RSA 比手工特征更高，
但 200-way top-1 **比手工基线还差**（S1a 离散 0.0053 ≈ 随机 vs 手工 0.0082）。
⇒ 「档位不足」这个结论目前**只在一个很窄的读出上成立**
（630-d 逐通道幅度 + 闭式 ridge + 1000 张训练图 + 单试次）。

而 NICE-EEG（arXiv:2308.13234, ICLR 2024）在**同一份 THINGS-EEG** 上报
**top-1 15.6% / top-5 42.8%**，方法是**图像编码器 + EEG 编码器对比学习**（+空间注意力）。

⇒ 在动手改训练目标之前，必须先把「1.1%」这个数字**拆开**：
它到底是 (a) 协议造成的、(b) 特征造成的、还是 (c) 模型/解码器造成的？
否则会把一个协议 bug 当成「模型能力不足」，去训一个根本不需要训的模型。

## 协议分类（本脚本的核心贡献：把三个协议分清）

| 协议 | 训练用的图 | 检索候选 | 难度 | 我们见过吗 |
|---|---|---|---|---|
| **P1 跨集检索** | train split 的 16540 张 | **测试 split 的 200 张（没见过）** | 高 | **A13 的 1.1% 就在这里**；NICE-EEG 的 15.6% 也是这里 |
| **P2 同集检索** | train split 的图 | **同一批 train 图**（重复外推） | 中 | exp73 的自检思路 |
| **P3 同集身份解码** | 测试 split 前 40 重复 | 同一批 200 张 | 低 | **exp64 / Space worker/train.py 的 0.0785（15.7× 随机）就在这里** |

⚠️ **P3 的 0.0785 不是「跨集检索」的数字**，它与 NICE-EEG 的 15.6% **不可比**。
本脚本把 P3 作为「同协议内的上限参照」单独报告，并**复现它**（自检 S2），
以此证明：差距不是「我们的解码实现坏了」。

## 因子表（在 P1 上一因子一因子地改）

| # | 因子 | 取值 |
|---|---|---|
| F1 | 训练图数 | 1000 / 4000 / 16540 |
| F2 | 解码器 | ridge（现状）/ RRR(k) / CCA(k) / ridge→单位化目标 |
| F3 | 试次聚合 | 单试次 / 试次平均 |
| F4 | 图像特征空间 | ViT-H-14 1024 原样 / PCA-128（训练集上拟合） |
| F5 | 协议 | P1 跨集 / P2 同集 / P3 同集身份解码 |

## 预先写死的自检（不通过就不看结论）

- **S1**：P2 同集检索（训练用 rep{0,1}、检索用 rep{2,3}）必须**显著高于随机**
  ⇒ 证明「EEG→图像」的线性映射本身是有信号的；
- **S2**：P3 同集身份解码必须**复现 exp64 的 ~0.0785（15.7× 随机）**
  ⇒ 证明数据口径与解码实现与历史结果一致（不是「跑歪了」）；
- **S3**：P1 跨集检索必须**显著高于随机 0.005**（否则 A13 的 1.1% 有问题）；
- **S4**：合成数据上「有信号 ⇒ 各解码器都显著高于随机；纯噪声 ⇒ 都回到随机」
  ⇒ 证明各解码器的实现本身没写反。

## 用法

    # 自测（本地合成数据）
    python -u research/brain_domains/exp/exp93a_level_audit.py --self-test

    # 云端
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93a_level_audit.py --subjects 1,2,3,4,5,6,7,8,9,10 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d --name t93a-audit
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
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp73_decode200 as E73  # noqa: E402
import exp89_posaxis as E89      # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93a")

SEED = 0
CHANCE = 1.0 / 200.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 特征
# ---------------------------------------------------------------------------

def feats_train(subject, n_images=None):
    """train split → (n_img, 4, 630) float32，**外加** `img` 路径列表与 `label`。

    ⚠️ 实测 `img` 是**每图 4 条相同路径**（不是索引），形如
    'train_images/00001_aardvark/aardvark_01b.jpg'；
    `label` 是概念号 0..1653。两者合起来能**证明** EEG 行序。
    """
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = obj["eeg"]
    img_paths, labels = None, None
    if isinstance(obj, dict) and "img" in obj:
        cand = np.asarray(obj["img"])
        if cand.ndim == 2 and cand.shape[1] == eeg.shape[1]:
            img_paths = [str(x) for x in cand[:, 0]]
        elif cand.ndim == 1 and len(cand) == len(eeg):
            img_paths = [str(x) for x in cand]
    if isinstance(obj, dict) and "label" in obj:
        lab = np.asarray(obj["label"])
        if lab.ndim == 2 and lab.shape[1] == eeg.shape[1]:
            labels = lab[:, 0].astype(np.int64)
        elif lab.ndim == 1 and len(lab) == len(eeg):
            labels = lab.astype(np.int64)
    if n_images is not None:
        eeg = eeg[:n_images]
        if img_paths is not None:
            img_paths = img_paths[:n_images]
        if labels is not None:
            labels = labels[:n_images]
    eeg = np.asarray(eeg, dtype=np.float16)
    X = E73.per_rep_targets(eeg)
    del obj, eeg
    return X, img_paths, labels


def concept_of_path(p):
    """'train_images/00001_aardvark/aardvark_01b.jpg' → 0（概念号减一）。"""
    try:
        return int(p.split("/")[1][:5]) - 1
    except Exception:
        return -1


def verify_alignment(X, img_paths, labels, IMG_train, rng, n_perm, n_rsa=1500):
    """**三段式对齐证明**（缺一不可，任何一段不过就不产出结论）。

    ① EEG 行 → 路径 → 概念号；概念号必须**逐行等于** `label`，
       且行序必须是「概念升序 × 每概念 10 张」⇒ EEG 行序确定。
    ② 图像特征文件的行序：在特征空间里，每行的最近邻必须落在**自己的 10 块**内
       （同概念 ⇒ CLIP 相似度高）。若顺序被打乱，命中率会塌到 ~0。
    ③ EEG 与图像特征之间的 RSA 必须显著 ⇒ 配对不是随机的。

    ⚠️ **③ 必须子采样**：RSA 的 RDM 是 n×n，n=16540 ⇒ 2.2 GB 矩阵 × 100 次置换
       = 2.7e12 flops ⇒ 永远跑不完（实测 exp93a 第一个 Job 卡死在这里）。
       ⇒ 只取前 `n_rsa` 张图算 RSA（置换检验的统计效力不受影响）。
    """
    out = {}
    if img_paths is not None and labels is not None:
        con = np.array([concept_of_path(p) for p in img_paths])
        out["label_match_frac"] = float((con == labels).mean())
        out["concept_block_order_ok"] = bool(
            np.array_equal(con, np.arange(len(con)) // 10))
    F = IMG_train / np.maximum(np.linalg.norm(IMG_train, axis=1, keepdims=True), 1e-12)
    rr = np.random.default_rng(7)
    rows = rr.choice(len(F), size=min(2000, len(F)), replace=False)
    sim = F[rows] @ F.T
    nn = sim.argmax(axis=1)
    out["nn_same_block_frac"] = float((nn // 10 == rows // 10).mean())
    m = min(n_rsa, len(X), len(IMG_train))
    r, p = E89.rsa(X[:m], IMG_train[:m], n_perm, rng)
    out["rsa_r"], out["rsa_p"] = float(r), float(p)
    out["rsa_n_images"] = int(m)
    return out


def feats_test(subject):
    """test split → (200, 80, 630) float32。"""
    fp = TEST_4D / f"sub-{subject:02d}.npz"
    eeg = np.load(fp)["eeg"]
    return E73.per_rep_targets(eeg)


def zscore_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(axis=0), flat.std(axis=0) + 1e-6


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# 解码器（全部闭式、零训练）
# ---------------------------------------------------------------------------

def _ridge_solve(A0, B, lam, scale):
    d = A0.shape[0]
    A = A0.copy()
    A.flat[:: d + 1] += lam * scale
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


def fit_ridge(Xrep, V, lam):
    """试次级 Gram 口径的 ridge（与 exp73 逐字一致：A0 = G_full / n_rep）。

    Xrep : (n_img, n_rep, d) 已 z-score
    V    : (n_img, m_out) 目标图像特征
    返回 (W, scale, Xbar, A0)
    """
    n_img, n_rep, d = Xrep.shape
    Xbar = Xrep.mean(axis=1, dtype=np.float64)
    X64 = Xrep.reshape(-1, d).astype(np.float64)
    A0 = (X64.T @ X64) / n_rep
    scale = float(np.trace(A0)) / d
    v_mean = V.mean(axis=0)
    Vc = V - v_mean
    W = _ridge_solve(A0, Xbar.T @ Vc, lam, scale)
    return W, scale, Xbar, A0, v_mean


def decode_cos(W, Xte, Vtest_c, n_rep_te):
    """单试次 / 试次平均两种口径的 top-1、top-5（余弦）。"""
    n_te, d = Xte.shape[0], Xte.shape[-1]
    flat = Xte.reshape(-1, d)
    pred = flat @ W
    pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    cn = Vtest_c / np.maximum(np.linalg.norm(Vtest_c, axis=1, keepdims=True), 1e-12)
    cos = pn @ cn.T
    true = np.repeat(np.arange(n_te), n_rep_te)
    out = {}
    out["acc1_cos"] = float((cos.argmax(axis=1) == true).mean())
    top5 = np.argsort(-cos, axis=1)[:, :5]
    out["acc5_cos"] = float((top5 == true[:, None]).any(axis=1).mean())
    avg = pred.reshape(n_te, n_rep_te, -1).mean(axis=1)
    an = avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)
    out["acc1_featavg_cos"] = float(((an @ cn.T).argmax(axis=1) == np.arange(n_te)).mean())
    return out


def rrr_project(W, Xbar, k):
    """缩减秩回归：把 ridge 解投影到拟合值的前 k 个右奇异向量上。"""
    P = Xbar @ W
    try:
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
    except np.linalg.LinAlgError:
        return W
    Vk = Vt[:k].T
    return W @ Vk @ Vk.T


def cca_fit(Xbar, V, k, reg=1e-3):
    """正则化 CCA：返回 (wx, wy, mu_x, mu_v, sxx_stats, syy_stats)。

    wx : (d, k)   EEG 侧典型变量
    wy : (m, k)   图像侧典型变量
    """
    mu_x = Xbar.mean(axis=0)
    mu_v = V.mean(axis=0)
    Xc = Xbar - mu_x
    Vc = V - mu_v
    m = Xc.shape[0]
    Sxx = (Xc.T @ Xc) / m
    Syy = (Vc.T @ Vc) / m
    Sxy = (Xc.T @ Vc) / m
    d = Sxx.shape[0]
    Sxx = Sxx + reg * (np.trace(Sxx) / d) * np.eye(d)
    Syy = Syy + reg * (np.trace(Syy) / Syy.shape[0]) * np.eye(Syy.shape[0])
    #  whitening
    ex, Ux = np.linalg.eigh(Sxx)
    ey, Uy = np.linalg.eigh(Syy)
    ex = np.maximum(ex, 1e-12)
    ey = np.maximum(ey, 1e-12)
    Ax = Ux @ np.diag(ex ** -0.5) @ Ux.T
    Ay = Uy @ np.diag(ey ** -0.5) @ Uy.T
    M = Ax @ Sxy @ Ay
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    kk = min(k, U.shape[1])
    wx = Ax @ U[:, :kk]
    wy = Ay @ Vt[:kk].T
    return wx, wy, mu_x, mu_v


def cca_decode(wx, wy, mu_x, mu_v, Xte, Vtest_c):
    """用典型变量分数做相关检索。"""
    Xc = Xte.reshape(-1, Xte.shape[-1]).astype(np.float64) - mu_x
    sx = Xc @ wx
    sy = (Vtest_c - mu_v) @ wy
    # 逐维标准化（用候选集自己的标准差即可，检索是相对的）
    sx = (sx - sx.mean(axis=0, keepdims=True)) / np.maximum(sx.std(axis=0, keepdims=True), 1e-12)
    sy = (sy - sy.mean(axis=0, keepdims=True)) / np.maximum(sy.std(axis=0, keepdims=True), 1e-12)
    n_te = Vtest_c.shape[0]
    n_rep_te = Xte.shape[1]
    a = sx / np.maximum(np.linalg.norm(sx, axis=1, keepdims=True), 1e-12)
    b = sy / np.maximum(np.linalg.norm(sy, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    true = np.repeat(np.arange(n_te), n_rep_te)
    return {"acc1_cos": float((sim.argmax(axis=1) == true).mean()),
            "acc5_cos": float((np.argsort(-sim, axis=1)[:, :5] == true[:, None]).any(axis=1).mean())}


def same_set_identity_decode(Xte, lam=1.0, seed=0):
    """**P3 同集身份解码**（exp64 / Space worker/train.py 的口径）。

    用前一半重复训练 k 类闭式岭分类器，后一半重复测试；图像集合相同。
    返回 (acc, ratio_to_chance)。
    """
    n_img, n_rep, d = Xte.shape
    rng = np.random.default_rng(seed)
    k = n_img
    n_tr = n_rep // 2
    Xs = Xte.reshape(-1, d).astype(np.float32)
    y = np.repeat(np.arange(k), n_rep)
    tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_tr) for i in range(k)])
    te = np.concatenate([np.arange(i * n_rep + n_rep - n_tr, (i + 1) * n_rep)
                         for i in range(k)])
    Xtr, ytr, Xte_, yte = Xs[tr], y[tr], Xs[te], y[te]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte_ = (Xte_ - mu) / sd
    Y = np.zeros((len(ytr), k), dtype=np.float32)
    Y[np.arange(len(ytr)), ytr] = 1.0
    A = (Xtr.T @ Xtr).astype(np.float64)
    A.flat[:: d + 1] += lam * np.trace(A) / d
    W = np.linalg.solve(A, (Xtr.T @ Y).astype(np.float64))
    pred = (Xte_ @ W).argmax(axis=1)
    acc = float((pred == yte).mean())
    return acc, acc / (1.0 / k)


def same_set_retrieval(Xrep, V, lam, n_images=2000, seed=0):
    """**P2 同集检索**：训练用 rep{0,1}，用 rep{2,3} 在同一批训练图里检索。"""
    n_img, n_rep, d = Xrep.shape
    rng = np.random.default_rng(seed)
    half = n_img // 2
    fit_img = np.arange(half)
    cand_img = np.arange(half, n_img)
    if len(cand_img) > n_images:
        cand_img = rng.choice(cand_img, size=n_images, replace=False)
        cand_img.sort()
    Xf = Xrep[fit_img, 0:2]
    Xc = Xrep[cand_img, 2:4]
    W, scale, Xbar, A0, v_mean = fit_ridge(Xf, V[fit_img], lam)
    Vc = V[cand_img] - v_mean
    return decode_cos(W, Xc, Vc, 2)


# ---------------------------------------------------------------------------
# 单被试
# ---------------------------------------------------------------------------

def run_subject(subject, IMG_train, IMG_test, args, rng):
    t0 = time.time()
    rec = {"subject": f"sub-{subject:02d}"}
    Xtr_all, img_paths, labels = feats_train(subject)
    Xte = feats_test(subject)
    mu, sd = zscore_fit(Xtr_all)
    Xtr_all = zscore_apply(Xtr_all, mu, sd)
    Xte = zscore_apply(Xte, mu, sd)
    n_img, n_rep, d = Xtr_all.shape
    rec["n_train_images"] = int(n_img)
    rec["n_test_images"] = int(Xte.shape[0])
    rec["n_feat"] = int(d)

    # ---- 对齐预检（三段式，见 verify_alignment；只跑一次，但**必须跑**）----
    if subject == args.align_check_subject:
        al = verify_alignment(Xtr_all, img_paths, labels, IMG_train,
                              rng, args.align_check_perm, args.align_check_images)
        rec["align_check"] = al
        log(f"  [对齐预检] ①路径概念号==label：{al.get('label_match_frac')} "
            f"｜行序=概念升序×10：{al.get('concept_block_order_ok')}"
            f"｜②特征行同块命中率：{al.get('nn_same_block_frac', 0):.4f}"
            f"｜③RSA={al.get('rsa_r', 0):+.4f}（p={al.get('rsa_p', 1):.4f}）")
        if not (al.get("nn_same_block_frac", 0) > 0.5
                and al.get("rsa_r", 0) > 0 and al.get("rsa_p", 1) < 0.05):
            raise RuntimeError(
                f"对齐预检失败：{al} ⇒ 图像特征与 EEG 不是一一对应 ⇒ 本被试结果作废")
    IMG_train = IMG_train[:len(Xtr_all)]

    # ---- 基准：exp73 的完整流程（含 α 的 5-fold CV 与置换零分布）----
    #      这一格必须复现 A13 的 1.1%，否则说明本脚本与 A13 不同口径。
    base = E73.fit_and_decode(Xtr_all, Xte, IMG_train, IMG_test, None,
                              E73.ALPHAS, rng, n_null=args.n_null)
    rec["baseline_ridge"] = {
        "acc1_cos": base["acc1_cos"], "acc5_cos": base["acc5_cos"],
        "acc1_featavg_cos": base["acc1_featavg_cos"],
        "alpha": base["alpha"],
        "null_acc1_cos_median": float(np.median(base["null_acc1_cos"])),
        "ratio_to_chance": base["acc1_cos"] / CHANCE,
    }
    log(f"  [基准 P1/ridge/16540] acc1={base['acc1_cos']:.4f} "
        f"({base['acc1_cos']/CHANCE:.1f}×) α={base['alpha']:g} "
        f"零中位={np.median(base['null_acc1_cos']):.4f}")

    lam = float(base["alpha"])

    # ---- F1 训练图数 ----
    f1 = {}
    for n in args.train_sizes:
        if n >= n_img:
            continue
        r = E73.fit_and_decode(Xtr_all[:n], Xte, IMG_train[:n], IMG_test, None,
                                E73.ALPHAS, rng, n_null=0)
        f1[str(n)] = {"acc1_cos": r["acc1_cos"], "alpha": r["alpha"],
                      "ratio_to_chance": r["acc1_cos"] / CHANCE}
        log(f"  [F1 n_train={n}] acc1={r['acc1_cos']:.4f} "
            f"({r['acc1_cos']/CHANCE:.1f}×)")
    rec["F1_train_size"] = f1

    # ---- F2 解码器（在全量训练图上）----
    W, scale, Xbar, A0, v_mean = fit_ridge(Xtr_all, IMG_train, lam)
    Vtest_c = IMG_test - v_mean
    f2 = {}
    base_dec = decode_cos(W, Xte, Vtest_c, Xte.shape[1])
    f2["ridge"] = base_dec
    for k in args.rrr_ks:
        Wr = rrr_project(W, Xbar, k)
        f2[f"rrr_k{k}"] = decode_cos(Wr, Xte, Vtest_c, Xte.shape[1])
    for k in args.cca_ks:
        wx, wy, mux, muv = cca_fit(Xbar, IMG_train, k)
        f2[f"cca_k{k}"] = cca_decode(wx, wy, mux, muv, Xte, Vtest_c)
    # 单位化目标：把目标归一化后再回归 ⇒ 与余弦匹配同口径
    Vn = IMG_train / np.maximum(np.linalg.norm(IMG_train, axis=1, keepdims=True), 1e-12)
    Wn, _, _, _, vmn = fit_ridge(Xtr_all, Vn, lam)
    f2["ridge_unit_target"] = decode_cos(Wn, Xte, IMG_test - vmn, Xte.shape[1])
    rec["F2_decoder"] = f2
    for k, v in f2.items():
        log(f"  [F2 {k:<20}] acc1={v['acc1_cos']:.4f} ({v['acc1_cos']/CHANCE:.1f}×)")

    # ---- F3 试次聚合（单试次 vs 试次平均）----
    rec["F3_trial_agg"] = {
        "trial_level_acc1": base_dec["acc1_cos"],
        "feature_avg_acc1": base_dec["acc1_featavg_cos"],
    }

    # ---- F4 图像特征空间 PCA-128（在训练图上拟合）----
    Vc_tr = IMG_train - IMG_train.mean(axis=0)
    U, S, Vt = np.linalg.svd(Vc_tr, full_matrices=False)
    P = Vt[:128].T
    Vtr_p = (IMG_train - IMG_train.mean(axis=0)) @ P
    Vte_p = (IMG_test - IMG_train.mean(axis=0)) @ P
    Wp, _, _, _, vmp = fit_ridge(Xtr_all, Vtr_p, lam)
    f4 = decode_cos(Wp, Xte, Vte_p - vmp, Xte.shape[1])
    rec["F4_target_pca128"] = f4
    log(f"  [F4 PCA-128] acc1={f4['acc1_cos']:.4f} ({f4['acc1_cos']/CHANCE:.1f}×)")

    # ---- F5 协议 ----
    acc3, ratio3 = same_set_identity_decode(Xte)
    rec["F5_protocol"] = {
        "P1_cross_set": base_dec["acc1_cos"],
        "P3_same_set_identity": acc3,
        "P3_ratio_to_chance": ratio3,
    }
    log(f"  [F5 P3 同集身份解码] acc={acc3:.4f} ({ratio3:.1f}× 随机) "
        f"← 应复现 exp64 的 ~0.0785 / 15.7×")

    # ---- 自检 S2：P2 同集检索 ----
    p2 = same_set_retrieval(Xtr_all, IMG_train, lam, n_images=args.p2_candidates)
    rec["self_check_S2_same_set_retrieval"] = p2
    log(f"  [S2 P2 同集检索] acc1={p2['acc1_cos']:.4f} "
        f"({p2['acc1_cos']/CHANCE:.1f}×)")

    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def self_test():
    log("=== 自测：四个解码器的实现方向 ===")
    rng = np.random.default_rng(3)
    n_img, n_rep, d, m_out, n_te = 300, 4, 40, 24, 60

    V = rng.normal(size=(n_img, m_out))
    Vt = rng.normal(size=(n_te, m_out))
    A_map = rng.normal(size=(m_out, d)) / np.sqrt(m_out)

    def make_eeg(Vx, nrep):
        n = len(Vx)
        e = (np.repeat(Vx, nrep, axis=0) @ A_map
             + 0.8 * rng.normal(size=(n * nrep, d)))
        return e.astype(np.float32).reshape(n, nrep, d)

    for tag, Xtr, Xte_ in (
            ("有信号", make_eeg(V, n_rep), make_eeg(Vt, 3)),
            ("纯噪声", make_eeg(V, n_rep), rng.normal(size=(n_te, 3, d)).astype(np.float32))):
        W, scale, Xbar, A0, v_mean = fit_ridge(Xtr, V, 1.0)
        Vc = Vt - v_mean
        d_ridge = decode_cos(W, Xte_, Vc, 3)
        d_rrr = decode_cos(rrr_project(W, Xbar, 8), Xte_, Vc, 3)
        wx, wy, mux, muv = cca_fit(Xbar, V, 8)
        d_cca = cca_decode(wx, wy, mux, muv, Xte_, Vc)
        accs = {"ridge": d_ridge["acc1_cos"], "rrr": d_rrr["acc1_cos"],
                "cca": d_cca["acc1_cos"]}
        log(f"  {tag}: " + "  ".join(f"{k}={v:.3f}({v*100/ (100/n_te):.1f}×)"
                                     for k, v in accs.items()))
        if tag == "有信号":
            for k, v in accs.items():
                assert v > 3.0 / n_te, f"{k} 在有信号时只有 {v:.4f}，实现方向可能写反"
        else:
            for k, v in accs.items():
                assert v < 2.5 / n_te, f"{k} 在纯噪声时高达 {v:.4f}，零分布口径有问题"

    log("  ✓ 三种解码器：有信号 ⇒ 显著高于随机；纯噪声 ⇒ 回到随机")
    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[1000, 4000])
    ap.add_argument("--rrr-ks", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--cca-ks", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--p2-candidates", type=int, default=2000)
    ap.add_argument("--align-check-subject", type=int, default=1)
    ap.add_argument("--align-check-perm", type=int, default=100)
    ap.add_argument("--align-check-images", type=int, default=1500,
                    help="对齐预检里算 RSA 的图像数（**必须子采样**：16540² 的 RDM 跑不完）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"图像特征 test {IMG_test.shape} | train {IMG_train.shape}")

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = run_subject(sid, IMG_train, IMG_test, args,
                              np.random.default_rng(SEED + sid))
            per_subject.append(rec)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)
        log(f"  用时 {time.time()-t0:.0f}s")

    n_err = sum(1 for r in per_subject if "error" in r)
    # ⚠️ 显式报「N 个被试全部报错」——交叉验证的教训（exp92 支线 A 曾静默出空表）
    log(f"\n完成：{len(per_subject)-n_err}/{len(per_subject)} 个被试成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))
    if len(per_subject) and n_err == len(per_subject):
        log("✗✗ 全部被试都报错 ⇒ 结果不可用")

    (out_dir / "exp93a_level_audit.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93a_level_audit.json")


if __name__ == "__main__":
    main()
