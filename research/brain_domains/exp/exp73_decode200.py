#!/usr/bin/env python3
"""exp73 — 解码方向：EEG 能认出是哪张图吗？（**单试次** 200 选 1）

## 为什么要有这个实验

exp67–exp72 全是**编码**方向：图像特征 → EEG。本实验是**解码**方向：
EEG → 图像特征 → 在 200 张候选图里取 argmax。

exp72 结尾自己列的"最该报的判别式数字"就是这个：**单试次**条件下，
从一次呈现的 EEG 里认出 200 张图中的哪一张（随机基线 1/200 = 0.5%）。

## 估计量：这一次的 Gram **必须**用试次级

    试次级 ridge：min_w Σ_{i,t} ||x_{i,t}·w − v_i||² + λ||w||²
    正则方程：  (G_full + λI) w = n_rep·X̄ᵀV
    两边 ÷ n_rep：(G_full/n_rep + λ'·I) w = X̄ᵀV,   λ' = λ/n_rep

    ⇒ **A0 = G_full / n_rep**，其中 G_full = Σ_{i,t} x_{i,t}x_{i,t}ᵀ

### ⚠ 方向审计（图级降维 ≡ 试次级 ridge，合成数据对拍）

等价性是**方向相关**的，本项目已用合成数据逐项验过：

| 方向 | 设计 | 响应 | 图级 ≡ 试次级？ |
|---|---|---|---|
| A（exp72 的编码方向）| 逐图像特征（同一图共享）| 逐试次 EEG | ✅ **精确成立**（偏差 ~1.8e-15）|
| B（本实验的解码方向）| 逐试次 EEG 特征 | 逐图像特征（同一图共享）| ❌ **不成立** |

方向 B 里，若把 Gram 换成图级的 G_bar = X̄ᵀX̄，就漏掉了**图内散布**
Δ = Σ_i Σ_t (x_{i,t}−x̄_i)(x_{i,t}−x̄_i)ᵀ，实测权重相对偏差 ~0.10–0.13、
单试次 200 选 1 准确率 0.920 vs 0.943（合成数据）。⇒ 本脚本用 G_full。

**方向 A 的结论同时说明：exp72 的"图级降维 ≡ 试次级"断言正确，无需更正。**

## α 的选择（不做选择偏置）

在 **train split 上做图像级 5-fold CV**，用**单试次预测**算 CV R²（与目标指标同口径），
**不使用 test split**。α 选定后才在 test split 上报数。

## 指标

- **单试次 top-1 / top-5**（200 图 × 80 重复 = 16000 个试次）← **主指标**
- 预测平均：80 次重复的**预测向量**取均值再 argmax
- 特征平均：EEG 特征取均值再解码
- 零分布：置换 train 的「图像 ↔ EEG」对应关系，8 次 ⇒ 期望 ≈ 0.5%

## 概念重叠：已核实**不存在**，探针转为一致性检验

（2026-10-22 核实，原假设被推翻，见 `exp73b_concept_overlap.py`）

本脚本最初假设 THINGS-EEG 的 200 张测试图与 1654 个训练概念**有重叠**
（同概念、不同图），因而加了一个"概念不相交版"以排除同概念泄漏。
**这个假设是错的**：

- 桶内图像清单直接给出：训练 1654 概念 × 10 图、测试 200 概念 × 1 图，
  **概念名重叠 = 0 / 200** ⇒ 标准协议本身就是 **concept-disjoint**
  （1654 + 200 = 1854 = THINGS 概念总数）。
- 文献亦如此表述："200 concept-disjoint test concepts"、"Y_seen ∩ Y_unseen = ∅"。

所以：

1. exp73 的 200 选 1 数字**本来就是新概念泛化**，不是同概念取巧；
2. 脚本内置的 ViT 近邻探针**不再用于剔除**（没有东西可剔），
   而转为**一致性检验**：若测试概念与训练概念重叠，测试图的 Top-10 近邻应
   几乎全部落在同一个训练概念组内（frac_single_group ≈ 1.0）；
   实测 **frac_single_group = 0.025** ⇒ 与"概念不相交"一致 ✅。
3. `exclude` 分支保留为安全网：若将来换成有重叠的数据集，它会自动生效。

## 预先指定的设计选择（不做事后挑选）

- 主分析：设计 = **全部 630 维**（10 窗 × 63 通道），**完全不做窗口选择**
- 次分析：设计 = 先验窗 200–400 ms（126 维，exp64/65/66/67 的 RSA 独立结果）
- 相似度：**余弦**（主）+ 原始点积（次），两者都报
"""

from __future__ import annotations

import argparse
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
N_NULL = 8
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
PRIOR_BINS = (2, 4)           # 200–400 ms（先验指定）
SEED = 0

TRAIN_4D = Path("/domains/omnibrain4d/eeg_train")   # 16540 图 × 4 重复
TEST_4D = Path("/domains/omnibrain4d/eeg")           # 200 图 × 80 重复
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/tier0-encoding")


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# 特征提取（与 exp67/70/71/72 **逐字一致**：基线校正 + 10 个 100ms 窗）
# ----------------------------------------------------------------------------
def per_rep_targets(eeg, chunk=400):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch*n_bin)，窗主序。"""
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_rep, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)
        e = e.transpose(0, 1, 3, 2)
        out[s:s + chunk] = e.reshape(e.shape[0], n_rep, n_ch * n_bin)
        del e
    return out


def zscore_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-6
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


# ----------------------------------------------------------------------------
# 试次级 Gram（**方向 B 的关键**：G_full，不是 G_bar）
# ----------------------------------------------------------------------------
def gram_blocks(Xrep, idx_groups):
    """按图像分组累加试次级 Gram。返回每个分组的 (d,d)。

    **float64 累加**：float32 累加会带来 ~1e-7 的相对误差，会淹没自测里
    对"Gram 约定是否正确"的判据（自测要求 1e-10 量级）。
    每组一次 float64 拷贝约 67 MB（3308 图 × 4 重复 × 630 维 × 8B），可接受。
    """
    d = Xrep.shape[-1]
    out = []
    for idx in idx_groups:
        Xi = np.ascontiguousarray(Xrep[idx].reshape(-1, d), dtype=np.float64)
        out.append(Xi.T @ Xi)
    return out


def solve_ridge(A0, B, lam, scale):
    import scipy.linalg as sla
    d = A0.shape[0]
    A = A0.copy()
    A.flat[:: d + 1] += lam * scale
    return sla.solve(A, B, assume_a="pos", overwrite_a=True, check_finite=False)


def r2_flat(pred, target, baseline):
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - baseline) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ----------------------------------------------------------------------------
# 核心：拟合 + 解码
# ----------------------------------------------------------------------------
def fit_and_decode(Xrep, Xte, V, Vtest, cols, alphas, rng,
                   n_null=0, exclude=None):
    """返回一个 dict（主指标 + 次指标）。

    Xrep : (n_img, n_rep, d)   已 z-score 的训练试次特征
    Xte  : (n_img_te, n_rep_te, d) 已 z-score 的测试试次特征
    V    : (n_img, 1024)       训练图像特征（行与 Xrep 的图像一一对应）
    Vtest: (n_img_te, 1024)    测试图像特征（候选）
    cols : 参与拟合的特征列（None = 全部）
    """
    n_img, n_rep, d_full = Xrep.shape
    # cols 不为 None 时**必须真的选列**：只改维度数而不切片，会让
    # "矩阵维度 630" 与 "以为的维度 126" 撞上（matmul core dimension mismatch）。
    # z-score 是逐维独立的，因此子集的 z 值已经是正确的，无需重算。
    if cols is not None:
        Xrep = Xrep[:, :, cols]
        Xte = Xte[:, :, cols]
    d = Xrep.shape[-1]

    if exclude is not None:
        keep = np.setdiff1d(np.arange(n_img), exclude)
    else:
        keep = np.arange(n_img)

    Xk = Xrep[keep]
    Vk = V[keep]
    # 图像均值必须用 float64 累加：float32 的均值与 float64 试次矩阵算出的
    # Σ_t x_{i,t} 会差 ~5e-8（float32 eps 量级），足以污染对拍判据。
    Xbar_k = Xk.mean(axis=1, dtype=np.float64)   # (m, d)
    m = len(keep)

    # --- 分组 Gram（m 张图分 5 组，总量 = 全量，省一半算力）---
    rng_f = np.random.default_rng(SEED)
    perm = rng_f.permutation(m)
    fold_img = np.array_split(perm, N_FOLDS)
    G_blocks = gram_blocks(Xk, fold_img)
    G_total = sum(G_blocks)

    # --- α 选择：图像级 5-fold CV，单试次预测的 R²（与主指标同口径）---
    cv_r2 = {a: [] for a in alphas}
    for fi, te_idx in enumerate(fold_img):
        tr_mask = np.ones(m, dtype=bool)
        tr_mask[te_idx] = False
        tr_idx = np.flatnonzero(tr_mask)
        G_tr = G_total - G_blocks[fi]
        A0 = G_tr / n_rep
        scale = float(np.trace(A0)) / d
        vbar_tr = Vk[tr_idx].mean(axis=0)
        Vc_tr = Vk[tr_idx] - vbar_tr
        Xbar_tr = Xbar_k[tr_idx]
        Xbar_te = Xbar_k[te_idx]
        Xte_fold = Xk[te_idx]
        V_te = Vk[te_idx]
        Vc_te = V_te - vbar_tr
        tgt_te = np.repeat(Vc_te, n_rep, axis=0)
        base_te = np.zeros_like(tgt_te)
        for a in alphas:
            W = solve_ridge(A0, Xbar_tr.T @ Vc_tr, a, scale)
            pred = (Xte_fold.reshape(-1, d) @ W).reshape(len(te_idx), n_rep, -1)
            pred = pred.reshape(-1, pred.shape[-1])
            cv_r2[a].append(r2_flat(pred, tgt_te, base_te))
    cv_curve = {a: float(np.mean(v)) for a, v in cv_r2.items()}
    best_a = max(alphas, key=lambda a: cv_curve[a])

    # --- 全量拟合（试次级 Gram）---
    A0 = G_total / n_rep
    scale = float(np.trace(A0)) / d
    v_mean = Vk.mean(axis=0)
    Vc = Vk - v_mean
    Xbar = Xbar_k
    W = solve_ridge(A0, Xbar.T @ Vc, best_a, scale)

    # --- 测试解码 ---
    def decode(W, Vtest_c):
        n_te, n_rep_te, _ = Xte.shape
        pred = (Xte.reshape(-1, d) @ W).reshape(n_te, n_rep_te, -1)
        flat = pred.reshape(-1, pred.shape[-1])
        pn = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        cn = Vtest_c / np.maximum(np.linalg.norm(Vtest_c, axis=1, keepdims=True), 1e-12)
        cos = pn @ cn.T
        dot = flat @ Vtest_c.T
        true = np.repeat(np.arange(n_te), n_rep_te)
        out = {}
        for name, sim in (("cos", cos), ("dot", dot)):
            pick = sim.argmax(axis=1)
            out[f"acc1_{name}"] = float((pick == true).mean())
            top5 = np.argsort(-sim, axis=1)[:, :5]
            out[f"acc5_{name}"] = float((top5 == true[:, None]).any(axis=1).mean())
        # 预测平均：80 次重复的**预测向量**取均值再 argmax
        avg = pred.mean(axis=1)
        an = avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)
        cn2 = Vtest_c / np.maximum(np.linalg.norm(Vtest_c, axis=1, keepdims=True), 1e-12)
        out["acc1_predavg_cos"] = float(((an @ cn2.T).argmax(axis=1) == np.arange(n_te)).mean())
        out["acc1_predavg_dot"] = float(((avg @ Vtest_c.T).argmax(axis=1) == np.arange(n_te)).mean())
        # 特征平均：EEG 特征取均值再解码
        favg = Xte.mean(axis=1, dtype=np.float64)
        fp = favg @ W
        fn_ = fp / np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
        out["acc1_featavg_cos"] = float(((fn_ @ cn2.T).argmax(axis=1) == np.arange(n_te)).mean())
        out["acc1_featavg_dot"] = float(((fp @ Vtest_c.T).argmax(axis=1) == np.arange(n_te)).mean())
        return out

    Vtest_c = Vtest - v_mean
    res = decode(W, Vtest_c)

    # --- 零分布：置换 train 的「图像 ↔ EEG」对应关系 ---
    nulls = []
    for pi in range(n_null):
        p = rng.permutation(m)
        Wn = solve_ridge(A0, Xbar.T @ Vc[p], best_a, scale)
        nulls.append(decode(Wn, Vtest_c)["acc1_cos"])
    res["null_acc1_cos"] = nulls
    res["alpha"] = best_a
    res["cv_r2_curve"] = {str(a): cv_curve[a] for a in alphas}
    res["n_train_images"] = int(m)
    res["n_excluded_images"] = int(n_img - m)
    return res


# ----------------------------------------------------------------------------
# 免标签的概念重叠探针
# ----------------------------------------------------------------------------
def concept_probe(Vtest, Vtrain, topk=10):
    """ViT 特征空间近邻 ⇒ 测试图与 train 的哪个 10 张组同概念。"""
    Vt = Vtest / np.maximum(np.linalg.norm(Vtest, axis=1, keepdims=True), 1e-12)
    Vr = Vtrain / np.maximum(np.linalg.norm(Vtrain, axis=1, keepdims=True), 1e-12)
    sim = Vt @ Vr.T
    idx = np.argsort(-sim, axis=1)[:, :topk]
    groups = idx // 10                       # train 概念 = 连续 10 张一组
    single = 0
    assign = {}
    for j in range(len(idx)):
        u = np.unique(groups[j])
        if len(u) == 1:
            single += 1
            assign.setdefault(int(u[0]), []).append(j)
    conflicts = sum(1 for v in assign.values() if len(v) > 1)
    return {"n_test": int(len(idx)),
            "frac_single_group": float(single / len(idx)),
            "n_groups_used": len(assign),
            "n_conflicts": int(conflicts),
            "match": {int(k): int(v[0]) for k, v in assign.items() if len(v) == 1}}


# ----------------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------------
def self_test():
    log("=== 自测 ===")
    rng = np.random.default_rng(1)
    n_img, n_rep, d, m_out, n_te = 120, 4, 40, 24, 60

    # (1) 试次级 Gram 约定与直接试次级最小二乘一致
    Xb = rng.normal(size=(n_img, d))
    Xrep = (Xb[:, None, :] + 0.3 * rng.normal(size=(n_img, n_rep, d))).astype(np.float32)
    V = rng.normal(size=(n_img, m_out))
    Xbar = Xrep.mean(axis=1, dtype=np.float64)
    Xtr64 = Xrep.reshape(-1, d).astype(np.float64)
    G_full = Xtr64.T @ Xtr64
    A0 = G_full / n_rep
    lam, scale = 1.0, float(np.trace(A0)) / d
    W = solve_ridge(A0, Xbar.T @ V, lam, scale)
    Xtr = Xtr64
    Vrep = np.repeat(V, n_rep, axis=0)
    A_dir = Xtr.T @ Xtr + (lam * n_rep) * scale * np.eye(d)
    W_dir = np.linalg.solve(A_dir, Xtr.T @ Vrep)
    dev = float(np.max(np.abs(W - W_dir)) / np.max(np.abs(W_dir)))
    log(f"  试次级 Gram 约定 vs 直接试次级：相对偏差 {dev:.3e}")
    assert dev < 1e-10, "Gram 约定不一致"

    # (2) 有信号 ⇒ 单试次 200 选 1 显著高于随机；纯信号缺失 ⇒ 回到随机
    #     关键：**训练与测试必须来自同一个生成模型**（图像特征 → EEG），
    #     否则解码器在训练时根本学不到映射，测什么都低于随机。
    V = rng.normal(size=(n_img, m_out))
    Vt = rng.normal(size=(n_te, m_out))
    A_map = rng.normal(size=(m_out, d)) / np.sqrt(m_out)

    def make_eeg(Vx, nrep):
        n = len(Vx)
        e = (np.repeat(Vx, nrep, axis=0) @ A_map
             + 0.8 * rng.normal(size=(n * nrep, d)))
        return e.astype(np.float32).reshape(n, nrep, d)

    Xsig = make_eeg(V, n_rep)                                  # 训练 EEG（有信号）
    Xte_sig = make_eeg(Vt, 3)                                  # 测试 EEG（有信号）
    Xte_noise = rng.normal(size=(n_te, 3, d)).astype(np.float32)  # 测试 EEG（纯噪声）

    for tag, Xte_, thr in (("有信号", Xte_sig, 3.0), ("纯噪声", Xte_noise, 2.0)):
        r = fit_and_decode(Xsig, Xte_, V, Vt, None, ALPHAS, rng, n_null=2)
        acc = r["acc1_cos"]
        ratio = acc / (1.0 / n_te)
        log(f"  {tag}: 单试次 acc1_cos = {acc:.4f} ({ratio:.1f}× 随机 {1/n_te:.4f})，"
            f"零分布中位 {np.median(r['null_acc1_cos']):.4f}，α={r['alpha']:g}")
        if tag == "有信号":
            assert ratio > thr, f"有信号时准确率只有 {ratio:.1f}× 随机，解码路径有 bug"
        else:
            assert ratio < thr, f"纯噪声时准确率 {ratio:.1f}× 随机，零分布口径有问题"

    # (3) cols 切片路径：维度必须**真的**变，且信号仍在
    #     （第一版只改维度数不切片 → matmul core dimension mismatch）
    cols = np.arange(0, 20)
    r = fit_and_decode(Xsig, Xte_sig, V, Vt, cols, ALPHAS, rng, n_null=1)
    log(f"  cols 切片: 630→{len(cols)} 维，单试次 acc1_cos = {r['acc1_cos']:.4f} "
        f"({r['acc1_cos'] / (1.0 / n_te):.1f}× 随机)")
    assert r["acc1_cos"] > 3.0 / n_te, "cols 切片后信号丢失"

    # (4) 概念探针：合成同概念数据应检出
    Vg = rng.normal(size=(n_te, m_out))
    Vblocks = np.repeat(Vg, 10, axis=0) + 0.05 * rng.normal(size=(n_te * 10, m_out))
    pr = concept_probe(Vg, Vblocks, topk=10)
    log(f"  概念探针（合成同概念）：frac_single_group = {pr['frac_single_group']:.3f}，"
        f"冲突 {pr['n_conflicts']}")
    assert pr["frac_single_group"] > 0.9
    log("=== 自测通过 ===")


# ----------------------------------------------------------------------------
def run_subject(sid, IMG_train, IMG_test, rng, args):
    t0 = time.time()
    tr_fp = TRAIN_4D / f"sub-{sid:02d}.npz"
    te_fp = TEST_4D / f"sub-{sid:02d}.npz"
    if not tr_fp.exists() or not te_fp.exists():
        return None

    eeg = np.load(tr_fp)["eeg"]
    Xrep = per_rep_targets(eeg)
    del eeg
    mu, sd = zscore_fit(Xrep)
    Xrep = zscore_apply(Xrep, mu, sd)

    eeg = np.load(te_fp)["eeg"]
    Xte = per_rep_targets(eeg)
    del eeg
    Xte = zscore_apply(Xte, mu, sd)

    prior_cols = np.arange(PRIOR_BINS[0] * 63, PRIOR_BINS[1] * 63)

    variants = {"all630": None, "prior126": prior_cols}

    # --- 概念重叠探针（免标签，只用特征）---
    probe = concept_probe(IMG_test, IMG_train)
    exclude = None
    if (probe["frac_single_group"] >= 0.9 and probe["n_conflicts"] == 0
            and len(probe["match"]) == len(IMG_test)):
        exclude = np.concatenate([np.arange(g * 10, g * 10 + 10)
                                  for g in probe["match"].keys()])
        exclude = np.unique(exclude)

    out = {"subject": f"sub-{sid:02d}", "concept_probe": probe,
           "n_excluded_images": int(0 if exclude is None else len(exclude)),
           "variants": {}}

    for vname, cols in variants.items():
        r = fit_and_decode(Xrep, Xte, IMG_train, IMG_test, cols,
                           ALPHAS, rng, n_null=N_NULL)
        # 概念不相交版
        if exclude is not None:
            rcd = fit_and_decode(Xrep, Xte, IMG_train, IMG_test, cols,
                                 ALPHAS, rng, n_null=0, exclude=exclude)
            r["concept_disjoint"] = {k: v for k, v in rcd.items()
                                     if k not in ("null_acc1_cos", "cv_r2_curve")}
        out["variants"][vname] = r
        log(f"  sub-{sid:02d} [{vname}] α={r['alpha']:g} "
            f"单试次 acc1_cos={r['acc1_cos']:.4f} acc5_cos={r['acc5_cos']:.4f} "
            f"dot={r['acc1_dot']:.4f} | 预测平均={r['acc1_predavg_cos']:.4f} "
            f"特征平均={r['acc1_featavg_cos']:.4f} "
            f"| 零中位={np.median(r['null_acc1_cos']):.4f}  [{time.time()-t0:.0f}s]")

    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def merge_write(rows, filename):
    """增量落盘：**按被试合并**，不覆盖别的分段跑出来的结果。

    超时补跑（--only-subjects）时，若直接写 `rows`，会把先前分段已写入的
    被试**覆盖掉**（本进程的 rows 只含本次跑的那些）。按 subject 合并即可避免。
    """
    path = OUT_DIR / filename
    merged = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            merged = {r["subject"]: r for r in old.get("per_subject", [])}
        except Exception:
            merged = {}
    for r in rows:
        merged[r["subject"]] = r
    ordered = [merged[k] for k in sorted(merged)]
    path.write_text(json.dumps({"per_subject": ordered}, indent=2,
                                ensure_ascii=False), encoding="utf-8")
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-subjects", type=str, default="",
                    help="逗号分隔的被试编号，如 1,2,3")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"特征 train {IMG_train.shape} / test {IMG_test.shape}")

    sids = ([int(s) for s in args.only_subjects.split(",") if s.strip()]
            if args.only_subjects else list(range(1, 11)))

    rows = []
    for sid in sids:
        rec = run_subject(sid, IMG_train, IMG_test, rng, args)
        if rec is None:
            log(f"  sub-{sid:02d}: 数据缺失，跳过")
            continue
        rows.append(rec)
        merge_write(rows, "exp73_decode200.json")

    log("\n" + "=" * 84)
    log("汇总：单试次 200 选 1 解码准确率（随机 0.5%）")
    log("=" * 84)
    summary = {}
    for vname in ("all630", "prior126"):
        vals = [r["variants"][vname]["acc1_cos"] for r in rows
                if vname in r["variants"]]
        if not vals:
            continue
        nulls = [np.median(r["variants"][vname]["null_acc1_cos"]) for r in rows
                 if vname in r["variants"]]
        summary[vname] = {
            "n_subjects": len(vals),
            "acc1_cos_median": float(np.median(vals)),
            "acc1_cos_min": float(np.min(vals)),
            "acc1_cos_max": float(np.max(vals)),
            "null_median": float(np.median(nulls)),
            "ratio_vs_random": float(np.median(vals) / 0.005),
            "n_above_null": int(sum(
                1 for r in rows if vname in r["variants"]
                and r["variants"][vname]["acc1_cos"] >
                np.max(r["variants"][vname]["null_acc1_cos"]))),
            "featavg_median": float(np.median(
                [r["variants"][vname]["acc1_featavg_cos"] for r in rows
                 if vname in r["variants"]])),
            "predavg_median": float(np.median(
                [r["variants"][vname]["acc1_predavg_cos"] for r in rows
                 if vname in r["variants"]])),
            "concept_disjoint_median": float(np.median(
                [r["variants"][vname]["concept_disjoint"]["acc1_cos"]
                 for r in rows if vname in r["variants"]
                 and "concept_disjoint" in r["variants"][vname]])) if any(
                "concept_disjoint" in r["variants"][vname] for r in rows) else None,
        }
        s = summary[vname]
        log(f"  {vname:9s}: 单试次 top1 中位 {s['acc1_cos_median']:.4f} "
            f"({s['ratio_vs_random']:.1f}× 随机) | 范围 {s['acc1_cos_min']:.4f}–{s['acc1_cos_max']:.4f} "
            f"| 零中位 {s['null_median']:.4f} | 高于零 {s['n_above_null']}/{s['n_subjects']} "
            f"| 特征平均 {s['featavg_median']:.4f} | 预测平均 {s['predavg_median']:.4f} "
            f"| 概念不相交 {s['concept_disjoint_median']}")

    (OUT_DIR / "exp73_decode200_full.json").write_text(
        json.dumps({"schema_version": "1.0", "script": "exp73_decode200",
                    "summary": summary, "per_subject": rows,
                    "elapsed_s": round(time.time() - t0, 1)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
