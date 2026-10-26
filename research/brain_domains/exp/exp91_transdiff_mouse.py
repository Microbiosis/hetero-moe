#!/usr/bin/env python3
"""exp91_transdiff_mouse.py — 判据①（转分化代价）在「**同记录 + 多脑区 + 各有行为标签**」
数据上的**可测性自检**（鼠 · Steinmetz 2019）。

## 为什么做这个（**不是新判据，是移植 + 自标定**）

判据① 至今**没有一次成功的测量**：exp90 在人 EEG 上因「前部无可读输出 + 自监督目标塌缩」作废
⇒ 我们**根本不知道这个判据有没有判别力**（exp90 的教训：「置换代价显著 > 0」不足以证明判据有效）。
⇒ 在鼠的**同一记录、多个脑区、各有行为标签**的数据上做一次移植：

- 若跨带转分化代价 **显著大于同带自对照** ⇒ 判据① **有判别力** ⇒ 为人 EEG 收 3×3 才有意义；
- 若两者没有差别 ⇒ 判据① **没判别力** ⇒ **收新数据也白搭**（是判据的问题，不是数据的问题）⇒ 要换判据。

⇒ ⇒ 这就是「**要不要收新数据**」那个决定的**前置实验**（见 `FINDINGS_TIER0_RSA.md` §三十五·四）。

## 移植了什么（**逐条对照 exp90**）

| exp90（人 EEG：一个模型、三个分支）| exp91（鼠：多区同记录）| 为什么这么改 |
|---|---|---|
| 三分支的潜变量 `a` 落在**同一模型**的同一空间 ⇒ 码本天然可跨分支比 | 用**两带联合 PCA**（train split 的 trial 特征，前 16 维）⇒ 共同空间 | 鼠上没有共享模型 ⇒ 必须自造共同空间 ⇒ **这是移植引入的新环节** ⇒ 靠**同带自对照**定标，不假装它不存在 |
| 每分支一个软原型集 `proto`（K=64），**被对比学习损失锚在图像原型上** | 每带一个**行为锚定**的原型集：按行为类分层 k-means（4 类 × 每类 4 原型 = 16）| exp90 的码本锚在**语义**上 ⇒ 鼠上锚在**行为**上 ⇒ ⚠️ **不锚在行为上就退化成「结构相似度」，而那不是判据** |
| `proto_read(a,P) = mean‖a − softmax(a·Pᵀ/τ)P‖²`（τ=0.1）| **同式逐字搬**（τ=0.1 固定）| — |
| `d(A→B) = proto_read(a_B, proto_A) − proto_read(a_B, proto_B)` | **同式逐字搬**（a = 该带在共同空间里的 trial 特征）| — |
| 「**自有原型必须先拟合到自己的潜变量上**」才是训练后的性质（exp90 自测 3 的教训）| 码本在 **train split** 上拟合、残差在 **test split** 上算 | 既守住那条前提，又避免 in-sample 偏袒 |
| （exp90 没有）| **同带自对照**：把 B 的单元**随机劈两半** ⇒ `d(B1→B2)` = 「同一带内部」的代价 = **地板** | 移植引入的新环节 ⇒ 必须有地板，否则「跨带代价大」可能只是「换个单元集就变大」 |

## 三个带（**不自己发明**）

用 Neuromatch 自己给的 `functional_domain`（他们的聚类，不是我们的）：
**`visual`（感觉侧 ≈ POST）/ `motor`（运动侧 ≈ CENT）/ `prefrontal`（执行侧 ≈ FRONT）**。
⚠️ 鼠 ≠ 人：这是**位置轴**在另一个系统上的对应，**只**用来回答「判据① 有没有判别力」，
**不回填**「三个原语还是两个」（人 EEG 的问题）。

## 行为标签（码本的锚）

Steinmetz 任务 = 视觉对比辨别 + 转轮选择 ⇒ 行为类 = **（刺激侧 × 选择）** ∈ 4 类
（`visual_stimulus_left_contrast` vs `right` ⇒ 哪一侧；`response_choice` = ±1）。
⚠️ **不是每个带都编码同一个变量**（视觉带编码刺激侧、运动带编码选择）⇒ C0 的可解码性检查
是「**该带至少能解码 4 类行为中的一个以上**」。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 通过 ⇒ |
|---|---|---|
| **C0** | **管线自检**（前置，不过 ⇒ 全实验作废）：① session 对齐质量达标（单元级相关中位 ≥ 0.3、>0.5 的单元占 ≥ 0.5）② 三个带齐全（各 ≥ 20 单元）③ 行为类 ≥ 4 个、每类 ≥ 8 trial ④ **每个带的行为可解码性**：该带活动解码 4 类行为的准确率 > 机会水平 0.25，且置换检验（50 次）p < 0.05（**这是 exp90「没有可读输出 ⇒ 3×3 退化成 1×1」的移植版守卫**）| 数据/标签/对齐没问题 ⇒ 继续；**不过 ⇒ 全实验作废** |
| **C1** | **判据① 有没有判别力**：逐 session 配对差 `Δ(X→Y) = d_跨带(X→Y) − d_同带(Y)`（同带自对照作地板）> 0；证据 = **session 级符号检验**（≥ 8 个 session ⇒ p ≤ 0.05）**且**中位 > 0 | 判据① **有判别力** ⇒ 为人 EEG 收 3×3 数据才有意义 |
| **C2** | **退化守卫**（exp90 的教训：塌缩后 d 仍是**非零常数** ⇒ 只看显著性会假阳性）：`d` 不得是**非零常数**（会话间离散度 > 0；**d ≡ 0 不算退化** —— 那是没有信号 ⇒ 归 C1_FAIL）| 不退化 ⇒ 判决有效；退化 ⇒ 判决作废（只报） |
| **C3** | **只报，不作判决**：① 3×3 转分化矩阵（6 个方向，逐 session 中位 + 符号检验）② 跨带**行为解码迁移**（用 X 带的解码器解 Y 带）③ 逐带可解码准确率 | — |

## 判决

- `C1_OK`：判据① 在这个数据上**有判别力**（⇒ 收数据这件事才成立）
- `C1_FAIL`：判据① **没判别力**（⇒ 收新数据无用 ⇒ 换判据）
- `C0_FAIL`：管线自检不过（⇒ 数据/标签/对齐的问题，**不是**关于判据的结论）

## 统计口径（与 exp89/exp90/exp96 逐字一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
- **零差剔除**：差值为 0 单独计数，**不参与**符号检验。
- **统计单元 = session**（不是 trial；trial 不独立 ⇒ trial 级检验一律只作描述性）。

## 用法（云端）

    python -u /domains/scripts/exp91_transdiff_mouse.py --data /domains/steinmetz_trials \
        --out /domains/runs/illusionbrain-align/exp91
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

TAU = 0.1                # exp90 的 proto_temp
N_PC = 16                # 共同空间维数
K_PROTO_PER_CLASS = 4    # 每行为类几个原型
N_CLASSES_MIN = 4       # 行为类至少这么多（刺激侧 × 选择 = 4）
MIN_TRIALS_PER_CLASS = 8
MIN_UNITS_PER_BAND = 20
N_PERM = 50              # 置换检验次数
CHANCE = 0.25            # 4 类行为的机会水平
EPS = 1e-9

BANDS = ("visual", "motor", "prefrontal")   # Neuromatch 自己的 functional_domain
BAND_LABEL = {"visual": "POST(感觉/视觉侧)", "motor": "CENT(运动侧)",
               "prefrontal": "FRONT(执行侧)"}


# ---------------------------------------------------------------------------
# 统计小工具（口径与 exp89/exp90/exp96 逐字一致）
# ---------------------------------------------------------------------------

def median_upper(xs):
    s = sorted(float(x) for x in xs if x == x)
    return s[len(s) // 2] if s else float("nan")


def sign_test_nonzero(diffs):
    d = [float(x) for x in diffs if x == x]
    n_zero = sum(1 for x in d if x == 0)
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 0, 0, 0, float("nan"), n_zero
    k = sum(1 for x in d if x > 0)
    m = n - k
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return n, k, m, float(min(1.0, p)), n_zero


def proto_read(a, P, tau=TAU):
    """exp90 的 `proto_read` 逐字搬：软分配 → 重构 → 残差（mean ‖a − rec‖²）。"""
    w = np.exp((a @ P.T) / tau - np.max((a @ P.T) / tau, axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    rec = w @ P
    return float(((a - rec) ** 2).sum(axis=1).mean())


def ridge_fit(X, Y, lam=1.0):
    """岭回归（多输出）⇒ 系数与截距。"""
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, X1.T @ Y)
    return W[:-1], W[-1]


def ridge_predict(X, W, b):
    return X @ W + b


def kmeans(X, k, seed=0, iters=50):
    """简易 k-means（k-means++ 初始化）⇒ 原型矩阵 (k, dim)。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n == 0:
        return np.zeros((k, d), dtype=np.float64)
    idx = rng.integers(0, n)
    C = [X[idx]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1), axis=1)
        pr = d2 / (d2.sum() + EPS)
        C.append(X[rng.choice(n, p=pr)])
    C = np.array(C)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return C


# ---------------------------------------------------------------------------
# 单个 session
# ---------------------------------------------------------------------------

def prep_session(z):
    """取出：带→单元索引、行为类、train/test 划分。"""
    feat = np.asarray(z["feat"], dtype=np.float32)
    dom = np.asarray(z["unit_domain"], dtype=object)
    choice = np.asarray(z["trial_choice"], dtype=np.float64)
    side = np.asarray(z["trial_stim_side"], dtype=np.float64)
    inc = np.asarray(z["trial_included"], dtype=bool)
    stim = np.asarray(z["trial_stim_time"], dtype=np.float64)
    n_tri = feat.shape[0]

    # 行为类 = 刺激侧 × 选择 ⇒ 0..3；-1 = 不可用
    cls = np.full(n_tri, -1, dtype=np.int64)
    ok = inc & np.isfinite(choice) & np.isfinite(side) & np.isfinite(stim)
    ok &= (np.abs(choice) > 0)
    cls[ok] = ((side[ok] > 0).astype(np.int64) * 2
               + (choice[ok] > 0).astype(np.int64))
    # 时间序切分：前 70% 训练，后 30% 测试（不跨时间泄漏）
    order = np.argsort(stim)
    n_tr = int(round(0.7 * n_tri))
    tr = np.zeros(n_tri, dtype=bool)
    tr[order[:n_tr]] = True
    te = ~tr & (cls >= 0)
    tr &= cls >= 0

    bands = {}
    for b in BANDS:
        idx = np.flatnonzero(np.array([str(x) == b for x in dom]))
        if len(idx) >= MIN_UNITS_PER_BAND:
            bands[b] = idx
    return feat, cls, tr, te, bands, str(z["session"]), z


def joint_pca(XA, XB, k=N_PC):
    """两带联合 PCA（train trial）⇒ 共同空间。"""
    U = np.hstack([XA, XB])
    mu = U.mean(axis=0, keepdims=True)
    Uc = U - mu
    # 随机抽样以限制 SVD 规模（trial × unit 可达 260×24000）
    if Uc.shape[0] > 400:
        sel = np.random.default_rng(0).choice(Uc.shape[0], 400, replace=False)
        Uc = Uc[sel]
    # 经济型 SVD
    try:
        _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
    except np.linalg.LinAlgError:
        Vt = np.linalg.pinv(Uc)[:k]
    W = Vt[:k].T
    return mu, W


def build_codebook(z_X, cls, tr, k_per_class=K_PROTO_PER_CLASS, seed=0):
    """行为锚定的原型集：按行为类分层 k-means ⇒ (K×dim) 矩阵；同时返回用到的类。"""
    protos, used = [], []
    for c in range(N_CLASSES_MIN):
        m = tr & (cls == c)
        if m.sum() < MIN_TRIALS_PER_CLASS:
            continue
        C = kmeans(z_X[m], k_per_class, seed=seed + c)
        protos.append(C)
        used.append(c)
    if not protos:
        return None, []
    return np.vstack(protos), used


def decode_acc(Xtr, ytr, Xte, yte, lam=1.0):
    """岭回归 + argmax ⇒ 多类解码准确率。"""
    Y = np.eye(int(ytr.max()) + 1)[ytr]
    W, b = ridge_fit(Xtr, Y, lam)
    pred = ridge_predict(Xte, W, b).argmax(axis=1)
    return float((pred == yte).mean())


def perm_p(Xtr, ytr, Xte, yte, n_perm=N_PERM, lam=1.0, seed=0):
    """置换检验：打乱训练标签重拟合 ⇒ 准确率的零分布。"""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(ytr)
        null.append(decode_acc(Xtr, yp, Xte, yte, lam))
    obs = decode_acc(Xtr, ytr, Xte, yte, lam)
    null = np.asarray(null)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


# ---------------------------------------------------------------------------
# 单 session 的转分化矩阵
# ---------------------------------------------------------------------------

def session_transdiff(z, seed=0):
    """⇒ {带对: {d_cross, d_ctrl, delta, n_tri, n_units}} + 每带可解码性。"""
    feat, cls, tr, te, bands, sess, _ = prep_session(z)
    out = {"session": sess, "bands": {b: int(len(v)) for b, v in bands.items()},
           "n_trials": int(feat.shape[0]), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "decode": {}, "pairs": {}}
    if len(bands) < 3 or tr.sum() < 40 or te.sum() < 20:
        out["skip"] = "带不全或 trial 不够"
        return out
    # 每带标准化（按 train 的均值方差）
    Z = {}
    for b, idx in bands.items():
        X = feat[:, idx].astype(np.float64)
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True) + EPS
        Z[b] = (X - mu) / sd
    # 行为可解码性（C0b）
    for b in bands:
        Xtr, ytr = Z[b][tr], cls[tr]
        Xte, yte = Z[b][te], cls[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            out["decode"][b] = {"acc": float("nan"), "p": float("nan")}
            continue
        acc, p = perm_p(Xtr, ytr, Xte, yte, seed=seed)
        out["decode"][b] = {"acc": acc, "p": p}
    # 每个带对：共同空间 + 码本 + 转分化
    for A in BANDS:
        for B in BANDS:
            if A == B or A not in bands or B not in bands:
                continue
            mu, W = joint_pca(Z[A][tr], Z[B][tr])
            zA = (Z[A] - mu) @ W
            zB = (Z[B] - mu) @ W
            PA, usedA = build_codebook(zA, cls, tr, seed=seed)
            PB, usedB = build_codebook(zB, cls, tr, seed=seed + 100)
            if PA is None or PB is None:
                continue
            used = sorted(set(usedA) & set(usedB))
            if len(used) < 2:
                continue
            # 只保留共有的类 ⇒ 两带码本同构
            def pick(P, n_class):
                out_rows = []
                for c in range(N_CLASSES_MIN):
                    if c in used:
                        out_rows.append(P[c * K_PROTO_PER_CLASS:(c + 1) * K_PROTO_PER_CLASS])
                return np.vstack(out_rows)
            PAu, PBu = pick(PA, len(used)), pick(PB, len(used))
            r_own = proto_read(zB[te], PBu)
            r_other = proto_read(zB[te], PAu)
            d_cross = r_other - r_own
            # 同带自对照：B 的单元随机劈两半
            rng = np.random.default_rng(seed + 7)
            idxB = bands[B]
            perm = rng.permutation(len(idxB))
            h = len(idxB) // 2
            B1, B2 = idxB[perm[:h]], idxB[perm[h:]]
            mu2, W2 = joint_pca(Z[B][tr], Z[B][tr])
            z1 = (Z[B][:, B1] - mu2) @ W2
            z2 = (Z[B][:, B2] - mu2) @ W2
            P1, u1 = build_codebook(z1, cls, tr, seed=seed + 11)
            P2, u2 = build_codebook(z2, cls, tr, seed=seed + 13)
            if P1 is None or P2 is None:
                continue
            used2 = sorted(set(u1) & set(u2))
            if len(used2) < 2:
                continue

            def pick2(P):
                rows = []
                for c in range(N_CLASSES_MIN):
                    if c in used2:
                        rows.append(P[c * K_PROTO_PER_CLASS:(c + 1) * K_PROTO_PER_CLASS])
                return np.vstack(rows)
            c1, c2 = pick2(P1), pick2(P2)
            d_ctrl = proto_read(z2[te], c1) - proto_read(z2[te], c2)
            out["pairs"][f"{A}->{B}"] = {
                "d_cross": d_cross, "d_ctrl": d_ctrl, "delta": d_cross - d_ctrl,
                "r_own": r_own, "r_other": r_other,
                "n_classes": len(used), "n_units_B": int(len(idxB)),
            }
    return out


# ---------------------------------------------------------------------------
# 判决
# ---------------------------------------------------------------------------

def decide(per_session, min_sessions=8, alpha=0.05):
    """从逐 session 原始记录重算判决（阈值的唯一出处 ⇒ 可离线重算）。"""
    problems = []
    # C0：管线自检
    c0_ok, c0_detail = True, {}
    for r in per_session:
        if r.get("skip"):
            c0_ok = False
            c0_detail[r["session"]] = r["skip"]
            continue
        for b, v in r.get("decode", {}).items():
            a, p = v.get("acc", float("nan")), v.get("p", float("nan"))
            if not (a == a) or not (p == p) or a <= CHANCE or p >= alpha:
                c0_ok = False
                c0_detail.setdefault(r["session"], []).append(
                    f"{b}: acc={a:.3f} p={p:.3f}")
    # C1：判据① 有没有判别力
    deltas = {}
    for key in sorted({k for r in per_session for k in r.get("pairs", {})}):
        vals = [r["pairs"][key]["delta"] for r in per_session if key in r.get("pairs", {})]
        if vals:
            deltas[key] = {"n": len(vals), "median": median_upper(vals), "vals": vals}
    pooled = []
    for key, v in deltas.items():
        pooled.extend(v["vals"])
    n_eff, n_pos, n_neg, p_sign, n_zero = sign_test_nonzero(pooled)
    med = median_upper(pooled)
    c1 = bool(n_eff >= min_sessions and p_sign <= alpha and med > 0)
    # C2：退化守卫（exp90 的教训：塌缩后 d 仍是**非零常数** ⇒ 只看显著性会假阳性）
    spread = {}
    for key, v in deltas.items():
        if len(v["vals"]) > 1:
            spread[key] = {"spread": float(max(v["vals"]) - min(v["vals"])),
                           "max_abs": float(max(abs(x) for x in v["vals"]))}
    # 退化 = d 是**非零常数**（d ≡ 0 不算退化：那是没有信号 ⇒ 归 C1_FAIL）
    degenerate = any(s["spread"] <= 1e-12 and s["max_abs"] > 1e-9
                     for s in spread.values())
    c2 = not degenerate
    if not c0_ok:
        verdict = "C0_FAIL"
    elif not c2:
        verdict = "DEGENERATE"
    elif c1:
        verdict = "C1_OK"
    else:
        verdict = "C1_FAIL"
    if not c2:
        problems.append("退化守卫不过（d 的会话间离散度为 0）⇒ 判决作废")
    return {
        "verdict": verdict,
        "C0_pipeline_selfcheck": c0_ok,
        "C0_detail": {k: (v if isinstance(v, str) else list(v)) for k, v in c0_detail.items()},
        "C1_has_discriminative_power": c1,
        "C1": {"n_eff": n_eff, "n_pos": n_pos, "n_neg": n_neg, "p": p_sign,
               "n_zero": n_zero, "median_delta": med, "min_sessions": min_sessions,
               "alpha": alpha},
        "C2_degenerate_guard": c2,
        "C2_spread": spread,
        "per_pair": {k: {"n": v["n"], "median": v["median"]} for k, v in deltas.items()},
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# 自测（判据实现也要被测）
# ---------------------------------------------------------------------------

def self_test():
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  ✓ {name}")
        else:
            bad.append(name)
            print(f"  ✗ {name}")

    chk("上中位(偶数 4 个取偏大)", median_upper([1.0, 2.0, 3.0, 4.0]) == 3.0)
    chk("上中位(空)=nan", math.isnan(median_upper([])))
    n, k, m, p, nz = sign_test_nonzero([0.1] * 6)
    chk("符号检验 6/6 正 ⇒ p=0.03125", (n, k, m) == (6, 6, 0) and abs(p - 0.03125) < 1e-12)
    n, k, m, p, nz = sign_test_nonzero([0.0] * 6)
    chk("符号检验 全零 ⇒ n_eff=0 且 n_neg=0", (n, k, m, nz) == (0, 0, 0, 6) and math.isnan(p))
    # proto_read：自有原型 ⇒ 残差更小（exp90 自测 3 的前提）
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 8))
    P = a[:8] + 0.01 * rng.normal(size=(8, 8))
    chk("proto_read 自有原型残差 < 乱原型残差",
        proto_read(a, P) < proto_read(a, rng.normal(size=(8, 8))))
    # 码本：按类分层 ⇒ 原型数 = 类数 × 每类原型数
    z = rng.normal(size=(200, 8))
    cls = np.repeat(np.arange(4), 50)
    tr = np.ones(200, dtype=bool)
    P, used = build_codebook(z, cls, tr)
    chk("码本形状 = 类数×每类原型数", P is not None and P.shape == (4 * K_PROTO_PER_CLASS, 8)
        and len(used) == 4)
    # 岭回归解码：可分的二类 ⇒ 准确率 ≈ 1
    X = np.vstack([rng.normal(-1, 0.3, (60, 5)), rng.normal(1, 0.3, (60, 5))])
    y = np.array([0] * 60 + [1] * 60)
    acc = decode_acc(X[:100], y[:100], X[100:], y[100:])
    chk("岭回归解码 可分二类 ⇒ 准确率 > 0.9", acc > 0.9)
    # decide：四条分支
    def fake(delta, ok0=True, jitter=0.05):
        rng = np.random.default_rng(7)
        ps = []
        for i in range(10):
            d = delta + (float(rng.normal(0, jitter)) if jitter else 0.0)
            ps.append({"session": f"s{i}", "skip": None if ok0 else "x",
                       "decode": {"visual": {"acc": 0.9, "p": 0.001}},
                       "pairs": {"visual->motor": {"delta": d, "d_cross": 1.0,
                                                   "d_ctrl": 0.1, "r_own": 1.0,
                                                   "r_other": 2.0, "n_classes": 4,
                                                   "n_units_B": 30}}})
        return ps
    v = decide(fake(0.5))
    chk("decide 10 个 session 全正 ⇒ C1_OK", v["verdict"] == "C1_OK"
        and v["C1_has_discriminative_power"])
    v = decide(fake(-0.5))
    chk("decide 全负 ⇒ C1_FAIL", v["verdict"] == "C1_FAIL")
    v = decide(fake(0.0, jitter=0))
    chk("decide 全零 ⇒ C1_FAIL（零差不参与符号检验）", v["verdict"] == "C1_FAIL")
    v = decide(fake(0.5, ok0=False))
    chk("decide C0 不过 ⇒ C0_FAIL", v["verdict"] == "C0_FAIL")
    v = decide(fake(0.5, jitter=0))
    chk("decide d 是**非零常数** ⇒ DEGENERATE（exp90 塌缩那个坑）",
        v["verdict"] == "DEGENERATE" and not v["C2_degenerate_guard"])
    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/domains/steinmetz_trials")
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp91")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("*.npz"))
    print(f"数据：{len(files)} 个 session（{data_dir}）", flush=True)
    if not files:
        print("✗ 没有抽取结果 ⇒ 先跑 steinmetz_extract.py")
        return 1

    per_session = []
    for fp in files:
        try:
            z = np.load(fp, allow_pickle=True)
            rec = session_transdiff(z)
            per_session.append(rec)
            ps = rec.get("pairs", {})
            print(f"  ✓ {rec['session']}：带 {rec['bands']}；train {rec['n_train']} / test {rec['n_test']}"
                  f"；解码 {({b: round(v['acc'], 3) for b, v in rec.get('decode', {}).items()})}"
                  f"；对 {len(ps)}", flush=True)
            for k, v in ps.items():
                print(f"      {k}: 跨带 {v['d_cross']:+.4f}  同带 {v['d_ctrl']:+.4f}"
                      f"  Δ {v['delta']:+.4f}", flush=True)
        except Exception as e:
            print(f"  ✗ {fp.name}: {type(e).__name__}: {e}", flush=True)

    verdict = decide(per_session)
    payload = {"config": vars(args), "bands": {b: BAND_LABEL[b] for b in BANDS},
               "criteria": {
                   "C0": "管线自检（对齐质量 / 三带齐全 / 行为类 / 每带行为可解码 > 机会 + 置换 p<0.05）",
                   "C1": "判据① 有没有判别力：Δ(跨带 − 同带) > 0，session 级符号检验 ≥8 session 且 p≤0.05 且中位 >0",
                   "C2": "退化守卫：d 的会话间离散度 > 0（不塌缩）",
                   "C3": "只报：3×3 矩阵 / 跨带解码迁移 / 逐带可解码准确率",
               },
               "per_session": per_session, "verdict": verdict}
    (out_dir / "exp91_transdiff_mouse.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n=== 判决：{verdict['verdict']} ===")
    print(f"  C0 自检：{verdict['C0_pipeline_selfcheck']}")
    print(f"  C1 判别力：{verdict['C1_has_discriminative_power']}"
          f"（n_eff={verdict['C1']['n_eff']}、n_pos={verdict['C1']['n_pos']}、"
          f"p={verdict['C1']['p']:.4f}、中位 Δ={verdict['C1']['median_delta']:+.4f}）")
    print(f"  C2 退化守卫：{verdict['C2_degenerate_guard']}")
    for k, v in verdict["per_pair"].items():
        print(f"    {k}: n={v['n']} 中位 Δ={v['median']:+.4f}")
    print(f"结果写入 {out_dir}/exp91_transdiff_mouse.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
