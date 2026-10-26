#!/usr/bin/env python3
"""exp97c_perband_labels.py — 判据① 的可测性自检**第三版**：**C0 的可读性判据改成「每带用它自己相关的二值行为变量」**（鼠 · Steinmetz 2019）。

## 为什么有 exp97c（**exp97b 的 C0 判死，可能是我们的设计错**，2026-10-26）

exp97b 的判决是 `C0_FAIL`（管线自检）⇒ 8 个「同记录多带」的 session 全被自检判死
⇒ ⇒ §三十七·四 指出：**这很可能不是数据的问题，是 C0 的设计问题** ——
exp97b 的 C0 用**同一个 4 类标签（刺激侧 × 选择）+ 同一个时间窗**去要求**所有带**，
而**不同带编码的本来就不是同一个变量**（视觉带编码刺激侧、运动带编码选择、执行带编码决策/选择）。

⇒ ⇒ **exp97c 只改这一件事**：C0 的可读性判据改成 **「每个带用它自己相关的二值行为变量」**：
- `visual` → **刺激侧**（side）
- `motor` → **选择**（choice）
- `prefrontal` → **选择**（choice）

⇒ 门槛：**acc > 0.5（二类机会水平）且置换 p < 0.05**。

⇒ ⇒ **其余一律不动**：码本（类条件均值，K=1/类）、共同空间（两带联合 PCA）+ 同带自对照、
主对（CENT↔FRONT = motor↔prefrontal）、阈值、统计口径、train/test 切分
⇒ ⇒ **单一改动 ⇒ 可归因**（这正是 exp97b 缺的东西）。

⚠️ **与 exp97b 的判决不可直接比**（C0 的口径变了）⇒ ⇒ 故 exp97c **同时把 exp97b 的旧口径算一遍**
（4 类联合标签）⇒ 作为**对照报告**（`C0_legacy_jointlabel`）⇒ ⇒ 这样能直接回答
「**可用 session 从 2 个回到几个**」。

⚠️ **排除规则（预登记，与结果无关）**：可读性是**数据属性**、不是结果属性 ⇒
⇒ 若主对涉及的两个带里**任一带**不过可读性门槛 ⇒ **该 session 不进入主判据**（被剔除）
⇒ ⇒ **必须同时报出**：进入主判据的 session 数、被剔除的 session 数与原因
⇒ ⇒ **不因为「结果好看」而保留 session**（排除规则与 Δ 无关）。

## 为什么有 exp97b（**exp97a 的实测结果 ⇒ 工具无效**，2026-10-26）

exp97a（第一版移植，码本 = 每类 **4** 个原型）跑完 ⇒ **测量本身无效**：逐 session 的 Δ
**符号在各方向间不一致**（实测一例：−68.9 / +61.3 / +21.9 / +51.0 / +110.9 / +253.4）⇒
**测到的不是「区差异」，而是移植引入的两个不对称**：
① **码本过拟合**：每类 4 个原型、每类只 ~25 个 trial ⇒ 自有码本**记住了训练集**
⇒ 在 test 上自有码本反而比外来码本**更差** ⇒ Δ 被过拟合不对称污染；
② **共同空间投影**：两带联合 PCA 的成分可能偏向某一带 ⇒ 两带的可比性被人为改变。

⇒ ⇒ **exp97a 的结果不得读作「判据① 原理上没判别力」**（那是**工具无效**，不是**原理无效**）⇒
⇒ ⇒ **exp97b 只改一件事：码本从「每类 4 个原型」改成「每类 1 个原型」**
⇒ 码本 = **类条件均值向量**（⇒ 与 exp90 的「语义原型 = 类条件均值」**同构** ⇒ 不过拟合）
⇒ **判据、阈值、统计口径一字不改**。

⚠️ **预登记的解释规则（防过度解读）**：若 Δ 的符号**系统性为负**（n_neg > n_pos）⇒
结论**只能**读作「这版移植仍无效」，**不得**读作「判据① 原理上没判别力」
⇒ ⇒ **「工具无效」与「原理无效」必须分开记**（exp90 的教训：把失败的工具说成失败的原理）。

## 为什么做这个（**不是新判据，是移植 + 自标定**）

判据① 至今**没有一次成功的测量**：exp90 在人 EEG 上因「前部无可读输出 + 自监督目标塌缩」作废
⇒ 我们**根本不知道这个判据有没有判别力**（exp90 的教训：「置换代价显著 > 0」不足以证明判据有效）。
⇒ 在鼠的**同一记录、多个脑区、各有行为标签**的数据上做一次移植：

- 若跨带转分化代价 **显著大于同带自对照** ⇒ 判据① **有判别力** ⇒ 为人 EEG 收 3×3 才有意义；
- 若两者没有差别 ⇒ 判据① **没判别力** ⇒ **收新数据也白搭**（是判据的问题，不是数据的问题）⇒ 要换判据。

⇒ ⇒ 这就是「**要不要收新数据**」那个决定的**前置实验**（见 `FINDINGS_TIER0_RSA.md` §三十五·四）。

## 移植了什么（**逐条对照 exp90**）

| exp90（人 EEG：一个模型、三个分支）| exp97a（鼠：多区同记录）| 为什么这么改 |
|---|---|---|
| 三分支的潜变量 `a` 落在**同一模型**的同一空间 ⇒ 码本天然可跨分支比 | 用**两带联合 PCA**（train split 的 trial 特征，前 16 维）⇒ 共同空间 | 鼠上没有共享模型 ⇒ 必须自造共同空间 ⇒ **这是移植引入的新环节** ⇒ 靠**同带自对照**定标，不假装它不存在 |
| 每分支一个软原型集 `proto`（K=64），**被对比学习损失锚在图像原型上** | 每带一个**行为锚定**的原型集：按行为类分层 k-means（**每类 1 个原型 ⇒ 码本 = 类条件均值**）| exp90 的码本锚在**语义**上 ⇒ 鼠上锚在**行为**上 ⇒ ⚠️ **不锚在行为上就退化成「结构相似度」，而那不是判据** |
| `proto_read(a,P) = mean‖a − softmax(a·Pᵀ/τ)P‖²`（τ=0.1）| **同式逐字搬**（τ=0.1 固定）| — |
| `d(A→B) = proto_read(a_B, proto_A) − proto_read(a_B, proto_B)` | **同式逐字搬**（a = 该带在共同空间里的 trial 特征）| — |
| 「**自有原型必须先拟合到自己的潜变量上**」才是训练后的性质（exp90 自测 3 的教训）| 码本在 **train split** 上拟合、残差在 **test split** 上算 | 既守住那条前提，又避免 in-sample 偏袒 |
| （exp90 没有）| **同带自对照**：把 B 的单元**随机劈两半** ⇒ `d(B1→B2)` = 「同一带内部」的代价 = **地板** | 移植引入的新环节 ⇒ 必须有地板，否则「跨带代价大」可能只是「换个单元集就变大」 |

## 三个带（**按解剖，不自造聚类**）

用 NWB 自带的**标准解剖标签**（`general/extracellular_ephys/electrodes/location`，Allen CCF）：
**`VIS*`（感觉/视觉侧 ≈ POST）/ `MO*`（运动侧 ≈ CENT）/ `PL`·`ACA`·`ILA`·`ORB*`（执行侧 ≈ FRONT）**。

⚠️ **留痕（2026-10-26，**出结果前**的正当修订）**：原写「用 Neuromatch 的 `functional_domain`
（他们的聚类，不是我们的）」⇒ 实测**桶里那份与 DANDI 的 NWB 对不上**（Neuromatch 版是原始数据的
**子集**：总放电 255 万 vs NWB 1328 万、单元 771 vs 2072；`orig_unit_idx` 直连匹配率仅 **0.02**）
⇒ ⇒ 改用 **NWB 自带的解剖标签** ⇒ 反而更好：**不借用任何外部聚类标准**
（分区标准就在数据里 ⇒ 一条事实只写一处）。
⚠️ 鼠 ≠ 人：这是**位置轴**在另一个系统上的对应，**只**用来回答「判据① 有没有判别力」，
**不回填**「三个原语还是两个」（人 EEG 的问题）。

## 数据可得性实测 ⇒ 主判据定在 **CENT↔FRONT 这一对**（⚠️ **出结果前**的正当修订，2026-10-26）

抽完 39 个 session 后的**数据事实**（不是结果，是可得性）：

| 每 session 覆盖（各带 ≥ 20 单元）| session 数 |
|---|---:|
| **三带齐全**（visual+motor+prefrontal）| **3**（Hench 20170615、Lederberg 20171205/1207）|
| **motor + prefrontal** | **5** |
| prefrontal + visual | 1 |
| 只有一带 | 30 |
| 一个带都没有 | 11 |

⇒ ⇒ **三带同记录只有 3 个 session** ⇒ 做不了 3×3 的 session 级统计（n=3 ≪ 8）⇒ 3×3 **只报不判**。
⇒ ⇒ 但 **motor↔prefrontal 这一对共 8 个 session**（5 + 3）⇒ 够做 session 级符号检验
（n=8 ⇒ 最低 p = 2/2⁸ = **0.0078**）。

⇒ ⇒ **主判据 = CENT↔FRONT 这一对**（`motor` ↔ `prefrontal`），理由有两条，**都留痕**：
① **实质性**：位置轴「三个原语还是两个」的关键分界**恰恰是 CENT 与 FRONT 的分界**
（C19：前部要再切成 CENT 运动 / FRONT 执行）⇒ 这一对最有信息量；
② **可行性**：只有这一对有足够的同记录 session（8 个）。
⇒ ⚠️ **诚实边界**：方向的选择是**实质性优先 + 可得性约束**，不是「哪个显著算哪个」
（判据在出结果前定死，且判决只认这一对）。
⇒ ⇒ 逐 session 的统计量 = **该 session 两个方向 Δ 的均值**（避免把同 session 当两个独立样本）。

## 数据制备（**预登记参数**）

- **bin = 10 ms**；**trial 窗 = 刺激后 [50 ms, 450 ms]**（400 ms）；
- 行为类 = **（刺激侧 × 选择）** ∈ 4 类（`visual_stimulus_left/right_contrast` 哪一侧非零；
  `response_choice` = ±1；`included` 为真且两值皆有限）；
- **train/test = 按时间前 70% / 后 30%**（不跨时间泄漏）；
- 共同空间 = **两带联合 PCA**（train trial，前 16 维）；码本 = **按行为类分层 k-means**
  （**每类 1 个原型 ⇒ 4 个原型 ⇒ 类条件均值**；exp97a 用的是每类 4 个 ⇒ 过拟合 ⇒ 无效）。

## 行为标签（码本的锚）

Steinmetz 任务 = 视觉对比辨别 + 转轮选择 ⇒ 行为类 = **（刺激侧 × 选择）** ∈ 4 类
（`visual_stimulus_left_contrast` vs `right` ⇒ 哪一侧；`response_choice` = ±1）。
⚠️ **不是每个带都编码同一个变量**（视觉带编码刺激侧、运动带编码选择）⇒ C0 的可解码性检查
是「**该带至少能解码 4 类行为中的一个以上**」。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 通过 ⇒ |
|---|---|---|
| **C0\u2032** | **逐带可读性**（**exp97c 改的就是这一条**）：**每个带用它自己相关的**二值行为变量（`visual`→刺激侧、`motor`→选择、`prefrontal`→选择）解码，须 **acc > 0.5**（二类机会水平）且**置换 p < 0.05**；主对涉及的两个带里**任一带**不过 ⇒ **该 session 被剔除**（不进入主判据）⇒ ⇒ 报告进入/剔除的 session 数与原因 | 可读性是**数据属性**、排除规则与 Δ 无关（预登记）|
| **C1** | **判据① 有没有判别力（主方向 = CENT↔FRONT）**：逐 session 配对差 `Δ(X→Y) = d_跨带(X→Y) − d_同带(Y)`（同带自对照作地板）；**主对 = `motor`↔`prefrontal`**，每个 session 取**两个方向 Δ 的均值**作统计量（不把同 session 当两个独立样本）；证据 = **session 级符号检验**（≥ 8 个 session ⇒ p ≤ 0.05）**且**中位 > 0 | 判据① **有判别力** ⇒ 为人 EEG 收 3×3 数据才有意义 |
| **C2** | **退化守卫**（exp90 的教训：塌缩后 d 仍是**非零常数** ⇒ 只看显著性会假阳性）：主判据的逐 session 统计量不得是**非零常数**（离散度 > 0；**恒为 0 不算退化** —— 那是没有信号 ⇒ 归 C1_FAIL）| 不退化 ⇒ 判决有效；退化 ⇒ 判决作废（只报） |
| **C3** | **只报，不作判决**：① **所有**可得方向的转分化（逐方向中位 + 符号检验；只有 3 个 session 的方向**只报**）② 3×3 矩阵（n=3 ⇒ 只报）③ 逐带可解码准确率 | — |

## 判决

- `C1_OK`：判据① 在这个数据上**有判别力**（⇒ 收数据这件事才成立）
- `C1_FAIL`：判据① **没判别力**（⇒ 收新数据无用 ⇒ 换判据）
- `C0_FAIL`：管线自检不过（⇒ 数据/标签/特征的问题，**不是**关于判据的结论）
- `DEGENERATE`：退化守卫不过（判决作废，只报）

## 统计口径（与 exp89/exp90/exp96 逐字一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
- **零差剔除**：差值为 0 单独计数，**不参与**符号检验。
- **统计单元 = session**（不是 trial；trial 不独立 ⇒ trial 级检验一律只作描述性）。

## 用法（云端）

    python -u /domains/scripts/exp91_transdiff_mouse.py --data /domains/steinmetz_trials \
        --out /domains/runs/illusionbrain-align/exp97c
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
K_PROTO_PER_CLASS = 1    # 每行为类几个原型（**exp97b：4 → 1** ⇒ 码本 = 类条件均值 ⇒ 不过拟合）
N_CLASSES_MIN = 4       # 行为类至少这么多（刺激侧 × 选择 = 4）
MIN_TRIALS_PER_CLASS = 8
MIN_UNITS_PER_BAND = 20
N_PERM = 50              # 置换检验次数
CHANCE = 0.25            # 4 类行为的机会水平
EPS = 1e-9

BANDS = ("visual", "motor", "prefrontal")   # Neuromatch 自己的 functional_domain
BAND_LABEL = {"visual": "POST(感觉/视觉侧)", "motor": "CENT(运动侧)",
               "prefrontal": "FRONT(执行侧)"}

# ⚠️ **exp97c 的唯一改动**：每个带的「可读性」用**它自己相关的**二值行为变量来判
BAND_LABEL_VAR = {"visual": "side",        # 刺激侧（视觉带编码的是感知）
                  "motor": "choice",       # 选择（运动带编码的是动作/选择）
                  "prefrontal": "choice"}  # 选择（执行带编码的是决策）
PRIMARY_BANDS = ("motor", "prefrontal")   # 主对涉及的两个带（= C0 的门槛对象）
CHANCE_BIN = 0.5                            # 二类标签的机会水平
ALPHA_READ = 0.05                           # 可读性门槛的显著性


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


def _ridge_fit_ref(X, Y, lam=1.0):
    """**参考实现**（原始形式：显式加截距列、截距不惩罚）⇒ 只用于自测比对。"""
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, X1.T @ Y)
    return W[:-1], W[-1]


def ridge_fit(X, Y, lam=1.0):
    """岭回归（多输出，**截距不惩罚**）⇒ 与 `_ridge_fit_ref` **数学等价**，但快很多。

    ⚠️ **为什么换实现（纯提速，不动判据）**：带上有上千单元时，直接在**单元空间**求解是
    O(n_units³)，再乘 50 次置换 ⇒ 实测**一个 session 跑了 8 分钟都没出来**（39 个必超时）。
    ⇒ 中心化 ⇒ 截距可分离；**样本数 < 特征数**时改在**样本空间**求解（O(n_trials³)）
    ⇒ 快几个数量级（自测里与参考实现逐位比对 ⇒ 数值不变）。
    """
    xm = X.mean(axis=0, keepdims=True)
    Xc = X - xm
    n, d = Xc.shape
    ym = Y.mean(axis=0, keepdims=True)
    if n < d:
        A = Xc @ Xc.T + lam * np.eye(n)
        W = Xc.T @ np.linalg.solve(A, Y - ym)
    else:
        A = Xc.T @ Xc + lam * np.eye(d)
        W = np.linalg.solve(A, Xc.T @ (Y - ym))
    b = ym - xm @ W
    return W, b


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
    dom = np.asarray(z["unit_band"], dtype=object)
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
    """置换检验：打乱训练标签重拟合 ⇒ 准确率的零分布。

    ⚠️ **加速**：Gram/核矩阵只算一次（置换只改标签、不改 X）⇒ 50 次置换只做矩阵乘。
    """
    rng = np.random.default_rng(seed)
    xm = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - xm
    n, d = Xc.shape
    if n < d:
        A = Xc @ Xc.T + lam * np.eye(n)
    else:
        A = Xc.T @ Xc + lam * np.eye(d)

    def acc_of(y):
        Y = np.eye(int(y.max()) + 1)[y]
        ym = Y.mean(axis=0, keepdims=True)
        if n < d:
            W = Xc.T @ np.linalg.solve(A, Y - ym)
        else:
            W = np.linalg.solve(A, Xc.T @ (Y - ym))
        b = ym - xm @ W
        pred = (Xte @ W + b).argmax(axis=1)
        return float((pred == yte).mean())

    obs = acc_of(ytr)
    null = np.asarray([acc_of(rng.permutation(ytr)) for _ in range(n_perm)])
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
           "decode": {}, "readable": {}, "pairs": {}}
    if len(bands) < 2 or tr.sum() < 40 or te.sum() < 20:
        out["skip"] = "带不足两个或 trial 不够"
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
    # ⚠️ **exp97c 的核心**：每带用**自己相关的**二值行为变量判「可读性」
    #    （二值标签由 4 类 cls 推出：side = cls>=2，choice = cls%2==1；cls<0 的 trial 无效）
    for b in bands:
        var = BAND_LABEL_VAR[b]
        yb = ((cls >= 2) if var == "side" else (cls % 2 == 1)).astype(np.int64)
        yb[cls < 0] = -1
        Xtr, ytr = Z[b][tr], yb[tr]
        Xte, yte = Z[b][te], yb[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            out["readable"][b] = {"var": var, "acc": float("nan"),
                                  "p": float("nan"), "ok": False}
            continue
        acc, p = perm_p(Xtr, ytr, Xte, yte, seed=seed + 1)
        out["readable"][b] = {"var": var, "acc": acc, "p": p,
                              "ok": bool(acc > CHANCE_BIN and p < ALPHA_READ)}
    # 每个带对：共同空间 + 码本 + 转分化（**只算实际存在的带对**）
    avail = [b for b in BANDS if b in bands]
    for A in avail:
        for B in avail:
            if A == B:
                continue
            mu, W = joint_pca(Z[A][tr], Z[B][tr])
            # ⚠️ **两次踩坑**（都只测零件时抓不到）：
            #    ① `mu` 是**拼接后**的均值（1, n_A+n_B）⇒ 必须按带切分再减
            #       （直接减单带矩阵 ⇒ ValueError: (214,381) vs (1,658)）
            #    ② `W` 的行对应**拼接后的单元** ⇒ 每带要用**属于自己那段**的行来投影
            #       （用整块 W ⇒ matmul 维度不匹配：120 vs 60）
            nA = Z[A].shape[1]
            k = W.shape[1]
            zA = (Z[A] - mu[:, :nA]) @ W[:nA, :]
            zB = (Z[B] - mu[:, nA:]) @ W[nA:, :]
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
            # ⚠️ **第四个踩坑**：劈出来的下标必须是**带内局部下标**（用来索引 Z[B] 的列），
            #    全局单元下标（bands[B]）不能直接去索引局部列 ⇒ 之前 index 76 out of bounds
            rng = np.random.default_rng(seed + 7)
            idxB = bands[B]
            perm = rng.permutation(len(idxB))
            h = len(idxB) // 2
            loc1, loc2 = perm[:h], perm[h:]        # 局部下标
            mu2, W2 = joint_pca(Z[B][tr], Z[B][tr])
            # joint_pca([B,B]) 的行按**整带**排 ⇒ 用「第一份」的均值/PC 行，再按子集取行
            nB = Z[B].shape[1]
            muB = mu2[:, :nB]
            WB = W2[:nB, :]
            z1 = (Z[B][:, loc1] - muB[:, loc1]) @ WB[loc1, :]
            z2 = (Z[B][:, loc2] - muB[:, loc2]) @ WB[loc2, :]
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
    # **主对（CENT↔FRONT）的逐 session 统计量 = 两个方向 Δ 的均值**
    #（不把同一个 session 当成两个独立样本 ⇒ session 级检验才成立）
    pm = [out["pairs"][k]["delta"] for k in ("motor->prefrontal", "prefrontal->motor")
          if k in out["pairs"]]
    if pm:
        out["primary_stat"] = float(np.mean(pm))
    return out


# ---------------------------------------------------------------------------
# 判决
# ---------------------------------------------------------------------------

def decide(per_session, min_sessions=8, alpha=0.05):
    """从逐 session 原始记录重算判决（阈值的唯一出处 ⇒ 可离线重算）。"""
    problems = []
    # ---- C0（exp97c 口径）：**主对涉及的两个带**用**各自的行为变量**过可读性门槛；
    #      不过 ⇒ 该 session **被剔除**（不进入主判据）⇒ 排除规则与 Δ 无关（预登记）
    c0_detail = {}
    prim_ok, prim_excluded = [], []
    for r in per_session:
        if r.get("skip"):
            c0_detail[r["session"]] = r["skip"]
            continue
        rd = r.get("readable") or {}
        bad = []
        for b in [x for x in PRIMARY_BANDS if x in rd]:
            v = rd[b]
            if not v.get("ok"):
                bad.append(f"{b}[{v.get('var')}]: acc={v.get('acc', 0):.3f} "
                           f"p={v.get('p', 0):.3f}")
        if r.get("primary_stat") is not None:
            if bad:
                prim_excluded.append(r["session"])
                c0_detail.setdefault(r["session"], []).extend(bad)
            else:
                prim_ok.append(r)
    # ---- 对照：exp97b 的旧口径（4 类联合标签，所有带同一套）⇒ 只作报告
    legacy_fail = []
    for r in per_session:
        if r.get("skip"):
            continue
        for b, v in (r.get("decode") or {}).items():
            a, p = v.get("acc", float("nan")), v.get("p", float("nan"))
            if not (a == a) or not (p == p) or a <= CHANCE or p >= alpha:
                legacy_fail.append(r["session"])
                break
    legacy_set = set(legacy_fail)
    n_legacy_used = sum(1 for r in per_session
                        if r.get("primary_stat") is not None
                        and r["session"] not in legacy_set)
    # C1：判据① 有没有判别力（**主方向 = CENT↔FRONT**：motor↔prefrontal）
    #     逐 session 统计量 = 该 session 两个方向 Δ 的均值（不把同 session 当两个独立样本）
    prim = [r["primary_stat"] for r in prim_ok]
    n_eff, n_pos, n_neg, p_sign, n_zero = sign_test_nonzero(prim)
    med = median_upper(prim)
    c1 = bool(n_eff >= min_sessions and p_sign <= alpha and med > 0)
    # C2：退化守卫（exp90 的教训：塌缩后 d 仍是**非零常数** ⇒ 只看显著性会假阳性）
    if len(prim) > 1:
        spread = {"primary": {"spread": float(max(prim) - min(prim)),
                               "max_abs": float(max(abs(x) for x in prim))}}
    else:
        spread = {"primary": {"spread": 0.0, "max_abs": 0.0}}
    # 退化 = 统计量是**非零常数**（恒为 0 不算退化：那是没有信号 ⇒ 归 C1_FAIL）
    degenerate = (spread["primary"]["spread"] <= 1e-12
                  and spread["primary"]["max_abs"] > 1e-9)
    c2 = not degenerate
    # C3（只报）：所有可得方向的逐方向统计
    deltas = {}
    for key in sorted({k for r in per_session for k in r.get("pairs", {})}):
        vals = [r["pairs"][key]["delta"] for r in per_session if key in r.get("pairs", {})]
        if vals:
            ne, np_, nn, ps, nz = sign_test_nonzero(vals)
            deltas[key] = {"n": len(vals), "median": median_upper(vals),
                           "n_pos": np_, "p_sign": ps, "n_zero": nz}
    if len(prim) < min_sessions:
        verdict = "INSUFFICIENT_SESSIONS"
    elif not c2:
        verdict = "DEGENERATE"
    elif c1:
        verdict = "C1_OK"
    else:
        verdict = "C1_FAIL"
    if len(prim) < min_sessions:
        problems.append(f"进入主判据的 session 只有 {len(prim)} 个（要求 ≥ {min_sessions}）")
    if not c2:
        problems.append("退化守卫不过（主判据统计量是非零常数）⇒ 判决作废")
    if n_eff < min_sessions:
        problems.append(f"进入主判据的 session 只有 {n_eff} 个（要求 ≥ {min_sessions}）")
    return {
        "verdict": verdict,
        "C0_pipeline_selfcheck": len(prim) >= min_sessions,
        "C0_detail": {k: (v if isinstance(v, str) else list(v)) for k, v in c0_detail.items()},
        "n_sessions_primary_total": sum(1 for r in per_session
                                        if r.get("primary_stat") is not None),
        "n_sessions_primary_used": len(prim),
        "sessions_excluded_by_readability": prim_excluded,
        "C0_legacy_jointlabel": {"n_failing": len(legacy_fail),
                                  "n_primary_used": n_legacy_used,
                                  "sessions_failing": legacy_fail},
        "C1_has_discriminative_power": c1,
        "C1": {"n_eff": n_eff, "n_pos": n_pos, "n_neg": n_neg, "p": p_sign,
               "n_zero": n_zero, "median_delta": med, "min_sessions": min_sessions,
               "alpha": alpha, "primary_pair": "motor<->prefrontal",
               "primary_stats": prim},
        "C1_direction": {"n_pos": n_pos, "n_neg": n_neg,
                         "systematically_negative": bool(n_neg > n_pos),
                         "rule": "系统性为负 ⇒ 只能读作「这版移植无效」，"
                                 "**不得**读作「判据① 原理上没判别力」"},
        "C2_degenerate_guard": c2,
        "C2_spread": spread,
        "per_pair": deltas,
        "n_sessions_total": len(per_session),
        "n_sessions_primary": len(prim),
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
    chk("码本形状 = 类数×每类原型数（exp97b：每类 1 个 ⇒ 类条件均值）", P is not None and P.shape == (4 * K_PROTO_PER_CLASS, 8)
        and len(used) == 4)
    # 岭回归解码：可分的二类 ⇒ 准确率 ≈ 1
    X = np.vstack([rng.normal(-1, 0.3, (60, 5)), rng.normal(1, 0.3, (60, 5))])
    y = np.array([0] * 60 + [1] * 60)
    acc = decode_acc(X[:100], y[:100], X[100:], y[100:])
    chk("岭回归解码 可分二类 ⇒ 准确率 > 0.9", acc > 0.9)
    # ⚠️ **加速版 ridge_fit 必须与参考实现等价**（否则「提速」偷偷改了判据）
    W1, b1 = _ridge_fit_ref(X, np.eye(2)[y], 1.0)
    W2, b2 = ridge_fit(X, np.eye(2)[y], 1.0)
    chk("加速版 ridge_fit ≡ 参考实现（数值一致）",
        float(np.abs(W1 - W2).max()) < 1e-6
        and float(np.abs(np.asarray(b1).ravel() - np.asarray(b2).ravel()).max()) < 1e-6)
    # decide：四条分支
    def fake(delta, ok0=True, jitter=0.05, bad_band=None, bad_idx=None):
        rng = np.random.default_rng(7)
        ps = []
        for i in range(10):
            d = delta + (float(rng.normal(0, jitter)) if jitter else 0.0)
            rd = {}
            for b in ("motor", "prefrontal"):
                bad_here = (bad_band == b) and (bad_idx is None or i in bad_idx)
                ok = ok0 and not bad_here
                rd[b] = {"var": "choice", "acc": 0.9 if ok else 0.3,
                         "p": 0.001 if ok else 0.7, "ok": ok}
            ps.append({"session": f"s{i}", "skip": None if ok0 else "x",
                       "decode": {"visual": {"acc": 0.9, "p": 0.001}},
                       "readable": rd,
                       "primary_stat": d,
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
    chk("decide 全部不可读 ⇒ 剔完不够 ⇒ INSUFFICIENT_SESSIONS",
        v["verdict"] == "INSUFFICIENT_SESSIONS" and v["n_sessions_primary_used"] == 0)
    v = decide(fake(0.5, bad_band="motor", bad_idx={0, 1}))
    chk("decide 部分 session 不可读 ⇒ **只剔这些**、其余照常（8 个够 ⇒ C1_OK）",
        v["verdict"] == "C1_OK" and v["n_sessions_primary_used"] == 8
        and len(v["sessions_excluded_by_readability"]) == 2)
    v = decide(fake(0.5, jitter=0))
    chk("decide d 是**非零常数** ⇒ DEGENERATE（exp90 塌缩那个坑）",
        v["verdict"] == "DEGENERATE" and not v["C2_degenerate_guard"])

    # ⑥ **端到端**（合成 session ⇒ 整条管线跑通）
    #    ⚠️ 这条是补的：前面的自测只测零件（统计小工具/decide）⇒ **漏掉了**
    #    「联合 PCA 的拼接均值没按带切分」这个真 bug（多带 session 全部崩）。
    def fake_session(seed=0, n_tri=200, n_per=60):
        rng = np.random.default_rng(seed)
        # ⚠️ **类必须在时间上打散**（真实 Steinmetz 的 trial 是交错的）：
        #    若类连续排布 ⇒ 时间切分（70/30）会把类劈开 ⇒ 解码必然随机水平
        cls = rng.permutation(np.repeat(np.arange(4), n_tri // 4))
        pat_m = rng.normal(size=(4, n_per))
        pat_p = rng.normal(size=(4, n_per))
        feat = np.zeros((n_tri, 2 * n_per), dtype=np.float32)
        feat[:, :n_per] = pat_m[cls] + 0.6 * rng.normal(size=(n_tri, n_per))
        feat[:, n_per:] = pat_p[cls] + 0.6 * rng.normal(size=(n_tri, n_per))
        return {"feat": feat,
                "unit_band": np.array(["motor"] * n_per + ["prefrontal"] * n_per,
                                       dtype=object),
                # ⚠️ 类编码必须与 exp97a 的 `cls = (side>0)*2 + (choice>0)` **一致**
                #    ⇒ side: c≥2 为 +1；choice: c 为奇数时 +1（否则解码器看不到结构）
                "trial_choice": np.where(cls % 2 == 1, 1.0, -1.0),
                "trial_stim_side": np.where(cls >= 2, 1.0, -1.0),
                "trial_included": np.ones(n_tri, dtype=bool),
                "trial_stim_time": np.arange(n_tri, dtype=np.float64),
                "session": f"syn{seed}"}

    rec = session_transdiff(fake_session(0))
    chk("端到端：合成 session 产出两个方向且有主对统计量",
        len(rec.get("pairs", {})) == 2 and rec.get("primary_stat") is not None)
    chk("端到端：两带行为都可解码且高于机会",
        bool(rec.get("decode")) and all(v["acc"] > CHANCE for v in rec["decode"].values()))
    chk("端到端：两带模式不同 ⇒ 跨带 Δ > 0",
        rec["pairs"]["motor->prefrontal"]["delta"] > 0)
    v = decide([session_transdiff(fake_session(i)) for i in range(10)])
    chk("端到端：10 个合成 session ⇒ decide ⇒ C1_OK", v["verdict"] == "C1_OK")

    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/domains/steinmetz_trials")
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp97a")
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
    for i, fp in enumerate(files):
        try:
            print(f"  → [{i+1}/{len(files)}] {fp.stem}", flush=True)
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
                   "C0prime": "逐带可读性：每带用它自己相关的二值行为变量，acc>0.5 且置换 p<0.05；不过 ⇒ 该 session 被剔除（排除规则与 Δ 无关）",
                   "C1": "判据① 有没有判别力：Δ(跨带 − 同带) > 0，session 级符号检验 ≥8 session 且 p≤0.05 且中位 >0",
                   "C2": "退化守卫：d 的会话间离散度 > 0（不塌缩）",
                   "C3": "只报：3×3 矩阵 / 跨带解码迁移 / 逐带可解码准确率",
               },
               "per_session": per_session, "verdict": verdict}
    (out_dir / "exp97c_perband_labels.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n=== 判决：{verdict['verdict']} ===")
    print(f"  C0 自检：{verdict['C0_pipeline_selfcheck']}")
    print(f"  C1 判别力：{verdict['C1_has_discriminative_power']}"
          f"（n_eff={verdict['C1']['n_eff']}、n_pos={verdict['C1']['n_pos']}、"
          f"p={verdict['C1']['p']:.4f}、中位 Δ={verdict['C1']['median_delta']:+.4f}）")
    print(f"  C2 退化守卫：{verdict['C2_degenerate_guard']}")
    for k, v in verdict["per_pair"].items():
        print(f"    {k}: n={v['n']} 中位 Δ={v['median']:+.4f}")
    print(f"结果写入 {out_dir}/exp97a_transdiff_mouse.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
