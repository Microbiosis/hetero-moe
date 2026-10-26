#!/usr/bin/env python3
"""exp98_posneg_control.py — 判据① 的**正/负对照验收**（Allen Visual Coding–Neuropixels）。

## 这是什么（承接 §三十七·三 / §三十八·六）

exp97a/b/c 在鼠（Steinmetz）上都没能回答「**判据① 有没有判别力**」：
- exp97a：工具无效（码本过拟合 ⇒ Δ 符号混乱）
- exp97b：码本改成类条件均值后方向对了，但 **C0 判死**
- exp97c：C0′（每带用它自己的二值行为变量）把可用 session 从 2 个救到 3 个，**仍 < 8** ⇒ 判决 `INSUFFICIENT_SESSIONS`
⇒ ⇒ **Steinmetz 这条线用尽**（不是「没判别力」，是**测不了**）。

§三十七·三 指出的出路：**Allen Visual Coding–Neuropixels 同一个数据集里就有正/负对照**
⇒ ⇒ **这一轮（exp98）就是那个验收实验**：判据① 若真有效，则
**跨层级/跨类型的带对**（正对照）的转分化代价必须**显著大于**同层级的带对（负对照）。

## 为什么换数据集就等于换「对照维度」（**这是本实验的设计核心**）

| | Steinmetz（exp97c）| Allen VC-NP（本实验）|
|---|---|---|
| 带对 | motor↔prefrontal（**只有一对**）| **多对**：VISp↔VISl（**层级距离最小**）/ VISp↔VISam（**层级距离更大**）；只报：VISp↔VISrl / VISp↔VISal / VISp↔VISpm / VISp↔CA1 |
| 对照 | 只有一个跨带对 ⇒ **无法回答「判据有效吗」** | **同 session 内同时有「最小对比」与「更大对比」** ⇒ **可以回答** |
| 码本锚 | 行为类（选择/刺激侧）| **刺激身份（自然图像，**K = 118** 张）** ⇒ **与 exp90（人 EEG 的语义原型）同构** |

⇒ ⇒ 判据① 的**定义不变**（换一个原语的码本，读出掉多少），只是**码本锚换了**（人 EEG 那侧锚在图像语义上）。

## 对照的层级依据（**外部事实，注册源核准**）

来源：**Allen Institute 官方 Visual Coding 页面**
（`https://brain-map.org/our-research/circuits-behavior/visual-coding`，原文）：

> "Data was collected from primary visual cortex (VISp) and five higher visual areas
> (VISl, VISal, VISpm, VISam, VISrl)."
> "In our Neuropixels experiments, **the same set of cortical visual areas** are targeted
> as in the 2-Photon Imaging Brain Observatory … Spikes in the hippocampal formation
> (CA1, CA3, and dentate gyrus) are also measured routinely."
> "A set of **natural images** and movies are also presented …"
> "The first release of Neuropixels data contains two separate stimulus sets. The first,
> known as **"Brain Observatory 1.1"**" ⇒ 即 DANDI `000021`。

**由此得到两条对本实验至关重要的事实：**

1. **VISp 是本数据集里唯一的 primary**；VISl / VISal / VISpm / VISam / VISrl **全部**属于
   Allen 所说的 "higher visual areas" ⇒ ⇒ **本数据集里根本不存在「两个同层级的皮层区」**
   ⇒ 负对照**不能**写成「同层级对照」，只能写成「**层级距离最小**的一对」。
   ⚠️ 因此**不得**声称「VISp/VISl/VISrl = L1、VISam/VISal = AL」——**那是错的**（VISrl 也被 Allen
   归为 higher visual area）。层级归属以 Allen 官方措辞为准（见上）。
2. 自然图像**在**标准刺激集里 ⇒ 判据① 的码本锚有真实语义（图像身份），不是人造标签。

高阶视觉区的功能定位另见标准综述：Glickfeld & Steinmetz (2017) *Higher-Order Areas of the
Mouse Visual Cortex*, **Neuron** 94:435–450（PMID 28746815）。

### ⇒ **解释规则的不对称性（本实验最关键的一条预登记）**

因为负对照是「**最小对比**」而不是「同层级对比」，两种结局的**信息量不同**：

- **分离成功** ⇒ **强证据**（连最小对比都能分开 ⇒ 判据① 确实有判别力）。
- **分离失败** ⇒ **不能**据此说「判据① 没判别力」（对比太弱 ⇒ 只说明「在这个最小对比上测不出来」）
  ⇒ 必须同时看**正对照自身的 Δ 中位**（> 0 ⇒ 「方向对、检验力不足」；≤ 0 ⇒ 才叫「没判别力」）。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 通过 ⇒ |
|---|---|---|
| **C0′** | **逐带可读性**：每个带必须能把**刺激身份**（自然图像，**K = 118** 张；`frame` 实测，空白屏已剔除）解码到**显著高于机会**（准确率 > **1/118** 且**置换 p < 0.05**，n_perm=100）。**主对涉及的三个带（VISp/VISl/VISam）任一不过 ⇒ 该 session 剔除**（排除规则与 Δ 无关）⇒ 报出进入/剔除的 session 数与原因 | 可读性是**数据属性** |
| **C1** | **正/负对照分离（主判据）**：逐 session 统计量 = `Δ(VISp↔VISam)` − `Δ(VISp↔VISl)`（每个「↔」取**两个方向 Δ 的均值**；Δ = d(跨带) − d(同带劈半自对照)）⇒ 证据 = **session 级符号检验**（**≥ 8 个 session** ⇒ p ≤ 0.05）**且**中位 > 0 | **判据① 有判别力**（⇒ 为人 EEG 收 3×3 才有意义）|
| **C2** | **退化守卫**（exp90 的教训）：主判据统计量不得是**非零常数**（离散度 > 0；恒为 0 不算退化 ⇒ 那是没信号 ⇒ 归 `C1_FAIL`）| 判决有效 |
| **C3** | **只报不判**：① 其余带对（VISp↔VISrl、VISp↔VISal、VISp↔CA1、VISp↔VISpm）的 Δ；② 每带解码准确率；③ 逐 session 的正/负对照分项 | — |

### ⚠️ 预登记的**解释规则**（防过度解读，与 exp97a 的教训同款）

- `C1_OK` ⇒ **判据① 有判别力**（**保守但有力**：负对照是「最小对比」，连它都能分开）。
- `C1_FAIL` **必须**再看两件事才能定性：
  ① **正对照自身的 Δ 中位**（`Δ(VISp↔VISam)`）：若 **> 0** 而配对差不显著 ⇒ 只能读作「**方向对、检验力不足**」；
  ② 若正对照 Δ 中位 **≤ 0** ⇒ 才读作「**判据① 在这个数据上没判别力**」。
- ⚠️ **并且**：因为负对照是「**最小对比**」而不是「同层级对比」（见上），
  `C1_FAIL` **本身就不等于**「没判别力」⇒ 只能读作「**在这个最小对比上测不出来**」。
- ⇒ ⇒ **「工具无效」与「原理无效」必须分开记**（exp90/exp97a 的教训）。

## 数据（**注册源核准**）

- **DANDI `000021`** = *Allen Institute – Visual Coding – Neuropixels (Brain Observatory 1.1)*
  ⇒ **许可 `CC-BY-4.0`** + Allen Institute terms of use；**32 个 session**、214 个 NWB 资产、477.6 GB。
- **只读** session 级 NWB 里的 `units`（放电时间）、`general/extracellular_ephys/electrodes`（单元→脑区）、
  `intervals/natural_scenes_presentations`（刺激块）⇒ **不下载 LFP / waveform**。
- **不挑 session**：全部 32 个都抽 ⇒ 只有**数据层面**的排除（C0′ / 单元数）才剔除。

## 协议（预登记参数）

- **刺激块**：自然图像（`natural_scenes`）⇒ 5950 次呈现 / session，**每次呈现 250 ms**。
- **类别标签来源**：`stimulus_name` **全同（1 类）⇒ 不可用**；`stimulus_index` **恒为 5**（块号）⇒ **也不可用**；
  类别取 **`frame` 列**（**实测**：`frame` ∈ {−1, 0, …, 117}，**共 119 个取值**；`frame = −1` 是**空白屏**，
  50 次；`frame` 0–117 = **118 张自然图像**，各 50 次 ⇒ 118×50 + 50 = **5950** 次呈现）
  ⇒ ⇒ **剔除空白屏** ⇒ **K = 118**（抽取器把 `frame` 重标为连续 **0..117**；机会水平 = **1/118 ≈ 0.85%**）。
  ⚠️ 这个 K 是**数据事实**（探针 `exp98_frames.py` 实测，32 个 session 一致），不是事后挑的。
- **窗**：刺激后 **[0, 200] ms** ⇒ ⚠️ **与 exp97c 的 [50, 450] 不同**，因为本数据集的呈现只有 250 ms
  ⇒ 窗必须留在呈现内（**数据决定的必然适配**，留痕在此）。
- **bin = 10 ms**（只用于计数）。
- **train/test = 按时间前 70% / 后 30%**（不跨时间泄漏；刺激序列是伪随机的 ⇒ 两边都含全部类）。
- **共同空间 = 两带联合 PCA**（train trial，前 16 维）。
- **码本 = 类条件均值**（每类 1 原型 ⇒ 与 exp90 的语义原型、exp97b 的码本**同构**）。
- **每带最少单元 = 20**（沿用 exp97c）。
- **同带自对照**：把目标带的单元**随机劈两半** ⇒ `d(B1→B2)` = 地板（移植引入的新环节，必须有）。
- **只算需要的带对**（VISp 与其余 6 个区，两个方向 ⇒ 14 个有向对）⇒ 不算无关对（省算力，不改判据）。

## 统计口径（与 exp89/exp90/exp96/exp97c 逐字一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
- **零差剔除**（差值为 0 单独计数，不参与符号检验）。
- **统计单元 = session**（trial 不独立 ⇒ trial 级检验一律只作描述性）。

## 用法

```powershell
python -u research/brain_domains/exp/exp98_posneg_control.py --data /domains/allen_trials --out /domains/runs/illusionbrain-align/exp98
python -u research/brain_domains/results/parse_exp98.py     # 离线重算（与 Job 判决逐字比对）
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TAU = 0.1                # exp90 的 proto_temp
N_PC = 16                # 共同空间维数
K_PROTO_PER_CLASS = 1    # 每类 1 个原型 ⇒ 码本 = 类条件均值
MIN_TRIALS_PER_CLASS = 8
MIN_UNITS_PER_BAND = 20
N_PERM = 100             # 置换次数（可读性检验；比 exp97c 的 50 更细）
CHANCE = None            # 运行时按类数算
ALPHA_READ = 0.05

# **本实验用到的脑区**（Allen CCF 标签）
# ⚠️ 层级归属以 **Allen 官方措辞**为准：VISp = primary visual cortex；
#    VISl/VISal/VISpm/VISam/VISrl = "higher visual areas"；CA1 = hippocampal formation。
#    ⇒ **不得**再写「VISp/VISl/VISrl = L1、VISam/VISal = AL」（那是错的：VISrl 也是 higher）。
BANDS = ("VISp", "VISl", "VISrl", "VISam", "VISal", "VISpm", "CA1")
# 负对照 = **层级距离最小**的一对（本数据集没有「两个同层级皮层区」⇒ 只能取最小对比）
NEG_PAIRS = (("VISp", "VISl"),)
# 正对照 = **层级距离更大**的一对（VISp 的 primary ↔ VISam 的 higher，前部/内侧）
POS_PAIRS = (("VISp", "VISam"),)
REPORT_PAIRS = (("VISp", "VISrl"), ("VISp", "VISal"), ("VISp", "CA1"), ("VISp", "VISpm"))
# 只算这 7 对（**含两个方向** ⇒ 14 个有向对）⇒ 无关对不算（省算力；**不改判据**）
PAIR_SET = set()
for _pr in (NEG_PAIRS + POS_PAIRS + REPORT_PAIRS):
    PAIR_SET.add(_pr)
    PAIR_SET.add((_pr[1], _pr[0]))


def median_upper(xs):
    s = sorted(float(x) for x in xs if x == x)
    return s[len(s) // 2] if s else float("nan")


def sign_test_nonzero(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    zero = n - pos - neg
    k = min(pos, neg)
    # 精确双侧符号检验
    from math import comb
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)) if n else 1.0
    return n, pos, neg, p, zero


def ridge_fit(X, Y, lam=1.0):
    """岭回归（中心化 + 对偶形式）⇒ 快（trial 数 < 单元数时在 trial 空间解）。"""
    xm = X.mean(axis=0, keepdims=True)
    Xc = X - xm
    n, d = Xc.shape
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if n < d:
        A = Xc @ Xc.T + lam * np.eye(n)
        W = Xc.T @ np.linalg.solve(A, Yc)
    else:
        A = Xc.T @ Xc + lam * np.eye(d)
        W = np.linalg.solve(A, Xc.T @ Yc)
    b = Y.mean(axis=0, keepdims=True) - xm @ W
    return W, b


def ridge_predict(X, W, b):
    return X @ W + b


def perm_p(Xtr, ytr, Xte, yte, n_perm=N_PERM, lam=1.0, seed=0):
    """置换检验：打乱训练标签重拟合 ⇒ 准确率零分布。

    ⚠️ **实现修订（出结果前做的；判据一字未动）**：
    原版每个置换都做一次 `np.linalg.solve(A, ·)`（d×d 分解），
    而本平台的 LAPACK 在**小矩阵 + 多线程**下极慢（实测 280×280 的 solve ≈ 78 ms）
    ⇒ 单 session 要几十分钟 ⇒ 8 个分片 Job 21 分钟连一个 session 都没算完。

    ⇒ 改成**一次分解 + 矩阵乘**：`Ainv = inv(A)` 只算一次，
    每个置换改成 `W = Ainv @ (Xcᵀ(Y−ym))` ⇒ **数学上等价**（`inv` 与 `solve` 只差舍入），
    **判据不变**：置换次数、seed、阈值、准确率定义全同。
    ⇒ 等价性由 `--self-test` 的**对拍用例**保证（与参考实现 `perm_p_ref` 逐项比对）。
    """
    rng = np.random.default_rng(seed)
    xm = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - xm
    n, d = Xc.shape
    if n < d:
        # 对偶形式（本实验走不到：n_train=4130 > d ≤ 559）⇒ 保持原样
        A = Xc @ Xc.T + lam * np.eye(n)

        def acc_of(y):
            Y = np.eye(K)[y]
            ym = Y.mean(axis=0, keepdims=True)
            W = Xc.T @ np.linalg.solve(A, Y - ym)
            b = ym - xm @ W
            pred = (Xte @ W + b).argmax(axis=1)
            return float((pred == yte).mean())

        obs = acc_of(ytr)
        null = np.asarray([acc_of(rng.permutation(ytr)) for _ in range(n_perm)])
    else:
        A = Xc.T @ Xc + lam * np.eye(d)
        Ainv = np.linalg.inv(A)
        K = int(ytr.max()) + 1

        def acc_of(y):
            Y = np.eye(K)[y]
            ym = Y.mean(axis=0, keepdims=True)
            W = Ainv @ (Xc.T @ (Y - ym))
            b = ym - xm @ W
            pred = (Xte @ W + b).argmax(axis=1)
            return float((pred == yte).mean())

        obs = acc_of(ytr)
        null = np.asarray([acc_of(rng.permutation(ytr)) for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def perm_p_ref(Xtr, ytr, Xte, yte, n_perm=N_PERM, lam=1.0, seed=0):
    """**参考实现**（原版，慢）⇒ 只给 `--self-test` 做**对拍**用；不参与任何判决路径。"""
    rng = np.random.default_rng(seed)
    xm = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - xm
    n, d = Xc.shape
    if n < d:
        A = Xc @ Xc.T + lam * np.eye(n)
    else:
        A = Xc.T @ Xc + lam * np.eye(d)
    K = int(ytr.max()) + 1

    def acc_of(y):
        Y = np.eye(K)[y]
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


def joint_pca(XA, XB, k=N_PC):
    """两带联合 PCA（train trial）⇒ 共同空间。"""
    U = np.hstack([XA, XB])
    mu = U.mean(axis=0, keepdims=True)
    Uc = U - mu
    if Uc.shape[0] > 400:
        sel = np.random.default_rng(0).choice(Uc.shape[0], 400, replace=False)
        Uc = Uc[sel]
    try:
        _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
    except np.linalg.LinAlgError:
        Vt = np.linalg.pinv(Uc)[:k]
    return mu, Vt[:k].T


def build_codebook(z_X, cls, tr, n_classes, seed=0):
    """码本 = **类条件均值**（每类 1 原型）⇒ 与 exp90 的语义原型、exp97b 的码本同构。"""
    protos, used = [], []
    for c in range(n_classes):
        m = tr & (cls == c)
        if m.sum() < MIN_TRIALS_PER_CLASS:
            continue
        protos.append(z_X[m].mean(axis=0, keepdims=True))
        used.append(c)
    if not protos:
        return None, []
    return np.vstack(protos), used


def proto_read(a, P, tau=TAU):
    """exp90 的读出式（原式搬）：mean‖a − softmax(a·Pᵀ/τ)P‖²。"""
    if P is None or len(P) == 0:
        return float("nan")
    logits = a @ P.T / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    W = np.exp(logits)
    W /= W.sum(axis=1, keepdims=True)
    rec = W @ P
    return float(np.mean(np.sum((a - rec) ** 2, axis=1)))


def prep_session(z):
    feat = np.asarray(z["feat"], dtype=np.float32)
    area = np.asarray(z["unit_area"], dtype=object)
    lab = np.asarray(z["trial_label"], dtype=np.int64)
    t = np.asarray(z["trial_time"], dtype=np.float64)
    n_tri = feat.shape[0]
    order = np.argsort(t)
    n_tr = int(round(0.7 * n_tri))
    tr = np.zeros(n_tri, dtype=bool)
    tr[order[:n_tr]] = True
    te = ~tr
    bands = {}
    for b in BANDS:
        idx = np.flatnonzero(np.array([str(x) == b for x in area]))
        if len(idx) >= MIN_UNITS_PER_BAND:
            bands[b] = idx
    n_classes = int(lab.max()) + 1 if len(lab) else 0
    return feat, lab, tr, te, bands, str(z["session"]), n_classes


def session_posneg(z, seed=0):
    """⇒ 该 session 的正/负对照 Δ 与 C0′ 可读性。"""
    feat, lab, tr, te, bands, sess, n_classes = prep_session(z)
    out = {"session": sess, "bands": {b: int(len(v)) for b, v in bands.items()},
           "n_trials": int(feat.shape[0]), "n_train": int(tr.sum()),
           "n_test": int(te.sum()), "n_classes": n_classes,
           "readable": {}, "pairs": {}}
    need = set()
    for a, b in NEG_PAIRS + POS_PAIRS:
        need.update((a, b))
    if not need.issubset(set(bands)) or tr.sum() < 40 or te.sum() < 20:
        out["skip"] = "主对所需脑区不全或 trial 不够"
        return out
    chance = 1.0 / max(n_classes, 2)
    Z = {}
    for b, idx in bands.items():
        X = feat[:, idx].astype(np.float64)
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True) + 1e-9
        Z[b] = (X - mu) / sd
    # C0′：每带能不能把**刺激身份**解码到显著高于机会
    for b in bands:
        Xtr, ytr = Z[b][tr], lab[tr]
        Xte, yte = Z[b][te], lab[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            out["readable"][b] = {"acc": float("nan"), "p": float("nan"), "ok": False}
            continue
        acc, p = perm_p(Xtr, ytr, Xte, yte, seed=seed)
        out["readable"][b] = {"acc": acc, "p": p, "chance": chance,
                              "ok": bool(acc > chance and p < ALPHA_READ)}
    # 每个带对：共同空间 + 码本 + 转分化 + 同带自对照（**只算 PAIR_SET 里的对** ⇒ 省算力）
    for A in BANDS:
        for B in BANDS:
            if A == B or (A, B) not in PAIR_SET:
                continue
            if A not in bands or B not in bands:
                continue
            mu, W = joint_pca(Z[A][tr], Z[B][tr])
            nA = Z[A].shape[1]
            zA = (Z[A] - mu[:, :nA]) @ W[:nA, :]
            zB = (Z[B] - mu[:, nA:]) @ W[nA:, :]
            PA, usedA = build_codebook(zA, lab, tr, n_classes, seed=seed)
            PB, usedB = build_codebook(zB, lab, tr, n_classes, seed=seed + 100)
            if PA is None or PB is None:
                continue
            used = sorted(set(usedA) & set(usedB))
            if len(used) < 2:
                continue
            PAu = np.vstack([PA[i] for i in range(len(used))])
            PBu = np.vstack([PB[i] for i in range(len(used))])
            r_own = proto_read(zB[te], PBu)
            r_other = proto_read(zB[te], PAu)
            d_cross = r_other - r_own
            rng = np.random.default_rng(seed + 7)
            idxB = bands[B]
            perm = rng.permutation(len(idxB))
            h = len(idxB) // 2
            loc1, loc2 = perm[:h], perm[h:]
            mu2, W2 = joint_pca(Z[B][tr], Z[B][tr])
            nB = Z[B].shape[1]
            muB = mu2[:, :nB]
            WB = W2[:nB, :]
            z1 = (Z[B][:, loc1] - muB[:, loc1]) @ WB[loc1, :]
            z2 = (Z[B][:, loc2] - muB[:, loc2]) @ WB[loc2, :]
            P1, u1 = build_codebook(z1, lab, tr, n_classes, seed=seed + 11)
            P2, u2 = build_codebook(z2, lab, tr, n_classes, seed=seed + 13)
            if P1 is None or P2 is None:
                continue
            used2 = sorted(set(u1) & set(u2))
            if len(used2) < 2:
                continue
            c1 = np.vstack([P1[i] for i in range(len(used2))])
            c2 = np.vstack([P2[i] for i in range(len(used2))])
            d_ctrl = proto_read(z2[te], c1) - proto_read(z2[te], c2)
            out["pairs"][f"{A}->{B}"] = {
                "d_cross": d_cross, "d_ctrl": d_ctrl, "delta": d_cross - d_ctrl,
                "n_classes": len(used), "n_units_B": int(len(idxB))}
    # 逐 session 统计量：正对照均值 − 负对照均值
    def pair_mean(a, b):
        vals = []
        for key in (f"{a}->{b}", f"{b}->{a}"):
            d = (out["pairs"].get(key) or {}).get("delta")
            if d is not None and d == d:
                vals.append(d)
        return float(np.mean(vals)) if vals else None
    neg = [pair_mean(a, b) for a, b in NEG_PAIRS]
    pos = [pair_mean(a, b) for a, b in POS_PAIRS]
    neg = [v for v in neg if v is not None]
    pos = [v for v in pos if v is not None]
    out["neg_stat"] = float(np.mean(neg)) if neg else None
    out["pos_stat"] = float(np.mean(pos)) if pos else None
    if out["neg_stat"] is not None and out["pos_stat"] is not None:
        out["primary_stat"] = out["pos_stat"] - out["neg_stat"]
    return out


def decide(per_session, min_sessions=8, alpha=0.05):
    """从逐 session 原始记录重算判决（阈值的唯一出处 ⇒ 可离线重算）。"""
    problems = []
    used, excluded = [], []
    for r in per_session:
        if r.get("skip"):
            excluded.append((r.get("session"), r.get("skip")))
            continue
        need = set()
        for a, b in NEG_PAIRS + POS_PAIRS:
            need.update((a, b))
        rd = r.get("readable") or {}
        bad = [f"{b}: acc={rd.get(b, {}).get('acc', float('nan')):.3f} "
               f"p={rd.get(b, {}).get('p', float('nan')):.3f}"
               for b in sorted(need) if not rd.get(b, {}).get("ok")]
        if bad or r.get("primary_stat") is None:
            excluded.append((r.get("session"), "；".join(bad) or "缺主对统计量"))
            continue
        used.append(r)
    prim = [r["primary_stat"] for r in used]
    n_eff, n_pos, n_neg, p_sign, n_zero = sign_test_nonzero(prim)
    med = median_upper(prim)
    c1 = bool(n_eff >= min_sessions and p_sign <= alpha and med > 0)
    if len(prim) > 1:
        spread = float(max(prim) - min(prim))
        max_abs = float(max(abs(x) for x in prim))
    else:
        spread, max_abs = 0.0, 0.0
    degenerate = (spread <= 1e-12 and max_abs > 1e-9)
    c2 = not degenerate
    if len(prim) < min_sessions:
        verdict = "INSUFFICIENT_SESSIONS"
    elif not c2:
        verdict = "DEGENERATE"
    elif c1:
        verdict = "C1_OK"
    else:
        verdict = "C1_FAIL"
    # 正对照自身的 Δ 中位（解释规则用）
    pos_vals = [r["pos_stat"] for r in used if r.get("pos_stat") is not None]
    neg_vals = [r["neg_stat"] for r in used if r.get("neg_stat") is not None]
    pos_med = median_upper(pos_vals)
    neg_med = median_upper(neg_vals)
    if verdict == "C1_FAIL":
        if pos_med == pos_med and pos_med > 0:
            interp = "方向对、检验力不足（**不得**读作「判据① 没判别力」）"
        else:
            interp = "正对照 Δ 中位 ≤ 0 ⇒ 读作「判据① 在这个数据上没判别力」"
    elif verdict == "C1_OK":
        interp = "判据① **有判别力**（正对照 Δ 显著大于负对照）"
    else:
        interp = "判决未成立（session 数不足 / 退化）"
    # 只报项：其余带对
    per_pair = {}
    for key in sorted({k for r in per_session for k in (r.get("pairs") or {})}):
        vals = [r["pairs"][key]["delta"] for r in per_session if key in (r.get("pairs") or {})]
        if vals:
            ne, np_, nn, ps, nz = sign_test_nonzero(vals)
            per_pair[key] = {"n": len(vals), "median": median_upper(vals),
                             "n_pos": np_, "p_sign": ps, "n_zero": nz}
    if len(prim) < min_sessions:
        problems.append(f"进入主判据的 session 只有 {len(prim)} 个（要求 ≥ {min_sessions}）")
    if not c2:
        problems.append("退化守卫不过（主判据统计量是非零常数）⇒ 判决作废")
    return {
        "verdict": verdict,
        "C0_pipeline_selfcheck": len(prim) >= min_sessions,
        "n_sessions_total": len(per_session),
        "n_sessions_primary_used": len(prim),
        "sessions_excluded": [{"session": s, "reason": w} for s, w in excluded],
        "C1_has_discriminative_power": c1,
        "C1": {"n_eff": n_eff, "n_pos": n_pos, "n_neg": n_neg, "p": p_sign,
               "n_zero": n_zero, "median_delta": med, "min_sessions": min_sessions,
               "alpha": alpha, "neg_pair": "VISp<->VISl", "pos_pair": "VISp<->VISam",
               "primary_stats": prim},
        "C1_pos_median": pos_med,
        "C1_neg_median": neg_med,
        "C1_interpretation": interp,
        "C2_degenerate_guard": c2,
        "C2_spread": {"primary": {"spread": spread, "max_abs": max_abs}},
        "per_pair": per_pair,
        "problems": problems,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/domains/allen_trials")
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp98")
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
        print("✗ 没有抽取结果 ⇒ 先跑 exp98_extract.py")
        return 1
    per_session = []
    for i, fp in enumerate(files):
        try:
            print(f"  → [{i+1}/{len(files)}] {fp.stem}", flush=True)
            z = np.load(fp, allow_pickle=True)
            rec = session_posneg(z)
            per_session.append(rec)
            print(f"  ✓ {rec['session']}：带 {rec['bands']}；"
                  f"可读 { {b: v.get('ok') for b, v in rec.get('readable', {}).items()} }；"
                  f"负 {rec.get('neg_stat')} 正 {rec.get('pos_stat')} "
                  f"主统计 {rec.get('primary_stat')}", flush=True)
        except Exception as e:
            print(f"  ✗ {fp.name}: {type(e).__name__}: {e}", flush=True)
    verdict = decide(per_session)
    payload = {"config": vars(args), "criteria": {
        "C0prime": "逐带可读性：能把刺激身份（118 张自然图像，空白屏已剔除）解码到显著高于机会"
                   "（acc>1/118 且置换 p<0.05）；主对三区任一不过 ⇒ 该 session 剔除（排除规则与 Δ 无关）",
        "C1": "正/负对照分离：逐 session 统计量 = Δ(VISp↔VISam) − Δ(VISp↔VISl)；"
              "session 级符号检验 ≥8 session 且 p≤0.05 且中位>0",
        "C2": "退化守卫：主判据统计量不得是非零常数",
        "C3": "只报：其余带对（VISp↔VISrl/VISal/CA1/VISpm）、每带解码准确率",
    }, "per_session": per_session, "verdict": verdict}
    (out_dir / "exp98_posneg_control.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # 回显通道（与 exp97c 同款）：**逐 session 原始记录 + 判决** ⇒ 桶写失败也能离线重算
    for _r in per_session:
        print("===REC===" + json.dumps(_r, ensure_ascii=False, default=str), flush=True)
    print("===VERDICT===" + json.dumps(verdict, ensure_ascii=False, default=str), flush=True)
    print(f"\n=== 判决：{verdict['verdict']} ===")
    print(f"  进入主判据：{verdict['n_sessions_primary_used']} / {verdict['n_sessions_total']}")
    print(f"  符号检验：{verdict['C1']['n_pos']}/{verdict['C1']['n_eff']} 正，"
          f"p={verdict['C1']['p']:.4f}，中位 Δ={verdict['C1']['median_delta']:+.4f}")
    print(f"  正对照中位 Δ={verdict['C1_pos_median']:+.4f}；"
          f"负对照中位 Δ={verdict['C1_neg_median']:+.4f}")
    print(f"  解释：{verdict['C1_interpretation']}")
    print(f"  C2 退化守卫：{verdict['C2_degenerate_guard']}")
    print(f"结果写入 {out_dir}/exp98_posneg_control.json")
    return 0


def self_test():
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # ① 统计小工具
    chk("sign_test 全正 n=3 ⇒ p=0.25", abs(sign_test_nonzero([1, 2, 3])[3] - 0.25) < 1e-12)
    chk("sign_test 全负 n=3 ⇒ p=0.25", abs(sign_test_nonzero([-1, -2, -3])[3] - 0.25) < 1e-12)
    chk("sign_test 全正 n=8 ⇒ p=0.0078",
        abs(sign_test_nonzero([1] * 8)[3] - 2 / 256) < 1e-12)
    chk("中位 = 上中位（偶数取偏大）", median_upper([1, 2, 3, 4]) == 3)

    # ② 读出/解码零件
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 0.3, (60, 5)), rng.normal(1, 0.3, (60, 5))])
    y = np.array([0] * 60 + [1] * 60)
    W, b = ridge_fit(X, np.eye(2)[y], 1.0)
    acc = float((ridge_predict(X, W, b).argmax(axis=1) == y).mean())
    chk("岭回归可分二类 ⇒ 准确率 > 0.9", acc > 0.9)
    # proto_read：**同一份数据**下，真原型（类均值）的残差必须小于乱原型
    cls4 = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cent = rng.normal(size=(4, 8))
    a = cent[cls4] + 0.1 * rng.normal(size=(8, 8))
    chk("proto_read：真原型残差 < 乱原型残差",
        proto_read(a, cent) < proto_read(a, rng.normal(size=(4, 8))))
    # ⑦ **perm_p 优化版 vs 参考实现**（对拍 ⇒ 等价性必须机器可判）
    Xr = rng.normal(size=(240, 14))
    yr = rng.integers(0, 5, 240)
    Xs = rng.normal(size=(80, 14))
    ys = rng.integers(0, 5, 80)
    o1, p1 = perm_p(Xr, yr, Xs, ys, n_perm=30)
    o2, p2 = perm_p_ref(Xr, yr, Xs, ys, n_perm=30)
    chk("perm_p 优化版与参考实现**逐位一致**（准确率）", o1 == o2)
    chk("perm_p 优化版与参考实现**置换 p 一致**", p1 == p2)
    # 信号更强时两者也要一致（准确率更高 ⇒ 不许因为实现差异翻面）
    ysig = rng.integers(0, 3, 240)
    Xsig = rng.normal(size=(240, 14)) + np.eye(3, 14)[ysig] * 1.5
    o3, p3 = perm_p(Xsig, ysig, Xs, ys, n_perm=30)
    o4, p4 = perm_p_ref(Xsig, ysig, Xs, ys, n_perm=30)
    chk("perm_p 优化版在**有信号**时也与参考实现一致", o3 == o4 and p3 == p4)

    # ③ decide() 的判决分支（**判决口径逐条测**）
    def fake_rec(stat=1.0, pos=None, neg=None, ok=True, jitter=0.3, seed=0):
        r = np.random.default_rng(seed)
        rd = {b: {"acc": 0.5 if ok else 0.02, "p": 0.001 if ok else 0.9, "ok": ok}
              for b in ("VISp", "VISl", "VISam")}
        p = stat if pos is None else pos
        n = stat if neg is None else neg
        return {"session": f"f{seed}", "readable": rd,
                "pairs": {"VISp->VISl": {"delta": n}, "VISl->VISp": {"delta": n},
                          "VISp->VISam": {"delta": p}, "VISam->VISp": {"delta": p}},
                "neg_stat": n, "pos_stat": p,
                "primary_stat": stat + jitter * r.normal()}

    v = decide([fake_rec(1.0, seed=i) for i in range(10)])
    chk("decide 10 个 session 全正 ⇒ C1_OK",
        v["verdict"] == "C1_OK" and v["C1_has_discriminative_power"])
    v = decide([fake_rec(-1.0, seed=i) for i in range(10)])
    chk("decide 全负 ⇒ C1_FAIL", v["verdict"] == "C1_FAIL")
    v = decide([fake_rec(0.0, seed=i) for i in range(10)])
    chk("decide 全零 ⇒ C1_FAIL（零差不参与符号检验）", v["verdict"] == "C1_FAIL")
    v = decide([fake_rec(1.0, jitter=0.0, seed=i) for i in range(10)])
    chk("decide 主统计量是**非零常数** ⇒ DEGENERATE（exp90 塌缩那个坑）",
        v["verdict"] == "DEGENERATE" and not v["C2_degenerate_guard"])
    v = decide([fake_rec(1.0, seed=i) for i in range(5)])
    chk("decide 只有 5 个 session ⇒ INSUFFICIENT_SESSIONS",
        v["verdict"] == "INSUFFICIENT_SESSIONS" and v["n_sessions_primary_used"] == 5)
    recs = [fake_rec(1.0, seed=i) for i in range(10)]
    recs[0]["readable"]["VISam"]["ok"] = False
    v = decide(recs)
    chk("decide 一个带不可读 ⇒ **只剔这个 session**、其余照常（9 个够 ⇒ C1_OK）",
        v["verdict"] == "C1_OK" and v["n_sessions_primary_used"] == 9
        and len(v["sessions_excluded"]) == 1)
    v = decide([fake_rec(-1.0, seed=i) for i in range(10)])
    chk("decide C1_FAIL 时给出解释（方向对/检验力不足 vs 没判别力）",
        isinstance(v["C1_interpretation"], str) and "正对照" in v["C1_interpretation"])

    # ④ **端到端**：合成 session（VISp/VISl 同编码 ⇒ 转分化便宜；VISam 编码被线性混合
    #    ⇒ 转分化贵，但**自己仍可解码** ⇒ 不会踩 C0′ 的可读性门槛）⇒ 主统计量 > 0。
    def fake_session(seed=0, n_tri=400, n_per=40, k=8, mix=True):
        rng = np.random.default_rng(seed)
        cls = rng.permutation(np.repeat(np.arange(k), n_tri // k))
        base = rng.normal(size=(k, n_per))
        M = rng.normal(size=(n_per, n_per))
        pats = [base, base, (base @ M.T) if mix else base]
        feat = np.zeros((n_tri, 3 * n_per), dtype=np.float32)
        for bi, pat in enumerate(pats):
            feat[:, bi * n_per:(bi + 1) * n_per] = pat[cls] + 0.6 * rng.normal(size=(n_tri, n_per))
        return {"feat": feat,
                "unit_area": np.array(["VISp"] * n_per + ["VISl"] * n_per + ["VISam"] * n_per,
                                       dtype=object),
                "trial_label": cls,
                "trial_time": np.arange(n_tri, dtype=np.float64),
                "session": f"syn{seed}"}

    rec = session_posneg(fake_session(0))
    chk("端到端：合成 session 产出正/负统计量",
        rec.get("neg_stat") is not None and rec.get("pos_stat") is not None
        and rec.get("primary_stat") is not None)
    chk("端到端：合成 session 的三个带都可解码（C0′ 不拦）",
        all(rec["readable"][b]["ok"] for b in ("VISp", "VISl", "VISam")))
    chk("端到端：**混合编码 ⇒ 正对照 Δ > 负对照 Δ**（主统计量 > 0）",
        rec["primary_stat"] > 0)
    rec_same = session_posneg(fake_session(0, mix=False))
    chk("端到端：**同编码 ⇒ 主统计量 ≈ 0**（不会凭空分出对照）",
        abs(rec_same["primary_stat"]) < abs(rec["primary_stat"]))
    v = decide([session_posneg(fake_session(i)) for i in range(10)])
    chk("端到端：10 个混合编码 session ⇒ decide ⇒ C1_OK", v["verdict"] == "C1_OK")
    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
