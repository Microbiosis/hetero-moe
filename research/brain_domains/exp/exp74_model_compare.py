#!/usr/bin/env python3
"""exp74 — 多模型特征对比：ViT-H-14 vs ViT-B-32（同一套 200 选 1 解码协议）

## 要回答的问题

> **更好的视觉特征 ⇒ 更好的脑解码吗？**

桶内已存在**两个** CLIP 在 THINGS-EEG 图像上的特征：

| 特征集 | 文件 | 维度 | 模型 |
|---|---|---:|---|
| ViT-H-14 | `eeg/eeg-image-decode/ViT-H-14_features_{train,test}.pt` | 1024 | CLIP ViT-H/14 |
| ViT-B-32 | `eeg/things-eeg/Preprocessed_data_250Hz_whiten/ViT-B-32_features_{train,test}.pt` | 512 | CLIP ViT-B/32 |

两者同族、同训练数据，计算量差 ~144×（632 vs 4.4 GFLOP/图），维度差 2×。
**两种结果都有意义**：

- ViT-H-14 明显更好 ⇒ 特征质量确实决定脑解码质量（与"更大的模型更脑对齐"的文献一致）；
- 两者相当 ⇒ **脑读出已经饱和**，瓶颈在 EEG 侧而不在特征侧
  （这本身就是一条重要结论：换更强的视觉模型救不了这件事）。

**不需要任何新的特征提取** —— 换个特征文件重跑同一套管线即可。

## 前置：ViT-B-32 的行序**从未被验证过**

这正是当年"缺陷 C（概念顺序与特征顺序的对齐从未验证）"的同类风险。
若行序错位，ViT-B-32 的解码准确率会掉到随机，届时**无法区分**
"模型更弱"与"行序错位"⇒ 结论会变成一句废话。

所以本脚本先做**只用特征、不碰 EEG**的对齐审计：

1. **对角线检验**：ViT-H-14 与 ViT-B-32 逐行配对的余弦相似度
   vs 随机配对。ViT-H-14 与 EEG 的 1:1 对齐已被 exp64/66/68p 独立验证
   （顺序对齐 p=0.005、522σ），因此**若两特征集逐行对齐，ViT-B-32 必然也对齐**。
2. **概念组检验**：训练集的概念 = 图像清单里的**连续 10 张一组**
   （exp73b 已从桶内清单直接核实）。若某特征集的行序与图像清单一致，
   则"连续 10 行的组内平均相似度"应显著高于随机配对。

两个检验都只用特征 ⇒ **不依赖 EEG，也不依赖任何未验证的假设**。

## 解码对比

完全沿用 exp73 的管线（`import exp73_decode200` 复用同一份实现，
保证两个模型走的是**同一套代码**，不是"看起来一样"的两套代码）：

- 估计量：试次级 ridge，`A0 = G_full / n_rep`（exp73a 审计确定）
- 设计维度：全部 630 维（不做窗口选择）
- α：train split 图像级 5-fold CV（单试次预测的 R²），不用 test
- 零分布：置换 train 的「图像 ↔ EEG」对应关系，8 次
- 指标：单试次 top-1/top-5、80 次平均后 top-1

**配对比较**：同一被试上比两个模型 ⇒ 用**配对符号检验**（n=10），
而不是比两个中位数（那是把 10 个被试当 20 个独立样本）。

## 合成自测

1. 对齐审计：构造"行对齐"与"行打乱"两组特征 ⇒ 对角线检验必须只对前者报警；
2. 概念组检验：构造"组内相似"与"随机"两组 ⇒ 必须区分；
3. 配对比较：构造"模型 A 恒优"与"无差异"两组 ⇒ 符号检验必须给对结论。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp73_decode200 as E  # noqa: E402  复用同一份解码实现

FS = 250.0
N_NULL = 8
ALPHAS = E.ALPHAS
SEED = E.SEED

TRAIN_4D = E.TRAIN_4D
TEST_4D = E.TEST_4D
OUT_DIR = E.OUT_DIR

FEATURE_SETS = {
    "ViT-H-14": ("/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt",
                 "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"),
    "ViT-B-32": ("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/ViT-B-32_features_train.pt",
                 "/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/ViT-B-32_features_test.pt"),
}

GROUP = 10          # 训练集：每概念 10 张连续（exp73b 已核实）


def log(*a):
    print(*a, flush=True)


def load_matrix(path, expect_rows):
    """读「矩阵式」特征文件（ViT-H-14：键 → (N,d) 张量）。

    兼容不同 key 名；形状必须与 expect_rows 一致。
    """
    import torch
    f = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(f, dict):
        key = None
        for k in ("img_features", "features", "image_features"):
            v = f.get(k)
            if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == expect_rows:
                key = k
                break
        if key is None:
            for k, v in f.items():
                if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == expect_rows:
                    key = k
                    break
        if key is None:
            raise KeyError(f"{path} 里找不到 {expect_rows} 行的特征矩阵；键={list(f)[:8]}")
        arr = f[key].numpy()
    else:
        arr = f.numpy()
    arr = arr.astype(np.float64)
    if arr.shape[0] != expect_rows:
        raise ValueError(f"{path}: 行数 {arr.shape[0]} ≠ 期望 {expect_rows}")
    return arr, {"kind": "matrix", "shape": tuple(arr.shape)}


PATH_RE = re.compile(
    r"^(?P<split>[A-Za-z_]*images)/(?P<cid>\d+)_(?P<cname>[^/]+)/(?P<file>[^/]+)$")


def path_dict_to_matrix(d, expect_rows, expect_per_concept=None, label=""):
    """路径字典 → (N,d) 矩阵，行序由 (概念号, 文件名) 排序推导；并校验结构。

    `expect_per_concept=None` 时**自行推断**每概念图数（只要求各概念一致），
    避免把"train 10 张 / test 1 张"这种差异写死。
    """
    if len(d) != expect_rows:
        raise ValueError(f"{label}: 图像数 {len(d)} ≠ 期望 {expect_rows}")
    items = []
    for k, v in d.items():
        m = PATH_RE.match(str(k))
        if not m:
            raise ValueError(f"{label}: 图像路径不符合预期格式：{k!r}")
        items.append((int(m.group("cid")), m.group("file"), str(k), v))
    items.sort(key=lambda t: (t[0], t[1]))          # == 桶内清单的字典序

    cids = sorted({t[0] for t in items})
    if cids != list(range(1, len(cids) + 1)):
        raise ValueError(f"{label}: 概念号不连续（{cids[:3]}…{cids[-3:]}）")
    per = {}
    for cid, _f, _k, _v in items:
        per[cid] = per.get(cid, 0) + 1
    counts = set(per.values())
    if len(counts) != 1:
        bad = [c for c, n in per.items() if n != counts.most_common(1)[0][0]][:5]
        raise ValueError(f"{label}: 各概念的图数不一致（{sorted(counts)}），异常概念 {bad}")
    inferred = counts.pop()
    if expect_per_concept is not None and inferred != expect_per_concept:
        raise ValueError(f"{label}: 每概念 {inferred} 图 ≠ 期望 {expect_per_concept}")
    if len(cids) * inferred != expect_rows:
        raise ValueError(f"{label}: {len(cids)} 概念 × {inferred} 图 ≠ {expect_rows} 行")

    dim = int(items[0][3].shape[-1])
    arr = np.stack([t[3].numpy() for t in items]).astype(np.float64)
    if arr.shape != (expect_rows, dim):
        raise ValueError(f"{label}: 拼出来是 {arr.shape}，期望 {(expect_rows, dim)}")
    meta = {"kind": "path_dict", "shape": tuple(arr.shape),
            "n_concepts": len(cids), "per_concept": inferred,
            "dim": dim, "concept_id_range": [cids[0], cids[-1]],
            "first_key": items[0][2], "last_key": items[-1][2]}
    return arr, meta


def load_path_dict(path, expect_rows, expect_per_concept):
    """读「路径字典式」特征文件（ViT-B-32：'image_features' = {图片路径 → 向量}）。

    关键：**行序不是假定的，而是由图片路径推导出来的**。
    排序键 = (概念号, 文件名)，与桶内 `Image_set` 清单的字典序完全一致
    （概念号 5 位补零 ⇒ 字典序 = 数值序），而 EEG 的行序就是这份清单的顺序
    （exp64/66/68p 已验证：ViT-H-14 与 EEG 1:1，图像路径为 `test_images/00002_antelope/...`）。

    ⇒ 因此这里得到的是**按名字核对**的对齐，而不是"位置刚好对上了"。
    同时校验结构：概念数、每概念图数、概念号是否连续 —— 任一不符直接报错。
    """
    import torch
    f = torch.load(path, map_location="cpu", weights_only=False)
    d = None
    for k in ("image_features", "img_features", "features"):
        if isinstance(f.get(k), dict):
            d = f[k]
            break
    if d is None:
        raise KeyError(f"{path} 里找不到 image_features 路径字典；顶层键={list(f)[:8]}")
    return path_dict_to_matrix(d, expect_rows, expect_per_concept, label=str(path))


# ---------------------------------------------------------------------------
# 对齐审计（只用特征，不碰 EEG）
# ---------------------------------------------------------------------------
def _unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def cross_predict_test(A, B, rng, n_perm=5, fold=0.8):
    """A 与 B 是否**逐行**对应（**维度可以不同**）。

    ViT-H-14 是 1024 维、ViT-B-32 是 512 维 ⇒ 没法逐元素做对角线检验。
    改用维度无关的**交叉预测**：用 B 线性预测 A（ridge，留出行评估 R²），
    对照组 = 把 B 的行打乱后做同样的事。
    ⇒ 若两套特征逐行对应，真配对的留出 R² 应远高于打乱对照。

    两个分支用**完全相同**的 λ 网格与折分，比较才公平。
    """
    n = len(A)
    Bn = _unit(B)
    idx = rng.permutation(n)
    n_tr = int(fold * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    mu_b = Bn[tr].mean(axis=0)
    sd_b = Bn[tr].std(axis=0) + 1e-9
    mu_a = A[tr].mean(axis=0)
    Ytr = A[tr] - mu_a
    Btr = (Bn[tr] - mu_b) / sd_b
    G = Btr.T @ Btr
    tr_scale = float(np.trace(G)) / G.shape[0]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    def r2_of(Bx_rows):
        Bx = (Bx_rows - mu_b) / sd_b
        Btr_ = Bx[tr]
        Bte_ = Bx[te]
        G_ = Btr_.T @ Btr_
        best = -np.inf
        for lam in lams:
            W = np.linalg.solve(G_ + lam * tr_scale * np.eye(G.shape[0]),
                                Btr_.T @ Ytr)
            pred = Bte_ @ W + mu_a
            ss_res = float(((A[te] - pred) ** 2).sum())
            ss_tot = float(((A[te] - mu_a) ** 2).sum())
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            best = max(best, r2)
        return best

    r2_true = r2_of(Bn)
    nulls = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        nulls.append(r2_of(Bn[p]))
    z = (r2_true - float(np.mean(nulls))) / (float(np.std(nulls)) + 1e-12)
    return {"r2_true": float(r2_true),
            "null_mean": float(np.mean(nulls)), "null_std": float(np.std(nulls)),
            "z": float(z), "n_perm": n_perm, "fold": fold}


def concept_group_test(F, rng, n_pairs=20000):
    """训练集的概念 = 连续 10 张一组。组内相似度应显著高于随机配对。

    返回 z 分数。**只用特征**，因此不需要 EEG。
    """
    Fn = _unit(F)
    n = len(Fn)
    n_grp = n // GROUP
    # 组内：每组取前 5 对（固定偏移，避免 O(n²)）
    within = []
    for g in range(n_grp):
        base = g * GROUP
        for o in range(1, GROUP):
            within.append(float(Fn[base] @ Fn[base + o]))
    within = np.asarray(within)
    # 随机：跨组随机配对
    gi = rng.integers(0, n_grp, n_pairs)
    gj = (gi + 1 + rng.integers(0, n_grp - 1, n_pairs)) % n_grp
    i = gi * GROUP
    j = gj * GROUP
    rand = np.einsum("ij,ij->i", Fn[i], Fn[j])
    z = (within.mean() - rand.mean()) / (rand.std() + 1e-12)
    return {"within_mean": float(within.mean()),
            "random_mean": float(rand.mean()),
            "random_std": float(rand.std()),
            "z": float(z),
            "n_within": int(len(within))}


def alignment_audit(FH, FB, rng):
    """两套特征的对齐审计（feature-only）。"""
    out = {}
    out["cross_predict_HB"] = cross_predict_test(FH, FB, rng)
    out["concept_group_H"] = concept_group_test(FH, rng)
    out["concept_group_B"] = concept_group_test(FB, rng)
    return out


# ---------------------------------------------------------------------------
# 解码对比
# ---------------------------------------------------------------------------
def run_subject(sid, feats, rng):
    """feats = {model_name: (V_train, V_test)}；跑完所有模型。"""
    t0 = time.time()
    tr_fp = TRAIN_4D / f"sub-{sid:02d}.npz"
    te_fp = TEST_4D / f"sub-{sid:02d}.npz"
    if not tr_fp.exists() or not te_fp.exists():
        return None

    eeg = np.load(tr_fp)["eeg"]
    Xrep = E.per_rep_targets(eeg)
    del eeg
    mu, sd = E.zscore_fit(Xrep)
    Xrep = E.zscore_apply(Xrep, mu, sd)

    eeg = np.load(te_fp)["eeg"]
    Xte = E.per_rep_targets(eeg)
    del eeg
    Xte = E.zscore_apply(Xte, mu, sd)

    out = {"subject": f"sub-{sid:02d}", "models": {}}
    for name, (V, Vtest) in feats.items():
        r = E.fit_and_decode(Xrep, Xte, V, Vtest, None, ALPHAS, rng, n_null=N_NULL)
        out["models"][name] = r
        log(f"  sub-{sid:02d} [{name:10s}] α={r['alpha']:g} "
            f"单试次 acc1_cos={r['acc1_cos']:.4f} acc5_cos={r['acc5_cos']:.4f} "
            f"dot={r['acc1_dot']:.4f} | 80次平均={r['acc1_featavg_cos']:.4f} "
            f"| 零中位={np.median(r['null_acc1_cos']):.4f}  [{time.time()-t0:.0f}s]")
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# 配对比较
# ---------------------------------------------------------------------------
def sign_test(diffs):
    """配对符号检验：H0 = 差值的中位为 0。

    用**标准精确二项检验**（与 R 的 `binom.test(alternative="two.sided")` 同义）：
    把所有"概率 ≤ 观测概率"的结局概率求和。
    （不要用"单尾 × 2"的加倍法：非对称情形下它与标准定义不同。）
    """
    from math import comb
    d = np.asarray(diffs)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "p_exact_two_sided": 1.0}
    k = int((d > 0).sum())
    tot = 2 ** n
    p_obs = comb(n, k) / tot
    p = sum(comb(n, i) / tot for i in range(n + 1)
            if comb(n, i) / tot <= p_obs + 1e-15)
    return {"n": n, "n_pos": k, "n_neg": n - k,
            "p_exact_two_sided": float(min(1.0, p))}


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def self_test():
    log("=== 自测 ===")
    rng = np.random.default_rng(11)
    n, d = 400, 64

    # (1) 交叉预测检验（维度可以不同）：对齐 vs 打乱
    base = rng.normal(size=(n, d))
    other = base[:, :d // 2] + 0.35 * rng.normal(size=(n, d // 2))   # 维度不同！
    c_ok = cross_predict_test(base, other, rng)["z"]
    shuffled = other[rng.permutation(n)]
    c_bad = cross_predict_test(base, shuffled, rng)["z"]
    log(f"  交叉预测检验：对齐 z={c_ok:.1f} | 打乱 z={c_bad:.1f}")
    assert c_ok > 10 and abs(c_bad) < 5, "交叉预测检验没区分开"

    # (2) 概念组检验：组内相似 vs 随机
    proto = rng.normal(size=(n // GROUP, d))
    F_grp = (proto[:, None, :] + 0.15 * rng.normal(size=(n // GROUP, GROUP, d))).reshape(n, d)
    F_rnd = rng.normal(size=(n, d))
    g_ok = concept_group_test(F_grp, rng)["z"]
    g_bad = concept_group_test(F_rnd, rng)["z"]
    log(f"  概念组检验：成组 z={g_ok:.1f} | 随机 z={g_bad:.1f}")
    assert g_ok > 5 and abs(g_bad) < 5, "概念组检验没区分开"

    # (3) 配对符号检验（n=10，与真实实验的被试数一致）
    s1 = sign_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])          # 全正 ⇒ p=2/1024
    s0 = sign_test([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])        # 交替 ⇒ p=1.0
    s8 = sign_test([1] * 8 + [-1, -1])                         # 8 正 2 负
    log(f"  符号检验：全正 p={s1['p_exact_two_sided']:.4f} | 交替 p={s0['p_exact_two_sided']:.4f} "
        f"| 8正2负 p={s8['p_exact_two_sided']:.4f}")
    assert s1["p_exact_two_sided"] < 0.01, "全正时 p 应为 0.00195"
    assert s0["p_exact_two_sided"] > 0.9, "交替时应为 1.0"
    assert 0.05 < s8["p_exact_two_sided"] < 0.2, "8正2负的标准精确 p 应约 0.109"
    # (4) 路径字典 loader：行序由路径推导 + 结构校验必须能抓错
    #     注意 train 用 `train_images/`、test 用 `test_images/` 前缀，正则都要认
    import torch
    d = {}
    for c in range(1, 4):
        for i in range(1, 11):
            d[f"train_images/{c:05d}_cc{c}/cc{c}_{i:02d}s.jpg"] = \
                torch.full((4,), float(c), dtype=torch.float32)
    arr, meta = path_dict_to_matrix(d, 30, 10)
    assert meta["first_key"] == "train_images/00001_cc1/cc1_01s.jpg", meta
    assert meta["last_key"] == "train_images/00003_cc3/cc3_10s.jpg", meta
    assert meta["per_concept"] == 10, meta
    # 另一前缀也要认
    d_test = {k.replace("train_images", "test_images"): v for k, v in d.items()}
    _a2, m2 = path_dict_to_matrix(d_test, 30, 10)
    assert m2["first_key"].startswith("test_images/"), m2
    # 行序必须按 (概念号, 文件名) —— 第 1 组的首行是 01s，第 2 组首行紧随其后
    assert abs(arr[0, 0] - 1.0) < 1e-9 and abs(arr[10, 0] - 2.0) < 1e-9
    # 自行推断每概念图数
    _a3, m3 = path_dict_to_matrix(d, 30, None)
    assert m3["per_concept"] == 10, m3
    d2 = dict(d)
    d2.pop("train_images/00002_cc2/cc2_05s.jpg")
    try:
        path_dict_to_matrix(d2, 30, 10)
        raise AssertionError("概念号不连续却没报错")
    except ValueError:
        pass
    try:
        path_dict_to_matrix(d, 30, 9)
        raise AssertionError("每概念图数不对却没报错")
    except ValueError:
        pass
    log(f"  路径字典 loader：行序推导 {meta['first_key']} … {meta['last_key']}，结构校验 OK")

    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-subjects", type=str, default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- 载入两套特征 ----
    feats = {}
    metas = {}
    VH_tr, m1 = load_matrix(FEATURE_SETS["ViT-H-14"][0], 16540)
    VH_te, m2 = load_matrix(FEATURE_SETS["ViT-H-14"][1], 200)
    feats["ViT-H-14"] = (VH_tr, VH_te)
    metas["ViT-H-14"] = {"train": m1, "test": m2}
    log(f"特征 ViT-H-14: train {VH_tr.shape} / test {VH_te.shape}（矩阵式）")

    VB_tr, m3 = load_path_dict(FEATURE_SETS["ViT-B-32"][0], 16540, None)
    VB_te, m4 = load_path_dict(FEATURE_SETS["ViT-B-32"][1], 200, None)
    feats["ViT-B-32"] = (VB_tr, VB_te)
    metas["ViT-B-32"] = {"train": m3, "test": m4}
    log(f"特征 ViT-B-32: train {VB_tr.shape} / test {VB_te.shape}（路径字典式，"
        f"行序由图片路径排序推导；每概念图数：train={m3['per_concept']} test={m4['per_concept']}；"
        f"首={m3['first_key']} 末={m3['last_key']}）")

    # ---- 对齐审计（只用特征）----
    log("\n--- 对齐审计（feature-only，不碰 EEG）---")
    audit = alignment_audit(feats["ViT-H-14"][0], feats["ViT-B-32"][0], rng)
    audit["load_meta"] = metas
    log(f"  交叉预测检验 H←B: R²={audit['cross_predict_HB']['r2_true']:.4f} "
        f"打乱对照={audit['cross_predict_HB']['null_mean']:.4f} "
        f"z={audit['cross_predict_HB']['z']:.1f}")
    log(f"  概念组检验 ViT-H-14: 组内={audit['concept_group_H']['within_mean']:.4f} "
        f"随机={audit['concept_group_H']['random_mean']:.4f} z={audit['concept_group_H']['z']:.1f}")
    log(f"  概念组检验 ViT-B-32: 组内={audit['concept_group_B']['within_mean']:.4f} "
        f"随机={audit['concept_group_B']['random_mean']:.4f} z={audit['concept_group_B']['z']:.1f}")

    # ---- 解码对比 ----
    sids = ([int(s) for s in args.only_subjects.split(",") if s.strip()]
            if args.only_subjects else list(range(1, 11)))
    rows = []
    for sid in sids:
        rec = run_subject(sid, feats, rng)
        if rec is None:
            log(f"  sub-{sid:02d}: 数据缺失，跳过")
            continue
        rows.append(rec)
        (OUT_DIR / "exp74_model_compare.json").write_text(
            json.dumps({"per_subject": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ---- 汇总 + 配对比较 ----
    log("\n" + "=" * 84)
    log("汇总：ViT-H-14 vs ViT-B-32（200 选 1，随机 0.5%）")
    log("=" * 84)
    summary = {"alignment_audit": audit, "models": {}}
    for name in FEATURE_SETS:
        a = [r["models"][name]["acc1_cos"] for r in rows]
        n = [float(np.median(r["models"][name]["null_acc1_cos"])) for r in rows]
        a5 = [r["models"][name]["acc5_cos"] for r in rows]
        fa = [r["models"][name]["acc1_featavg_cos"] for r in rows]
        summary["models"][name] = {
            "n_subjects": len(a),
            "acc1_cos_median": float(np.median(a)),
            "acc1_cos_min": float(np.min(a)), "acc1_cos_max": float(np.max(a)),
            "ratio_vs_random": float(np.median(a) / 0.005),
            "null_median": float(np.median(n)),
            "acc5_cos_median": float(np.median(a5)),
            "featavg_median": float(np.median(fa)),
            "per_subject_acc1_cos": {r["subject"]: r["models"][name]["acc1_cos"]
                                     for r in rows},
        }
        s = summary["models"][name]
        log(f"  {name:10s}: 单试次 top1 中位 {s['acc1_cos_median']:.4f} "
            f"({s['ratio_vs_random']:.1f}× 随机) 范围 {s['acc1_cos_min']:.4f}–{s['acc1_cos_max']:.4f} "
            f"| 零中位 {s['null_median']:.4f} | top5 {s['acc5_cos_median']:.4f} "
            f"| 80次平均 {s['featavg_median']:.4f}")

    if len(rows) >= 2:
        A = "ViT-H-14"
        B = "ViT-B-32"
        diffs = [r["models"][A]["acc1_cos"] - r["models"][B]["acc1_cos"] for r in rows]
        st = sign_test(diffs)
        summary["paired"] = {
            "comparison": f"{A} − {B}（单试次 acc1_cos，被试内配对）",
            "median_diff": float(np.median(diffs)),
            "mean_diff": float(np.mean(diffs)),
            "per_subject_diff": {r["subject"]: d for r, d in zip(rows, diffs)},
            "sign_test": st,
        }
        log(f"  配对差 ({A} − {B}): 中位 {np.median(diffs):+.4f} "
            f"均值 {np.mean(diffs):+.4f} | 正 {st['n_pos']} 负 {st['n_neg']} "
            f"精确双侧 p = {st['p_exact_two_sided']:.4f}")

    (OUT_DIR / "exp74_model_compare_full.json").write_text(
        json.dumps({"schema_version": "1.0", "script": "exp74_model_compare",
                    "summary": summary, "per_subject": rows,
                    "elapsed_s": round(time.time() - t0, 1)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
