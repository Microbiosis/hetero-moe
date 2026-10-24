#!/usr/bin/env python3
"""exp89_posaxis.py — 位置轴三测量（**零训练**，剃刀后最便宜的判别实验）。

## 目的（D20）

剃刀之后要决定：**做不做两原语模型**（后部感觉原语 + 前部执行原语）。
判据是**位置轴上的分工**：

- 后部（枕/顶/颞）带图像语义，而前部（前额/中央）**不带或弱得多**
  ⇒ 分工假设成立 ⇒ 值得做两原语模型；
- 前部带**同等**语义 ⇒ 分工假设被证伪 ⇒ **不做**（省掉最贵的一轮）。

顺带回答第二个问题：**位置轴是不是一条真轴**（脑域线的三套正交轴之一）。
若 63 个通道的语义对齐度沿「后→前」单调下降 ⇒ 位置轴有真实结构。

## 三测量（**全部零训练**：不训练任何新参数）

| 编号 | 测量 | 做法 | 回答什么 |
|---|---|---|---|
| (a) | 通道组消融 | 用 **630-d 原始特征**（基线表同一协议）在通道子集上算 RSA | 后部 vs 前部的语义对齐差 |
| (b) | 视觉诱发幅度反推位置 | 跨 split 逐通道 ERP 相关 + 幅度图 ⇒ 补 test split 没有 `ch_names` 的缺口 | test split 通道顺序是否与 train 一致；诱发幅度是否沿位置轴单调 |
| (c) | 通道对照（两个） | ① **同尺寸随机通道组**零分布 ② **打乱通道顺序**后的「后−前」差零分布 | 证明 (a) 的组间差来自**通道身份/位置**，而不是「维度多/少」或任何与位置无关的成分 |

## ⚠️ (c) 为什么不是「打乱通道顺序 ⇒ RSA 塌到 0」

**打错会得到一个假断言。** 本管线（`per_rep_targets` → 逐维 z-score → `rdm_full`
= 1−Pearson 距离 → RSA）**对通道置换是严格不变的**：

- `per_rep_targets` 逐通道做基线校正与分箱 ⇒ 置换只改维度顺序；
- 逐维 z-score ⇒ 每个维度的均值/方差跟着走 ⇒ 不变；
- `rdm_full` 用行内 Pearson 相关 ⇒ **对维度置换不变**（点积求和换序不换值）。

⇒ 打乱通道顺序后 **ALL 组的 RSA 一个数都不会变**（本脚本里有断言验证）。
真正的对照必须打在**组别归属**上：

| 对照 | 做法 | 期望 |
|---|---|---|
| (c1) 同尺寸随机组 | 抽 N 个**随机**通道（N = 该解剖组的通道数）算 RSA | 解剖组的 RSA 应显著高于随机组的中位数 |
| (c2) 打乱通道顺序 | 置换 EEG 的通道顺序后，重算「POST−ANT」 | 差值应落回随机组零分布内 ⇒ 说明组间差来自「通道↔位置的对应」 |

## 为什么用「630-d 原始特征」而不是自训 tokenizer 的读出

1. 基线表里 **630-d 原始特征 = +0.0504（10/10 显著）**，比任何基模都高
   ⇒ 原始信号口径本身就够灵敏；
2. tokenizer 读出要把非选中通道**置零**再喂给 encoder ⇒ **输入分布被破坏**，
   测到的是「OOD 稳健性」，不是「位置轴本身有没有语义」；
3. 零训练 + 与基线表同口径 ⇒ 数字**可直接横向比**（+0.0504 就是 ALL 组的参照线）。

## 口径纪律

- **特征逐字复用 `exp73.per_rep_targets`**（基线校正 12 采样 + 10×100 ms 窗），
  ⇒ 630-d 与基线表那一行是同一个量；
- **RSA 逐字复用 `eval_brain_alignment.metric_a_rsa`**（试次平均 → 跨图去共同成分
  → 逐维 z-score → RDM → 置换零分布，200 次置换）；
- 通道组的特征**不是重新跑一遍 `per_rep_targets`**，而是从 ALL 的 630-d 特征里
  **按维度切片**（布局是 bin-major：`dim = bin*n_ch + ch`）——
  同一份预处理 ⇒ 组间差只来自通道选择，不会混进预处理差异。
  ⚠️ 这个布局**必须实测验证**（见自测里的断言），否则会静默切错。

## 已知坑（本脚本自带消解）

1. **`per_rep_targets` 的维度是 bin-major**（不是 channel-major）
   ⇒ 单通道切片 = `[b*n_ch + c for b in range(10)]`，自测里有断言；
2. **通道组里可能有 0 个通道**（名字认不出来）⇒ 每次都打印分组表，肉眼可查；
3. **诱发幅度窗口**用 48–500 ms（峰峰值），因为数据是从刺激起点裁的
   ⇒ 没有真正的刺激前基线（`per_rep_targets` 用的 0–48 ms 只是项目约定）；
4. **test split 的 npz 没有 `ch_names`** ⇒ 通道名从 train split 取，
   而**两 split 的顺序一致性由 (b) 实测**（逐通道 ERP 波形相关），
   不靠假设。

## 用法

    # 自测（本地，合成数据）
    python -u research/brain_domains/exp/exp89_posaxis.py --self-test

    # 云端
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp89_posaxis.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 120m -d
"""

from __future__ import annotations

import argparse
import json
import math
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

import eval_brain_alignment as EVAL   # noqa: E402
import exp73_decode200 as E73          # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp89")
SEED = 0

FS = 250.0
BASELINE = 12               # 与 exp73.per_rep_targets 逐字一致（0–48 ms）
AMP_LO, AMP_HI = 12, 125    # 诱发幅度窗口：48–500 ms（峰峰值）
N_BIN = 10                  # 10 个 100 ms 窗（exp73 口径）

GROUPS = ("ALL", "OCC", "PARI", "TEMP", "POST", "CENT", "FRONT", "ANT")
MB_GROUPS = ("ALL", "POST", "ANT", "OCC")
NULL_GROUPS = ("OCC", "POST", "ANT")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 通道分组
# ---------------------------------------------------------------------------

def classify_channel(name):
    """把 10-20/10-10 通道名归到解剖组。

    ⚠️ **顺序敏感**：长前缀必须先判（`TPO` 先于 `TP` 先于 `T`，
    `PO` 先于 `P`，`CP` 先于 `C`，`FC` 先于 `F`，`FT` 先于 `F`）。
    """
    n = str(name).strip().upper().replace("-", "").replace("_", "")
    if n.startswith("TPO") or n.startswith("FTP"):
        return "temporal"
    if n.startswith("FT") or n.startswith("TP"):
        return "temporal"
    if n.startswith("FP") or n.startswith("AF"):
        return "frontal"
    if n.startswith("PO"):
        return "occipital"
    if n.startswith("CP"):
        return "parietal"
    if n.startswith("FC"):
        return "central"
    if n.startswith("T"):
        return "temporal"
    if n.startswith("P"):
        return "parietal"
    if n.startswith("O"):
        return "occipital"
    if n.startswith("C"):
        return "central"
    if n.startswith("F"):
        return "frontal"
    return "other"


def build_groups(ch_names):
    """通道名 → {组名: 通道下标列表}。"""
    lab = [classify_channel(c) for c in ch_names]
    idx = {k: [i for i, x in enumerate(lab) if x == k]
           for k in ("occipital", "parietal", "temporal", "central", "frontal",
                     "other")}
    g = {
        "OCC": idx["occipital"],
        "PARI": idx["parietal"],
        "TEMP": idx["temporal"],
        "CENT": idx["central"],
        "FRONT": idx["frontal"],
        "POST": idx["occipital"] + idx["parietal"] + idx["temporal"],
        "ANT": idx["central"] + idx["frontal"],
        "OTHER": idx["other"],
    }
    g["ALL"] = list(range(len(ch_names)))
    return g, lab


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    xs = np.asarray(x)[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    return pearson(_rank(a), _rank(b))


def sign_test_p(diffs):
    """双侧符号检验（精确二项）。返回 (n_pos, n_neg, p)。"""
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return k, m, float(min(1.0, p))


def channel_dims(ch_idx, n_ch, n_bin=N_BIN):
    """单通道/通道组在 630-d 特征里的维度下标（**bin-major**）。

    `per_rep_targets` 最后一步是 `transpose(0,1,3,2).reshape(..., n_ch*n_bin)`
    ⇒ 下标 = `bin*n_ch + ch`（**bin 在外**）。
    """
    return [b * n_ch + c for b in range(n_bin) for c in ch_idx]


def slice_group(feats, ch_idx, n_ch, n_bin=N_BIN):
    """从 (n_img, n_rep, n_ch*n_bin) 里切出某个通道组的特征。"""
    d = channel_dims(ch_idx, n_ch, n_bin)
    return feats[:, :, d]


# ---------------------------------------------------------------------------
# 诱发响应 / 位置
# ---------------------------------------------------------------------------

def grand_erp(eeg):
    """(n_img, n_rep, C, T) → (C, T) 全体试次平均（= 诱发响应）。"""
    x = eeg.mean(axis=1, dtype=np.float32)      # (n_img, C, T)
    return x.mean(axis=0, dtype=np.float32)     # (C, T)


def amp_map(erp, lo=AMP_LO, hi=AMP_HI):
    """逐通道的诱发幅度（窗口内峰峰值），基线用 0–48 ms 的均值扣除。"""
    e = erp - erp[:, :BASELINE].mean(axis=1, keepdims=True)
    w = e[:, lo:hi]
    return w.max(axis=1) - w.min(axis=1)


def anat_score(labels):
    """解剖位置分（后部 +3 → 前部 −1），用于「幅度图是否沿位置轴单调」。"""
    m = {"occipital": 3.0, "parietal": 2.0, "temporal": 1.0,
         "central": 0.0, "frontal": -1.0, "other": 0.0}
    return np.array([m.get(x, 0.0) for x in labels], dtype=np.float64)


# ---------------------------------------------------------------------------
# RSA
# ---------------------------------------------------------------------------

def rsa(feats, img, n_perm, rng):
    """(n_img, n_rep, D) → (r, p)；口径 = metric_a_rsa。"""
    a = EVAL.metric_a_rsa(feats, img, n_perm, rng)
    return float(a["r_img"]), float(a["p_img"])


def per_channel_rsa(feats, img, n_ch, n_perm, rng):
    """逐通道 RSA（63 个数）⇒ 语义对齐的「位置轴」曲线。"""
    out = np.zeros(n_ch, dtype=np.float64)
    for c in range(n_ch):
        d = channel_dims([c], n_ch)
        r, _p = rsa(feats[:, :, d], img, n_perm, rng)
        out[c] = r
    return out


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_test_eeg(subject):
    """test split 的 npz → (200, 80, C, T)。同时把 npz 的键名打出来（防坑）。"""
    fp = TEST_4D / f"sub-{subject:02d}.npz"
    z = np.load(fp)
    keys = list(z.keys())
    log(f"  npz 键：{keys}")
    eeg = z["eeg"]
    return eeg, keys


def load_train_chunk(subject, n_images):
    """train split 的前 n_images 张 + `ch_names`。

    ⚠️ `train.pt` 里的 `eeg` 是 **numpy 数组**（实测）⇒ 用 `np.asarray` 统一。
    ⚠️ 整个文件 ≈1.94 GiB ⇒ 切片后立刻丢掉原对象，别留着。
    """
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    ch_names = list(obj["ch_names"])
    eeg = np.asarray(obj["eeg"][:n_images], dtype=np.float16)
    del obj
    return ch_names, eeg


# ---------------------------------------------------------------------------
# 单被试
# ---------------------------------------------------------------------------

def run_subject(subject, IMG_test, IMG_train, args, rng):
    """一个被试的全部三测量。返回记录 dict。"""
    rec = {"subject": f"sub-{subject:02d}"}
    t0 = time.time()

    # ---- 数据 ----
    eeg_te, npz_keys = load_test_eeg(subject)
    rec["npz_keys"] = npz_keys
    log(f"  test split：{eeg_te.shape} {eeg_te.dtype}")
    n_img, n_rep, n_ch, T = eeg_te.shape

    need_train = args.train_images > 0
    ch_names, eeg_tr = (None, None)
    if need_train:
        ch_names, eeg_tr = load_train_chunk(subject, args.train_images)
        log(f"  train split：{eeg_tr.shape} | 通道名 {len(ch_names)} 个")
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]
        rec["ch_names_source"] = "fallback"
    else:
        rec["ch_names_source"] = "train.pt"
    rec["ch_names"] = list(ch_names)

    groups, labels = build_groups(ch_names)
    rec["group_labels"] = labels
    rec["groups"] = {k: list(map(int, v)) for k, v in groups.items()}
    log("  通道分组：" + "  ".join(
        f"{k}={len(groups[k])}" for k in GROUPS if k in groups))

    # ---- (b) 位置：诱发幅度 + 跨 split 顺序一致性 ----
    erp_te = grand_erp(eeg_te)
    amp_te = amp_map(erp_te)
    b = {"amp_te": amp_te.tolist()}
    if eeg_tr is not None and eeg_tr.shape[0] > 0:
        erp_tr = grand_erp(eeg_tr)
        amp_tr = amp_map(erp_tr)
        w = slice(0, min(AMP_HI, T))
        wave_r = np.array([pearson(erp_tr[c, w], erp_te[c, w])
                           for c in range(n_ch)])
        b.update({
            "amp_tr": amp_tr.tolist(),
            "wave_r": wave_r.tolist(),
            "wave_median": float(np.median(wave_r)),
            "wave_n_gt_0.8": int((wave_r > 0.8).sum()),
            "amp_r": pearson(amp_tr, amp_te),
            "amp_spearman": spearman(amp_tr, amp_te),
        })
        b["amp_vs_anat_spearman"] = spearman(amp_te, anat_score(labels))
        b["amp_group_mean"] = {
            g: float(amp_te[groups[g]].mean()) for g in GROUPS
            if groups.get(g)}
        log(f"  ORDER 逐通道 ERP 相关：中位 {b['wave_median']:+.4f} "
            f"| >0.8 的通道 {b['wave_n_gt_0.8']}/{n_ch} "
            f"| 幅度图 r={b['amp_r']:+.4f} sp={b['amp_spearman']:+.4f} "
            f"| 幅度~解剖位 sp={b['amp_vs_anat_spearman']:+.4f}")
        log("  AMP  " + "  ".join(f"{k}={b['amp_group_mean'][k]:.3f}"
                                  for k in b["amp_group_mean"]))
    rec["position"] = b

    # ---- 特征：ALL 的 630-d（一次算完，组间切片）----
    feats_te = E73.per_rep_targets(eeg_te, chunk=args.chunk)
    feats_tr = None
    if eeg_tr is not None and eeg_tr.shape[0] > 0:
        feats_tr = E73.per_rep_targets(eeg_tr, chunk=args.chunk)
    log(f"  特征：test {feats_te.shape}" +
        (f" | train {feats_tr.shape}" if feats_tr is not None else ""))

    # ---- (a) 通道组消融 ----
    gr = {}
    for gname in GROUPS:
        idx = groups.get(gname, [])
        if not idx:
            log(f"  RSA {gname:<6} 组内没有通道 ⇒ 跳过")
            continue
        f = slice_group(feats_te, idx, n_ch)
        r, p = rsa(f, IMG_test, args.n_perm, rng)
        gr[gname] = {"r": r, "p": p, "n_ch": len(idx)}
        log(f"  RSA {gname:<6} n_ch={len(idx):<3} r={r:+.4f} p={p:.4f}")
    rec["group_rsa"] = gr

    # ---- 逐通道 RSA（位置轴曲线）----
    # ⚠️ 只用 r（r_obs 与置换次数无关）⇒ 置换次数可以给小，省时间
    ch_r = per_channel_rsa(feats_te, IMG_test, n_ch, args.n_perm_channel, rng)
    rec["per_channel_rsa"] = ch_r.tolist()
    order = np.argsort(-np.abs(ch_r))[:8]
    log("  CHMAP " + "  ".join(
        f"{ch_names[c]}={ch_r[c]:+.4f}({labels[c][:4]})" for c in order))
    # 通道名表 + 逐通道曲线（**整条**位置轴曲线，供头皮图用；只打 CHMAP 前 8 名不够）
    log("  CHLIST " + ",".join(str(c) for c in ch_names))
    log("  CURVE " + ",".join(f"{v:+.4f}" for v in ch_r))

    # ---- (c1) 同尺寸随机通道组零分布 ----
    rnd = {}
    for gname in NULL_GROUPS:
        idx = groups.get(gname, [])
        if not idx or gname not in gr:
            continue
        k = len(idx)
        rs = []
        for _ in range(args.n_random):
            sub = rng.choice(n_ch, size=k, replace=False).tolist()
            r, _ = rsa(slice_group(feats_te, sub, n_ch), IMG_test,
                       args.n_perm_null, rng)
            rs.append(r)
        r_true = gr[gname]["r"]
        # ⚠️ `frac_ge_true` = **随机组里 ≥ 真值** 的比例
        #    ⇒ 0% ⇒ 真值高于**全部**随机组；100% ⇒ 真值低于**全部**随机组
        rnd[gname] = {"size": k, "r_true": r_true,
                      "r_null_median": float(np.median(rs)),
                      "r_null_max": float(np.max(rs)),
                      "frac_null_ge_true": float(np.mean(
                          [1.0 if x >= r_true else 0.0 for x in rs])),
                      "rs": rs}
        log(f"  NULL {gname:<5} 同尺寸随机组中位 {rnd[gname]['r_null_median']:+.4f} "
            f"最大 {rnd[gname]['r_null_max']:+.4f} "
            f"| 真值 {r_true:+.4f} "
            f"| 随机组中≥真值的占 {rnd[gname]['frac_null_ge_true']:.0%}")
    rec["random_group_null"] = rnd

    # ---- (c2) 打乱通道顺序：ALL 组 RSA 不变（置换不变性），组间差塌回零分布 ----
    sh = []
    r_all_true = gr["ALL"]["r"] if "ALL" in gr else None
    for k in range(args.n_shuffle):
        perm = rng.permutation(n_ch)
        f_sh = E73.per_rep_targets(eeg_te[:, :, perm, :], chunk=args.chunk)
        r_all, _ = rsa(f_sh, IMG_test, args.n_perm, rng)
        r_post, _ = rsa(slice_group(f_sh, groups["POST"], n_ch), IMG_test,
                        args.n_perm, rng)
        r_ant, _ = rsa(slice_group(f_sh, groups["ANT"], n_ch), IMG_test,
                       args.n_perm, rng)
        sh.append({"r_all": r_all, "r_post": r_post, "r_ant": r_ant,
                   "diff": r_post - r_ant})
        log(f"  SHUF #{k} ALL r={r_all:+.4f}（真值 {r_all_true:+.4f}） "
            f"POST {r_post:+.4f} ANT {r_ant:+.4f} 差 {r_post - r_ant:+.4f}")
    rec["shuffled"] = sh

    # ---- 指标 B（次要确认：ridge 200 选 1）----
    if feats_tr is not None and args.metric_b:
        mb = {}
        for gname in MB_GROUPS:
            idx = groups.get(gname, [])
            if not idx:
                continue
            Xtr = slice_group(feats_tr, idx, n_ch)
            Xte = slice_group(feats_te, idx, n_ch)
            mu, sd = EVAL.zscore_fit(Xtr)
            rb = EVAL.metric_b_decode(EVAL.zscore_apply(Xtr, mu, sd),
                                      EVAL.zscore_apply(Xte, mu, sd),
                                      IMG_train[:args.train_images], IMG_test,
                                      rng, n_null=args.n_null)
            mb[gname] = {"acc1": float(rb["acc1_cos"]),
                         "null_acc1": float(np.median(rb["null_acc1_cos"]))}
            log(f"  MB {gname:<6} acc1={mb[gname]['acc1']:.4f} "
                f"(零中位 {mb[gname]['null_acc1']:.4f})")
        rec["metric_b"] = mb

    rec["elapsed_s"] = round(time.time() - t0, 1)
    log(f"  用时 {rec['elapsed_s']}s")
    del eeg_te, eeg_tr, feats_te, feats_tr
    return rec


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def summarize(per_subject):
    """跨被试汇总：各组中位 / 显著数 / 后前差 / 对照零分布。"""
    out = {}
    names = set()
    for r in per_subject:
        names |= set(r.get("group_rsa", {}).keys())
    for gname in sorted(names):
        rs = [r["group_rsa"][gname]["r"] for r in per_subject
              if gname in r.get("group_rsa", {})]
        ps = [r["group_rsa"][gname]["p"] for r in per_subject
              if gname in r.get("group_rsa", {})]
        if not rs:
            continue
        srt = sorted(rs)
        out[gname] = {"n": len(rs), "r_median": srt[len(srt) // 2],
                      "n_sig": sum(1 for p in ps if p < 0.05),
                      "r_mean": float(np.mean(rs)),
                      "r_all": rs}

    # 后 vs 前 的配对差
    diffs = []
    for r in per_subject:
        g = r.get("group_rsa", {})
        if "POST" in g and "ANT" in g:
            diffs.append(g["POST"]["r"] - g["ANT"]["r"])
    if diffs:
        k, m, p = sign_test_p(diffs)
        out["POST_minus_ANT"] = {"n": len(diffs), "n_pos": k, "n_neg": m,
                                 "p_sign": p,
                                 "median": float(np.median(diffs)),
                                 "mean": float(np.mean(diffs)),
                                 "diffs": diffs}

    # (c2) 打乱通道顺序后的「后−前」差零分布（汇总所有被试）
    null_diffs = [s["diff"] for r in per_subject for s in r.get("shuffled", [])]
    if null_diffs and diffs:
        nd = np.array(null_diffs, dtype=np.float64)
        beat = int(np.sum(np.array(diffs) > nd.max()))
        out["POST_minus_ANT_null"] = {
            "n_null": len(null_diffs),
            "null_median": float(np.median(nd)),
            "null_max": float(nd.max()),
            "null_p95": float(np.percentile(nd, 95)),
            "n_true_above_null_max": beat,
            "n_true": len(diffs),
        }
    # (c1) 随机组零分布汇总
    rnd_agg = {}
    for gname in NULL_GROUPS:
        meds = [r["random_group_null"][gname]["r_null_median"]
                for r in per_subject if gname in r.get("random_group_null", {})]
        fr = [r["random_group_null"][gname]["frac_null_ge_true"]
              for r in per_subject if gname in r.get("random_group_null", {})]
        if meds:
            rnd_agg[gname] = {"null_median_of_medians": float(np.median(meds)),
                               "mean_frac_ge_true": float(np.mean(fr))}
    if rnd_agg:
        out["random_group_null_summary"] = rnd_agg

    sh_all = [s["r_all"] for r in per_subject for s in r.get("shuffled", [])]
    sh_true = [r.get("group_rsa", {}).get("ALL", {}).get("r", float("nan"))
               for r in per_subject for _ in r.get("shuffled", [])]
    if sh_all and "ALL" in out and len(sh_true) == len(sh_all):
        d = np.abs(np.array(sh_all) - np.array(sh_true))
        out["permutation_invariance_ALL"] = {
            "r_true_median": out["ALL"]["r_median"],
            "r_shuffled_median": float(np.median(sh_all)),
            "max_abs_delta": float(np.nanmax(d)),
        }
    curves = [r["per_channel_rsa"] for r in per_subject
              if r.get("per_channel_rsa")]
    if curves:
        out["per_channel_median"] = np.median(np.array(curves), axis=0).tolist()
    return out


# ---------------------------------------------------------------------------
# 自测（合成数据，本地可跑）
# ---------------------------------------------------------------------------

def synthetic_ch_names():
    base = ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Fz",
            "FC1", "FC2", "FC5", "FC6",
            "Cz", "C3", "C4",
            "T3", "T4", "T7", "T8", "T9", "T10", "T11", "T12",
            "FT7", "FT8", "FT9", "FT10",
            "TP9", "TP10", "TPO9", "TPO10",
            "CP1", "CP2", "CP5", "CP6",
            "P1", "P2", "P3", "P4", "P7", "P8", "Pz",
            "PO3", "PO4", "PO7", "PO8", "PO9", "PO10", "POz",
            "O1", "O2", "Oz", "M1", "M2"]
    assert len(base) == 63, len(base)
    assert len(set(base)) == 63, "通道名必须互不相同"
    return base


def _synth_eeg(rng, IMG, ch_names, groups, n_img, n_rep, n_ch, T):
    """合成 EEG：
    - 每个通道一个**形状不同**的诱发成分（tau 随通道变）⇒ 顺序检查才有分辨力；
    - 枕叶组植入**图像特异**信号；
    - 额叶组给一个更大的诱发幅度（无图像信息）⇒ 幅度图可查。
    """
    eeg = rng.normal(size=(n_img, n_rep, n_ch, T)).astype(np.float32) * 0.3
    occ = groups["OCC"]
    front = groups["FRONT"]
    t = np.arange(T)
    for c in range(n_ch):
        tau = 15 + (c % 60)
        amp = 4.0 if c in front else (1.0 + 0.02 * c)
        eeg[:, :, c, :] += amp * np.exp(-0.5 * ((t - tau) / 8.0) ** 2)
    W = np.zeros((IMG.shape[1], n_ch * N_BIN))
    # ⚠️ 维度布局是 **bin-major**（`dim = bin*n_ch + ch`）⇒ W 也必须按这个排
    # ⚠️ 植入幅度必须**远小于**诱发成分（0.3 vs ~2.1），否则图像特异信号会污染
    #    诱发幅度图 ⇒ (b) 就成了循环论证（用图像信号「发现」枕叶位置）。
    for b in range(N_BIN):
        for c in occ:
            W[:, b * n_ch + c] = rng.normal(size=IMG.shape[1]) * 0.3
    win = T // N_BIN
    for i in range(n_img):
        sig = (IMG[i] @ W).reshape(N_BIN, n_ch)          # (bin, ch)
        blk = np.repeat(sig.T[:, :, None], win, axis=2)   # (ch, bin, WIN)
        eeg[i] += blk.reshape(n_ch, T)[None, :, :]        # 广播到所有重复
    return eeg


def self_test():
    """用**合成数据**把三测量的每一步都验一遍（含维度布局与置换不变性的断言）。"""
    log("=== 自测：位置轴三测量（合成数据）===")
    rng = np.random.default_rng(7)
    n_img, n_rep, n_ch, T, d_img = 40, 8, 63, 250, 32
    ch_names = synthetic_ch_names()
    groups, labels = build_groups(ch_names)
    log(f"  分组：OCC={len(groups['OCC'])} PARI={len(groups['PARI'])} "
        f"TEMP={len(groups['TEMP'])} POST={len(groups['POST'])} "
        f"CENT={len(groups['CENT'])} FRONT={len(groups['FRONT'])} "
        f"ANT={len(groups['ANT'])} OTHER={len(groups['OTHER'])}")
    assert len(groups["ALL"]) == 63
    assert not (set(groups["POST"]) & set(groups["ANT"])), "POST/ANT 不能重叠"
    assert set(groups["OTHER"]) == {61, 62}, groups["OTHER"]

    IMG = rng.normal(size=(n_img, d_img))
    eeg = _synth_eeg(rng, IMG, ch_names, groups, n_img, n_rep, n_ch, T)

    feats = E73.per_rep_targets(eeg, chunk=20)
    assert feats.shape == (n_img, n_rep, n_ch * N_BIN), feats.shape

    # --- 断言 1：bin-major 布局（单通道切片必须等于「只喂该通道」的结果）---
    for c in (0, 5, 37, 62):
        f_single = E73.per_rep_targets(eeg[:, :, [c], :], chunk=20)
        f_slice = slice_group(feats, [c], n_ch)
        assert f_single.shape == f_slice.shape, (f_single.shape, f_slice.shape)
        assert np.allclose(f_single, f_slice, atol=1e-5), f"通道 {c} 切片错了"
    log("  ✓ 维度布局断言通过（bin-major，切片 == 单独喂该通道）")

    # --- 断言 2：植入通道的 RSA 必须高于噪声通道，额叶组不应超过枕叶组 ---
    ch_r = per_channel_rsa(feats, IMG, n_ch, n_perm=40,
                           rng=np.random.default_rng(3))
    occ, front = groups["OCC"], groups["FRONT"]
    noise = [c for c in range(n_ch) if c not in occ and c not in front]
    r_occ = float(np.mean([ch_r[c] for c in occ]))
    r_noise = float(np.mean([ch_r[c] for c in noise]))
    r_front = float(np.mean([ch_r[c] for c in front]))
    log(f"  逐通道 RSA：枕叶 {r_occ:+.4f} / 额叶 {r_front:+.4f} / 噪声 {r_noise:+.4f}")
    assert r_occ > r_noise, "植入的图像信号没被检出来"
    assert r_front < r_occ, "额叶不该比枕叶更贴图像"

    # --- 断言 3：置换不变性（ALL 组 RSA 不因通道置换而变）---
    r_before, _ = rsa(feats, IMG, 40, np.random.default_rng(5))
    perm = np.random.default_rng(11).permutation(n_ch)
    f_sh = E73.per_rep_targets(eeg[:, :, perm, :], chunk=20)
    r_after, _ = rsa(f_sh, IMG, 40, np.random.default_rng(5))
    log(f"  置换不变性：ALL r {r_before:+.6f} → {r_after:+.6f}")
    assert abs(r_after - r_before) < 1e-9, (r_before, r_after)
    log("  ✓ 通道置换下 ALL 组 RSA 严格不变（⇒ 对照必须打在组别归属上）")

    # --- 断言 4：置换后「解剖组」的 RSA 会变（组别归属被打乱）---
    r_post_true, _ = rsa(slice_group(feats, groups["POST"], n_ch), IMG, 40,
                          np.random.default_rng(6))
    r_post_shuf, _ = rsa(slice_group(f_sh, groups["POST"], n_ch), IMG, 40,
                          np.random.default_rng(6))
    log(f"  置换后 POST 组 r {r_post_true:+.4f} → {r_post_shuf:+.4f}")
    assert abs(r_post_shuf - r_post_true) > 1e-6, "组别归属打乱后 RSA 不该不变"

    # --- 断言 5：跨 split 顺序检查（同序高相关 / 换序低相关）---
    erp1 = grand_erp(eeg[:20])
    erp2 = grand_erp(eeg[20:])
    w = slice(0, AMP_HI)
    r_same = float(np.median([pearson(erp1[c, w], erp2[c, w]) for c in range(n_ch)]))
    perm2 = np.random.default_rng(13).permutation(n_ch)
    r_diff = float(np.median([pearson(erp1[c, w], erp2[perm2[c], w])
                              for c in range(n_ch)]))
    log(f"  顺序检查：同序中位 r={r_same:+.4f} / 换序中位 r={r_diff:+.4f}")
    assert r_same > 0.9 and r_diff < 0.5, (r_same, r_diff)

    # --- 断言 6：诱发幅度图应把「大诱发」的额叶排到前面 ---
    amp = amp_map(erp2)
    assert amp[front].mean() > amp[occ].mean(), "幅度图没反映植入的诱发成分"

    # --- 断言 7：符号检验 ---
    k, m, p = sign_test_p([0.1] * 10)
    assert (k, m) == (10, 0) and abs(p - 2 / 1024) < 1e-9, (k, m, p)
    log("  ✓ 符号检验：10/10 → p=0.00195")
    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=1000,
                    help="train split 取前多少张（(b) 的诱发图 + 指标 B 的拟合集）")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-perm-null", type=int, default=50,
                    help="随机组零分布用的置换次数（只取 r，不看 p）")
    ap.add_argument("--n-perm-channel", type=int, default=50,
                    help="逐通道 RSA 的置换次数（只取 r）")
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=20,
                    help="(c1) 每个尺寸的随机通道组个数")
    ap.add_argument("--n-shuffle", type=int, default=5,
                    help="(c2) 打乱通道顺序的重复次数")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--metric-b", type=int, default=1,
                    help="1 = 算指标 B（ridge 200 选 1）")
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
    IMG_train = None
    if args.metric_b and args.train_images > 0:
        ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
        IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"图像特征：test {IMG_test.shape}"
        + (f" | train {IMG_train.shape}" if IMG_train is not None else ""))

    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    per_subject = []
    for sid in sids:
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = run_subject(sid, IMG_test, IMG_train, args,
                              np.random.default_rng(SEED + sid))
            per_subject.append(rec)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)

    summary = summarize(per_subject)
    for gname, e in summary.items():
        if gname == "per_channel_median":
            continue
        if "r_median" in e:
            log(f"  {gname:<12} 中位 r={e['r_median']:+.4f} "
                f"({e['n_sig']}/{e['n']} 显著)")
        else:
            log(f"  {gname:<12} {e}")

    (out_dir / "exp89_posaxis.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject,
                    "summary": summary},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/exp89_posaxis.json")


if __name__ == "__main__":
    main()
