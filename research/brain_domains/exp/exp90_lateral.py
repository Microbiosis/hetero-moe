#!/usr/bin/env python3
"""exp90_lateral.py — **左右轴（外侧轴）**的零训练测量：我们的数据到底能不能分辨左右？

## 为什么做这个（2026-10-24 的一处纠正）

裂脑实验切的是**左右轴**（胼胝体 = 左右半球之间的连合纤维），
而我们此前所有结构主张都落在**前后轴（位置轴）**上
⇒ **两条轴不能互借证据**（左右半球的专门化 ≠ 前部/后部的分工）。

⇒ 于是要回答一个**边界问题**，而不是去找偏侧：
> **我们这批 63 电极的 EEG 记录，在空间上能不能把左右分开？**

- 63 个 10-20 电极是**双侧**的：每个电极都同时采到两半球的投射
  ⇒ 半球偏侧是**幅度约 10–20% 的统计性不对称** ⇒ EEG 尺度上很可能**根本看不见**；
- 猕猴那套 spike 数据是**单侧**记录（M1/PMd/DMFC/S1）⇒ 也没有左右可比性。
⇒ 所以左右轴在我们的数据里很可能**测不了**。但「很可能」不是数 ⇒ **把它测出来**。

## 三个测量（全部零训练，协议与 exp89 完全一致：630-d 原始特征 + `metric_a_rsa`）

| # | 测量 | 做法 | 读法 |
|---|---|---|---|
| **L1** | **左右偏侧** | LEFT 28 / RIGHT 28 / MID 7 各自的 RSA | 显著左>右 ⇒ 偏侧在 EEG 尺度上**可见**；左右相等 ⇒ 不可见 |
| **L2** | **左右结构冗余度** | 左侧组的「图像相似度结构」与右侧组的做 Spearman，**并给同侧分半可靠度作上限** | 冗余度 / 上限 = 两侧共享结构的比例；上限本身很重要（没有它，这个相关系数无法解释） |
| **L3** | **跨半球 transfer**（裂脑那一题） | ridge 200 选 1：左拟合→右解码、右拟合→左解码，与同侧（左→左、右→右）比；**再跑一条错位对照** | 跨侧 / 同侧 = **可迁移比例**；对齐版 vs 错位版之差 = 「位点对应」真正贡献了多少 |

⚠️ **L3 的前提：左右两组必须按「同一位点」对齐**（第一版栽在这里）
第一版让左右两组各自按蒙太奇顺序排 ⇒ 第 i 维在两侧代表**不同的位点**
⇒ 「左学到的映射用到右侧」这件事**没有意义**（测的是一个假对照）。
⇒ 修正：右侧组的顺序 = 其**镜像伙伴**在蒙太奇里的位置（F3↔F4、O1↔O2、FT9↔FT10…）
⇒ 并加一条**错位对照**（打乱右侧的位点对应）作为地板。
⚠️ 顺带修掉一个真 bug：两位数电极名（**FT10 / TP10**）会被朴素的「剥掉末位数字」规则切错
⇒ 前缀切成 "FT1" ⇒ 整条镜像序列错位（自测里用断言抓到了）。

⚠️ **L3 的方向性说明**：我们不是在做「左右半球损伤」，我们量的是
**同一批电极的两半之间的信息可迁移性** ⇒ 这是**数据侧的纠缠度**，
不是「裂脑」本身（裂脑是**解剖**上的分离；我们是**采样**上的纠缠）。

## 通道左右怎么分（从 `ch_names` 推，不靠假设）

真实蒙太奇（`train.pt` 的 `ch_names`，63 个）恰好 **28 左 + 28 右 + 7 中线**：
- 末位**奇数**数字 → 左（Fp1, AF3, F7, FC5, T7, C3, TP9, CP5, P3, PO7, O1 …）
- 末位**偶数**数字 → 右
- 末位 **`z`** → 中线（AFz, Fz, FCz, Cz, CPz, Pz, POz, Oz —— 实测 7 个）
- `M1/M2`（乳突参考，真实蒙太奇里**没有**）→ 归 other，不进任何组

## 预登记判据

| 结果 | 结论 |
|---|---|
| 左右 RSA 相等 + 冗余度高 + 跨/同侧转移比 ≈ 1 | **左右轴在我们数据里测不了** ⇒ 记为「未测·当前数据测不了」，与「谱系轴缺 pNE 侧」同类处理；**不得**用裂脑文献充当我们的证据 |
| 左右 RSA 显著不等 | 偏侧在 EEG 尺度上可见 ⇒ 左右轴在我们的数据里**可测**，应当进骨架 |

## 用法

    # 自测（本地，合成数据）
    python -u research/brain_domains/exp/exp90_lateral.py --self-test

    # 云端
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp90_lateral.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 120m -d
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
import exp89_posaxis as E89            # noqa: E402

OUT_DIR = Path("/domains/runs/illusionbrain-align/exp90")
SEED = 0
E65 = EVAL.E65


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 通道左右
# ---------------------------------------------------------------------------

def laterality(name):
    """通道名 → 'left' / 'right' / 'mid' / 'other'。

    规则：末位奇数数字 = 左，偶数 = 右，`z` = 中线，`M1/M2`（乳突）= other。
    """
    n = str(name).strip().upper().replace("-", "").replace("_", "")
    if n in ("M1", "M2"):
        return "other"
    if not n:
        return "other"
    last = n[-1]
    if last == "Z":
        return "mid"
    if last.isdigit():
        return "left" if (int(last) % 2 == 1) else "right"
    return "other"


def build_lat_groups(ch_names):
    """→ {'LEFT': [...], 'RIGHT': [...], 'MID': [...], 'ALL': [...], 'OTHER': [...]}"""
    lab = [laterality(c) for c in ch_names]
    g = {
        "LEFT": [i for i, x in enumerate(lab) if x == "left"],
        "RIGHT": [i for i, x in enumerate(lab) if x == "right"],
        "MID": [i for i, x in enumerate(lab) if x == "mid"],
        "OTHER": [i for i, x in enumerate(lab) if x == "other"],
    }
    g["ALL"] = list(range(len(ch_names)))
    return g, lab


def site_key(name):
    """电极位点键：把左右**镜像对**映射到同一个键。

    「F3」与「F4」是同一位点的左右两个电极 ⇒ 键 = (字母前缀, 对序号)：
    F1/F2 ⇒ 1，F3/F4 ⇒ 2，F5/F6 ⇒ 3，F7/F8 ⇒ 4；T7/T8、O1/O2、PO3/PO4 同理。

    ⚠️ **必须处理两位数电极名**（FT10 / TP10）：
    用正则 `^(.*?)(\\d+)$` 剥掉**末尾的数字串** ⇒ "FT10" 的前缀是 "FT"、数字是 10
    （否则会切成 "FT1" + 0 ⇒ 与 FT9 配不上 ⇒ 整条镜像序列错位）。
    """
    n = str(name).strip().upper().replace("-", "").replace("_", "")
    if not n:
        return ("", 0)
    m = re.match(r"^(.*?)(\d+)$", n)
    if not m:
        return (n, 0)
    pre, d = m.group(1), int(m.group(2))
    return (pre, (d + 1) // 2 if d % 2 else d // 2)


def mirrored_orders(ch_names, groups):
    """左右两组按**同一位点**对齐的通道顺序。

    ⚠️ **为什么必须对齐**：L3（跨半球 transfer）要把「在左侧学到的映射」用到右侧特征上。
    若左右两组各按自己的顺序排，第 i 维在两侧代表**不同的位点** ⇒ 映射没有意义
    （这是第一版 exp90 的缺陷，2026-10-24 修正）。
    ⇒ 右侧组的顺序 = 其**镜像伙伴**在蒙太奇里的位置 ⇒ 第 i 维两侧同位点。
    """
    pos = {c: i for i, c in enumerate(ch_names)}
    left_pos = {site_key(c): pos[c] for c in ch_names
                if laterality(c) == "left"}
    left = sorted([c for c in ch_names if laterality(c) == "left"],
                   key=lambda c: pos[c])
    right = sorted([c for c in ch_names if laterality(c) == "right"],
                   key=lambda c: left_pos.get(site_key(c), 10 ** 6))
    return left, right


# ---------------------------------------------------------------------------
# RDM（与 metric_a_rsa 同一归一化链，才能和 RSA 同口径）
# ---------------------------------------------------------------------------

def img_rdm(feats):
    """(n_img, n_rep, D) → (n_img, n_img) 图像相似度结构。

    归一化链**逐字复制** `metric_a_rsa`：试次平均 → 跨图去共同成分 → 逐维 z-score。
    """
    X = feats.mean(axis=1).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    X = (X - mu) / sd
    return E65.rdm_full(X)


def rdm_redundancy(feats_a, feats_b):
    """两组特征的「图像相似度结构」之间的 Spearman ⇒ 1 = 携带同一份信息。"""
    ra = E65.tri(img_rdm(feats_a))
    rb = E65.tri(img_rdm(feats_b))
    return float(E65.spearman(ra, rb))


def _bin_split(feats, ch_idx, n_ch, parity, n_bin=E89.N_BIN):
    """取偶数 / 奇数时间窗的特征（**分半可靠度**用：同一侧的两半 ⇒ 可靠度上限）。"""
    dims = [b * n_ch + c for b in range(n_bin) if b % 2 == parity
            for c in ch_idx]
    return feats[:, :, dims]


# ---------------------------------------------------------------------------
# 单被试
# ---------------------------------------------------------------------------

def run_subject(subject, IMG_test, IMG_train, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    t0 = time.time()

    eeg_te, npz_keys = E89.load_test_eeg(subject)
    rec["npz_keys"] = npz_keys
    n_img, n_rep, n_ch, T = eeg_te.shape
    log(f"  test split：{eeg_te.shape}")

    ch_names, eeg_tr = E89.load_train_chunk(subject, args.train_images)
    groups, lab = build_lat_groups(ch_names)
    rec["ch_names"] = list(ch_names)
    rec["laterality"] = lab
    rec["groups"] = {k: list(map(int, v)) for k, v in groups.items()}
    log("  左右分组：" + "  ".join(f"{k}={len(groups[k])}"
                                   for k in ("LEFT", "RIGHT", "MID", "OTHER", "ALL")))

    feats_te = E73.per_rep_targets(eeg_te, chunk=args.chunk)
    feats_tr = E73.per_rep_targets(eeg_tr, chunk=args.chunk)
    log(f"  特征：test {feats_te.shape} | train {feats_tr.shape}")

    # ---- L1：左右偏侧（RSA）----
    rs = {}
    for gname in ("LEFT", "RIGHT", "MID", "ALL"):
        idx = groups[gname]
        if not idx:
            continue
        f = E89.slice_group(feats_te, idx, n_ch)
        r, p = E89.rsa(f, IMG_test, args.n_perm, rng)
        rs[gname] = {"r": r, "p": p, "n_ch": len(idx)}
        log(f"  RSA {gname:<6} n_ch={len(idx):<3} r={r:+.4f} p={p:.4f}")
    rec["group_rsa"] = rs

    # ---- L2：左右结构冗余度（+ 同侧分半可靠度作上限）----
    f_left = E89.slice_group(feats_te, groups["LEFT"], n_ch)
    f_right = E89.slice_group(feats_te, groups["RIGHT"], n_ch)
    f_mid = E89.slice_group(feats_te, groups["MID"], n_ch)
    red_lr = rdm_redundancy(f_left, f_right)
    red_lm = rdm_redundancy(f_left, f_mid)
    red_rm = rdm_redundancy(f_right, f_mid)
    # 同侧分半（偶数窗 vs 奇数窗）⇒ 同一份信息的「可靠度上限」
    # ⚠️ 没有这个上限，L-R 的相关系数无法解释（它被信号强度压着）
    red_ceiling = rdm_redundancy(
        _bin_split(feats_te, groups["LEFT"], n_ch, 0),
        _bin_split(feats_te, groups["LEFT"], n_ch, 1))
    rec["rdm_redundancy"] = {"L_R": red_lr, "L_M": red_lm, "R_M": red_rm,
                             "CEILING": red_ceiling}
    log(f"  RDMRED L-R={red_lr:+.4f}  L-M={red_lm:+.4f}  R-M={red_rm:+.4f} "
        f"| 同侧分半上限={red_ceiling:+.4f}")

    # ---- L3：跨半球 transfer（**位点对齐版** + 错位对照）----
    left_order, right_order = mirrored_orders(ch_names, groups)
    li = [ch_names.index(c) for c in left_order]
    ri = [ch_names.index(c) for c in right_order]
    perm_r = [ri[k] for k in rng.permutation(len(ri))]     # 错位对照
    log(f"  对齐：左 {len(li)} 位点 / 右 {len(ri)} 位点；"
        f"前 5 对 = {[(left_order[i], right_order[i]) for i in range(5)]}")
    mb = {}

    def decode(idx_fit, idx_test, tag):
        if not idx_fit or not idx_test:
            return None
        Xtr = E89.slice_group(feats_tr, idx_fit, n_ch)
        Xte = E89.slice_group(feats_te, idx_test, n_ch)
        mu, sd = EVAL.zscore_fit(Xtr)
        rb = EVAL.metric_b_decode(EVAL.zscore_apply(Xtr, mu, sd),
                                  EVAL.zscore_apply(Xte, mu, sd),
                                  IMG_train[:args.train_images], IMG_test,
                                  rng, n_null=args.n_null)
        out = {"acc1": float(rb["acc1_cos"]),
               "null_acc1": float(np.median(rb["null_acc1_cos"]))}
        mb[tag] = out
        log(f"  MB {tag:<9} acc1={out['acc1']:.4f} (零中位 {out['null_acc1']:.4f})")
        return out

    decode(li, li, "SL2L")
    decode(ri, ri, "SR2R")
    decode(li, ri, "XL2R")          # 位点对齐的跨半球
    decode(li, perm_r, "XM2R")      # 错位对照（同一通道数，位点对应被打乱）
    decode(ri, li, "XR2L")
    decode(groups["ALL"], groups["ALL"], "ALL2ALL")
    rec["metric_b"] = mb

    # 跨 / 同 侧可迁移比例
    def ratio(a, b):
        x, y = mb.get(a), mb.get(b)
        if not x or not y or y["acc1"] <= 0:
            return None
        return x["acc1"] / y["acc1"]

    rec["transfer_ratio"] = {
        "XL2R_over_SL2L": ratio("XL2R", "SL2L"),
        "XR2L_over_SR2R": ratio("XR2L", "SR2R"),
        "XL2R_over_XM2R": ratio("XL2R", "XM2R"),
    }
    log("  TRANSFER " + "  ".join(
        f"{k}={v:.3f}" for k, v in rec["transfer_ratio"].items() if v is not None))

    rec["elapsed_s"] = round(time.time() - t0, 1)
    log(f"  用时 {rec['elapsed_s']}s")
    del eeg_te, eeg_tr, feats_te, feats_tr
    return rec


# ---------------------------------------------------------------------------
# 汇总（KEY 行，方便从日志直接重建）
# ---------------------------------------------------------------------------

def summarize(per_subject):
    out = {}

    def med(vals):
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return None
        s = sorted(vals)
        return float(s[len(s) // 2])

    for g in ("LEFT", "RIGHT", "MID", "ALL"):
        rs = [r["group_rsa"][g]["r"] for r in per_subject if g in r.get("group_rsa", {})]
        if rs:
            out[f"{g}_RSA_MEDIAN"] = med(rs)
            out[f"{g}_RSA_NSIG"] = sum(
                1 for r in per_subject
                if g in r.get("group_rsa", {}) and r["group_rsa"][g]["p"] < 0.05)
            out[f"{g}_RSA_N"] = len(rs)
    d = [r["group_rsa"]["LEFT"]["r"] - r["group_rsa"]["RIGHT"]["r"]
         for r in per_subject
         if "LEFT" in r.get("group_rsa", {}) and "RIGHT" in r.get("group_rsa", {})]
    if d:
        out["LEFT_MINUS_RIGHT_MEDIAN"] = med(d)
        out["LEFT_MINUS_RIGHT_NPOS"] = sum(1 for x in d if x > 0)
        out["LEFT_MINUS_RIGHT_N"] = len(d)
    for k in ("L_R", "L_M", "R_M", "CEILING"):
        v = [r["rdm_redundancy"][k] for r in per_subject if "rdm_redundancy" in r]
        if v:
            out[f"RDMRED_{k}_MEDIAN"] = med(v)
    for tag in ("SL2L", "SR2R", "XL2R", "XM2R", "XR2L", "ALL2ALL"):
        v = [r["metric_b"][tag]["acc1"] for r in per_subject
             if tag in r.get("metric_b", {})]
        if v:
            out[f"MB_{tag}_MEDIAN"] = med(v)
        nv = [r["metric_b"][tag]["null_acc1"] for r in per_subject
              if tag in r.get("metric_b", {})]
        if nv:
            out[f"MB_{tag}_NULL_MEDIAN"] = med(nv)
    for k in ("XL2R_over_SL2L", "XR2L_over_SR2R", "XL2R_over_XM2R"):
        v = [r["transfer_ratio"][k] for r in per_subject
             if k in r.get("transfer_ratio", {}) and r["transfer_ratio"][k] is not None]
        if v:
            out[f"TRANSFER_{k}_MEDIAN"] = med(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fte = torch.load(E89.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(E89.FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"图像特征：test {IMG_test.shape} | train {IMG_train.shape}")

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        log(f"\n=== sub-{sid:02d} ===")
        try:
            per_subject.append(run_subject(sid, IMG_test, IMG_train, args,
                                           np.random.default_rng(SEED + sid)))
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)

    summary = summarize(per_subject)
    log("\n=== 汇总（KEY）===")
    for k, v in summary.items():
        log(f"  KEY {k} {v}")
    (out_dir / "exp90_lateral.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject,
                    "summary": summary}, indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/exp90_lateral.json")


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _synth(n_img=30, n_rep=4, n_ch=63, T=250, d_img=24, mode="lateralized"):
    """合成 EEG。

    mode='lateralized'：只有**左**侧通道携带图像信息（右侧是噪声）
    mode='symmetric'：两侧都携带**同一份**图像信息
    """
    rng = np.random.default_rng(11)
    ch = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
          "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
          "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
          "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
          "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
          "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
          "O1", "Oz", "O2"]
    assert len(ch) == 63
    groups, lab = build_lat_groups(ch)
    IMG = rng.normal(size=(n_img, d_img))
    eeg = rng.normal(size=(n_img, n_rep, n_ch, T)).astype(np.float32) * 0.3
    W = rng.normal(size=(d_img, n_ch * E89.N_BIN)) * 0.0
    carry = []
    if mode == "lateralized":
        carry = groups["LEFT"]
    elif mode == "symmetric":
        carry = groups["LEFT"] + groups["RIGHT"]
    for c in carry:
        for b in range(E89.N_BIN):
            W[:, b * n_ch + c] = rng.normal(size=d_img) * 0.5
    win = T // E89.N_BIN
    for i in range(n_img):
        sig = (IMG[i] @ W).reshape(E89.N_BIN, n_ch)
        blk = np.repeat(sig.T[:, :, None], win, axis=2)   # (ch, bin, WIN)
        eeg[i] += blk.reshape(n_ch, T)[None, :, :]
    return eeg, IMG, ch, groups


def self_test():
    log("=== 自测：左右轴测量（合成数据）===")
    # --- 断言 1：左右分组必须是 28/28/7 ---
    eeg, IMG, ch, groups = _synth()
    assert len(groups["LEFT"]) == 28, len(groups["LEFT"])
    assert len(groups["RIGHT"]) == 28, len(groups["RIGHT"])
    assert len(groups["MID"]) == 7, len(groups["MID"])
    assert len(groups["OTHER"]) == 0
    log(f"  ✓ 左右分组：L={len(groups['LEFT'])} R={len(groups['RIGHT'])} "
        f"MID={len(groups['MID'])}")

    # --- 断言 2：偏侧模式下 LEFT 的 RSA 必须高于 RIGHT ---
    feats = E73.per_rep_targets(eeg, chunk=10)
    r_left, _ = E89.rsa(E89.slice_group(feats, groups["LEFT"], 63), IMG, 40,
                         np.random.default_rng(1))
    r_right, _ = E89.rsa(E89.slice_group(feats, groups["RIGHT"], 63), IMG, 40,
                          np.random.default_rng(1))
    log(f"  偏侧模式：LEFT r={r_left:+.4f} vs RIGHT r={r_right:+.4f}")
    assert r_left > r_right + 0.01, (r_left, r_right)

    # --- 断言 3：对称模式下 冗余度必须显著更高 ---
    eeg2, IMG2, ch2, g2 = _synth(mode="symmetric")
    f2 = E73.per_rep_targets(eeg2, chunk=10)
    red_lat = rdm_redundancy(E89.slice_group(feats, groups["LEFT"], 63),
                             E89.slice_group(feats, groups["RIGHT"], 63))
    red_sym = rdm_redundancy(E89.slice_group(f2, g2["LEFT"], 63),
                             E89.slice_group(f2, g2["RIGHT"], 63))
    log(f"  冗余度：偏侧模式 {red_lat:+.4f} vs 对称模式 {red_sym:+.4f}")
    assert red_sym > red_lat, (red_sym, red_lat)

    # --- 断言 4：RDM 冗余度自身可算（对称同一组 = 1.0）---
    same = rdm_redundancy(E89.slice_group(f2, g2["LEFT"], 63),
                          E89.slice_group(f2, g2["LEFT"], 63))
    assert abs(same - 1.0) < 1e-9, same
    log("  ✓ 自比冗余度 = 1.0（算对了）")

    # --- 断言 5：laterality 规则 ---
    assert laterality("O1") == "left" and laterality("O2") == "right"
    assert laterality("Oz") == "mid" and laterality("M1") == "other"
    log("  ✓ laterality 规则通过")

    # --- 断言 6：**位点镜像对齐**（L3 的前提；第一版就是栽在这里）---
    left_o, right_o = mirrored_orders(ch, groups)
    assert len(left_o) == len(right_o) == 28, (len(left_o), len(right_o))
    for a, b in zip(left_o, right_o):
        assert site_key(a) == site_key(b), (a, b)
    assert (left_o[0], right_o[0]) == ("Fp1", "Fp2"), (left_o[0], right_o[0])
    assert (left_o[-1], right_o[-1]) == ("O1", "O2"), (left_o[-1], right_o[-1])
    log(f"  ✓ 位点镜像对齐：前 3 对 "
        f"{[(left_o[i], right_o[i]) for i in range(3)]}，末对 "
        f"{(left_o[-1], right_o[-1])}")

    # --- 断言 7：分半（偶/奇窗）能切出同维数的两半 ---
    f_even = _bin_split(feats, groups["LEFT"], 63, 0)
    f_odd = _bin_split(feats, groups["LEFT"], 63, 1)
    assert f_even.shape == f_odd.shape == (feats.shape[0], feats.shape[1], 140), \
        (f_even.shape, f_odd.shape)
    log("  ✓ 分半切分：偶/奇窗各 140 维（28 通道 × 5 窗）")
    log("=== 自测通过 ===")


if __name__ == "__main__":
    main()
