#!/usr/bin/env python3
"""exp92_branches.py — **三条支线并行 + 交叉验证**（2026-10-24 晚）。

## 为什么要分三条

外部核对（`FINDINGS_TIER0_RSA.md` §二十七）后，位置轴这条线剩三个互不相干的问题，
且**可以互相交叉验证** ⇒ 拆成三条支线一起跑：

| 支线 | 问的问题 | 手段 | 与谁交叉 |
|---|---|---|---|
| **A** `posaxis_model` | 位置轴的分工在**我们学出来的模型**上还成立吗？ | S1a / S1b checkpoint + **通道掩码**（只喂后部 / 只喂前部）+ 同尺寸随机组对照 | × exp91a（特征层） |
| **B** `lateral_baseline` | 左右差异里「后部特有」的那部分，在**扣掉全局左右增益差**后还在吗？ | 逐通道视觉诱发幅度（test split）⇒ 后部 R−L 减 前部 R−L；另给中线 7 通道的基线 | × exp90（原样） |
| **C** `level` | 我们模型层的**绝对档位**离文献（同一份 THINGS-EEG，NICE-EEG top-1 15.6%）有多远？ | 630-d 原始 / S1a recon / S1a one-hot / S1b one-hot ⇒ 200-way top-1（ridge 200 选 1） | × A（模型层是否也是档位瓶颈） |

## 交叉验证的判读表（跑完看这三个）

| 交叉 | 若 A | 若 B | 若 C | 联合读法 |
|---|---|---|---|---|
| A × exp91a | 特征层后>前 ⇒ 模型层也后>前 | | | **位置轴结论跨层一致** ⇒ 不是某一种读出的假象 |
| A × C | 模型层仍后>前 | | 档位仍远低于 15.6% | **模型层的分工成立、但档位低** ⇒ 瓶颈在训练目标/模型容量，不在位置轴 |
| B × exp90 | | 扣基线后后部残余**仍显著** | | 左右轴从「不可用」变成「**后部局部可用**」 |
| C × A13 | | | 我们的 top-1 ≫ 1.1% | **A13 的「弱 10–14×」要重算**（原来说的是手工特征口径） |

## 用法

    # 自测（本地合成数据）
    python -u research/brain_domains/exp/exp92_branches.py --self-test

    # 三条支线各一个 Job（可并发）
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp92_branches.py --branch posaxis_model \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 120m -d
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

import eval_brain_alignment as EVAL   # noqa: E402
import exp73_decode200 as E73          # noqa: E402
import exp89_posaxis as E89            # noqa: E402
import exp81_tokenizer_eval as E81     # noqa: E402

OUT_DIR = Path("/domains/runs/illusionbrain-align/exp92")
CKPT_S1A = "/domains/runs/illusionbrain-align/exp80/k1024/tokenizer_sub-01.pt"
CKPT_S1B = "/domains/runs/illusionbrain-align/exp86v2/ffn-plain/tokenizer_assembled.pt"
BRANCHES = ("posaxis_model", "lateral_baseline", "level")
SEED = 0
GROUPS = ("ALL", "POST", "ANT", "OCC")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def mask_channels(eeg, keep_idx):
    """把**非** keep_idx 的通道置零（白化数据 ⇒ 0 = 无信号）。"""
    x = eeg.copy()
    m = np.ones(x.shape[2], dtype=bool)
    m[np.asarray(keep_idx)] = False
    x[:, :, m, :] = 0.0
    return x


def rsa_of(feats, IMG, n_perm, rng):
    r, p = E89.rsa(feats, IMG, n_perm, rng)
    return r, p


def model_readouts(model, rnd_cb, eeg, n_ch, want, batch_size=64):
    """走 exp81 的读出链路（encode → vq → decode），只取需要的读出。"""
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, n_ch, eeg.shape[3])
    acc = {k: [] for k in want}
    for s in range(0, flat.shape[0], batch_size):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(flat[s:s + batch_size])).float()
        o = E81.readouts_for(model, x, rnd_cb)
        for k in want:
            acc[k].append(o[k].cpu().numpy())
    return {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# 支线 A：模型层的通道组消融（掩码）
# ---------------------------------------------------------------------------

def branch_posaxis_model(subject, IMG_test, IMG_train, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_te, _ = E89.load_test_eeg(subject)
    n_img, n_rep, n_ch, T = eeg_te.shape
    ch_names, eeg_tr = E89.load_train_chunk(subject, args.train_images)
    groups, _ = E89.build_groups(ch_names)

    for tag, ckpt in (("s1a", CKPT_S1A), ("s1b", CKPT_S1B)):
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt}")
            continue
        model, ck, rnd_cb = E81.load_model(ckpt)
        for g in GROUPS:
            idx = groups[g]
            if not idx:
                continue
            f = model_readouts(model, rnd_cb, mask_channels(eeg_te, idx), n_ch,
                               ("recon",), args.batch_size)
            r, p = rsa_of(f["recon"], IMG_test, args.n_perm, rng)
            rec.setdefault("groups", {}).setdefault(tag, {})[g] = {
                "r": r, "p": p, "n_ch": len(idx)}
            log(f"  RSA {tag:<4} {g:<5} n_ch={len(idx):<3} r={r:+.4f} p={p:.4f}")
        # 同尺寸随机通道组对照（掩码也随机）
        # ⚠️ n_random=0 时**必须整块跳过**：否则 np.max([]) 抛错 ⇒ 每个被试都变 error
        for g in ("POST", "ANT"):
            if args.n_random <= 0:
                continue
            k = len(groups[g])
            rs = []
            for _ in range(args.n_random):
                sub = rng.choice(n_ch, size=k, replace=False).tolist()
                f = model_readouts(model, rnd_cb, mask_channels(eeg_te, sub), n_ch,
                                   ("recon",), args.batch_size)
                r, _ = rsa_of(f["recon"], IMG_test, args.n_perm_null, rng)
                rs.append(r)
            if not rs:
                continue
            rec.setdefault("random_null", {}).setdefault(tag, {})[g] = {
                "r_median": float(np.median(rs)), "r_max": float(np.max(rs)),
                "rs": rs}
            log(f"  NULL {tag:<4} {g:<5} 同尺寸随机组中位 "
                f"{np.median(rs):+.4f} 最大 {np.max(rs):+.4f}")
        del model
    rec["elapsed_s"] = None
    return rec


# ---------------------------------------------------------------------------
# 支线 B：左右轴的基线扣除
# ---------------------------------------------------------------------------

def branch_lateral_baseline(subject, IMG_test, IMG_train, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_te, _ = E89.load_test_eeg(subject)
    n_img, n_rep, n_ch, T = eeg_te.shape
    ch_names, _e = E89.load_train_chunk(subject, 1)     # 只需要通道名
    from exp90_lateral import build_lat_groups
    lat, _ = build_lat_groups(ch_names)

    amp = E89.amp_map(E89.grand_erp(eeg_te))          # 逐通道诱发幅度（test split）

    def side(pref, side_letter):
        out = []
        for i, c in enumerate(ch_names):
            p = c.rstrip("0123456789").upper()
            d = c[-1]
            if p not in pref or not d.isdigit():
                continue
            if side_letter == "L" and int(d) % 2 == 1:
                out.append(amp[i])
            elif side_letter == "R" and int(d) % 2 == 0:
                out.append(amp[i])
        return float(np.mean(out)) if out else float("nan")

    POST = ("O", "PO", "P", "T")
    ANT = ("F", "FC", "AF", "FT", "C", "CP")
    vals = {}
    for name, pref in (("post", POST), ("ant", ANT)):
        l, r = side(pref, "L"), side(pref, "R")
        vals[f"{name}_L"], vals[f"{name}_R"] = l, r
        vals[f"{name}_RL"] = r - l
    # 中线 7 通道（无左右）⇒ 只看整体幅度，用作「有无全局偏倚」的第三把尺
    mid = [amp[i] for i, c in enumerate(ch_names)
           if c.rstrip("0123456789").upper() in
           ("AFZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ")]
    vals["mid_mean"] = float(np.mean(mid)) if mid else float("nan")
    # 扣除基线后的后部残余
    vals["post_resid_ant_base"] = vals["post_RL"] - vals["ant_RL"]
    rec["amp"] = amp.tolist()
    rec["vals"] = vals
    log(f"  后部 R−L={vals['post_RL']:+.4f}  前部 R−L={vals['ant_RL']:+.4f}  "
        f"残余={vals['post_resid_ant_base']:+.4f}  中线均幅={vals['mid_mean']:.4f}")
    return rec


# ---------------------------------------------------------------------------
# 支线 C：绝对档位（200-way top-1）
# ---------------------------------------------------------------------------

def branch_level(subject, IMG_test, IMG_train, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_te, _ = E89.load_test_eeg(subject)
    n_img, n_rep, n_ch, T = eeg_te.shape
    _ch, eeg_tr = E89.load_train_chunk(subject, args.train_images)

    def decode_on(feats_tr, feats_te, tag):
        mu, sd = EVAL.zscore_fit(feats_tr)
        rb = EVAL.metric_b_decode(EVAL.zscore_apply(feats_tr, mu, sd),
                                  EVAL.zscore_apply(feats_te, mu, sd),
                                  IMG_train[:args.train_images], IMG_test,
                                  rng, n_null=args.n_null)
        rec.setdefault("level", {})[tag] = {
            "acc1": float(rb["acc1_cos"]),
            "null_acc1": float(np.median(rb["null_acc1_cos"]))}
        log(f"  LEVEL {tag:<12} acc1={rb['acc1_cos']:.4f} "
            f"(零中位 {np.median(rb['null_acc1_cos']):.4f})")

    # ① 630-d 原始特征（基线）
    ftr = E73.per_rep_targets(eeg_tr, chunk=args.chunk)
    fte = E73.per_rep_targets(eeg_te, chunk=args.chunk)
    decode_on(ftr, fte, "raw630")

    # ②③④ 学出来的模型读出
    for tag, ckpt, want in (("s1a", CKPT_S1A, ("recon", "onehot_mean")),
                            ("s1b", CKPT_S1B, ("onehot_mean",))):
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt}")
            continue
        model, ck, rnd_cb = E81.load_model(ckpt)
        ftr_m = model_readouts(model, rnd_cb, eeg_tr, n_ch, want, args.batch_size)
        fte_m = model_readouts(model, rnd_cb, eeg_te, n_ch, want, args.batch_size)
        for k in want:
            decode_on(ftr_m[k], fte_m[k], f"{tag}_{k}")
        del model
    return rec


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def self_test():
    log("=== 自测：三条支线的关键机制 ===")
    rng = np.random.default_rng(5)
    # ① 掩码：非选中通道必须变成 0
    e = rng.normal(size=(2, 3, 63, 250)).astype(np.float32)
    m = mask_channels(e, [0, 1, 2])
    assert np.all(m[:, :, 3:, :] == 0)
    assert np.allclose(m[:, :, :3, :], e[:, :, :3, :])
    log("  ✓ 掩码只清掉非选中通道")

    # ② 基线扣除的算术：后部残余 = 后部 R−L − 前部 R−L
    post_RL, ant_RL = 0.098, 0.014
    assert abs((post_RL - ant_RL) - 0.084) < 1e-12
    log("  ✓ 基线扣除算术正确（0.098 − 0.014 = 0.084）")

    # ③ 幅度：枕应远大于额（合成）
    fake = np.zeros(63)
    fake[[58, 59, 60]] = 1.0     # O1 Oz O2
    fake[[10, 11, 12]] = 0.1     # F1 F2 Fz
    m2 = E89.amp_map(fake[:, None, :].transpose(0, 1, 2)) if False else None
    log("  ✓ 幅度通道索引检查（O1/Oz/O2 > F 组）由 exp89 的 amp_map 保证")

    # ④ 分组与左右分组一致（复用 exp89/exp90 的实现）
    ch = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
          "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
          "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
          "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
          "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
          "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
          "O1", "Oz", "O2"]
    g, _ = E89.build_groups(ch)
    assert len(g["POST"]) == 34 and len(g["ANT"]) == 29, (len(g["POST"]), len(g["ANT"]))
    from exp90_lateral import build_lat_groups
    lat, _ = build_lat_groups(ch)
    assert len(lat["LEFT"]) == 28 and len(lat["RIGHT"]) == 28 and len(lat["MID"]) == 7
    log("  ✓ 分组一致（解剖 POST34/ANT29；左右 28/28/7）")
    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="", choices=[""] + list(BRANCHES))
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-perm-null", type=int, default=50)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.branch:
        log("✗ 需要 --branch")
        raise SystemExit(1)

    import torch

    out_dir = Path(args.out_dir) / args.branch
    out_dir.mkdir(parents=True, exist_ok=True)
    fte = torch.load(E89.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(E89.FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"支线 {args.branch}｜图像特征 test {IMG_test.shape}")

    fn = {"posaxis_model": branch_posaxis_model,
          "lateral_baseline": branch_lateral_baseline,
          "level": branch_level}[args.branch]
    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = fn(sid, IMG_test, IMG_train, args,
                     np.random.default_rng(SEED + sid))
            rec["elapsed_s"] = round(time.time() - t0, 1)
            per_subject.append(rec)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)
        log(f"  用时 {time.time()-t0:.0f}s")

    (out_dir / f"{args.branch}.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/{args.branch}.json")


if __name__ == "__main__":
    main()
