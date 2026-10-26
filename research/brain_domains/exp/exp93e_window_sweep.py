#!/usr/bin/env python3
"""exp93e_window_sweep.py — **特征窗口**的扫描（位置轴 + 绝对档位，两件事一起查）。

## 为什么

exp93a 的 P3（同集身份解码）得 0.0432（8.6× 随机），**没复现** exp64 的
0.0785（15.7× 随机）。查下来不是实现坏了，而是**特征窗口不同**：

| | 窗口 | 每通道箱 |
|---|---|---|
| exp64 / Space worker | **50–500 ms**（12–125 样本） | 10 × 11 样本 |
| 我们的 630-d（exp73 口径） | **0–250 ms**（0–250 样本） | 10 × 25 样本 |

而 exp64 自己的逐窗剖面是：**100–200 ms 达峰（10.3×）、200–300 ms 8.8×、300–400 ms 3.0×、
400 ms 后回到随机** ⇒ 我们的 0–250 ms 窗口**切掉了 250–500 ms 的一大半信号**，
还塞进了 0–50 ms 的基线段。

⇒ 两个后果，都必须查：
1. **绝对档位**：手工 630-d 基线可能因为窗口选错而**被低估** ⇒ 我们与文献的差距被高估；
2. **位置轴**：所有位置轴读数（exp89 / exp91a / exp92-A）**都用 0–250 ms**
   ⇒ 「后部带语义、前部不带」这个结论**是否受窗口影响**，没查过。

⇒ 本脚本在**多个窗口**上同时测这两件事。

## 窗口（按样本下标，250 Hz）

| 名称 | 范围 | 说明 |
|---|---|---|
| `w0_250` | 0–250 | **现状**（exp73 口径）|
| `w50_500` | 12–125 | exp64 口径（信号窗）|
| `w100_400` | 25–100 | exp64 的 100–400 ms 合并窗 |
| `w50_250` | 12–62 | 去掉 0–50 ms 基线段 |
| `w100_250` | 25–62 | 100–250 ms（峰值段）|

每个窗口统一切成 **10 个等宽箱**（与现状「10 箱」同构）⇒ 630 维。

## 判据（预先写死）

1. **绝对档位**：哪个窗口的 P1 跨集 200-way top-1 最高？比现状高多少？
2. **位置轴**：每个窗口下的 POST − ANT 配对差（10 个被试、双侧符号检验）；
   ⇒ **结论必须在所有窗口下同向**，否则位置轴结论要加「窗口限定」；
3. **ANT 是否仍低于全部同尺寸随机组**（每窗各算自己的随机组地板）。

## 用法

    python -u research/brain_domains/exp/exp93e_window_sweep.py --self-test

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93e_window_sweep.py --subjects 1,2,3,4,5 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d --name t93e-win-s1-5
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

import exp73_decode200 as E73   # noqa: E402
import exp89_posaxis as E89      # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93e")

SEED = 0
N_BINS = 10
BASELINE = 12           # 0–48 ms（与 exp73 一致）
CHANCE = 1.0 / 200.0

# 窗口名 → (起样本, 止样本)
WINDOWS = {
    "w0_250": (0, 250),
    "w50_500": (12, 125),
    "w100_400": (25, 100),
    "w50_250": (12, 62),
    "w100_250": (25, 62),
}


def log(*a):
    print(*a, flush=True)


def windowed_targets(eeg, lo, hi, n_bins=N_BINS, baseline=BASELINE, chunk=200):
    """(n_img, n_rep, C, T) → (n_img, n_rep, C*n_bins)：去基线 + 取窗 + 等宽分箱。"""
    n_img, n_rep, n_ch, T = eeg.shape
    seg = (hi - lo) // n_bins
    out = np.empty((n_img, n_rep, n_ch * n_bins), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :baseline].mean(axis=3, keepdims=True)
        w = e[:, :, :, lo:lo + seg * n_bins]
        w = w.reshape(w.shape[0], n_rep, n_ch, n_bins, seg).mean(axis=4)
        w = w.transpose(0, 1, 3, 2)
        out[s:s + chunk] = w.reshape(w.shape[0], n_rep, n_ch * n_bins)
        del e, w
    return out


def mask_channels(eeg, keep_idx):
    x = eeg.copy()
    m = np.ones(x.shape[2], dtype=bool)
    m[np.asarray(keep_idx)] = False
    x[:, :, m, :] = 0.0
    return x


def self_test():
    log("=== 自测：窗口分箱 ===")
    rng = np.random.default_rng(0)
    e = rng.normal(size=(4, 2, 63, 250)).astype(np.float32)
    for name, (lo, hi) in WINDOWS.items():
        seg = (hi - lo) // N_BINS
        assert seg >= 1, (name, seg)
        X = windowed_targets(e, lo, hi)
        assert X.shape == (4, 2, 630), (name, X.shape)
    log(f"  ✓ {len(WINDOWS)} 个窗口都得到 (n_img, n_rep, 630)")
    # 基线扣除生效：常数偏移应被消掉
    e2 = e + 5.0
    a = windowed_targets(e, 12, 125)[0, 0]
    b = windowed_targets(e2, 12, 125)[0, 0]
    assert np.allclose(a, b, atol=1e-3), "基线扣除没生效"
    log("  ✓ 基线扣除生效（加常数不改变特征）")
    log("=== 自测通过 ===")


def run_subject(subject, IMG_train, IMG_test, args, rng):
    import torch
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_te = np.load(TEST_4D / f"sub-{subject:02d}.npz")["eeg"]
    n_img, n_rep, n_ch, T = eeg_te.shape
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    ch_names = list(obj["ch_names"])
    eeg_tr_all = np.asarray(obj["eeg"], dtype=np.float16)
    del obj
    groups, _ = E89.build_groups(ch_names)

    n_tr = min(args.train_images, len(eeg_tr_all))
    eeg_tr = eeg_tr_all[:n_tr]
    del eeg_tr_all

    for wname, (lo, hi) in WINDOWS.items():
        t0 = time.time()
        Xtr = windowed_targets(eeg_tr, lo, hi)
        Xte = windowed_targets(eeg_te, lo, hi)
        mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
        sd = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

        w = {"window": wname, "lo": lo, "hi": hi}
        # --- 位置轴：分组 RSA ---
        grp = {}
        for g in ("ALL", "POST", "ANT", "OCC"):
            idx = groups[g]
            if not idx:
                continue
            Xg = windowed_targets(mask_channels(eeg_te, idx), lo, hi)
            Xg = (Xg - mu) / sd
            r, p = E89.rsa(Xg, IMG_test, args.n_perm, rng)
            grp[g] = {"r": r, "p": p, "n_ch": len(idx)}
        w["groups"] = grp
        # --- 绝对档位：P1 跨集 200-way ---
        rb = E73.fit_and_decode(Xtr, Xte, IMG_train[:n_tr], IMG_test, None,
                                E73.ALPHAS, rng, n_null=0)
        w["level_acc1"] = rb["acc1_cos"]
        w["level_acc1_featavg"] = rb["acc1_featavg_cos"]
        w["level_alpha"] = rb["alpha"]
        # --- 该窗口自己的随机组地板（同尺寸，掩码也随机）---
        nulls = []
        for _ in range(args.n_random):
            sub = rng.choice(n_ch, size=len(groups["POST"]), replace=False)
            Xg = windowed_targets(mask_channels(eeg_te, sub), lo, hi)
            Xg = (Xg - mu) / sd
            r, _p = E89.rsa(Xg, IMG_test, args.n_perm_null, rng)
            nulls.append(r)
        w["null_median"] = float(np.median(nulls))
        w["null_max"] = float(np.max(nulls))
        w["POST_above_all_null"] = bool(grp["POST"]["r"] > w["null_max"])
        w["ANT_below_all_null"] = bool(grp["ANT"]["r"] < w["null_max"])
        w["elapsed_s"] = round(time.time() - t0, 1)
        rec.setdefault("windows", {})[wname] = w
        log(f"  [{wname:<9}] POST={grp['POST']['r']:+.4f} ANT={grp['ANT']['r']:+.4f} "
            f"| level acc1={w['level_acc1']:.4f} ({w['level_acc1']/CHANCE:.1f}×) "
            f"| 随机组中位 {w['null_median']:+.4f} 最大 {w['null_max']:+.4f} "
            f"| POST>{w['null_max']:+.4f}={w['POST_above_all_null']} "
            f"ANT<{w['null_max']:+.4f}={w['ANT_below_all_null']} [{w['elapsed_s']}s]")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=16540)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-perm-null", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    IMG_test = torch.load(FEAT_TEST, map_location="cpu",
                          weights_only=False)["img_features"].numpy().astype(np.float64)
    IMG_train = torch.load(FEAT_TRAIN, map_location="cpu",
                           weights_only=False)["img_features"].numpy().astype(np.float64)

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
    log(f"\n完成：{len(per_subject)-n_err}/{len(per_subject)} 成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))
    if per_subject and n_err == len(per_subject):
        log("✗✗ 全部被试都报错 ⇒ 结果不可用")

    (out_dir / "exp93e_window_sweep.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93e_window_sweep.json")


if __name__ == "__main__":
    main()
