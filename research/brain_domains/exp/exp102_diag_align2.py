#!/usr/bin/env python3
r"""exp102_diag_align2.py — **对齐闸门失败的第二个诊断：sub-05 到底是哪种「坏」**。

## 前一个诊断（`exp102_diag_align.py`）已经排除的

- ❌ **行序错**：`omnibrain4d/eeg/meta.parquet` 的「行号 → 概念号」映射对 **10/10 被试都是恒等**
  ⇒ 4D 的行序与自发现的概念顺序**一致** ⇒ 不是重排能修的。
- ❌ **NaN / 退化**：‖z‖ 中位数 81~115（正常），无 NaN。
- ❌ **4D 文件结构异常**：`axis_info.json` 里 sub-05 的 rep 可靠性 0.0561 在正常区间。

## 本脚本要分的四种「坏」

| 症状 | 判据 |
|---|---|
| ① **常数方向主导**（加性结构） | sim 可写成 $a_i+b_j$ ⇒ 每行 argmax 落在**同一列**；去均值后 200-way 立刻回来 |
| ② **试次问题** | 换别的 rep、或 80 试次平均后 200-way 恢复 |
| ③ **编码信息本身没有** | Hungarian 最佳匹配（= 信息上限）也 ≈ 随机 ⇒ 这批 EEG 与这批图无关 |
| ④ **只是弱** | 对角线排名不差但没到第一 ⇒ 200-way 低是「弱」不是「坏」 |

## 输出

逐被试一行：`raw / 去均值 / 逐行标准化 / 试次平均 / 对角线平均名次 / 不同 argmax 列数 / 贪心匹配上限`。

## 用法

```powershell
python research\\brain_domains\\exp\\exp102_diag_align2.py
python -u /domains/scripts/exp102_diag_align2.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

TEST_4D = Path("/domains/omnibrain4d/eeg")
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
DEFAULT_OUT = Path("/domains/runs/illusionbrain-align/exp102/gen2")
N_IMG = 200
TEST_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def log(*a):
    print(*a, flush=True)


def l2norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def load_true_clip(out_dir: Path):
    tp = out_dir / "true_clip.npy"
    if not tp.exists():
        raise SystemExit(f"!! {tp} 不存在 ⇒ 先跑一次 --step generate")
    return l2norm(np.load(tp).astype(np.float64))


def get_encoder(device: str):
    import torch
    import exp93b_contrastive as E93B
    ck = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    enc = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    enc.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    return enc.to(device).eval(), E93B


def encode(model, E93B, flat, device, batch=16):
    return E93B.encode_all(model, flat, device, batch).astype(np.float64)


def acc_of(cz, true_clip):
    sim = cz @ true_clip.T
    am = sim.argmax(axis=1)
    n = len(am)
    idx = np.arange(n)
    # 对角线平均名次（1 = 第一名；随机 ≈ (n+1)/2）
    order = np.argsort(-sim, axis=1)
    rank = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
    return {
        "acc": float((am == idx).mean()),
        "mean_diag_rank": float(rank.mean()),
        "n_distinct_argmax": int(len(set(am.tolist()))),
        "sim": sim,
        "am": am,
    }


def greedy_upper_bound(sim):
    """贪心一对一匹配能拿到的上限（信息上界的粗估；不是判据）。"""
    s = sim.copy()
    n = s.shape[0]
    hits = 0
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(s), s.shape)
        if i == j:
            hits += 1
        s[i, :] = -np.inf
        s[:, j] = -np.inf
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 20, 40, 60])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    true_clip = load_true_clip(out_dir)
    n = min(N_IMG, true_clip.shape[0])
    true_clip = true_clip[:n]
    model, E93B = get_encoder(args.device)

    rows = []
    for sid in TEST_SUBJECTS:
        eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
        n_img, n_rep = eeg.shape[0], eeg.shape[1]
        flat0 = np.asarray(eeg[:n, 0], dtype=np.float32)
        z = encode(model, E93B, flat0, args.device)
        base = acc_of(l2norm(z), true_clip)

        # ① 去均值（去常数方向）
        cz_c = l2norm(z - z.mean(axis=0, keepdims=True))
        cen = acc_of(cz_c, true_clip)

        # ② 逐行标准化
        zs = z - z.mean(axis=1, keepdims=True)
        zs = zs / np.maximum(zs.std(axis=1, keepdims=True), 1e-8)
        std = acc_of(l2norm(zs), true_clip)

        # ③ 试次平均（80 次重复一起平均 ⇒ 信噪比最高）
        avg_flat = np.asarray(eeg[:n].mean(axis=1), dtype=np.float32)
        z_avg = encode(model, E93B, avg_flat, args.device)
        avgc = acc_of(l2norm(z_avg), true_clip)

        # ④ 换几个试次
        rep_accs = {}
        for r in args.reps:
            if r >= n_rep:
                continue
            zr = encode(model, E93B, np.asarray(eeg[:n, r], dtype=np.float32), args.device)
            rep_accs[r] = acc_of(l2norm(zr), true_clip)["acc"]

        ub = greedy_upper_bound(base["sim"])
        rec = {
            "subject": sid, "n_img": int(n_img), "n_rep": int(n_rep),
            "acc_raw": base["acc"], "acc_centered": cen["acc"],
            "acc_rowstd": std["acc"], "acc_repavg": avgc["acc"],
            "rep_accs": rep_accs,
            "mean_diag_rank": base["mean_diag_rank"],
            "mean_diag_rank_centered": cen["mean_diag_rank"],
            "n_distinct_argmax": base["n_distinct_argmax"],
            "greedy_upper_bound": ub,
        }
        rows.append(rec)
        log(f"[sub-{sid:02d}] raw={rec['acc_raw']:.4f} 去均值={rec['acc_centered']:.4f} "
            f"逐行标准化={rec['acc_rowstd']:.4f} 试次平均={rec['acc_repavg']:.4f} "
            f"｜对角线名次 {rec['mean_diag_rank']:.0f}/{n}（随机 ≈{n // 2}）"
            f"｜不同 argmax 列 {rec['n_distinct_argmax']}｜贪心上限={ub:.4f}")
        log(f"           各试次 200-way：{rep_accs}")

    log("")
    log("=== 结论 ===")
    for r in rows:
        flag = ""
        if r["acc_raw"] < 0.01:
            flag = ("⇒ 去均值救回来了" if r["acc_centered"] >= 0.01 else "")
            flag += ("｜试次平均救回来了" if r["acc_repavg"] >= 0.01 else "")
            flag += ("｜贪心上限也≈随机 ⇒ 这批 EEG 与这批图无关" if r["greedy_upper_bound"] < 0.02 else "")
        log(f"  sub-{r['subject']:02d}: raw={r['acc_raw']:.4f} "
            f"center={r['acc_centered']:.4f} repavg={r['acc_repavg']:.4f} "
            f"ub={r['greedy_upper_bound']:.4f} {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
