#!/usr/bin/env python3
"""exp88_codebook_geometry.py — 诊断：**学出来的码本**为什么不如**随机码本**。

## 背景（FINDINGS §二十三·四）

在「有语义监督」的配方下，`emb_mean`（学出来的码本，64 维）**低于**
`random_codebook`（随机码本，64 维）：

| 臂 | 学出来的码本 | 随机码本 | Δr |
|---|---:|---:|---:|
| v2-none | +0.0294 | +0.0495 | −0.0180（1/10） |
| v2-plain | +0.0221 | +0.0149 | +0.0068（8/10，p=0.055） |

⇒ **学出来的码本没有被语义组织**（它的几何至今只被重建目标组织）。

## 本脚本要回答的问题

在**不改任何东西**的前提下，看清码本的几何出了什么事：

| 量 | 为什么看它 |
|---|---|
| 平均两两余弦相似度 | 若码向量彼此高度相似 ⇒ 码本塌成「几乎同一个向量」⇒ 读出几乎无方差 ⇒ RSA 低 |
| 有效秩（奇异值的参与比） | 码本实际张成几个维度 |
| 每维方差剖面 | 若极度不均衡 ⇒ RSA（对每维线性敏感）会被大方差维主导 ⇒ 语义维被压掉 |
| 码向量范数分布 | 若与随机码本差一个量级 ⇒ 读出的尺度结构完全不同 |
| 码向量与随机码本的**同形对照** | 用同范数、同各向同性的随机矩阵，看 RSA 会不会一样 |

⇒ 这决定下一步该修什么：
- 塌缩/低秩 ⇒ 需要**多样性**约束（不是「让码本吃语义梯度」那么简单）；
- 各向异性 ⇒ 需要**归一化/白化**读出（评估口径问题，不是模型问题）；
- 尺度问题 ⇒ 需要修读出口径。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp88_codebook_geometry.py \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 60m -d \
      --ckpt /domains/runs/illusionbrain-align/exp86v2/ffn-plain/tokenizer_assembled.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
for _p in (str(EXP_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 0


def log(*a):
    print(*a, flush=True)


def geometry(E):
    """E: (K, D) 码向量矩阵 → 几何摘要。"""
    K, D = E.shape
    norms = np.linalg.norm(E, axis=1)
    En = E / np.maximum(norms[:, None], 1e-12)
    # 平均两两余弦（随机抽样，K=1024 时全算 52 万对也不贵，直接全算）
    G = En @ En.T
    iu = np.triu_indices(K, 1)
    cos_mean = float(G[iu].mean())
    cos_p95 = float(np.percentile(G[iu], 95))
    # 有效秩：奇异值的参与比 (Σλ)² / Σλ²
    sv = np.linalg.svd(E, compute_uv=False)
    lam = sv ** 2
    pr = float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12))
    # 每维方差
    var_d = E.var(axis=0)
    var_ratio = float(var_d.max() / max(var_d.mean(), 1e-12))
    return {
        "K": int(K), "D": int(D),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "cos_mean": cos_mean, "cos_p95": cos_p95,
        "eff_rank_participation": pr,
        "var_max_over_mean": var_ratio,
        "var_per_dim_min": float(var_d.min()), "var_per_dim_max": float(var_d.max()),
        "sing_top1_ratio": float(lam[0] / max(lam.sum(), 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    keys = [k for k in sd if k.endswith("vq.embedding.weight")]
    if not keys:
        raise RuntimeError(f"checkpoint 里没有 vq.embedding.weight（键：{list(sd)[:8]}…）")
    E = sd[keys[0]].float().numpy()
    log(f"checkpoint: {args.ckpt}")
    log(f"  arch={ck.get('arch')} donor_ffn={ck.get('donor_ffn')} "
        f"args.attn_donor={ck.get('args', {}).get('attn_donor')}")
    log(f"  码本 {E.shape}")

    g = geometry(E)
    log("\n=== 学出来的码本（几何）===")
    for k, v in g.items():
        log(f"  {k:<26} {v}")

    # --- 随机码本对照：三种生成方式 ---
    rng = np.random.default_rng(SEED)
    K, D = E.shape
    variants = {
        "random_N(0,0.5) [exp81 用的那个]": rng.normal(size=(K, D)) * 0.5,
        "random_N(0,1)": rng.normal(size=(K, D)),
        "random_匹配学出来的码本范数": (rng.normal(size=(K, D))
                                       * g["norm_mean"] / np.sqrt(D)),
    }
    log("\n=== 随机码本对照（几何）===")
    for name, R in variants.items():
        gr = geometry(R)
        log(f"  {name}")
        log(f"    norm_mean {gr['norm_mean']:.3f} | cos_mean {gr['cos_mean']:+.4f} "
            f"| eff_rank {gr['eff_rank_participation']:.1f} "
            f"| var_max/mean {gr['var_max_over_mean']:.2f}")

    log("\n=== 判读 ===")
    if g["cos_mean"] > 0.5:
        log("  ⚠ 码向量彼此高度相似（cos_mean > 0.5）⇒ **码本塌成近重复向量**"
            " ⇒ 读出几乎无方差 ⇒ RSA 低。修法是**多样性**约束，不是语义梯度。")
    elif g["eff_rank_participation"] < 0.3 * D:
        log(f"  ⚠ 有效秩只有 {g['eff_rank_participation']:.1f}/{D}"
            " ⇒ 码本张成的子空间太窄 ⇒ 语义方向可能被挤掉。")
    else:
        log(f"  ✓ 码本几何未见塌缩（cos_mean {g['cos_mean']:+.3f}，"
            f"有效秩 {g['eff_rank_participation']:.1f}/{D}）"
            " ⇒ 问题更可能在**读出口径/各向异性**，或**码本根本没吃到语义梯度**。")
    log(f"  · 学出来的码本：cos_mean {g['cos_mean']:+.4f}，"
        f"var_max/mean {g['var_max_over_mean']:.2f}，"
        f"范数 {g['norm_mean']:.3f}±{g['norm_std']:.3f}")
    log("  · exp81 的随机码本：N(0,0.5²)，各向同性、范数约 4.0")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"ckpt": args.ckpt, "learned": g}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        log(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"✗ {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1500:])
        raise SystemExit(1)
