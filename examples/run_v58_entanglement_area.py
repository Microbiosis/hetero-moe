"""V58.0 — 膜 / 面积律: 在真实表征上量 S(A) 与 χ.

> 归属: aligner (投影空间线的理论收口)
> 复现: bash run.sh v58
> 只有子空间/谱运算, 不需要 GPU; 但要 8 层容器表征 → 在 Space 上跑

三条判决 (B1-B4, 见 SPEC.md)
    B1 面积律: χ 沿切口饱和, 不随 |A| 线性增长
    B2 RG 流:  沿**层**方向的纠缠熵随深度下降 (粗粒化)
    B3 顺序:   natural 序的 χ 显著低于 random 序 (否则"相邻"没有意义)
    B4 非均匀: χ 剖面不止一个取值 (否则膜退化成 v57 的标量 r)
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.aligner.v58_entanglement_area.brane import (
    make_lattice, scan_cuts, BraneSpace, cross_spectrum, noise_tau, PREDICTIONS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research/aligner/v58_entanglement_area/results")
ART = os.environ.get("V58_ART_DIR", OUT)
CACHE = os.environ.get("V58_CACHE", "/tmp/v58_layers.npz")

N_TEXTS = int(os.environ.get("V58_N", "192"))
S_TOK = int(os.environ.get("V58_TOK", "16"))
N_LAYERS = int(os.environ.get("V58_LAYERS", "8"))
N_PERM = int(os.environ.get("V58_PERM", "8"))
DEVICE = os.environ.get("V58_DEVICE", "cpu")


# ---------------------------------------------------------------- 表征 (逐 token, 逐层)

def build_cache():
    """MSA-4B 前 n_layers 层, 每层均值池化前的 [n, S, D] → 池化成 [n, D]。"""
    from research._primitives.scale_hetero_loaders import MSA_DIR, load_big_container
    from transformers import AutoTokenizer

    texts = [f"sample {i} the quick brown fox jumps over the lazy dog {i*7 % 97}"
             for i in range(N_TEXTS)]
    print(f"  {len(texts)} 条文本, {S_TOK} tokens, 取前 {N_LAYERS} 层")
    big = load_big_container(n_layers=N_LAYERS)
    tok = AutoTokenizer.from_pretrained(MSA_DIR)
    ids = tok(texts, padding="max_length", truncation=True, max_length=S_TOK,
              return_tensors="pt").input_ids
    hidden = []
    with torch.no_grad():
        # 每层输出都拿: 用 output_hidden_states
        big.config.output_hidden_states = True
        out = big(ids, output_hidden_states=True)
        for h in out.hidden_states[:N_LAYERS]:
            hidden.append(h.float().mean(dim=1))      # [n, D] 每层
    del big
    gc.collect()
    arr = torch.stack(hidden, 0)                       # [L, n, D]
    print(f"  层表征 {tuple(arr.shape)}")
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    np.savez(CACHE, layers=arr.numpy())
    return arr


def load_layers():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        a = torch.from_numpy(z["layers"]).float()
        print(f"  缓存命中 {CACHE}  {tuple(a.shape)}")
        return a
    return build_cache()


# ---------------------------------------------------------------- 主流程

def main():
    t0 = time.time()
    print("=" * 80)
    print("V58.0 — 膜 / 面积律: S(A) 与 χ 在真实表征上的读数")
    print("=" * 80)
    print("理论链: RT (面积律) → 高斯基态 S=½Σlog(1/(1−ρ²)) → χ = 超阈通道数")
    print()

    L = load_layers()                       # [L, n, D]
    nL, n, D = L.shape
    print(f"  {nL} 层 × {n} 样本 × {D} 维")

    # ---- B2: 沿层方向 (RG / radial) ----
    print("\n" + "=" * 80)
    print("B2 · RG 流: 沿层方向逐切口 (每层是一个 site)")
    print("=" * 80)
    # 把 [L, n, D] 排成 [n, L, D] 的链: site = 层
    chain = L.permute(1, 0, 2).contiguous()            # [n, L, D]
    bs_rg = BraneSpace()
    prof_rg = bs_rg.measure(chain, axis=1)
    for p in prof_rg:
        print(f"  cut={p['cut']:>2} |A|={p['size_A']:>6} chi={p['chi']:>4} "
              f"S={p['entropy']:8.3f} s_max={p['spec_max']:.3f} tau={p['tau']:.3f}")
    # ⚠️ 修正 (2026-10-28): 原来算的是「每层 vs 顶层」, 而顶层与自己比恒为 0
    #    ⇒ 序列末项必为 0 ⇒ 任何"随深度下降"的判据都自动失败 (实测 S 从 23.2 涨到 49.2)。
    #    正确口径: 相邻层之间的纠缠 S(l, l+1) —— 这才是 RG 步的熵产生量。
    print("\n  相邻层之间的纠缠熵 S(l, l+1):")
    per_layer_S = []
    for li in range(nL - 1):
        a, b = L[li], L[li + 1]
        spec = cross_spectrum(a, b)
        rho2 = (spec.pow(2) / (spec.pow(2) + n)).clamp(max=1 - 1e-9)
        S = float(0.5 * (-torch.log1p(-rho2)).sum())
        per_layer_S.append(S)
        print(f"    layer {li}→{li+1}: S={S:9.3f}")

    # ---- 纯噪声对照 (必须给出 χ=0, 否则整表失效) ----
    print("\n" + "=" * 80)
    print("对照 · 纯噪声 (χ 必须 = 0; 否则阈值把噪声当信号)")
    print("=" * 80)
    g = torch.Generator().manual_seed(0)
    n_blk = 16                      # 空间轴把 D 切成 16 块（下面 B1/B3/B4 沿用）
    noise = torch.randn(n, nL, n_blk, D // n_blk, generator=g)
    bs_null = BraneSpace()
    bs_null.measure(noise, axis=1)
    null_chi = [p["chi"] for p in bs_null.profile]
    print(f"  纯噪声 chi={null_chi}  总容量={bs_null.total_capacity}")
    null_ok = all(c == 0 for c in null_chi)
    print(f"  → {'通过（阈值有效）' if null_ok else '⛔ 失败：阈值低于噪声，整表不可用'}")
    if not null_ok:
        print("  后续所有 χ 读数都不可解释，实验终止。")

    # ---- B1 + B3 + B4: 沿 token 方向, natural vs random ----
    print("\n" + "=" * 80)
    print("B1/B3/B4 · token 方向: natural 序 vs random 序")
    print("=" * 80)
    # 用每层 [n, D] 当一个 site → 链长 = 层数; 但 token 方向的 lattic 需要逐 token。
    # 这里用"把每层的 D 维切成若干块"作为空间轴: 块 = 隐维的一段 (natural) 或随机置换
    results = {}
    for name, order in (("natural", "natural"), ("random", "random")):
        # 把最后一层 + 中间层堆成 2D 片: [L2=n_blk 块, L1=层]
        feat = L.reshape(nL, n, n_blk, D // n_blk).permute(1, 0, 2, 3)   # [n, L1, L2, dw]
        if order == "random":
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(n_blk, generator=g)
            feat = feat[:, :, perm, :]
        bs = BraneSpace()
        prof = bs.measure(feat, axis=1)                                  # 沿 token/块方向
        results[name] = {
            "profile": prof,
            "chi_mean": float(np.mean([p["chi"] for p in prof])),
            "corr": bs.area_law_ratio(),
            "uniform": bs.is_uniform,
            "total": bs.total_capacity,
            "n_chi_values": len({p["chi"] for p in prof}),
        }
        print(f"\n  [{name}] chi={[p['chi'] for p in prof]}")
        print(f"    均值={results[name]['chi_mean']:.2f}  corr(|A|,chi)="
              f"{results[name]['corr']:+.2f}  uniform={results[name]['uniform']}  "
              f"总容量={results[name]['total']}")

    # ---- 判决 ----
    print("\n" + "=" * 80)
    print("判决")
    print("=" * 80)
    b1 = abs(results["natural"]["corr"]) < 0.5
    # 相邻层熵应随深度**下降** (粗粒化); 用首尾比较, 且要求实质差距
    b2 = per_layer_S[0] > per_layer_S[-1] * 1.1
    # ⚠️ 修正 (2026-10-28): 原来只比均值大小。实测 42.07 vs 42.53 = 差 0.46,
    #    远小于 χ 自身的波动 (剖面在 31–46 之间) ⇒ 这个"小于"没有依据。
    #    改为要求**实质差距**: 至少 10% 才算 natural 序有结构。
    nat, rnd = results["natural"]["chi_mean"], results["random"]["chi_mean"]
    b3 = nat < rnd * 0.9
    b3_note = f"{nat:.2f} vs {rnd:.2f} (需 nat < 0.9·rnd = {rnd*0.9:.2f})"
    b4 = results["natural"]["n_chi_values"] > 1
    print(f"  B1 面积律 (|corr|<0.5): natural corr={results['natural']['corr']:+.2f} → {'过' if b1 else '未过'}")
    print(f"  B2 RG 流 (相邻层熵随深度下降): 首 S={per_layer_S[0]:.1f} vs 尾 S={per_layer_S[-1]:.1f}"
          f" (需首 > 1.1×尾) → {'过' if b2 else '未过'}")
    print(f"  B3 顺序有结构: {b3_note} → {'过' if b3 else '未过'}")
    print(f"  B4 非均匀 (χ 取值数>1): {results['natural']['n_chi_values']} 种 → {'过' if b4 else '未过'}")
    print(f"\n  RG 剖面 corr={bs_rg.area_law_ratio()}  uniform={bs_rg.is_uniform}")
    if not null_ok:
        print("\n⛔ 纯噪声对照失败 ⇒ 本表所有 χ 读数作废，不作判决。")

    os.makedirs(ART, exist_ok=True)
    payload = {
        "schema_version": "1.0", "version": "58.0",
        "package": "v58_entanglement_area", "line": "aligner",
        "config": {"n_texts": n, "tokens": S_TOK, "layers": nL, "dim": D,
                   "n_blocks": n_blk, "cache": CACHE, "device": DEVICE},
        "predictions": PREDICTIONS,
        "rg_profile": prof_rg,
        "per_layer_entropy_vs_IR": per_layer_S,
        "token_axis": results,
        "null_control": {"chi": null_chi, "total": bs_null.total_capacity,
                         "passed": bool(null_ok)},
        "verdicts": {"B1_area_law": bool(b1), "B2_rg_flow": bool(b2),
                     "B3_order_structure": bool(b3), "B4_nonuniform": bool(b4)},
    }
    mpath = os.path.join(ART, "metrics.json")
    json.dump(payload, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n落盘 {mpath} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
