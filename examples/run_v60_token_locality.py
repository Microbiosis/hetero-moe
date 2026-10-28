"""V60 — 换观测量：逐 token 对的耦合强度 vs 距离。

## 为什么这一代值得跑

v58（任意轴）与 v59（token 轴）用"切口谱 → χ"都得到平凡结果。
v59 查明原因：χ 度量"两侧是否同源"，而同一文本的所有 token 天然同源，
于是它对"怎么切"不敏感。

本代不用 χ，直接量**局部性**：token i 与 j 的耦合是否随 |i−j| 衰减。
这是"空间有几何"能被检验的最小陈述；面积律只是它的推论。

## 预登记判决（P0–P3）

    P0 正控:     纯噪声上 |corr(d, M̄(d))| < 0.3
    P1 距离衰减: corr(d, M̄(d)) < −0.5
    P2 置换可分: 真实 corr < 40 次打乱 token 顺序零分布的 2.5% 分位
    P3 局部性比: M̄(1) / M̄(S−1) > 1.5

P0 不过 ⇒ 整表作废；P2 不过 ⇒ P1 可能只是单调趋势的副产品。
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.aligner.v60_token_locality.locality import (
    token_grams, coupling_matrix, distance_profile, corr, permuted_profile,
    center_over_sites,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research/aligner/v60_token_locality/results")
ART = os.environ.get("V60_ART_DIR", OUT)
CACHE = os.environ.get("V60_CACHE", "/tmp/v60_tokens.npz")

N_TEXTS = int(os.environ.get("V60_N", "192"))
S_TOK = int(os.environ.get("V60_TOK", "16"))
N_LAYERS = int(os.environ.get("V60_LAYERS", "8"))
N_PERM = int(os.environ.get("V60_PERM", "40"))
TOPK = int(os.environ.get("V60_TOPK", "3"))

PREDICTIONS = {
    "P0": "正控: 纯噪声上 |corr(d, M̄(d))| < 0.3",
    "P1": "距离衰减: corr(d, M̄(d)) < −0.5",
    "P2": "置换可分: 真实 corr < 40 次打乱 token 顺序零分布的 2.5% 分位",
    "P3": "局部性比: M̄(1)/M̄(S−1) > 1.5",
}


def build_cache():
    from research._primitives.scale_hetero_loaders import MSA_DIR, load_big_container
    from transformers import AutoTokenizer

    texts = [f"sample {i} the quick brown fox jumps over the lazy dog {i*7 % 97}"
             for i in range(N_TEXTS)]
    print(f"  {len(texts)} 条文本, {S_TOK} tokens, 前 {N_LAYERS} 层（不池化 token）")
    big = load_big_container(n_layers=N_LAYERS)
    tok = AutoTokenizer.from_pretrained(MSA_DIR)
    ids = tok(texts, padding="max_length", truncation=True, max_length=S_TOK,
              return_tensors="pt").input_ids
    hidden = []
    with torch.no_grad():
        big.config.output_hidden_states = True
        out = big(ids, output_hidden_states=True)
        for h in out.hidden_states[:N_LAYERS]:
            hidden.append(h.float())
    del big
    gc.collect()
    arr = torch.stack(hidden, 0)
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


def analyse_arm(X, tag, n_perm, topk):
    """单臂：耦合矩阵 → 距离剖面 → P1/P2/P3。"""
    n, S, D = X.shape
    G = token_grams(X)
    M = coupling_matrix(G, n, topk)
    ds, prof = distance_profile(M)
    r = corr(ds, prof)

    null = []
    for t in range(n_perm):
        _, pv = permuted_profile(M, seed=7000 + t)
        null.append(corr(ds, pv))
    null.sort()
    lo = null[int(0.025 * (len(null) - 1))]
    ratio = prof[0] / prof[-1] if prof[-1] > 0 else float("inf")

    return {
        "arm": tag,
        "profile": [round(float(v), 5) for v in prof],
        "distances": ds,
        "corr": round(r, 4),
        "null_lo": round(lo, 4),
        "null_mean": round(sum(null) / len(null), 4),
        "locality_ratio": round(float(ratio), 4),
        "matrix": [[round(float(M[i, j]), 4) for j in range(S)] for i in range(S)],
        "verdict_P1": r < -0.5,
        "verdict_P2": r < lo,
        "verdict_P3": ratio > 1.5,
    }


def main():
    t0 = time.time()
    print("=" * 84)
    print("V60 — token 轴局部性（换观测量：逐对耦合 vs 距离）")
    print("=" * 84)
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print()

    L = load_layers()
    nL, n, S, D = L.shape
    print(f"  {nL} 层 × {n} 样本 × {S} tokens × {D} 维")
    print(f"  耦合 = 互协方差谱前 {TOPK} 项之和；置换零分布 {N_PERM} 次\n")

    # ---- P0 正控: 纯噪声 ----
    g = torch.Generator().manual_seed(11)
    Xn = torch.randn(n, S, D, generator=g)
    p0 = analyse_arm(Xn, "null", 20, TOPK)
    p0_ok = abs(p0["corr"]) < 0.3
    print("=" * 84)
    print("P0 · 纯噪声正控")
    print("=" * 84)
    print(f"  profile = {p0['profile']}")
    print(f"  corr    = {p0['corr']}  →  |corr|<0.3 ? {p0_ok}")
    print()

    layers = []
    for li in range(nL):
        X = L[li].contiguous()
        print("=" * 84)
        print(f"layer {li}")
        print("=" * 84)
        arms = {}
        for tag, XX in (("raw", X), ("centered", center_over_sites(X))):
            a = analyse_arm(XX, tag, N_PERM, TOPK)
            arms[tag] = a
            print(f"  [{tag:8s}] corr={a['corr']:+.4f}  零分布下界={a['null_lo']:+.4f}  "
                  f"局部性比={a['locality_ratio']:.2f}")
            print(f"             M̄(d) = {a['profile']}")
            print(f"             P1={a['verdict_P1']} P2={a['verdict_P2']} P3={a['verdict_P3']}")
        layers.append({"layer": f"layer{li}", "raw": arms["raw"], "centered": arms["centered"]})
        print()

    def cnt(arm, key):
        return sum(1 for r in layers if r[arm][key])

    print("=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"  P0 正控(纯噪声)     {'通过' if p0_ok else '⛔失败'}")
    for arm in ("raw", "centered"):
        a, b, c = cnt(arm, "verdict_P1"), cnt(arm, "verdict_P2"), cnt(arm, "verdict_P3")
        print(f"  [{arm:8s}] P1 距离衰减={a}/{nL}  P2 置换可分={b}/{nL}  P3 局部性={c}/{nL}")
    both = sum(1 for r in layers
               if r["centered"]["verdict_P1"] and r["centered"]["verdict_P2"])
    print(f"\n  centered 臂 P1+P2 同时过: {both}/{nL}")
    if both > 0:
        print("  ⇒ token 轴存在局部结构（相邻 token 的耦合显著高于远距）")
    else:
        print("  ⇒ token 轴无局部结构：耦合与距离无关")

    body = {
        "version": "60.0", "package": "v60_token_locality", "line": "aligner",
        "config": {"n_texts": N_TEXTS, "tokens": S_TOK, "layers": N_LAYERS,
                   "dim": D, "n_perm": N_PERM, "topk": TOPK},
        "predictions": PREDICTIONS,
        "null_control": {"corr": p0["corr"], "profile": p0["profile"], "passed": p0_ok},
        "layers": layers,
        "summary": {
            "P0": p0_ok,
            "raw": {k: cnt("raw", f"verdict_{k}") for k in ("P1", "P2", "P3")},
            "centered": {k: cnt("centered", f"verdict_{k}") for k in ("P1", "P2", "P3")},
            "centered_P1_and_P2": both,
            "n_layers": nL,
        },
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(body, f, ensure_ascii=False, indent=1)
    print(f"\n落盘 {ART}/metrics.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
