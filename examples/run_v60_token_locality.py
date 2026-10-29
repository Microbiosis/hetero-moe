"""V60 — 换观测量：同一文本内位置对的余弦相似度 vs 距离。

## 为什么这一代值得跑

v58（任意轴）与 v59（token 轴）用"切口谱 → χ"都平凡。
v59 查明原因：χ 度量「两侧是否同源」，而同一文本的所有 token 天然同源。

本代换观测量：`M[i,j] = 跨样本平均 cos(X[s,i,:], X[s,j,:])`。
共同因只抬高截距、不改斜率；斜率（相邻是否更相似）才是位置结构的信号。

## 预登记判决（P0–P3）

    P0 正控:     纯噪声上 |corr(d, M̄(d))| < 0.5
    P1 距离衰减: corr(d, M̄(d)) < −0.5
    P2 置换可分: 真实 corr < 40 次打乱 token 顺序零分布的 2.5% 分位
    P3 局部性差: M̄(1) − M̄(S−1) > 0.01

## 三臂

| 臂 | 变换 | 用意 |
| --- | --- | --- |
| `raw` | 原样 | 含全部成分 |
| `centered` | `X[s,t,:] − mean_t X[s,:,:]` | 对应 v59 的做法；**已知会引入符号问题** |
| `detrend` | `X[s,t,:] − mean_s X[:,t,:]` | 去掉"位置 t 的通用方向"，保留样本特异偏离 |
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.aligner.v60_token_locality.locality import cos_profile, cos_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research/aligner/v60_token_locality/results")
ART = os.environ.get("V60_ART_DIR", OUT)
CACHE = os.environ.get("V60_CACHE", "/tmp/v60_tokens.npz")

N_TEXTS = int(os.environ.get("V60_N", "192"))
S_TOK = int(os.environ.get("V60_TOK", "16"))
N_LAYERS = int(os.environ.get("V60_LAYERS", "8"))
N_PERM = int(os.environ.get("V60_PERM", "40"))

PREDICTIONS = {
    "P0": "正控: 纯噪声上 |corr(d, M̄(d))| < 0.5",
    "P1": "距离衰减: corr(d, M̄(d)) < −0.5",
    "P2": "置换可分: 真实 corr < 40 次打乱 token 顺序零分布的 2.5% 分位",
    "P3": "局部性差: M̄(1) − M̄(S−1) > 0.01",
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


def main():
    t0 = time.time()
    print("=" * 84)
    print("V60 — token 轴局部性（观测量 = 同一文本内位置对的余弦相似度）")
    print("=" * 84)
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print()

    L = load_layers()
    nL, n, S, D = L.shape
    print(f"  {nL} 层 × {n} 样本 × {S} tokens × {D} 维\n")

    # ---- P0 正控 ----
    g = torch.Generator().manual_seed(11)
    Xn = torch.randn(n, S, D, generator=g)
    p0 = cos_profile(Xn, n_perm=20)
    p0_ok = abs(p0["corr"]) < 0.5
    print("=" * 84)
    print("P0 · 纯噪声正控")
    print("=" * 84)
    print(f"  corr = {p0['corr']:+.4f}   M̄(d) = {p0['profile']}")
    print(f"  |corr| < 0.5 ? {p0_ok}")
    print()

    layers = []
    for li in range(nL):
        X = L[li].contiguous()
        variants = {
            "raw": X,
            "centered": X - X.mean(dim=1, keepdim=True),
            "detrend": X - X.mean(dim=0, keepdim=True),
        }
        print("=" * 84)
        print(f"layer {li}")
        print("=" * 84)
        arms = {}
        for tag, XX in variants.items():
            a = cos_profile(XX, n_perm=N_PERM, seed=li * 10)
            arms[tag] = a
            print(f"  [{tag:9s}] corr={a['corr']:+.4f}  零下界={a['null_lo']:+.4f}  "
                  f"零均值={a['null_mean']:+.4f}  差={a['gap']:+.5f}")
            print(f"              P1={a['verdict_P1']} P2={a['verdict_P2']} "
                  f"P3={a['verdict_P3']}   M̄(d)={a['profile'][:5]}...")
        layers.append({"layer": f"layer{li}", **arms})
        print()

    def cnt(arm, key):
        return sum(1 for r in layers if r[arm][key])

    print("=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"  P0 正控(纯噪声)  {'通过' if p0_ok else '⛔失败'}")
    for arm in ("raw", "centered", "detrend"):
        print(f"  [{arm:9s}] P1={cnt(arm, 'verdict_P1')}/{nL}  "
              f"P2={cnt(arm, 'verdict_P2')}/{nL}  P3={cnt(arm, 'verdict_P3')}/{nL}")
    print()
    for arm in ("raw", "centered", "detrend"):
        both = sum(1 for r in layers
                   if r[arm]["verdict_P1"] and r[arm]["verdict_P2"])
        print(f"  {arm:9s} P1+P2 同时过: {both}/{nL}")
    hits = [arm for arm in ("raw", "centered", "detrend")
            if any(r[arm]["verdict_P1"] and r[arm]["verdict_P2"] for r in layers)]
    if hits:
        print(f"\n  ⇒ token 轴存在局部结构（{hits} 臂通过）")
    else:
        print("\n  ⇒ token 轴无局部结构：位置对的相似度与距离无关")
    print()
    for arm in ("raw", "centered", "detrend"):
        means = [sum(r[arm]["profile"]) / len(r[arm]["profile"]) for r in layers]
        corrs = [r[arm]["corr"] for r in layers]
        print(f"  {arm:9s} 每层 M̄ 均值: {[round(v,3) for v in means]}")
        print(f"  {'':9s} 每层 corr  : {[round(v,3) for v in corrs]}")

    body = {
        "version": "60.0", "package": "v60_token_locality", "line": "aligner",
        "config": {"n_texts": N_TEXTS, "tokens": S_TOK, "layers": N_LAYERS,
                   "dim": D, "n_perm": N_PERM},
        "predictions": PREDICTIONS,
        "null_control": {"corr": p0["corr"], "profile": p0["profile"], "passed": p0_ok},
        "layers": layers,
        "summary": {
            "P0": p0_ok,
            **{arm: {k: cnt(arm, f"verdict_{k}") for k in ("P1", "P2", "P3")}
               for arm in ("raw", "centered", "detrend")},
            "arms_with_locality": hits,
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
