"""V60 补充分析：**只测 d=1**（最邻近），不用线性相关。

## 为什么需要

V60 预登记判据全败，但 `detrend` 臂出现一个未被判据捕捉的信号：
`M̄(1) ≈ −0.29`，而 `M̄(2..15)` 全在 −0.06…+0.06，**八层几乎不变**。

原因：`corr(d, M̄(d))` 是线性相关，d=2..15 的缓慢上升抵消了 d=1 的离群值
（总相关于是变成 +0.44，方向还反了）。**只在最邻近出现的结构会被线性相关稀释。**

## 判据（D0–D3，运行前固定）

    D0 正控:     纯噪声上 M̄(1) 的偏离不显著
    D1 显著:     真实 Δ ≡ M̄(1) − mean(M̄(2..S−1)) 落在置换零分布 [2.5%, 97.5%] 之外
    D2 一致:     ≥6/8 层的 Δ 同号
    D3 效应量:   |Δ| > 0.05

零分布：打乱 token 顺序（同时重排 M 的行列），重算 Δ。40 次。

**这是对已有数据的再分析，不是新实验**：只改观测量，不改数据。
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.aligner.v60_token_locality.locality import cos_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.environ.get("V60D1_ART_DIR", "v60d1_artifacts")
CACHE = os.environ.get("V60_CACHE", "/tmp/v60_tokens.npz")

N_TEXTS = int(os.environ.get("V60_N", "192"))
S_TOK = int(os.environ.get("V60_TOK", "16"))
N_LAYERS = int(os.environ.get("V60_LAYERS", "8"))
N_PERM = int(os.environ.get("V60D1_PERM", "40"))

PREDICTIONS = {
    "D0": "正控: 纯噪声上 M̄(1) 的偏离不显著",
    "D1": "显著: Δ 落在置换零分布 [2.5%,97.5%] 之外",
    "D2": "一致: ≥6/8 层 Δ 同号",
    "D3": "效应量: |Δ| > 0.05",
}


def build_cache():
    from research._primitives.scale_hetero_loaders import MSA_DIR, load_big_container
    from transformers import AutoTokenizer

    texts = [f"sample {i} the quick brown fox jumps over the lazy dog {i*7 % 97}"
             for i in range(N_TEXTS)]
    print(f"  {len(texts)} 条文本, {S_TOK} tokens, 前 {N_LAYERS} 层")
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


def delta_d1(M: torch.Tensor) -> float:
    """Δ = M̄(1) − mean(M̄(2..S−1))。正值 = 相邻更相似。"""
    S = M.shape[0]
    d1 = sum(float(M[i, i + 1]) for i in range(S - 1)) / (S - 1)
    rest = [float(M[i, i + d]) for d in range(2, S) for i in range(S - d)]
    return d1 - sum(rest) / len(rest)


def analyse(X, tag, n_perm):
    n, S, D = X.shape
    M = cos_matrix(X)
    d = delta_d1(M)

    g = torch.Generator().manual_seed(4242)
    null = []
    for _ in range(n_perm):
        p = torch.randperm(S, generator=g)
        null.append(delta_d1(M[p][:, p]))
    null.sort()
    lo = null[int(0.025 * (len(null) - 1))]
    hi = null[int(0.975 * (len(null) - 1))]
    outside = not (lo <= d <= hi)

    return {
        "arm": tag, "delta": round(d, 5),
        "null_lo": round(lo, 5), "null_hi": round(hi, 5),
        "null_mean": round(sum(null) / len(null), 5),
        "outside": bool(outside),
        "verdict_D1": bool(outside),
        "verdict_D3": abs(d) > 0.05,
    }


def main():
    t0 = time.time()
    print("=" * 84)
    print("V60 补充: 只测 d=1（最邻近）")
    print("=" * 84)
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print()

    L = load_layers()
    nL, n, S, D = L.shape
    print(f"  {nL} 层 × {n} 样本 × {S} tokens × {D} 维，置换 {N_PERM} 次\n")

    g = torch.Generator().manual_seed(11)
    Xn = torch.randn(n, S, D, generator=g)
    p0 = analyse(Xn, "null", 20)
    p0_ok = not p0["outside"]
    print("=" * 84)
    print("D0 · 纯噪声正控")
    print("=" * 84)
    print(f"  Δ={p0['delta']:+.5f}  零分布[{p0['null_lo']:+.5f}, {p0['null_hi']:+.5f}]")
    print(f"  不显著 ? {p0_ok}\n")

    layers = []
    for li in range(nL):
        X = L[li].contiguous()
        variants = {
            "raw": X,
            "centered": X - X.mean(dim=1, keepdim=True),
            "detrend": X - X.mean(dim=0, keepdim=True),
        }
        arms = {t: analyse(XX, t, N_PERM) for t, XX in variants.items()}
        layers.append({"layer": f"layer{li}", **arms})
        print("=" * 84)
        print(f"layer {li}")
        print("=" * 84)
        for t, a in arms.items():
            mark = "★显著" if a["outside"] else "      "
            print(f"  [{t:9s}] Δ={a['delta']:+.5f}  {mark}  "
                  f"零分布[{a['null_lo']:+.5f}, {a['null_hi']:+.5f}]  "
                  f"D1={a['verdict_D1']} D3={a['verdict_D3']}")
        print()

    def cnt(arm, key):
        return sum(1 for r in layers if r[arm][key])

    print("=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"  D0 正控 {p0_ok}\n")
    for arm in ("raw", "centered", "detrend"):
        ds = [r[arm]["delta"] for r in layers]
        neg = sum(1 for v in ds if v < 0)
        pos = sum(1 for v in ds if v > 0)
        sig = cnt(arm, "verdict_D1")
        same = max(neg, pos)
        print(f"  [{arm:9s}] D1 显著={sig}/{nL}  D2 同号={same}/{nL}  "
              f"D3 效应量={cnt(arm, 'verdict_D3')}/{nL}")
        print(f"              Δ 逐层: {[round(v, 4) for v in ds]}")
        print(f"              方向: 负(远端更相似)={neg}, 正(相邻更相似)={pos}")
    print()

    best = None
    for arm in ("raw", "centered", "detrend"):
        ds = [r[arm]["delta"] for r in layers]
        sig = cnt(arm, "verdict_D1")
        same = max(sum(1 for v in ds if v < 0), sum(1 for v in ds if v > 0))
        if sig >= 6 and same >= 6:
            best = arm
            print(f"  ⇒ {arm} 臂通过 D1+D2: 存在稳定的最邻近效应")
    if best is None:
        print("  ⇒ 无任何臂同时满足 D1(≥6/8) 与 D2(≥6/8)")
        print("     ⇒ v60 的 d=1 信号不构成稳定效应，FINDINGS 中的'信号1'应撤回")

    body = {
        "version": "60.1", "package": "v60_token_locality", "line": "aligner",
        "analysis": "d1_only",
        "config": {"n_texts": N_TEXTS, "tokens": S_TOK, "layers": N_LAYERS,
                   "dim": D, "n_perm": N_PERM},
        "predictions": PREDICTIONS,
        "null_control": {"delta": p0["delta"], "passed": p0_ok},
        "layers": layers,
        "summary": {
            "D0": p0_ok,
            "arms": {a: {"D1": cnt(a, "verdict_D1"), "D3": cnt(a, "verdict_D3")}
                     for a in ("raw", "centered", "detrend")},
            "confirmed_arm": best,
            "n_layers": nL,
        },
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "d1.json"), "w") as f:
        json.dump(body, f, ensure_ascii=False, indent=1)
    print(f"\n落盘 {ART}/d1.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
