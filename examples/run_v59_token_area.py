"""V59 — 面积律的**共同因**诊断：空间轴换成 token 位置。

## 这一代要回答的问题

v58 得到一个矛盾：面积律(B1)字面通过，但顺序检验(B3)两臂完全相同。
本轮单测复现了同一个现象：在位置信号很强的构造数据上，
natural 与 30 个置换**全部同分**（22）。这说明 χ 对"在哪里切"不敏感。

不敏感的原因是一个**共同因**：同一文本的所有 token 共享"这段文本的语义"。
它同时出现在任何二分的两侧 ⇒ χ 与切口位置无关 ⇒ 面积律成为平凡性质。

## 所以 v59 不再只测面积律，而是**分离共同因**

三臂：

| 臂 | 处理 | 回答 |
| --- | --- | --- |
| raw | 原样 | 共同因有多大 |
| **centered** | 去掉逐样本 token 均值 | **去掉共同因后，token 特有部分还有结构吗** |
| sample-shuffled | 打乱 B 的样本行 | 相关性是不是真的来自同源 |

## 判决（N0–N4，运行前固定）

    N0 同源对照:   sample-shuffled 的全部切口 χ = 0
    N1 切口敏感:   ≥3/15 切口上 natural χ 落在置换零分布 [2.5%,97.5%] 之外
    N2 面积律:     corr(|A|, χ_natural) < 0.5
    N3 非均匀:     χ 剖面 ≥ 3 种取值
    N4 正控:       纯噪声全部切口 χ = 0

**判读顺序 N4 → N0 → N1 → N2/N3。**
N1 不过 ⇒ N2 不可解释（v58 的教训：平凡性质也会"过"）。
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.aligner.v59_token_area.token_area import (
    cross_spectrum, noise_ceiling, cut_tokens, center_over_sites, shuffle_rows,
    chi_matched, chi_iid, TokenGrams, chi_from_grams, matched_tau_from_grams,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research/aligner/v59_token_area/results")
ART = os.environ.get("V59_ART_DIR", OUT)
CACHE = os.environ.get("V59_CACHE", "/tmp/v59_tokens.npz")

N_TEXTS = int(os.environ.get("V59_N", "192"))
S_TOK = int(os.environ.get("V59_TOK", "16"))
N_LAYERS = int(os.environ.get("V59_LAYERS", "8"))
N_PERM = int(os.environ.get("V59_PERM", "40"))
N_NULL = int(os.environ.get("V59_NULL", "10"))

PREDICTIONS = {
    "N0": "零分布匹配: iid 天花板会低估尾部(实测约 4 个假通道), 故主统计量用匹配零分布",
    "N1": "切口敏感: ≥3/15 切口上 natural χ 落在置换零分布 [2.5%,97.5%] 之外",
    "N2": "面积律: corr(|A|, χ_natural) < 0.5",
    "N3": "非均匀: χ 剖面 ≥ 3 种取值",
    "N4": "正控: 纯噪声全部切口 χ = 0",
}


def build_cache():
    """保留 token 轴: 每层 [n, S, D]（v58 在这里 mean(dim=1)，把 token 池掉了）。"""
    from research._primitives.scale_hetero_loaders import MSA_DIR, load_big_container
    from transformers import AutoTokenizer

    texts = [f"sample {i} the quick brown fox jumps over the lazy dog {i*7 % 97}"
             for i in range(N_TEXTS)]
    print(f"  {len(texts)} 条文本, {S_TOK} tokens, 前 {N_LAYERS} 层（**不池化 token**）")
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


def profile_arm(X, taus_iid, n_perm, seed_perm=2000, seed_mat=4000, tg=None):
    """一条臂的读数。用 TokenGrams 预编译，单次谱计算降到 O(n³)。"""
    n, S, D = X.shape
    if tg is None:
        tg = TokenGrams(X)
    nat, nat_iid, shuf_mat, perm = [], [], [], {k: [] for k in range(1, S)}

    for k in range(1, S):
        idx = list(range(k))
        G_a = tg.G(idx)
        G_b = tg.G_complement(idx)
        tau_m = matched_tau_from_grams(G_a, G_b, n, reps=10, seed=seed_mat + k)
        nat.append(chi_from_grams(G_a, G_b, n, tau_m))
        nat_iid.append(chi_from_grams(G_a, G_b, n, taus_iid[k - 1]))
        # 同源对照: 打乱 B 的行 → G_b 的行列同置换
        gp = torch.Generator().manual_seed(9000 + k)
        p = torch.randperm(n, generator=gp)
        G_bp = G_b[p][:, p]
        tau_s = matched_tau_from_grams(G_a, G_bp, n, reps=8, seed=seed_mat + 500 + k)
        shuf_mat.append(chi_from_grams(G_a, G_bp, n, tau_s))

    gp = torch.Generator().manual_seed(seed_perm)
    for _ in range(n_perm):
        for k in range(1, S):
            idx = torch.randperm(S, generator=gp)[:k].tolist()
            G_a = tg.G(idx)
            G_b = tg.G_complement(idx)
            tau = matched_tau_from_grams(G_a, G_b, n, reps=4, seed=seed_mat + 900 + k)
            perm[k].append(chi_from_grams(G_a, G_b, n, tau))

    detail, outside = [], 0
    for k in range(1, S):
        v = sorted(perm[k])
        lo = v[int(0.025 * (len(v) - 1))]
        hi = v[int(0.975 * (len(v) - 1))]
        out_flag = not (lo <= nat[k - 1] <= hi)
        outside += int(out_flag)
        detail.append({"k": k, "natural": nat[k - 1], "perm_lo": lo, "perm_hi": hi,
                       "perm_mean": round(float(np.mean(v)), 2), "outside": out_flag})

    sizes = [k * D for k in range(1, S)]
    corr = None
    if len(set(nat)) > 1:
        corr = float(np.corrcoef(sizes, nat)[0, 1])

    return {
        "natural": nat,
        "natural_iid_null": nat_iid,
        "shuffled": shuf_mat,
        "permutation_detail": detail,
        "n_outside": outside,
        "corr_size_chi": None if corr is None else round(corr, 4),
        "n_chi_values": len(set(nat)),
        "verdict_N0": all(c == 0 for c in shuf_mat),
        "verdict_N1": outside >= 3,
        "verdict_N2": (corr is not None and abs(corr) < 0.5),
        "verdict_N3": len(set(nat)) >= 3,
    }


def analyse_layer(X, tag, n_perm, n_null):
    n, S, D = X.shape

    taus = [noise_ceiling(n, k * D, (S - k) * D, reps=n_null, seed=1000 + k)
            for k in range(1, S)]

    g = torch.Generator().manual_seed(7)
    Xn = torch.randn(n, S, D, generator=g)
    null_chi = [int((cross_spectrum(*cut_tokens(Xn, k)) > taus[k - 1]).sum())
                for k in range(1, S)]

    Xc = center_over_sites(X)
    return {
        "layer": tag, "n": n, "S": S, "D": D,
        "taus": [round(t, 4) for t in taus],
        "raw": profile_arm(X, taus, n_perm),
        "centered": profile_arm(Xc, taus, n_perm, seed_perm=2100, seed_mat=5000),
        "null_chi": null_chi,
        "verdict_N4": all(c == 0 for c in null_chi),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("V59 — 面积律的共同因诊断（空间轴 = token 位置）")
    print("=" * 80)
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print()

    L = load_layers()
    nL, n, S, D = L.shape
    print(f"  {nL} 层 × {n} 样本 × {S} tokens × {D} 维\n")

    out_layers = []
    for li in range(nL):
        r = analyse_layer(L[li].contiguous(), f"layer{li}", N_PERM, N_NULL)
        out_layers.append(r)
        print("=" * 80)
        print(f"layer {li}")
        print("=" * 80)
        for arm in ("raw", "centered"):
            a = r[arm]
            print(f"  [{arm:8s}] chi = {a['natural']}")
            print(f"             置换均值 = {[d['perm_mean'] for d in a['permutation_detail']]}")
            print(f"             样本打乱 = {a['shuffled']}")
            print(f"             (iid 口径 = {a['natural_iid_null']})")
            print(f"             N0={a['verdict_N0']} N1={a['verdict_N1']} "
                  f"N2={a['verdict_N2']} N3={a['verdict_N3']}  "
                  f"corr={a['corr_size_chi']} 取值={a['n_chi_values']}")
        print(f"  纯噪声 chi = {r['null_chi']} → {'通过' if r['verdict_N4'] else '⛔失败'}")
        print()

    def cnt(arm, key):
        return sum(1 for r in out_layers if r[arm][key])

    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"  N4 正控(纯噪声,iid)  {sum(r['verdict_N4'] for r in out_layers)}/{nL}")
    for arm in ("raw", "centered"):
        print(f"  [{arm:8s}] N0 同源={cnt(arm,'verdict_N0')}/{nL}  "
              f"N1 切口敏感={cnt(arm,'verdict_N1')}/{nL}  "
              f"N2 面积律={cnt(arm,'verdict_N2')}/{nL}  "
              f"N3 非均匀={cnt(arm,'verdict_N3')}/{nL}")
    print()
    both = sum(1 for r in out_layers
               if r["centered"]["verdict_N1"] and r["centered"]["verdict_N2"])
    print(f"  centered 臂 N1+N2 同时通过: {both}/{nL}")
    if both == 0:
        print("  ⇒ 去掉共同因后，token 轴上没有可被该统计量读出的切口结构。")
    print()
    raw_mean = [float(np.mean(r["raw"]["natural"])) for r in out_layers]
    cen_mean = [float(np.mean(r["centered"]["natural"])) for r in out_layers]
    print(f"  raw      每层 χ 均值: {[round(v,1) for v in raw_mean]}")
    print(f"  centered 每层 χ 均值: {[round(v,1) for v in cen_mean]}")

    body = {
        "version": "59.0", "package": "v59_token_area", "line": "aligner",
        "config": {"n_texts": N_TEXTS, "tokens": S_TOK, "layers": N_LAYERS,
                   "dim": D, "n_perm": N_PERM, "n_null": N_NULL},
        "predictions": PREDICTIONS,
        "layers": out_layers,
        "summary": {
            "N4": sum(r["verdict_N4"] for r in out_layers),
            "raw": {k: cnt("raw", f"verdict_{k}") for k in ("N0", "N1", "N2", "N3")},
            "centered": {k: cnt("centered", f"verdict_{k}") for k in ("N0", "N1", "N2", "N3")},
            "centered_N1_and_N2": both,
            "raw_chi_mean": [round(v, 2) for v in raw_mean],
            "centered_chi_mean": [round(v, 2) for v in cen_mean],
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
