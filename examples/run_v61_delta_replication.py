"""V61 — Δ（最邻近效应）跨模型复制。

判据见 research/aligner/v61_delta_replication/SPEC.md（运行前固定）。

3 模型 × 2 长度 = 6 条件。
    big: EverMind-AI/MSA-4B
    emb: EverMind-AI/skillcorpus-embedding-0.6b
    rer: EverMind-AI/skillcorpus-reranker-0.6b

观测量与 v60.1 完全一致：
    M[i,j] = (1/n) Σ_s cos(X[s,i,:], X[s,j,:])
    Δ      = M̄(1) − mean(M̄(2..S−1))
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
ART = os.environ.get("V61_ART_DIR", "v61_artifacts")
CACHE_DIR = os.environ.get("V61_CACHE_DIR", "/tmp/v61_cache")

N_TEXTS = int(os.environ.get("V61_N", "192"))
LAYERS = int(os.environ.get("V61_LAYERS", "8"))
N_PERM = int(os.environ.get("V61_PERM", "40"))
LENGTHS = [int(x) for x in os.environ.get("V61_LENGTHS", "16,32").split(",")]
MODELS = os.environ.get("V61_MODELS", "big,emb,rer").split(",")

PREDICTIONS = {
    "R0": "正控: 6 个条件的纯噪声 Δ 全部不显著",
    "R1": "跨模型复制: detrend 臂 Δ<0 在 ≥5/6 条件成立, 每条件内 ≥6/8 层为负",
    "R2": "反向对照: raw 臂 Δ>0 在 ≥5/6 条件成立",
    "R3": "效应量: 全部条件的中位 |Δ| > 0.05",
    "R4": "非交替: 每条件 |mean(偶d) − mean(奇d≥3)| < 0.05",
}

# 文本必须长到 S=32 不触发 padding —— padding 位置全相同会污染 Δ
TEXT_TMPL = ("sample {i} the quick brown fox jumps over the lazy dog and then it "
             "runs away into the green forest near the river bank while {j} birds "
             "sing loudly every single morning of the week without fail")


def get_model(name):
    """返回 (encode_fn, D, n_layers)。encode_fn(texts, S) -> [n, S, D]。"""
    from research._primitives.scale_hetero_loaders import (
        MSA_DIR, EMB_DIR, RER_DIR, load_big_container, load_small,
        load_tokenizer, encode,
    )
    from transformers import AutoTokenizer

    if name == "big":
        big = load_big_container(n_layers=LAYERS)

        def enc(texts, S):
            tok = AutoTokenizer.from_pretrained(MSA_DIR)
            ids = tok(texts, padding="max_length", truncation=True,
                      max_length=S, return_tensors="pt")
            with torch.no_grad():
                big.config.output_hidden_states = True
                out = big(ids.input_ids, output_hidden_states=True)
            hs = [h.float() for h in out.hidden_states[:LAYERS]]
            mask = ids.attention_mask
            return torch.stack(hs, 0), int(mask.sum(1).min())
        return enc, None, None

    path = {"emb": EMB_DIR, "rer": RER_DIR}[name]
    model, D, _ = load_small(name, path)
    tok = load_tokenizer(path)

    def enc(texts, S):
        ids = tok(texts, padding="max_length", truncation=True,
                  max_length=S, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=ids.input_ids, output_hidden_states=True)
        hs = [h.float() for h in out.hidden_states[:LAYERS]]
        return torch.stack(hs, 0), int(ids.attention_mask.sum(1).min())

    return enc, D, None


def _stable_seed(*parts) -> int:
    """确定性 seed。不能用内置 hash(): 字符串 hash 每进程随机，会毁掉可复现性。"""
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) % 100003
    return h


def delta_d1(M: torch.Tensor) -> float:
    S = M.shape[0]
    d1 = sum(float(M[i, i + 1]) for i in range(S - 1)) / (S - 1)
    rest = [float(M[i, i + d]) for d in range(2, S) for i in range(S - d)]
    return d1 - sum(rest) / len(rest)


def parity_gap(M: torch.Tensor) -> float:
    """R4: mean(偶 d) − mean(奇 d≥3)。周期 2 的分量会把它推离 0。"""
    S = M.shape[0]
    ev, od = [], []
    for d in range(2, S):
        vals = [float(M[i, i + d]) for i in range(S - d)]
        (ev if d % 2 == 0 else od).append(sum(vals) / len(vals))
    if not ev or not od:
        return 0.0
    return sum(ev) / len(ev) - sum(od) / len(od)


def analyse(X, n_perm, seed):
    n, S, D = X.shape
    M = cos_matrix(X)
    d = delta_d1(M)
    pg = parity_gap(M)

    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_perm):
        p = torch.randperm(S, generator=g)
        null.append(delta_d1(M[p][:, p]))
    null.sort()
    lo = null[int(0.025 * (len(null) - 1))]
    hi = null[int(0.975 * (len(null) - 1))]
    return {
        "delta": round(d, 5), "null_lo": round(lo, 5), "null_hi": round(hi, 5),
        "outside": bool(not (lo <= d <= hi)),
        "parity_gap": round(pg, 5),
    }


def run_condition(mname, S):
    enc, D, _ = get_model(mname)
    texts = [TEXT_TMPL.format(i=i, j=i * 7 % 97) for i in range(N_TEXTS)]
    stack, min_len = enc(texts, S)
    nL, n, Sgot, Dgot = stack.shape
    padding_ok = (min_len == S)
    print(f"\n  [{mname} S={S}] 层={nL} n={n} S={Sgot} D={Dgot} "
          f"最短有效长度={min_len} padding?={'无' if padding_ok else '⚠有'}")
    if not padding_ok:
        print(f"    ⚠ S={S} 超过文本长度，存在 padding —— 该条件记录但需在解读时降权")

    arms = {"raw": [], "centered": [], "detrend": []}
    for li in range(nL):
        X = stack[li].contiguous()
        variants = {
            "raw": X,
            "centered": X - X.mean(dim=1, keepdim=True),
            "detrend": X - X.mean(dim=0, keepdim=True),
        }
        for tag, XX in variants.items():
            r = analyse(XX, N_PERM, seed=li * 10 + _stable_seed(mname, S, tag))
            r["layer"] = li
            arms[tag].append(r)

    g = torch.Generator().manual_seed(11)
    Xn = torch.randn(n, S, Dgot, generator=g)
    null_ctl = analyse(Xn, 20, 999)

    print(f"    detrend Δ: {[a['delta'] for a in arms['detrend']]}")
    print(f"    raw     Δ: {[a['delta'] for a in arms['raw']]}")
    print(f"    纯噪声  Δ={null_ctl['delta']:+.4f} 显著={null_ctl['outside']}")

    del stack
    gc.collect()
    return {
        "model": mname, "S": S, "layers": nL, "n": n, "D": Dgot,
        "padding_free": padding_ok, "min_effective_len": min_len,
        "arms": arms,
        "null_control": null_ctl,
    }


def main():
    t0 = time.time()
    print("=" * 88)
    print("V61 — Δ（最邻近效应）跨模型复制")
    print("=" * 88)
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print()

    conds = []
    for m in MODELS:
        for S in LENGTHS:
            try:
                conds.append(run_condition(m, S))
            except Exception as e:
                print(f"  ⛔ 条件 {m}/S={S} 失败: {type(e).__name__}: {e}")
                conds.append({"model": m, "S": S, "failed": str(e)[:300]})

    ok = [c for c in conds if "failed" not in c]
    n_ok = len(ok)

    print("\n" + "=" * 88)
    print("汇总")
    print("=" * 88)

    r0 = all(not c["null_control"]["outside"] for c in ok)
    print(f"  R0 正控(纯噪声 Δ 全不显著)  {'通过' if r0 else '⛔失败'}  ({n_ok}/{len(conds)} 条件可用)")

    def cond_pass(c, arm, sign):
        d = [a["delta"] for a in c["arms"][arm]]
        n_neg = sum(1 for v in d if v < 0)
        n_pos = sum(1 for v in d if v > 0)
        need = (len(d) * 6 + 7) // 8      # ≥6/8
        return (n_neg >= need) if sign < 0 else (n_pos >= need)

    r1_conds = [c for c in ok if cond_pass(c, "detrend", -1)]
    r2_conds = [c for c in ok if cond_pass(c, "raw", +1)]
    thresh = (n_ok * 5 + 5) // 6
    r1 = len(r1_conds) >= max(thresh, 1)
    r2 = len(r2_conds) >= max(thresh, 1)
    print(f"  R1 detrend Δ<0: {len(r1_conds)}/{n_ok} 条件  (需 ≥{thresh})  {'通过' if r1 else '未通过'}")
    print(f"  R2 raw     Δ>0: {len(r2_conds)}/{n_ok} 条件  (需 ≥{thresh})  {'通过' if r2 else '未通过'}")

    all_abs = sorted(abs(a["delta"]) for c in ok for arm in ("raw", "centered", "detrend")
                     for a in c["arms"][arm])
    med = all_abs[len(all_abs) // 2] if all_abs else 0.0
    r3 = med > 0.05
    print(f"  R3 中位 |Δ| = {med:.4f}  (需 >0.05)  {'通过' if r3 else '未通过'}")

    r4_bad = []
    for c in ok:
        for arm in ("raw", "centered", "detrend"):
            for a in c["arms"][arm]:
                if abs(a["parity_gap"]) >= 0.05:
                    r4_bad.append((c["model"], c["S"], arm, a["layer"], a["parity_gap"]))
    r4 = len(r4_bad) == 0
    print(f"  R4 非交替: {len(r4_bad)} 个 (条件,臂,层) 的奇偶差 ≥0.05  {'通过' if r4 else '未通过'}")
    if r4_bad[:5]:
        print(f"     前几例: {r4_bad[:5]}")

    print("\n  逐条件（detrend / raw 的 Δ 符号计数）:")
    for c in ok:
        dd = [a["delta"] for a in c["arms"]["detrend"]]
        rd = [a["delta"] for a in c["arms"]["raw"]]
        print(f"    {c['model']:4s} S={c['S']:<3} "
              f"detrend 负 {sum(1 for v in dd if v<0)}/{len(dd)} "
              f"raw 正 {sum(1 for v in rd if v>0)}/{len(rd)} "
              f"padding={'无' if c['padding_free'] else '有'}")

    print("\n" + "=" * 88)
    if not r0:
        print("  判决: R0 失败 ⇒ 整表作废")
    elif r1 and r2:
        print("  判决: R1 与 R2 同时成立 ⇒ 位置结构有两层且方向相反")
    elif r1 and not r2:
        print("  判决: 只有 R1 成立 ⇒ 残差层成立, raw 的正 Δ 需另找解释")
    elif not r1:
        print("  判决: R1 未通过 ⇒ v60.1 的观察**未能复制**，记为未复现")
    if not r4:
        print("  ⚠ R4 未过 ⇒ 部分效应可由周期 2 的交替分量解释")

    body = {
        "version": "61.0", "package": "v61_delta_replication", "line": "aligner",
        "config": {"n_texts": N_TEXTS, "layers": LAYERS, "lengths": LENGTHS,
                   "models": MODELS, "n_perm": N_PERM},
        "predictions": PREDICTIONS,
        "conditions": conds,
        "summary": {"R0": r0, "R1": r1, "R2": r2, "R3": r3, "R4": r4,
                    "R1_n": f"{len(r1_conds)}/{n_ok}", "R2_n": f"{len(r2_conds)}/{n_ok}",
                    "median_abs_delta": round(med, 5), "n_conditions": n_ok,
                    "r4_violations": len(r4_bad)},
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(body, f, ensure_ascii=False, indent=1)
    print(f"\n落盘 {ART}/metrics.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
