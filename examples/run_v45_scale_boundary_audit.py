"""v45.0 — 语料规模边界的方向与样本外存活性审计 (runner).

审计 ``paper.tex`` 的**理论主张**::

    "establish a corpus-scale boundary showing that the optimal hub mode
     switches from gate-style to adaptive-EMA as data scale grows"

该论文主表自身给出的数据却是 **adaptive-ema -> gate**(小语料 adaptive-ema 优、
中大语料 gate 优, 翻转点在 medium)。摘要与自己的表方向相反。

本代在两点边界上做 held-out 检验:

- **small**  (v26 配置: MiniCorpus, STEPS=100)
- **large**  (v29 配置: LargeCorpus,  STEPS=500)

两个被审计的 runner (``run_v26_real_corpus.py`` / ``run_v29_large_corpus_v26.py``)
**本来就已经同时计算 train_fuse 与 ho_fuse**, 只是 held-out 从未被汇总进结论。
本代把那份已经算好却没人看的数字取出来用。

协议: 层构造、训练器、语料、训练与评估循环全部直接复用那两个 runner, 不重写。

用法::

    python3 examples/run_v45_scale_boundary_audit.py              # 2 规模 × 2 模式 × 5 seeds
    V45_SEEDS=1 python3 examples/run_v45_scale_boundary_audit.py  # 冒烟
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from research.corpus_scale.v45_scale_boundary_audit import (
    boundary_survives_heldout, scale_table,
)

import run_v26_real_corpus as V26
import run_v29_large_corpus_v26 as V29

MODES = ["gate", "adaptive-ema"]          # 边界主张涉及的两个模式
SCALES = ["small", "large"]


def build_corpora(encoders):
    """按两档规模各构造一份真实语料 (与两个原 runner 完全一致的构造)。"""
    # 直接复用两个被审计 runner 模块里**已经导入好**的语料类, 不自己猜路径
    from research._primitives.real_corpus import MiniCorpus
    from research.corpus_scale.v26_real_corpus import MiniCodeCorpus, MiniImageCorpus
    from research.corpus_scale.v29_large_corpus import (
        LargeTextCorpus, LargeCodeCorpus, LargeImageCorpus,
    )
    from examples.run_v8_full import S

    bert_tok, llama_tok = encoders[0][0], encoders[1][0]
    small = (MiniCorpus(bert_tok, max_length=S),
             MiniCodeCorpus(llama_tok, max_length=S),
             MiniImageCorpus(image_size=224))
    large = (LargeTextCorpus(bert_tok, max_length=S),
             LargeCodeCorpus(llama_tok, max_length=S),
             LargeImageCorpus(image_size=224))
    return {"small": small, "large": large}


def main():
    from examples.run_v8_full import load_encoders, make_pools

    seeds = int(os.environ.get("V45_SEEDS", 5))
    print("=" * 88)
    print("V45.0 — 语料规模边界: 方向判定 + 样本外存活性")
    print("=" * 88)
    print("主张: 最优中枢模式随规模 **gate -> adaptive-ema** (paper.tex 摘要)")
    print("主表: 小语料 adaptive-ema 优 / 中大语料 gate 优, 翻转点 medium")
    print("  ⇒ 摘要与自己的表**方向相反**")
    print(f"检验: {len(SCALES)} 档规模 × {len(MODES)} 模式 × {seeds} seeds (train + held-out)")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    corpora = build_corpora(encoders)
    runner = {"small": V26, "large": V29}

    raw = {sc: {m: [] for m in MODES} for sc in SCALES}
    for scale in SCALES:
        for mode in MODES:
            for seed in range(seeds):
                try:
                    tr, ho = runner[scale].run_seed(
                        mode, seed, encoders, attn_pools, *corpora[scale])
                    raw[scale][mode].append((tr, ho))
                    print(f"  [{scale:<5} {mode:<12} seed {seed}] train={tr:.4f} held-out={ho:.4f}")
                except Exception as e:
                    raw[scale][mode].append((float("nan"), float("nan")))
                    print(f"  [{scale:<5} {mode:<12} seed {seed}] FAILED: {type(e).__name__}: {e}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("-" * 88)
    rows = []
    tbl_in, tbl_out = [], []
    for scale in SCALES:
        med_in, med_out = {}, {}
        for mode in MODES:
            v = [(a, b) for a, b in raw[scale][mode] if not np.isnan(a)]
            if not v:
                continue
            med_in[mode] = statistics.median([a for a, _ in v])
            med_out[mode] = statistics.median([b for _, b in v])
            rows.append((scale, mode, med_in[mode], med_out[mode]))
        if len(med_in) == 2:
            tbl_in.append((scale, med_in))
            tbl_out.append((scale, med_out))

    print(scale_table(rows))

    if tbl_in and tbl_out:
        r = boundary_survives_heldout(tbl_in, tbl_out)
        print(f"\n=== 边界方向 ===")
        print(f"  in-sample 口径: {r['in_sample']['direction']}  "
              f"(翻转于 {r['in_sample']['switch_at']})  逐档 {r['in_sample']['per_scale']}")
        print(f"  held-out  口径: {r['heldout']['direction']}  "
              f"(翻转于 {r['heldout']['switch_at']})  逐档 {r['heldout']['per_scale']}")
        print(f"  方向一致: {r['direction_agrees']}   逐档最优者一致: {r['winners_agree']}")
        print(f"  裁决: **{r['verdict']}**")

    print("\n=== 摘要主张 (gate -> adaptive-ema) 是否成立 ===")
    if tbl_out:
        d = {sc: tbl_out[i][1] for i, sc in enumerate([t[0] for t in tbl_out])}
        low, high = SCALES[0], SCALES[-1]
        lo_w = min(d[low].items(), key=lambda kv: kv[1])[0]
        hi_w = min(d[high].items(), key=lambda kv: kv[1])[0]
        ok = (lo_w == "gate" and hi_w == "adaptive-ema")
        print(f"  held-out 实测: {low} 档优 = {lo_w}; {high} 档优 = {hi_w}")
        print(f"  摘要主张成立: **{ok}**"
              + ("" if ok else f"  ← 实测方向是 **{lo_w} -> {hi_w}**, 与摘要相反"))
    print("=" * 88)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "corpus_scale", "v45_scale_boundary_audit", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "v45_scale_boundary_audit", "seeds": seeds,
                   "scales": SCALES, "modes": MODES,
                   "raw": raw, "table_in": tbl_in, "table_out": tbl_out},
                  f, indent=2, ensure_ascii=False)
    print(f"写入: {out}")


if __name__ == "__main__":
    main()
