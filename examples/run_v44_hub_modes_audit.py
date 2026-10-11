"""v44.0 — 四种中枢模式的容量-泄漏审计 (runner).

审计 ``v22_central_theory`` 的四模式比较, 即 ``paper.tex`` 主表
(``Adaptive-EMA achieves the lowest MSE on v22 (0.2972)``) 的依据。

两条:
1. **容量**  gate 的中枢参数 (512) 是其余三种 (1,024) 的一半, 而主表赢的是
   adaptive-ema (1,024) —— 混杂方向需实测确认。表见包内 ``capacity_table``。
2. **泄漏**  ``examples/run_v22_theory_compare.py:357-365`` 在训练同源数据上评估
   (``h_list = h_by_modal``)。本 runner 对每个模式同时报 in-sample 与 held-out,
   并给出**同 seed 配对**增益 —— v40/v43 两次教训: 均值会被单种子主导。

协议复用: 层构造、训练器、数据构造、评估口径全部 import
``examples/run_v22_theory_compare.py``, 只新增 held-out 评估。

用法::

    python3 examples/run_v44_hub_modes_audit.py                # 4 模式 × 5 seeds
    V44_SEEDS=1 python3 examples/run_v44_hub_modes_audit.py    # 冒烟
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.central_stability.v44_hub_modes_audit import (
    MODES, capacity_table, generalization_gap, paired_gain,
)

from examples.run_v22_theory_compare import TheoryLayer
from examples.run_v8_full import (
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
    load_encoders, make_pools, encode_text, encode_code, encode_image,
)

HELDOUT_OFFSET = 1000


def build_data(seed, encoders):
    """与 v22 的 run_seed 同构; 换足够大的 seed 即得同分布不同抽样。"""
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    h_text = encode_text(["cat dog bird", "animal pet wild", "feline canine fowl"],
                         bert_tok, bert)
    h_code = encode_code(["def hello():", "return True", "pass None"], llama_tok, llama)
    h_img = encode_image(np.random.randint(0, 256, (224, 224, 3)), vit_proc, vit)

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    return modal_seqs, modal_indices, targets


def bucket(modal_seqs, modal_indices, targets):
    h_by = [[] for _ in range(N_CLS)]
    t_by = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by[m].append(modal_seqs[i])
        t_by[m].append(targets[i])
    for c in range(N_CLS):
        h_by[c] = torch.stack(h_by[c], dim=0)
        t_by[c] = torch.stack(t_by[c], dim=0)
    return h_by, t_by


def evaluate(layer, h_by, t_by):
    """与 v22 run_seed 的评估口径逐行一致 (line 357-365)。"""
    with torch.no_grad():
        y = layer(h_by)
        total = 0.0
        for c in range(N_CLS):
            for s in range(6):
                total += F.mse_loss(y[s:s + 1], t_by[c][s:s + 1]).item()
    return total / 18


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = build_data(seed, encoders)
    layer = TheoryLayer(d_shared=D_SHARED, num_experts=N_CLS,
                        modal_dims=[e[2] for e in encoders],   # v8_full 的编码器是 (tok, model, X, D)
                        attn_pools=attn_pools, mode=mode)

    h_by, t_by = bucket(modal_seqs, modal_indices, targets)
    params = [p for p in layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = sum(F.mse_loss(layer(h_by), t_by[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        opt.step()

    mse_in = evaluate(layer, h_by, t_by)
    ho_seqs, ho_idx, ho_tgt = build_data(seed + HELDOUT_OFFSET, encoders)
    h_ho, t_ho = bucket(ho_seqs, ho_idx, ho_tgt)
    mse_out = evaluate(layer, h_ho, t_ho)
    return mse_in, mse_out


def main():
    seeds = int(os.environ.get("V44_SEEDS", SEEDS))
    print("=" * 92)
    print("V44.0 — 四种中枢模式: 容量 / 泄漏")
    print("=" * 92)
    print(f"{'模式':<14} {'中枢参数':>9} {'相对 gate':>10}   构成")
    for r in capacity_table(D_SHARED, N_CLS):
        comp = " + ".join(f"{k}[{v}]" for k, v in
                          (("c", r["c"]), ("W_t", r["W_t"]), ("U", r["U"]),
                           ("decay", r["decay"])) if v)
        print(f"{r['mode']:<14} {r['total']:>9,} {r['ratio_vs_gate']:>9.2f}×   {comp}")
    print()
    print(f"泄漏: v22 runner 在训练同源数据上评估(line 357-365), 本 runner 另报 held-out")
    print(f"{len(MODES)} 模式 × {seeds} seeds; 读数一律用同 seed 配对中位数")
    print("-" * 92)

    encoders = load_encoders()
    attn_pools = make_pools(encoders)

    raw = {m: [] for m in MODES}
    for seed in range(seeds):
        for mode in MODES:
            try:
                mi, mo = run_seed(mode, seed, encoders, attn_pools)
                raw[mode].append((mi, mo))
                print(f"  [seed {seed}] {mode:<14} in={mi:.4f}  held-out={mo:.4f}")
            except Exception as e:
                raw[mode].append((float("nan"), float("nan")))
                print(f"  [seed {seed}] {mode:<14} FAILED: {type(e).__name__}: {e}")

    print("-" * 92)
    base_in = [a for a, _ in raw["broadcast"] if not np.isnan(a)]
    base_out = [b for _, b in raw["broadcast"] if not np.isnan(b)]
    print(f"{'模式':<14} | {'in 中位':>8} {'out 中位':>9} {'泛化落差':>9} | "
          f"{'vs broadcast in':>17} {'vs broadcast out':>18}")
    print("-" * 92)
    verdict = {}
    for mode in MODES:
        vals = [(a, b) for a, b in raw[mode] if not np.isnan(a)]
        if not vals:
            continue
        ins = [a for a, _ in vals]; outs = [b for _, b in vals]
        gap = generalization_gap(ins, outs)
        gi = paired_gain(base_in, ins); go = paired_gain(base_out, outs)
        verdict[mode] = {"gap": gap, "gain_in": gi, "gain_out": go}
        print(f"{mode:<14} | {gap['in_sample_median']:>8.4f} {gap['heldout_median']:>9.4f} "
              f"{gap['relative_gap_pct']:>8.1f}% | {gi['median_gain_pct']:>+16.1f}% "
              f"{go['median_gain_pct']:>+17.1f}%")

    print("\n=== 主表主张: adaptive-ema 最优, 是否在 held-out 上存活 ===")
    ae = verdict.get("adaptive-ema")
    if ae:
        print(f"  in-sample : +{ae['gain_in']['median_gain_pct']:.1f}% vs broadcast "
              f"(胜 {ae['gain_in']['seeds_won']}/{ae['gain_in']['n']})")
        print(f"  held-out  : +{ae['gain_out']['median_gain_pct']:.1f}% vs broadcast "
              f"(胜 {ae['gain_out']['seeds_won']}/{ae['gain_out']['n']})")
    print("\n=== 每模式对 broadcast 的配对胜出 (held-out) ===")
    for mode in MODES:
        if mode == "broadcast":
            continue
        g = verdict.get(mode)
        if g:
            print(f"  {mode:<14} {g['gain_out']['seeds_won']}/{g['gain_out']['n']}  "
                  f"逐 seed {[f'{x:+.0f}' for x in g['gain_out']['per_seed']]}")
    print("=" * 92)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "central_stability", "v44_hub_modes_audit", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "v44_hub_modes_audit", "seeds": seeds, "modes": MODES,
                   "capacity": capacity_table(D_SHARED, N_CLS),
                   "raw": {k: v for k, v in raw.items()}, "verdict": verdict},
                  f, indent=2, ensure_ascii=False, default=float)
    print(f"写入: {out}")


if __name__ == "__main__":
    main()
