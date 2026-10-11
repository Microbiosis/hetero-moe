"""v46.0 — 「增益集中在 attention 通路」的样本外审计 (runner).

审计 ``microfusion.tex:77/151`` 的主张::

    "a path decomposition shows the gain concentrates in the attention path:
     cross-architecture differences surface mainly in attention patterns"

``examples/run_v7_cross_arch.py`` 的四个通路分解量 ``fuse / attn_only /
ffn_only / avg`` **全部在训练同源数据上评估**(第 100-122 行与训练共用
``x_shared_list`` / ``targets``), 且该 runner **没有任何 held-out 构造**。

本 runner 复用 v7 的全部构件(层构造 / 训练器 / 编码 / 数据构造), 只把评估
扩到两个口径:

- ``in_sample`` —— 与 v7 原口径一致(便于对齐既有数字)
- ``heldout``  —— 同分布不同抽样(``seed + 1000``: 换噪声 + 换随机图像)

读数一律用**配对逐样本中位数**(v40 起的纪律: 均值会被单种子主导)。

用法::

    python3 examples/run_v46_attn_path_audit.py              # 5 seeds
    V46_SEEDS=1 python3 examples/run_v46_attn_path_audit.py  # 冒烟
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.routing_evolution.v46_attn_path_audit import (
    decomposition_table, path_claim_verdict, path_gap,
)

# v7 的全部构件直接复用, 保证与被审计对象逐行同口径
from examples.run_v7_cross_arch import (
    B, D_SHARED, LR, N_CLS, S, SEEDS, STEPS, V7Trainer,
    build_layer, encode_code, encode_image, encode_text,
    load_encoders_and_attn,
)

HELDOUT_OFFSET = 1000


def _build(encoders, seed):
    """构造该 seed 的 18 个样本 + 目标 (与 v7 run_one 的第 1/3 步同构)。"""
    torch.manual_seed(seed)
    (bert_tok, bert, D_bert, _, _), \
    (llama_tok, llama, D_llama, _, _), \
    (vit_proc, vit, D_vit, _, _) = encoders

    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)

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


def _project(modal_seqs, modal_indices, layer):
    return [F.linear(h.unsqueeze(0), layer.P_m[cls])
            for h, cls in zip(modal_seqs, modal_indices)]


def _eval4(layer, x_list, targets):
    """v7 的四个通路分解量, 与 run_v7_cross_arch.py 第 100-122 行逐行一致。"""
    with torch.no_grad():
        f = a = ff = av = 0.0
        for x, t in zip(x_list, targets):
            t_b = t.unsqueeze(0)
            f += F.mse_loss(layer(x), t_b).item()
            attn_sum, _ = layer._attn_path(x)
            a += F.mse_loss(x + attn_sum, t_b).item()
            ffn_sum, _ = layer._ffn_path(x)
            ff += F.mse_loss(x + ffn_sum, t_b).item()
            x_det = x.detach()
            tot = torch.zeros_like(x)
            for m in range(layer.num_experts_ffn):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                gate = F.linear(h_a, layer.w_gates[m])
                up = F.linear(h_a, layer.w_ups[m])
                fv = F.linear(F.silu(gate) * up, layer.w_downs[m])
                tot = tot + fv
            av += F.mse_loss(x + tot / layer.num_experts_ffn, t_b).item()
    n = len(x_list)
    return {"fuse": f / n, "attn_only": a / n, "ffn_only": ff / n, "avg": av / n}


def run_seed(seed, encoders):
    modal_seqs, modal_indices, targets = _build(encoders, seed)
    layer = build_layer(encoders)
    x_train = _project(modal_seqs, modal_indices, layer)

    trainer = V7Trainer(layer, lr=LR, phase=2)
    for _ in range(STEPS):
        trainer.step(x_train, [t.unsqueeze(0) for t in targets])

    in_s = _eval4(layer, x_train, targets)

    ho_seqs, ho_idx, ho_tgt = _build(encoders, seed + HELDOUT_OFFSET)
    x_ho = _project(ho_seqs, ho_idx, layer)
    out_s = _eval4(layer, x_ho, ho_tgt)
    return {"seed": seed, "in_sample": in_s, "heldout": out_s}


def main():
    seeds = int(os.environ.get("V46_SEEDS", SEEDS))
    print("=" * 84)
    print("V46.0 — 「增益集中在 attention 通路」的样本外审计")
    print("=" * 84)
    print(f"审计对象: microfusion.tex:77/151 的通路分解主张")
    print(f"v7 原口径: 四个量全部 in-sample, 且无任何 held-out 构造")
    print(f"本代: {seeds} seeds × (in-sample + held-out), 配对逐样本中位数")
    print()

    encoders = load_encoders_and_attn()
    rows = []
    for seed in range(seeds):
        try:
            r = run_seed(seed, encoders)
            rows.append(r)
            print(f"  [seed {seed}] in  attn={r['in_sample']['attn_only']:.4f} "
                  f"ffn={r['in_sample']['ffn_only']:.4f}  |  "
                  f"out attn={r['heldout']['attn_only']:.4f} ffn={r['heldout']['ffn_only']:.4f}")
        except Exception as e:
            print(f"  [seed {seed}] FAILED: {type(e).__name__}: {e}")

    if not rows:
        print("无有效 seed"); return

    print("\n" + "=" * 84)
    print(decomposition_table(rows))
    print()
    verdicts = {}
    for scope in ("in_sample", "heldout"):
        a = [r[scope]["attn_only"] for r in rows]
        f = [r[scope]["ffn_only"] for r in rows]
        g = path_gap(a, f)
        v = path_claim_verdict(g)
        verdicts[scope] = {"gap": g, "verdict": v}
        print(f"  {scope:<10} 通路差中位 {g['median_pct']:+.1f}%  "
              f"attn 胜 {g['attn_wins']}/{g['n']}  ⇒ 裁决: {v}")
    same = verdicts["in_sample"]["verdict"] == verdicts["heldout"]["verdict"]
    print(f"\n  两口径结论一致: {same}")
    if not same:
        print("  ⇒ 该主张只在其原口径(in-sample)下成立, 必须按 held-out 重述。")
    print("=" * 84)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "routing_evolution", "v46_attn_path_audit", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "v46_attn_path_audit", "seeds": seeds,
                   "rows": rows, "verdicts": verdicts}, f, indent=2, ensure_ascii=False)
    print(f"写入: {out}")


if __name__ == "__main__":
    main()
