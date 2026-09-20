"""V25.0 — v22 CentralTheory + v18 MiniCorpus 跨版本集成端到端 (4 mode × 5 seeds).

4 变体 × 5 seeds = 20 run:
    broadcast     — v8.0 默认 (z += U @ c)
    gate          — v9.0 修正 C (z / temperature)
    router-norm   — v22 新 (LayerNorm(z + U @ c))
    adaptive-ema  — v22 新 (broadcast + decay = sigmoid(α·||c||))

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

数据:
    - 训练: text 模态用 MiniCorpus.train_sentences (16 句 Wikipedia 真实文本)
            code/image 模态保留 v22 合成扰动 (诚实标记)
    - 评估:
        - train fuse: 训练集 (MiniCorpus.train_sentences 抽样) 的 18 个样本
        - held-out fuse: 全新 MiniCorpus.held_out 6 句 → 重复到 18 个样本, 测真实泛化

核心判断: 在真实 Wikipedia 文本上, CentralTheory 4 mode 的稳定性和排序是否保持?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.real_corpus import MiniCorpus
from research.corpus_scale.v25_corpus_central import (
    RealCorpusTheoryLayer,
    build_v25_param_groups,
    prepare_real_corpus_data,
    prepare_held_out_data,
)
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)

MODES = ["broadcast", "gate", "router-norm", "adaptive-ema"]


def group_by_modal(modal_seqs, modal_indices, targets, N_CLS):
    """把 (modal_seqs, modal_indices, targets) 按类分组, 得到 h_by_modal / t_by_modal."""
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    return h_by_modal, t_by_modal


def run_seed(mode: str, seed: int, encoders, attn_pools, corpus):
    """单 seed 训练 + 训练集评估 + held-out 评估."""
    torch.manual_seed(seed)
    print(f"  [seed {seed}] mode={mode} ...")
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders

    # 数据准备
    modal_seqs, modal_indices, targets = prepare_real_corpus_data(
        seed=seed, encoders=encoders, corpus=corpus,
        n_per_class=6, S=S, D_SHARED=D_SHARED, N_CLS=N_CLS,
    )
    h_by_modal, t_by_modal = group_by_modal(modal_seqs, modal_indices, targets, N_CLS)

    # held-out 数据
    ho_seqs, ho_indices, ho_targets = prepare_held_out_data(
        encoders=encoders, corpus=corpus,
        S=S, D_SHARED=D_SHARED, N_CLS=N_CLS, n_per_class=6,
    )
    ho_by_modal, ho_t_by_modal = group_by_modal(ho_seqs, ho_indices, ho_targets, N_CLS)

    # 模型
    layer = RealCorpusTheoryLayer(
        d_shared=D_SHARED, num_experts=3,
        modal_dims=[D_bert, D_llama, D_vit],
        attn_pools=attn_pools, mode=mode,
    )

    # 训练
    layer.train()
    opt = torch.optim.AdamW(build_v25_param_groups(layer), lr=LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        h_list = h_by_modal
        y = layer(h_list)
        total_loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()

    # 训练集评估
    layer.eval()
    with torch.no_grad():
        y = layer(h_by_modal)
        train_fuse = 0.0
        for c in range(N_CLS):
            for s in range(6):
                train_fuse += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        train_fuse /= 18

    # held-out 评估
    with torch.no_grad():
        y_ho = layer(ho_by_modal)
        heldout_fuse = 0.0
        for c in range(N_CLS):
            for s in range(6):
                heldout_fuse += F.mse_loss(y_ho[s:s+1], ho_t_by_modal[c][s:s+1]).item()
        heldout_fuse /= 18
    return train_fuse, heldout_fuse


def main():
    print("=" * 78)
    print("V25.0 — v22 CentralTheory + v18 MiniCorpus 跨版本集成")
    print("=" * 78)
    print(f"4 变体 × {SEEDS} seeds = 20 run")
    print("text 模态用 MiniCorpus.train_sentences 真实 Wikipedia; code/image 合成")
    print("评估: 训练集 fuse + held-out 6 句真实泛化 fuse")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    corpus = MiniCorpus(bert_tok, max_length=S)
    print(f"  MiniCorpus: train={len(corpus.train_sentences)} 句, held_out={len(corpus.held_out)} 句\n")

    train_results = {m: [] for m in MODES}
    heldout_results = {m: [] for m in MODES}

    for mode in MODES:
        for seed in range(SEEDS):
            try:
                train_fuse, ho_fuse = run_seed(mode, seed, encoders, attn_pools, corpus)
                train_results[mode].append(train_fuse)
                heldout_results[mode].append(ho_fuse)
                print(f"  [{mode:>14} seed {seed}] train={train_fuse:.4f}  held-out={ho_fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>14} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                train_results[mode].append(float("nan"))
                heldout_results[mode].append(float("nan"))

    # 汇总: 训练集
    print("\n" + "=" * 78)
    print(f"{'mode':>14} | {'train fuse':>11} | {'held-out fuse':>13} | {'vs broadcast':>14}")
    print("-" * 78)
    b_train_vals = [m for m in train_results["broadcast"] if not np.isnan(m)]
    if not b_train_vals:
        print("全部失败, 无法汇总")
        return
    b_train_mean = statistics.mean(b_train_vals)
    for m in MODES:
        t_vals = [x for x in train_results[m] if not np.isnan(x)]
        h_vals = [x for x in heldout_results[m] if not np.isnan(x)]
        if not t_vals:
            print(f"{m:>14} | {'N/A':>11} | {'N/A':>13} | {'N/A':>14}")
            continue
        t_mean = statistics.mean(t_vals)
        h_mean = statistics.mean(h_vals)
        h_std = statistics.stdev(h_vals) if len(h_vals) > 1 else 0.0
        gain = (b_train_mean - t_mean) / max(b_train_mean, 1e-9) * 100
        print(f"{m:>14} | {t_mean:>11.4f} | {h_mean:>13.4f} (±{h_std:.3f}) | {gain:>+13.1f}%")
    print("=" * 78)

    # 结论
    print("\n结论:")
    f_b_t = statistics.mean([m for m in train_results["broadcast"] if not np.isnan(m)])
    f_g_t = statistics.mean([m for m in train_results["gate"] if not np.isnan(m)])
    f_r_t = statistics.mean([m for m in train_results["router-norm"] if not np.isnan(m)])
    f_a_t = statistics.mean([m for m in train_results["adaptive-ema"] if not np.isnan(m)])
    print(f"  broadcast (训练集):     {f_b_t:.4f}")
    print(f"  gate (训练集):           {f_g_t:.4f}  (vs broadcast: {(f_b_t - f_g_t) / f_b_t * 100:+.1f}%)")
    print(f"  router-norm (训练集):    {f_r_t:.4f}  (vs broadcast: {(f_b_t - f_r_t) / f_b_t * 100:+.1f}%)")
    print(f"  adaptive-ema (训练集):  {f_a_t:.4f}  (vs broadcast: {(f_b_t - f_a_t) / f_b_t * 100:+.1f}%)")

    f_b_h = statistics.mean([m for m in heldout_results["broadcast"] if not np.isnan(m)])
    f_g_h = statistics.mean([m for m in heldout_results["gate"] if not np.isnan(m)])
    f_r_h = statistics.mean([m for m in heldout_results["router-norm"] if not np.isnan(m)])
    f_a_h = statistics.mean([m for m in heldout_results["adaptive-ema"] if not np.isnan(m)])
    print(f"\n  broadcast (held-out):   {f_b_h:.4f}")
    print(f"  gate (held-out):         {f_g_h:.4f}  (vs broadcast: {(f_b_h - f_g_h) / f_b_h * 100:+.1f}%)")
    print(f"  router-norm (held-out):  {f_r_h:.4f}  (vs broadcast: {(f_b_h - f_r_h) / f_b_h * 100:+.1f}%)")
    print(f"  adaptive-ema (held-out):{f_a_h:.4f}  (vs broadcast: {(f_b_h - f_a_h) / f_b_h * 100:+.1f}%)")

    best_train = min(f_g_t, f_r_t, f_a_t)
    best_heldout = min(f_g_h, f_r_h, f_a_h)
    print()
    if best_train == f_a_t:
        print("✅ 训练集: adaptive-ema 仍是最佳")
    elif best_train == f_r_t:
        print("✅ 训练集: router-norm 是最佳")
    else:
        print("✅ 训练集: gate 是最佳")
    if best_heldout == f_a_h:
        print("✅ held-out: adaptive-ema 仍是最佳")
    elif best_heldout == f_r_h:
        print("✅ held-out: router-norm 是最佳")
    else:
        print("✅ held-out: gate 是最佳")


if __name__ == "__main__":
    main()
