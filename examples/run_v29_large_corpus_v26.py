"""V29.0 — 大语料 v26 对比 (4 mode × 5 seeds = 20 run).

数据: v29 LargeTextCorpus (64+24) + LargeCodeCorpus (32+12) + LargeImageCorpus (50+16)
对比: v26 (16+6 / 16+6 / 16+6)

STEPS: 500 (5× v26, 来自 v19 H3 发现)

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

核心判断: 大语料 + 长训练下, router-norm / adaptive-ema 是否仍退步?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from experiment_env import log_experiment_env

from research._primitives.real_corpus import MiniCorpus
from research.corpus_scale.v25_corpus_central import (
    RealCorpusTheoryLayer,
    build_v25_param_groups,
)
from research.corpus_scale.v29_large_corpus import (
    LargeTextCorpus, LargeCodeCorpus, LargeImageCorpus,
    prepare_v29_real_corpus_data, prepare_v29_held_out_data,
)
from examples.run_v8_full import (
    load_encoders, make_pools,
    D_SHARED, N_CLS, B, S, SEEDS,
)

# v29 大语料: STEPS=500 (v19 H3 发现长训练改善 +9.5%)
STEPS_V29 = 500
LR = 1e-2
MODES = ["broadcast", "gate", "router-norm", "adaptive-ema"]


def group_by_modal(modal_seqs, modal_indices, targets, N_CLS):
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    return h_by_modal, t_by_modal


def run_seed(mode, seed, encoders, attn_pools, text_corpus, code_corpus, image_corpus):
    """单 seed 训练 + 训练集评估 + held-out 评估."""
    torch.manual_seed(seed)
    print(f"  [seed {seed}] mode={mode:>14} ...")
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders

    # 数据 (v29 大语料)
    modal_seqs, modal_indices, targets = prepare_v29_real_corpus_data(
        seed=seed, encoders=encoders,
        text_corpus=text_corpus, code_corpus=code_corpus, image_corpus=image_corpus,
        n_per_class=6, S=S, D_SHARED=D_SHARED, N_CLS=N_CLS,
    )
    h_by_modal, t_by_modal = group_by_modal(modal_seqs, modal_indices, targets, N_CLS)
    ho_seqs, ho_indices, ho_targets = prepare_v29_held_out_data(
        encoders=encoders,
        text_corpus=text_corpus, code_corpus=code_corpus, image_corpus=image_corpus,
        S=S, D_SHARED=D_SHARED, N_CLS=N_CLS, n_per_class=6,
    )
    ho_by_modal, ho_t_by_modal = group_by_modal(ho_seqs, ho_indices, ho_targets, N_CLS)

    # 模型
    layer = RealCorpusTheoryLayer(
        d_shared=D_SHARED, num_experts=3,
        modal_dims=[D_bert, D_llama, D_vit],
        attn_pools=attn_pools, mode=mode,
    )

    # 训练 (STEPS=500, 长训练)
    layer.train()
    opt = torch.optim.AdamW(build_v25_param_groups(layer), lr=LR)
    for _ in range(STEPS_V29):
        opt.zero_grad(set_to_none=True)
        y = layer(h_by_modal)
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
        ho_fuse = 0.0
        for c in range(N_CLS):
            for s in range(6):
                ho_fuse += F.mse_loss(y_ho[s:s+1], ho_t_by_modal[c][s:s+1]).item()
        ho_fuse /= 18
    return train_fuse, ho_fuse


def main():
    _env = log_experiment_env("run_v29_large_corpus_v26")
    print(f"  时间:       {_env['timestamp']}")
    print(f"  Python:     {_env['python_version'].split()[0]}")
    print(f"  PyTorch:    {_env['pytorch_version']}")
    print(f"  CUDA:       {_env['cuda_available']} ({_env['cuda_version']})")
    print(f"  设备:       {_env['device']}")
    print(f"  CPU 核心:   {_env['cpu_count']}")
    print("=" * 78)
    print()

    print("=" * 78)
    print("V29.0 — 大语料 v26 对比 (text 64/24, code 32/12, image 50/16, STEPS=500)")
    print("=" * 78)
    print(f"4 modes × {SEEDS} seeds = {len(MODES) * SEEDS} run")
    print(f"对比: v26 小语料 (16/6/16/6 + 16/6, STEPS=100)")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    llama_tok = encoders[1][0]
    text_corpus = LargeTextCorpus(bert_tok, max_length=S)
    code_corpus = LargeCodeCorpus(llama_tok, max_length=S)
    image_corpus = LargeImageCorpus(image_size=224)
    print(f"  text_corpus: train={len(text_corpus.train_sentences)}, held_out={len(text_corpus.held_out)}")
    print(f"  code_corpus: train={len(code_corpus.train_snippets)}, held_out={len(code_corpus.held_out)}")
    print(f"  image_corpus: train={len(image_corpus.train_themes)}, held_out={len(image_corpus.held_out)}\n")

    results = {m: [] for m in MODES}

    for mode in MODES:
        for seed in range(SEEDS):
            try:
                train_fuse, ho_fuse = run_seed(mode, seed, encoders, attn_pools,
                                                 text_corpus, code_corpus, image_corpus)
                results[mode].append((train_fuse, ho_fuse))
                print(f"  train={train_fuse:.4f}  held-out={ho_fuse:.4f}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append((float("nan"), float("nan")))

    # 汇总
    print("\n" + "=" * 78)
    print("v29 大语料 vs v26 小语料 (held-out fuse, 越低越好)")
    print("-" * 78)
    print(f"{'mode':>14} | {'v29 large':>11} | {'v26 small':>11} | {'delta':>10}")
    v26_small = {
        "broadcast": 0.4624,
        "gate": 0.4516,
        "router-norm": 0.4859,
        "adaptive-ema": 0.4931,
    }
    for mode in MODES:
        vals = [h for _, h in results[mode] if not np.isnan(h)]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            v26_val = v26_small[mode]
            delta = (v26_val - mean) / v26_val * 100
            marker = "✅" if mean < v26_val else ("⚠" if mean == v26_val else "❌")
            print(f"{mode:>14} | {mean:>11.4f} (±{std:.3f}) | {v26_val:>11.4f} | {delta:>+9.1f}% {marker}")
    print("=" * 78)


if __name__ == "__main__":
    main()
