"""V29.0 — 大语料 v27 对比 (5 patch × 4 mode × 5 seeds = 100 run).

数据: v29 LargeTextCorpus + LargeCodeCorpus + LargeImageCorpus (大语料)
STEPS: 500 (长训练, v19 H3)

对比: v27 小语料 (16/6/16/6 + 16/6, STEPS=100)

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

核心判断: 大语料 + 长训练下, v27 的 4 个 patch (l2/dropout/warmup/noise) 是否有效?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from experiment_env import log_experiment_env
import torch
import torch.nn.functional as F

from research.corpus_scale.v25_corpus_central import build_v25_param_groups
from research.corpus_scale.v27_router_norm_fix import (
    RegularizedTheoryLayer, compute_total_loss,
    DEFAULT_PATCH_CONFIGS, PATCH_NAMES,
)
from research.corpus_scale.v29_large_corpus import (
    LargeTextCorpus, LargeCodeCorpus, LargeImageCorpus,
    prepare_v29_real_corpus_data, prepare_v29_held_out_data,
)
from examples.run_v8_full import (
    load_encoders, make_pools,
    D_SHARED, N_CLS, B, S, SEEDS,
)

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


def run_seed(patch_name, mode, seed, encoders, attn_pools, text_corpus, code_corpus, image_corpus):
    torch.manual_seed(seed)
    patch_config = DEFAULT_PATCH_CONFIGS[patch_name]
    print(f"  [seed {seed}] patch={patch_name:>7} mode={mode:>14} ...")
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders

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

    layer = RegularizedTheoryLayer(
        d_shared=D_SHARED, num_experts=3,
        modal_dims=[D_bert, D_llama, D_vit],
        attn_pools=attn_pools, mode=mode,
        patch_config=patch_config,
    )

    layer.train()
    opt = torch.optim.AdamW(build_v25_param_groups(layer), lr=LR)
    for _ in range(STEPS_V29):
        opt.zero_grad(set_to_none=True)
        total_loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=N_CLS)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()

    layer.eval()
    with torch.no_grad():
        y = layer(h_by_modal)
        train_fuse = 0.0
        for c in range(N_CLS):
            for s in range(6):
                train_fuse += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        train_fuse /= 18

    with torch.no_grad():
        y_ho = layer(ho_by_modal)
        ho_fuse = 0.0
        for c in range(N_CLS):
            for s in range(6):
                ho_fuse += F.mse_loss(y_ho[s:s+1], ho_t_by_modal[c][s:s+1]).item()
        ho_fuse /= 18
    return train_fuse, ho_fuse


def main():
    print("=" * 78)
    print("V29.0 — 大语料 v27 对比 (5 patch × 4 mode × 5 seeds = 100 run)")
    print("=" * 78)
    print(f"{len(PATCH_NAMES)} patch × {len(MODES)} mode × {SEEDS} seeds = "
          f"{len(PATCH_NAMES) * len(MODES) * SEEDS} run")
    print(f"对比: v27 小语料 (16/6/16/6 + 16/6, STEPS=100)")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    llama_tok = encoders[1][0]
    text_corpus = LargeTextCorpus(bert_tok, max_length=S)
    code_corpus = LargeCodeCorpus(llama_tok, max_length=S)
    image_corpus = LargeImageCorpus(image_size=224)

    results = {p: {m: [] for m in MODES} for p in PATCH_NAMES}

    for patch_name in PATCH_NAMES:
        for mode in MODES:
            for seed in range(SEEDS):
                try:
                    train_fuse, ho_fuse = run_seed(
                        patch_name, mode, seed, encoders, attn_pools,
                        text_corpus, code_corpus, image_corpus,
                    )
                    results[patch_name][mode].append((train_fuse, ho_fuse))
                    print(f"  train={train_fuse:.4f}  held-out={ho_fuse:.4f}")
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    results[patch_name][mode].append((float("nan"), float("nan")))

    # 汇总
    print("\n" + "=" * 78)
    print("v29 大语料 held-out fuse (各 patch × mode)")
    print("-" * 78)
    print(f"{'patch':>10} | " + " | ".join(f"{m:>14}" for m in MODES))
    for patch_name in PATCH_NAMES:
        line = f"{patch_name:>10} | "
        for mode in MODES:
            vals = [h for _, h in results[patch_name][mode] if not np.isnan(h)]
            if vals:
                mean = statistics.mean(vals)
                line += f" {mean:>14.4f}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    # 找每个 mode 的最佳 patch (对比 v27 "none 是最佳" 的负面发现)
    print("\n" + "=" * 78)
    print("每个 mode 在 held-out 上的最佳 patch (v29)")
    print("-" * 78)
    for mode in MODES:
        best_patch = None
        best_heldout = float("inf")
        for patch_name in PATCH_NAMES:
            vals = [h for _, h in results[patch_name][mode] if not np.isnan(h)]
            if vals:
                mean = statistics.mean(vals)
                if mean < best_heldout:
                    best_heldout = mean
                    best_patch = patch_name
        baseline_vals = [h for _, h in results["none"][mode] if not np.isnan(h)]
        baseline_heldout = statistics.mean(baseline_vals) if baseline_vals else None
        if baseline_heldout:
            gain = (baseline_heldout - best_heldout) / baseline_heldout * 100
            print(f"  {mode:>14}: best={best_patch:>7} (held-out={best_heldout:.4f}, "
                  f"vs none={baseline_heldout:.4f}, gain={gain:+.1f}%)")
    print("=" * 78)


if __name__ == "__main__":
    main()
