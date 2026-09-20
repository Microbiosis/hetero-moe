"""V27.0 — router-norm / adaptive-ema held-out 修正端到端.

5 patch × 4 mode × 5 seeds = 80 run:
    patches ∈ {none, l2, dropout, warmup, noise}
    modes   ∈ {broadcast, gate, router-norm, adaptive-ema}

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

数据: 复用 v26 fully-real-corpus (text/code/image 全部真实)
评估: 训练集 fuse + held-out fuse 双评估

核心判断:
    - router-norm / adaptive-ema 在 v26 中 held-out 退步 (-5.1% / -6.6%)
    - v27 的 4 个 patch 能否让 router-norm / adaptive-ema 在 held-out 上不退步?
    - 找出对每个 mode 最有效的 patch 组合
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.real_corpus import MiniCorpus
from research.corpus_scale.v26_real_corpus import (
    MiniCodeCorpus, MiniImageCorpus,
    prepare_v26_real_corpus_data, prepare_v26_held_out_data,
)
from research.corpus_scale.v25_corpus_central import build_v25_param_groups
from research.corpus_scale.v27_router_norm_fix import (
    RegularizedTheoryLayer,
    compute_total_loss,
    DEFAULT_PATCH_CONFIGS, PATCH_NAMES,
)
from examples.run_v8_full import (
    load_encoders, make_pools,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)

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
    """单 seed 训练 + 训练集评估 + held-out 评估."""
    torch.manual_seed(seed)
    patch_config = DEFAULT_PATCH_CONFIGS[patch_name]
    print(f"  [seed {seed}] patch={patch_name:>7} mode={mode:>14} ...")
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders

    # 数据
    modal_seqs, modal_indices, targets = prepare_v26_real_corpus_data(
        seed=seed, encoders=encoders,
        text_corpus=text_corpus, code_corpus=code_corpus, image_corpus=image_corpus,
        n_per_class=6, S=S, D_SHARED=D_SHARED, N_CLS=N_CLS,
    )
    h_by_modal, t_by_modal = group_by_modal(modal_seqs, modal_indices, targets, N_CLS)
    ho_seqs, ho_indices, ho_targets = prepare_v26_held_out_data(
        encoders=encoders,
        text_corpus=text_corpus, code_corpus=code_corpus, image_corpus=image_corpus,
        S=S, D_SHARED=D_SHARED, N_CLS=N_CLS, n_per_class=6,
    )
    ho_by_modal, ho_t_by_modal = group_by_modal(ho_seqs, ho_indices, ho_targets, N_CLS)

    # 模型 (带 v27 patch)
    layer = RegularizedTheoryLayer(
        d_shared=D_SHARED, num_experts=3,
        modal_dims=[D_bert, D_llama, D_vit],
        attn_pools=attn_pools, mode=mode,
        patch_config=patch_config,
    )

    # 训练
    layer.train()
    opt = torch.optim.AdamW(build_v25_param_groups(layer), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        total_loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=N_CLS)
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
    print("=" * 78)
    print("V27.0 — router-norm / adaptive-ema held-out 修正")
    print("=" * 78)
    print(f"{len(PATCH_NAMES)} patch × {len(MODES)} mode × {SEEDS} seeds = "
          f"{len(PATCH_NAMES) * len(MODES) * SEEDS} run")
    print(f"patches: {PATCH_NAMES}")
    print(f"modes:   {MODES}")
    print("数据: 复用 v26 fully-real-corpus (text/code/image 全部真实)")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    llama_tok = encoders[1][0]
    text_corpus = MiniCorpus(bert_tok, max_length=S)
    code_corpus = MiniCodeCorpus(llama_tok, max_length=S)
    image_corpus = MiniImageCorpus(image_size=224)

    # 嵌套 dict: results[patch][mode] = [(train, held_out), ...]
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

    # 汇总: 按 mode × patch 展示训练集 + held-out
    print("\n" + "=" * 78)
    print("训练集 fuse (各 patch × mode)")
    print("-" * 78)
    print(f"{'patch':>10} | " + " | ".join(f"{m:>14}" for m in MODES))
    for patch_name in PATCH_NAMES:
        line = f"{patch_name:>10} | "
        for mode in MODES:
            vals = [t for t, _ in results[patch_name][mode] if not np.isnan(t)]
            if vals:
                mean = statistics.mean(vals)
                line += f" {mean:>14.4f}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    print("\n" + "=" * 78)
    print("held-out fuse (各 patch × mode)")
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

    # 关键判断: 找每个 mode 的最佳 patch (held-out 最低)
    print("\n" + "=" * 78)
    print("每个 mode 在 held-out 上的最佳 patch")
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
        else:
            print(f"  {mode:>14}: best={best_patch:>7} (held-out={best_heldout:.4f})")

    # v26 baseline 对比
    print("\n" + "=" * 78)
    print("vs v26 baseline (patch='none'):")
    print("-" * 78)
    v26_baseline = {
        "broadcast": 0.4624,
        "gate": 0.4516,
        "router-norm": 0.4859,
        "adaptive-ema": 0.4931,
    }
    for mode in MODES:
        for patch_name in PATCH_NAMES:
            vals = [h for _, h in results[patch_name][mode] if not np.isnan(h)]
            if vals and patch_name != "none":
                mean = statistics.mean(vals)
                v26_b = v26_baseline[mode]
                improvement = (v26_b - mean) / v26_b * 100
                marker = "✅" if improvement > 0 else ("⚠" if improvement < -2 else "→")
                print(f"  {mode:>14} + {patch_name:>7}: held-out={mean:.4f} ({marker} {improvement:+.1f}% vs v26)")
    print("=" * 78)


if __name__ == "__main__":
    main()
