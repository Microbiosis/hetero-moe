"""V28.0 — 架构层面修正 router-norm / adaptive-ema 端到端.

11 modes × 5 seeds = 55 run:
    v22 4 个原始 mode (基线):
        - broadcast, gate, router-norm, adaptive-ema
    v28 7 个架构变体:
        - router-norm-affine-false, router-norm-zero-u, router-norm-rms, router-norm-pure
        - adaptive-ema-clamped, adaptive-ema-bounded, adaptive-ema-gate

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

数据: 复用 v26 fully-real-corpus (text/code/image 全部真实)
评估: 训练集 fuse + held-out fuse 双评估

核心判断:
    - v28 架构变体能否让 router-norm / adaptive-ema 在 held-out 上不退步 (≥ broadcast 0.4624)?
    - 哪些变体能达到 gate 水平 (held-out ≤ 0.4516)?
    - 是否有变体严格优于 gate (held-out < 0.4516)?
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
from research.corpus_scale.v28_architectural_fix import ArchitecturalFixLayer, ALL_MODES
from examples.run_v8_full import (
    load_encoders, make_pools,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


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
    print(f"  [seed {seed}] mode={mode:>24} ...")
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

    # 模型 (v28 ArchitecturalFixLayer 支持 11 modes)
    layer = ArchitecturalFixLayer(
        d_shared=D_SHARED, num_experts=3,
        modal_dims=[D_bert, D_llama, D_vit],
        attn_pools=attn_pools, mode=mode,
    )

    # 训练
    layer.train()
    opt = torch.optim.AdamW(build_v25_param_groups(layer), lr=LR)
    for _ in range(STEPS):
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
    _env = log_experiment_env("run_v28_architectural_fix")
    print(f"  时间:       {_env['timestamp']}")
    print(f"  Python:     {_env['python_version'].split()[0]}")
    print(f"  PyTorch:    {_env['pytorch_version']}")
    print(f"  CUDA:       {_env['cuda_available']} ({_env['cuda_version']})")
    print(f"  设备:       {_env['device']}")
    print(f"  CPU 核心:   {_env['cpu_count']}")
    print("=" * 78)
    print()

    print("=" * 78)
    print("V28.0 — 架构层面修正 router-norm / adaptive-ema")
    print("=" * 78)
    print(f"{len(ALL_MODES)} modes × {SEEDS} seeds = {len(ALL_MODES) * SEEDS} run")
    print(f"modes ({len(ALL_MODES)}): {ALL_MODES}")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    llama_tok = encoders[1][0]
    text_corpus = MiniCorpus(bert_tok, max_length=S)
    code_corpus = MiniCodeCorpus(llama_tok, max_length=S)
    image_corpus = MiniImageCorpus(image_size=224)

    results = {m: [] for m in ALL_MODES}

    for mode in ALL_MODES:
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

    # 汇总: 训练集 fuse
    print("\n" + "=" * 78)
    print("训练集 fuse (各 mode, 越低越好)")
    print("-" * 78)
    for mode in ALL_MODES:
        vals = [t for t, _ in results[mode] if not np.isnan(t)]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {mode:>24}: {mean:.4f} (±{std:.3f})")

    # 汇总: held-out fuse
    print("\n" + "=" * 78)
    print("held-out fuse (各 mode, 越低越好)")
    print("-" * 78)
    for mode in ALL_MODES:
        vals = [h for _, h in results[mode] if not np.isnan(h)]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {mode:>24}: {mean:.4f} (±{std:.3f})")

    # 关键判断
    print("\n" + "=" * 78)
    print("关键判断:")
    print("-" * 78)
    gate_vals = [h for _, h in results["gate"] if not np.isnan(h)]
    gate_heldout = statistics.mean(gate_vals) if gate_vals else None
    broadcast_vals = [h for _, h in results["broadcast"] if not np.isnan(h)]
    broadcast_heldout = statistics.mean(broadcast_vals) if broadcast_vals else None
    if gate_heldout:
        print(f"  gate 基准 (held-out):       {gate_heldout:.4f}")
    if broadcast_heldout:
        print(f"  broadcast 基准 (held-out):   {broadcast_heldout:.4f}")

    print()
    print("  变体 vs v22 baseline (router-norm/adaptive-ema):")
    for mode in ["router-norm", "adaptive-ema"] + [m for m in ALL_MODES if "v28" not in m]:
        continue  # 已在 ALL_MODES, 跳过重复
    # 简化版: 每个 mode 都和 gate/broadcast 对比
    print()
    print("  每个 mode vs gate (held-out, gate 越低越好):")
    for mode in ALL_MODES:
        vals = [h for _, h in results[mode] if not np.isnan(h)]
        if vals and gate_heldout:
            mean = statistics.mean(vals)
            gap = (mean - gate_heldout) / gate_heldout * 100
            marker = "✅" if gap <= 0 else ("⚠" if gap < 5 else "❌")
            print(f"  {marker} {mode:>24}: {mean:.4f} (gap to gate: {gap:+.1f}%)")

    # 找出 router-norm / adaptive-ema 的最佳变体
    print("\n" + "=" * 78)
    print("每个 base mode 的最佳 v28 变体:")
    print("-" * 78)
    router_variants = [m for m in ALL_MODES if m.startswith("router-norm") and m != "router-norm"]
    adaptive_variants = [m for m in ALL_MODES if m.startswith("adaptive-ema") and m != "adaptive-ema"]
    for base, variants in [("router-norm", router_variants), ("adaptive-ema", adaptive_variants)]:
        # base baseline
        baseline_vals = [h for _, h in results[base] if not np.isnan(h)]
        baseline_heldout = statistics.mean(baseline_vals) if baseline_vals else None
        # 最佳变体
        best_variant = None
        best_heldout = float("inf")
        for variant in variants:
            vals = [h for _, h in results[variant] if not np.isnan(h)]
            if vals:
                mean = statistics.mean(vals)
                if mean < best_heldout:
                    best_heldout = mean
                    best_variant = variant
        if baseline_heldout and best_variant:
            gain = (baseline_heldout - best_heldout) / baseline_heldout * 100
            print(f"  {base:>14}: best variant = {best_variant:>24} (held-out={best_heldout:.4f}, "
                  f"vs base {baseline_heldout:.4f}, gain={gain:+.1f}%)")
    print("=" * 78)


if __name__ == "__main__":
    main()
