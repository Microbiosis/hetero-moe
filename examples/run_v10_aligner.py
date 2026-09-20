"""V10.0 语义对齐器 vs v9.0 线性 P_m — 跨架构基准 (5 seeds)。

3 变体 × 5 seeds = 15 个 run, 对照:
    v9-linear       — v9.0 修正 C (gate-style) + 线性 P_m (v6/v7 基线)
    v10-mean        — AlignedFusionLayer + CrossArchMeanAligner
    v10-attn        — AlignedFusionLayer + CrossArchAttnAligner (共享 K/V)

判断: 如果 attn < mean < linear, 则"语义对齐"路线有效.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import CentralAugmentedFusionLayer, V8Trainer
from research.aligner.v10_embedding import (
    CrossArchMeanAligner, CrossArchAttnAligner, AlignedFusionLayer,
)
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_layer(mode, attn_pools, encoders):
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]

    if mode == "v9-linear":
        # v9 基线: 线性 P_m (v8 broadcast 中枢, 由 v9 接管 C-1 通过 make_v9_layer)
        layer = CentralAugmentedFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
        )
        from research.routing_evolution.v9_gate_central import make_v9_layer
        return make_v9_layer(
            lambda **kw: layer,  # 已构造, 仅升级 C-1
            d_shared=D_SHARED,
            num_experts_attn=len(attn_pools),
            num_experts_ffn=len(modal_dims),
            c1_alpha=0.1,
        )

    if mode == "v10-mean":
        aligner = CrossArchMeanAligner(modal_dims=modal_dims, d_shared=D_SHARED)
    elif mode == "v10-attn":
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
    return AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools, modal_dims=modal_dims,
        aligner=aligner,
        c1_alpha=0.1,
    )


def prepare_data(seed, encoders):
    """返回 (per_sample_x_list, per_sample_targets, modal_indices), v9 和 v10 共用."""
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
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


def train_v9(layer, modal_seqs, modal_indices, targets):
    """v9 用 CentralAugmentedFusionLayer, 输入是 [1, S, D_shared] (per sample)."""
    trainer = V8Trainer(layer, lr=LR, phase=2)
    x_list = [F.linear(h.unsqueeze(0), layer.P_m[cls])
              for h, cls in zip(modal_seqs, modal_indices)]
    t_list = [t.unsqueeze(0) for t in targets]
    for _ in range(STEPS):
        trainer.step(x_list, t_list)


def train_v10(layer, modal_seqs, modal_indices, targets):
    """v10 用 AlignedFusionLayer, 输入是 h_list=[h_A, h_B, h_C] (3 元素, 每元素 [6, S, D_m]).

    每 step 1 个 forward: aligner 联合处理 3 模态的所有 6 样本.
    """
    trainer = V8Trainer(layer, lr=LR, phase=2)
    # 按模态分 batch (每模态 6 个样本)
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)  # [6, S, D_m]
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)  # [6, S, D_shared]
    for _ in range(STEPS):
        trainer.optimizer.zero_grad(set_to_none=True)
        h_list = h_by_modal  # 3 元素 list, 每元素 [6, S, D_m]
        y = layer(h_list)  # [6, S, D_shared]
        # 用所有 3 模态 target 的均值作为损失
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        trainer.optimizer.step()


def eval_v9(layer, modal_seqs, modal_indices, targets):
    """v9 评估: per sample 投影后送 layer."""
    fuse_loss = 0.0
    x_list = [F.linear(h.unsqueeze(0), layer.P_m[cls])
              for h, cls in zip(modal_seqs, modal_indices)]
    with torch.no_grad():
        for x, t in zip(x_list, targets):
            y = layer(x)
            fuse_loss += F.mse_loss(y, t.unsqueeze(0)).item()
        fuse_loss /= B
    return fuse_loss


def eval_v10(layer, modal_seqs, modal_indices, targets):
    """v10 评估: batched forward, 一次输出所有样本的 fuse."""
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    fuse_loss = 0.0
    with torch.no_grad():
        h_list = h_by_modal
        y = layer(h_list)  # [6, S, D_shared]
        # 18 个样本分别计 loss
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] mode={mode} ...")

    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)
    layer = make_layer(mode, attn_pools, encoders)

    if mode == "v9-linear":
        train_v9(layer, modal_seqs, modal_indices, targets)
        fuse = eval_v9(layer, modal_seqs, modal_indices, targets)
    else:
        train_v10(layer, modal_seqs, modal_indices, targets)
        fuse = eval_v10(layer, modal_seqs, modal_indices, targets)

    return fuse


MODES = ["v9-linear", "v10-mean", "v10-attn"]


def main():
    print("=" * 78)
    print("V10.0 语义对齐器 vs v9.0 线性 P_m — 跨架构基准")
    print("=" * 78)
    print("3 变体 × 5 seeds = 15 个 run")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {m: [] for m in MODES}

    print(f"\n每个 mode × {SEEDS} seeds")
    for mode in MODES:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(mode, seed, encoders, attn_pools)
                results[mode].append(fuse)
                print(f"  [{mode:>10} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>10} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>12} | {'fuse MSE':>9} | {'vs linear':>11}")
    print("-" * 78)
    base_fuse = statistics.mean([f for f in results["v9-linear"] if not np.isnan(f)])

    for m in MODES:
        vals = [f for f in results[m] if not np.isnan(f)]
        if not vals:
            print(f"{m:>12} | {'N/A':>9} | {'N/A':>11}")
            continue
        fuse_mean = statistics.mean(vals)
        vs_lin = (base_fuse - fuse_mean) / max(base_fuse, 1e-9) * 100
        print(f"{m:>12} | {fuse_mean:>9.4f} | {vs_lin:>+10.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    f_lin = statistics.mean([f for f in results["v9-linear"] if not np.isnan(f)])
    f_mean = statistics.mean([f for f in results["v10-mean"] if not np.isnan(f)])
    f_attn = statistics.mean([f for f in results["v10-attn"] if not np.isnan(f)])
    print(f"  v9-linear (P_m):          {f_lin:.4f}")
    print(f"  v10-mean (跨底座均值):     {f_mean:.4f}  (Δ vs linear: {f_mean - f_lin:+.4f})")
    print(f"  v10-attn (跨底座注意力):   {f_attn:.4f}  (Δ vs linear: {f_attn - f_lin:+.4f})")
    print()
    if f_attn < f_mean < f_lin:
        print("✅ 语义对齐路线有效: attn < mean < linear")
    elif f_attn < f_lin:
        print(f"✅ attn 改善 ({f_attn:.4f} < {f_lin:.4f}), mean 表现次之")
    else:
        print("❌ 语义对齐未改善 v9 基线 — 需要更巧妙的对齐机制")


if __name__ == "__main__":
    main()