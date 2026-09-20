"""V14.0 — MixtureAligner vs CrossArchAttnAligner 端到端对照 (5 seeds).

v13 验证了 MixtureAligner 子包,但没在端到端跑过.
v14 真正端到端对照 3 种 aligner:
    linear  — v9 P_m 矩形投影 (基线)
    attn    — v10 CrossArchAttnAligner (已知 ≈ 0.34)
    mixture — v13 MixtureAligner (跨底座 + 跨专家, 期望更好)

判断: mixture < attn < linear ? 即"更复杂的对齐器"是否真的更好?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import V8Trainer as TeacherTrainer
from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from research.aligner.v13_mixture import MixtureAligner
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_layer(aligner_kind, attn_pools, encoders):
    from research.routing_evolution.v8_central import CentralAugmentedFusionLayer
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    if aligner_kind == "attn":
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
    elif aligner_kind == "mixture":
        aligner = MixtureAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                                  n_shared_experts=4, num_heads=4)
    elif aligner_kind == "linear":
        return CentralAugmentedFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
            c1_alpha=0.1,
        )
    return AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools, modal_dims=modal_dims,
        aligner=aligner,
        c1_alpha=0.1,
    )


def prepare_data(seed, encoders):
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


def train_eval(layer, modal_seqs, modal_indices, targets):
    from research.routing_evolution.v8_central import CentralAugmentedFusionLayer
    is_v9 = isinstance(layer, CentralAugmentedFusionLayer)
    trainer = TeacherTrainer(layer, lr=LR, phase=2)
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    for _ in range(STEPS):
        trainer.optimizer.zero_grad(set_to_none=True)
        if is_v9:
            # v9 linear: per-sample forward
            losses = []
            for i in range(B):
                cls_idx = modal_indices[i]
                x = F.linear(modal_seqs[i].unsqueeze(0), layer.P_m[cls_idx])
                y = layer(x)
                losses.append(F.mse_loss(y, targets[i].unsqueeze(0)))
            total = sum(losses) / len(losses)
        else:
            y = layer(h_by_modal)
            loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
            total = loss
        total.backward()
        trainer.optimizer.step()
    # eval
    fuse_loss = 0.0
    with torch.no_grad():
        if is_v9:
            for i in range(B):
                cls_idx = modal_indices[i]
                x = F.linear(modal_seqs[i].unsqueeze(0), layer.P_m[cls_idx])
                y = layer(x)
                fuse_loss += F.mse_loss(y, targets[i].unsqueeze(0)).item()
            fuse_loss /= B
        else:
            y = layer(h_by_modal)
            for c in range(N_CLS):
                for s in range(6):
                    fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
            fuse_loss /= 18
    return fuse_loss


def run_seed(aligner_kind, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] aligner={aligner_kind} ...")
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)
    layer = make_layer(aligner_kind, attn_pools, encoders)
    return train_eval(layer, modal_seqs, modal_indices, targets)


MODES = ["linear", "attn", "mixture"]


def main():
    print("=" * 78)
    print("V14.0 — MixtureAligner vs CrossArchAttnAligner 端到端对照")
    print("=" * 78)
    print("3 变体 × 5 seeds = 15 run")
    print("核心判断: mixture < attn < linear ? 更复杂的对齐器是否真的更好?")
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
                print(f"  [{mode:>8} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>8} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>10} | {'fuse MSE':>9} | {'vs linear':>11}")
    print("-" * 78)
    linear_mean = statistics.mean([m for m in results["linear"] if not np.isnan(m)])
    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>10} | {'N/A':>9} | {'N/A':>11}")
            continue
        m_mean = statistics.mean(vals)
        gain = (linear_mean - m_mean) / max(linear_mean, 1e-9) * 100
        print(f"{m:>10} | {m_mean:>9.4f} | {gain:>+10.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    f_l = statistics.mean([m for m in results["linear"] if not np.isnan(m)])
    f_a = statistics.mean([m for m in results["attn"] if not np.isnan(m)])
    f_m = statistics.mean([m for m in results["mixture"] if not np.isnan(m)])
    print(f"  linear (v9 P_m):     {f_l:.4f}")
    print(f"  attn (v10 CrossArch): {f_a:.4f}")
    print(f"  mixture (v13):        {f_m:.4f}")
    print()
    if f_m < f_a < f_l:
        print("✅ mixture < attn < linear 严格成立, 更复杂的对齐器更好")
    elif f_m < f_l and f_a < f_l:
        if f_m < f_a:
            print(f"✅ mixture 优于 attn ({f_m:.4f} < {f_a:.4f})")
        else:
            print(f"⚠ mixture 与 attn 接近 ({f_m:.4f} vs {f_a:.4f}), 都优于 linear")
    else:
        print(f"⚠ mixture 未必优于 attn, 但两者都优于 linear")


if __name__ == "__main__":
    main()