"""V15.0 — Mixture of Mixture 嵌套对齐器 端到端对照 (5 seeds).

3 变体 × 5 seeds = 15 run:
    attn    — v10 CrossArchAttnAligner (单层, 已知 ≈ 0.34)
    mixture — v13 MixtureAligner (单层跨底座 + 跨专家, 已知 ≈ 0.26)
    mom     — v15 MoMAligner (两级嵌套, 内层底座级 + 外层池级, 期望更好)

核心问题: mom < mixture < attn ? 嵌套对齐器是否比单层更好?
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
from research.aligner.v15_nested import MoMAligner
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_layer(aligner_kind, attn_pools, encoders):
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    if aligner_kind == "attn":
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
    elif aligner_kind == "mixture":
        aligner = MixtureAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                                  n_shared_experts=4, num_heads=4)
    elif aligner_kind == "mom":
        aligner = MoMAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                              n_inner=2, n_outer=2, num_heads=4)
    return AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools, modal_dims=modal_dims,
        aligner=aligner, c1_alpha=0.1,
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
        y = layer(h_by_modal)
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        trainer.optimizer.step()
    fuse_loss = 0.0
    with torch.no_grad():
        y = layer(h_by_modal)
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


def run_seed(aligner_kind, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] aligner={aligner_kind} ...")
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)
    layer = make_layer(aligner_kind, attn_pools, encoders)
    return train_eval(layer, modal_seqs, modal_indices, targets)


MODES = ["attn", "mixture", "mom"]


def main():
    print("=" * 78)
    print("V15.0 — Mixture of Mixture 嵌套对齐器 端到端对照")
    print("=" * 78)
    print("3 变体 × 5 seeds = 15 run")
    print("核心判断: mom < mixture < attn ? 嵌套对齐器是否比单层更好?")
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
    print(f"{'mode':>10} | {'fuse MSE':>9} | {'vs attn':>10}")
    print("-" * 78)
    attn_mean = statistics.mean([m for m in results["attn"] if not np.isnan(m)])
    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>10} | {'N/A':>9} | {'N/A':>10}")
            continue
        m_mean = statistics.mean(vals)
        gain = (attn_mean - m_mean) / max(attn_mean, 1e-9) * 100
        print(f"{m:>10} | {m_mean:>9.4f} | {gain:>+9.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    f_a = statistics.mean([m for m in results["attn"] if not np.isnan(m)])
    f_m = statistics.mean([m for m in results["mixture"] if not np.isnan(m)])
    f_mom = statistics.mean([m for m in results["mom"] if not np.isnan(m)])
    print(f"  attn (v10):       {f_a:.4f}")
    print(f"  mixture (v13):    {f_m:.4f}")
    print(f"  mom (v15):        {f_mom:.4f}")
    print()
    if f_mom < f_m < f_a:
        print("✅ mom < mixture < attn 严格成立, 嵌套对齐器更好")
    elif f_mom < f_a:
        print(f"✅ mom 比 attn 改善, 但可能不优于 mixture (饱和信号)")
    else:
        print(f"⚠ mom 与 mixture/attn 接近 (饱和或回退)")


if __name__ == "__main__":
    main()