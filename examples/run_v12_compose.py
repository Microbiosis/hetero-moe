"""V12.0 — v9 + v10 组合验证 (语义对齐器 + gate-style 中枢).

4 变体 × 5 seeds = 20 run, 对照:
    v9-linear-gate        — v9.0 修正 C (P_m 线性 + gate 中枢)         # v9 baseline
    v10-attn-broadcast     — v10 attn aligner + v8 broadcast 中枢         # 退化 (broadcast 中枢在 v9 已知有 bug)
    v12-attn-gate          — v10 attn aligner + v9 gate 中枢             # v12 = v9 + v10 组合 (核心)
    v10-attn-gate-original — v10 attn aligner + v9 gate 中枢 (与 v12 同, 作对照)

核心判断: v12-attn-gate 是否优于 v9-linear-gate (叠加增益)?
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
from archive.negative_findings.v12_compose import ComposedFusionLayer
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_layer(mode, attn_pools, encoders):
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]

    if mode == "v9-linear-gate":
        return CentralAugmentedFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
            c1_alpha=0.1,
        )

    if mode in ("v10-attn-broadcast", "v12-attn-gate", "v10-attn-gate-original"):
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
        c1_mode = "broadcast" if mode == "v10-attn-broadcast" else "gate"
        return AlignedFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
            aligner=aligner,
            c1_alpha=0.1, c1_mode=c1_mode,
        )


def train_v9(layer, modal_seqs, modal_indices, targets):
    trainer = V8Trainer(layer, lr=LR, phase=2)
    x_list = [F.linear(h.unsqueeze(0), layer.P_m[cls])
              for h, cls in zip(modal_seqs, modal_indices)]
    t_list = [t.unsqueeze(0) for t in targets]
    for _ in range(STEPS):
        trainer.step(x_list, t_list)


def train_v10(layer, modal_seqs, modal_indices, targets):
    trainer = V8Trainer(layer, lr=LR, phase=2)
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
        h_list = h_by_modal
        y = layer(h_list)
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        trainer.optimizer.step()


def eval_v9(layer, modal_seqs, modal_indices, targets):
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
        y = layer(h_list)
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] mode={mode} ...")

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

    layer = make_layer(mode, attn_pools, encoders)

    if mode == "v9-linear-gate":
        train_v9(layer, modal_seqs, modal_indices, targets)
        return eval_v9(layer, modal_seqs, modal_indices, targets)
    else:
        train_v10(layer, modal_seqs, modal_indices, targets)
        return eval_v10(layer, modal_seqs, modal_indices, targets)


MODES = ["v9-linear-gate", "v10-attn-broadcast", "v12-attn-gate", "v10-attn-gate-original"]


def main():
    print("=" * 78)
    print("V12.0 — v9 + v10 组合 (语义对齐器 + gate-style 中枢)")
    print("=" * 78)
    print("4 变体 × 5 seeds = 20 run")
    print("核心判断: v12-attn-gate 是否优于 v9-linear-gate (叠加增益)?")
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
                print(f"  [{mode:>26} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>26} seed {seed}] FAILED: {type(e).__name__}: {e}")
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>28} | {'fuse MSE':>9} | {'vs v9-gate':>11}")
    print("-" * 78)
    v9_mean = statistics.mean([m for m in results["v9-linear-gate"] if not np.isnan(m)])

    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>28} | {'N/A':>9} | {'N/A':>11}")
            continue
        m_mean = statistics.mean(vals)
        gain = (v9_mean - m_mean) / max(v9_mean, 1e-9) * 100
        print(f"{m:>28} | {m_mean:>9.4f} | {gain:>+10.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    f_v9 = statistics.mean([m for m in results["v9-linear-gate"] if not np.isnan(m)])
    f_bc = statistics.mean([m for m in results["v10-attn-broadcast"] if not np.isnan(m)])
    f_v12 = statistics.mean([m for m in results["v12-attn-gate"] if not np.isnan(m)])
    f_orig = statistics.mean([m for m in results["v10-attn-gate-original"] if not np.isnan(m)])
    print(f"  v9-linear-gate (基线):      {f_v9:.4f}")
    print(f"  v10-attn-broadcast (退化):  {f_bc:.4f}  (vs v9: {(f_v9 - f_bc)/max(f_v9, 1e-9)*100:+.1f}%)")
    print(f"  v12-attn-gate (组合):       {f_v12:.4f}  (vs v9: {(f_v9 - f_v12)/max(f_v9, 1e-9)*100:+.1f}%)")
    print(f"  v10-attn-gate (原始):       {f_orig:.4f}  (vs v9: {(f_v9 - f_orig)/max(f_v9, 1e-9)*100:+.1f}%)")
    print()
    if f_v12 < f_v9:
        improvement = (f_v9 - f_v12) / f_v9 * 100
        if improvement > 50:
            print(f"✅✅ 叠加增益显著: v12 比 v9 改善 {improvement:.1f}%")
        else:
            print(f"✅ 叠加增益存在: v12 比 v9 改善 {improvement:.1f}%")
    else:
        print(f"⚠ 无叠加增益: v12 ({f_v12:.4f}) 不优于 v9 ({f_v9:.4f})")


if __name__ == "__main__":
    main()