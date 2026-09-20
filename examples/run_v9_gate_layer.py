"""V9.0 增强版 — 使用 v9_gate_central 子包的标准接口.

相对原 run_v9_fix_bc.py 的差异:
    原版: 用 v8 的 CentralWorkspace (mode="gate") + custom 参数.
    本版: 用 v9_gate_central.GateStyleWorkspace (独立类, 干净接口).

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

3 变体 × 5 seeds:
    v7-baseline     — 无中枢
    v9-gate-c1      — 只用 GateStyleWorkspace 替代 C-1
    v9-gate-full    — v9 gate C-1 + v8 C-2/C-3 全开

目标: 验证 v9_gate_central 标准接口与 v8 历史结果一致
      (c1-fix-c +44.5%, c3-fix-c +88.1% 的数字应在 v9 包装下复现).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v7_attention import CrossArchAttnFFNFusionLayer, V7Trainer
from research.routing_evolution.v8_central import CentralAugmentedFusionLayer, HierarchicalCentralLayer, V8Trainer
from research._primitives.central_mechanism import GateStyleWorkspace, make_v9_layer
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_layer(mode: str, attn_pools, encoders):
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]

    if mode == "v7-baseline":
        return CrossArchAttnFFNFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
        )
    elif mode == "v9-gate-c1":
        # v9 gate C-1 + v8 C-2 协同 (保留 V_coop)
        layer = make_v9_layer(
            CentralAugmentedFusionLayer,
            d_shared=D_SHARED,
            num_experts_attn=len(attn_pools),
            num_experts_ffn=len(modal_dims),
            d_ff=4 * D_SHARED,
            attn_pools=attn_pools,
            modal_dims=modal_dims,
            c1_alpha=0.1,
        )
        return layer
    elif mode == "v9-gate-full":
        # v9 gate C-1 + v8 C-2 + C-3 层级中枢
        layer = make_v9_layer(
            HierarchicalCentralLayer,
            d_shared=D_SHARED,
            num_experts_attn=len(attn_pools),
            num_experts_ffn=len(modal_dims),
            d_ff=4 * D_SHARED,
            attn_pools=attn_pools,
            modal_dims=modal_dims,
            c1_alpha=0.1,
        )
        return layer
    else:
        raise ValueError(f"unknown mode {mode}")


def make_trainer(layer):
    if isinstance(layer, CrossArchAttnFFNFusionLayer):
        return V7Trainer(layer, lr=LR, phase=2)
    return V8Trainer(layer, lr=LR, phase=2)


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)

    layer = make_layer(mode, attn_pools, encoders)

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

    x_shared_list = [F.linear(h.unsqueeze(0), layer.P_m[cls])
                     for h, cls in zip(modal_seqs, modal_indices)]

    trainer = make_trainer(layer)
    for step in range(STEPS):
        trainer.step(x_shared_list, [t.unsqueeze(0) for t in targets])

    with torch.no_grad():
        fuse_loss = sum(F.mse_loss(layer(x), t.unsqueeze(0)).item()
                        for x, t in zip(x_shared_list, targets)) / B
    return fuse_loss


MODES = ["v7-baseline", "v9-gate-c1", "v9-gate-full"]


def main():
    print("=" * 78)
    print("V9.0 (v9_gate_central 标准接口) — 3 变体 × 5 seeds = 15 个 run")
    print("=" * 78)
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {m: [] for m in MODES}

    for mode in MODES:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(mode, seed, encoders, attn_pools)
                results[mode].append(fuse)
                print(f"  [{mode:>14} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>14} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    print("\n" + "=" * 78)
    print(f"{'mode':>14} | {'fuse MSE':>9} | {'vs v7-baseline':>14}")
    print("-" * 78)
    base_mean = statistics.mean([m for m in results["v7-baseline"] if not np.isnan(m)])
    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>14} | {'N/A':>9} | {'N/A':>14}")
            continue
        m_mean = statistics.mean(vals)
        gain = (base_mean - m_mean) / max(base_mean, 1e-9) * 100
        print(f"{m:>14} | {m_mean:>9.4f} | {gain:>+13.1f}%")
    print("=" * 78)
    print()
    print("对照历史 (run_v9_fix_bc.py):")
    print("  c1-fix-c: +44.5%  | c3-fix-c: +88.1%")


if __name__ == "__main__":
    main()
