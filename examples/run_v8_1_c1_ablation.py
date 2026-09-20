"""V8.1 — C-1 单独负增益的根因分析。

C-1: z_router += U @ c  (c 中枢 token, U 广播矩阵)

C-1 单加 v7 baseline 反而让 fuse MSE 从 1.35 升到 1.86 (-38.1%)。
两个潜在根因 (来自 EVOLUTION.md 方向 B):
  H1: U 训练初期任意扰动破坏 v7 已收敛的路由
      (c 初始 0, U · 0 = 0, 但 U 自己有非零训练扰动,
       c 在 EMA 累积前 c 仍小, U 已变大, 故 z 被任意扰动)
  H2: EMA decay 0.9 太慢, c 在 100 步训练里没积累够信息

诊断 6 变体:
  0. baseline            — v7 (无 C-1)
  1. c-only              — c 训练, U 冻结为 0  → 排除 H1 中 U 的扰动
  2. U-only              — U 训练, c 冻结为 0  → 单独 U 是否破坏 (验证 H1)
  3. c+U (C-1 原版)      — 两者都训练 + decay=0.9
  4. c+U, decay=0.5     — 加快 EMA → 验证 H2
  5. c+U, warmup-5      — 前 5 step 把 c 冻结为 0 → 延迟启用中枢

每个变体 5 seeds. 共 30 个 run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, ViTModel, AutoTokenizer, ViTImageProcessor

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import CentralAugmentedFusionLayer, V8Trainer
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def configure_layer_for_variant(layer: CentralAugmentedFusionLayer, variant: str):
    """根据 variant 配置 C-1 相关参数 (cw_attn 和 cw_ffn)."""
    # 复制衰减: 变体 4 用 0.5, 其它默认 0.9
    ema = 0.5 if variant == "decay-0.5" else 0.9
    for cw in (layer.cw_attn, layer.cw_ffn):
        cw.ema_decay = ema
        cw.U.data = torch.randn_like(cw.U) * 0.01  # 重置 U (默认初始化)
        cw.c.data.zero_()                            # c 重置为 0

    if variant == "baseline":
        # 完全关掉 C-1: 中枢 U 冻结为 0 (等价于没有中枢 broadcast)
        for cw in (layer.cw_attn, layer.cw_ffn):
            cw.U.data.zero_()
            cw.U.requires_grad_(False)
        layer.disable_ema()

    elif variant == "c-only":
        # 只训练 c, U 冻结为 0 → U·c = 0·c = 0 → 中枢 broadcast = 0
        # c 单独能积累信息, 但不影响路由
        for cw in (layer.cw_attn, layer.cw_ffn):
            cw.U.data.zero_()
            cw.U.requires_grad_(False)
            cw.c.requires_grad_(True)

    elif variant == "U-only":
        # 只训练 U, c 冻结为 0 → 中枢 broadcast = U·0 = 0
        # 但 U 单独训练扰动会进入 augment_router_logits
        for cw in (layer.cw_attn, layer.cw_ffn):
            cw.c.data.zero_()
            cw.c.requires_grad_(False)
            cw.U.requires_grad_(True)

    elif variant == "c+U":
        # C-1 原版
        for cw in (layer.cw_attn, layer.cw_ffn):
            cw.c.requires_grad_(True)
            cw.U.requires_grad_(True)

    elif variant == "decay-0.5":
        # EMA 加快 (衰减 0.5)
        for cw in (layer.cw_attn, layer.cw_ffn):
            cw.c.requires_grad_(True)
            cw.U.requires_grad_(True)

    elif variant == "warmup-5":
        # 前 5 step 冻结 c (c=0), 之后解冻
        for cw in (layer.cw_attn, layer.cw_ffn):
            cw.c.requires_grad_(False)   # 暂时冻结
            cw.U.requires_grad_(True)
        # 通过全局 step 计数器控制 warmup
        layer._warmup_remaining = 5
        # 包装 forward 让前 5 step 自动跳过 EMA
        original_forward = layer.forward
        def forward_with_warmup(x_shared):
            if getattr(layer, "_warmup_remaining", 0) > 0:
                # 在 warmup 期间, 临时禁掉 EMA
                layer._ema_enabled = False
                out = original_forward(x_shared)
                layer._ema_enabled = True
                layer._warmup_remaining -= 1
                if layer._warmup_remaining == 0:
                    # 解冻 c
                    for cw in (layer.cw_attn, layer.cw_ffn):
                        cw.c.requires_grad_(True)
                return out
            else:
                return original_forward(x_shared)
        layer.forward = forward_with_warmup

    else:
        raise ValueError(f"unknown variant {variant}")


def run_seed(variant: str, seed: int, encoders, attn_pools) -> tuple:
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] variant={variant} ...")

    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)

    layer = CentralAugmentedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools,
        modal_dims=[312, 768, 192],
    )
    # 关掉 V_coop (变体 4 = 衰减, 与 C-1 无关, 全部跑 C-2 默认配置)
    # 但本题只关心 C-1 部分, 所以保留 V_coop (0.1*I) 减少其他变量
    configure_layer_for_variant(layer, variant)

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            perturbed = src + 0.1 * torch.randn(src.shape, generator=g)
            modal_seqs.append(perturbed)
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)

    x_shared_list = []
    for h, cls in zip(modal_seqs, modal_indices):
        x = h.unsqueeze(0)
        x_shared = F.linear(x, layer.P_m[cls])
        x_shared_list.append(x_shared)

    trainer = V8Trainer(layer, lr=LR, phase=2)
    for step in range(STEPS):
        trainer.step(x_shared_list, [t.unsqueeze(0) for t in targets])

    with torch.no_grad():
        fuse_loss = 0.0
        for x_shared, t in zip(x_shared_list, targets):
            y = layer(x_shared)
            fuse_loss += F.mse_loss(y, t.unsqueeze(0)).item()
        fuse_loss /= B
    return fuse_loss


VARIANTS = ["baseline", "c-only", "U-only", "c+U", "decay-0.5", "warmup-5"]


def main():
    print("=" * 78)
    print("V8.1 — C-1 单独负增益的根因分析 (6 变体 × 5 seeds)")
    print("=" * 78)
    print("目标: 区分 H1 (U 扰动) vs H2 (EMA 太慢)")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {v: [] for v in VARIANTS}

    print(f"\n每个 variant × {SEEDS} seeds")
    print("-" * 78)
    for variant in VARIANTS:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(variant, seed, encoders, attn_pools)
                results[variant].append(fuse)
            except Exception as e:
                print(f"  [seed {seed}] variant={variant} FAILED: {e}")
                results[variant].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'variant':>12} | {'fuse MSE':>9} | {'vs baseline':>13} | {'H1/H2 判定':>16}")
    print("-" * 78)
    base_mean = statistics.mean([m for m in results["baseline"] if not np.isnan(m)])
    diagnosis = {
        "baseline":  "—",
        "c-only":    "→ H1/U 不是",
        "U-only":    "→ H1 确认 (U 单独破坏)",
        "c+U":       "→ H1+H2 共同",
        "decay-0.5": "→ H2 验证 (快 EMA 救回)",
        "warmup-5":  "→ H1 验证 (延迟启用救回)",
    }
    for v in VARIANTS:
        vals = [m for m in results[v] if not np.isnan(m)]
        if not vals:
            print(f"{v:>12} | {'N/A':>9} | {'N/A':>13} | {'':>16}")
            continue
        mean_mse = statistics.mean(vals)
        gain = (base_mean - mean_mse) / max(base_mean, 1e-9) * 100
        print(f"{v:>12} | {mean_mse:>9.4f} | {gain:>+12.1f}% | {diagnosis[v]:>16}")
    print("=" * 78)
    print()
    print("结论:")
    print(f"  baseline fuse MSE = {base_mean:.4f}")
    for v in VARIANTS:
        if v == "baseline":
            continue
        vals = [m for m in results[v] if not np.isnan(m)]
        if not vals:
            continue
        d = statistics.mean(vals) - base_mean
        sign = "+" if d > 0 else ""
        print(f"  {v:>12}  Δ_MSE = {sign}{d:+.4f}  ({'更差' if d > 0 else '更好'})")
    print()
    print("✅ V8.1 诊断完成 — 详见 docs/V8_1_C1_ROOT_CAUSE.md")


if __name__ == "__main__":
    main()
