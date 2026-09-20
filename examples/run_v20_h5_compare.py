"""V20.0 — H5 中枢广播位置端到端对照 (5 变体 × 5 seeds).

5 变体 × 5 seeds = 25 run:
    none              — 无中枢 (基线)
    ffn-broadcast     — C-1 broadcast 加到 ffn 路由 (v8.0 默认)
    attn-broadcast    — C-1 broadcast 加到 attn 路由 (H5 假设)
    both-broadcast    — 同时加到 attn 和 ffn
    signal            — C-1 作为 attn 路由的"信号" (调制 x, 再算 attn)

端到端: 跨架构 3 模态 (TinyBERT + TinyLlama + ViT), 用 v10 attn aligner +
v9 gate-style 中枢, 但 c 的 broadcast 位置按变体切换.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import V8Trainer
from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from research.central_diagnostics.v20_c1_position import CentralBroadcaster
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


class PositionLayer(nn.Module):
    """H5 测试层: 可配置中枢 broadcast 位置.

    简化版 AlignedFusionLayer: 用 attn aligner + ffn 路径 + 可配置 c 广播位置.
    """

    def __init__(self, d_shared, num_experts_attn, num_experts_ffn, modal_dims,
                 attn_pools, position="ffn"):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts_attn = num_experts_attn
        self.num_experts_ffn = num_experts_ffn
        self.position = position
        # aligner
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=d_shared, num_heads=4)
        # 路由器
        self.W_router_attn = nn.Parameter(torch.randn(num_experts_attn, d_shared) * 0.01)
        self.W_router_ffn = nn.Parameter(torch.randn(num_experts_ffn, d_shared) * 0.01)
        # 中枢 (单 c, 但 broadcast 位置可变)
        self.cb = CentralBroadcaster(d_shared, max(num_experts_attn, num_experts_ffn),
                                      broadcast_position=position)
        # 兼容 V8Trainer: 给它 cw_attn/cw_ffn 别名
        self.cw_attn = self.cb
        self.cw_ffn = self.cb
        # attn 通路
        self.attn_pools = nn.ModuleList(attn_pools)
        # ffn (简化, 冻结)
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(d_shared)) for _ in range(num_experts_ffn)])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(d_shared)) for _ in range(num_experts_ffn)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts_ffn)])
        d_ff = 4 * d_shared
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(d_ff, d_shared) * 0.02, requires_grad=False) for _ in range(num_experts_ffn)])
        self.w_ups = nn.ParameterList([nn.Parameter(torch.randn(d_ff, d_shared) * 0.02, requires_grad=False) for _ in range(num_experts_ffn)])
        self.w_downs = nn.ParameterList([nn.Parameter(torch.randn(d_shared, d_ff) * 0.02, requires_grad=False) for _ in range(num_experts_ffn)])
        # aligner
        self.aligner = aligner

    def forward(self, x_list):
        # 1. aligner 对齐到 D_shared
        x_shared = self.aligner(x_list)
        x_det = x_shared.detach()
        # 2. attn 路由 (可能含中枢)
        z_attn = F.linear(x_det, self.W_router_attn)
        z_attn = self.cb.augment_router_logits(z_attn, position="attn")
        if self.position == "signal":
            # c 作为信号: 调制 x 而非 logits
            # z_attn_signal = W_attn · (x + U·c·spread)
            signal = (self.cb.U @ self.cb.c)              # [M]
            # 把 signal 扩展到 [B, S, D_shared] 用于调制 x
            # 但 signal 维度是 [M=3], x 是 [B, S, D=256] — 维度不匹配
            # 实际含义: c 应该是 D_shared 维度而非 M 维度
            # 这里简化: 把 c broadcast 到 [1, 1, D_shared] 加到 x
            x_signal = x_det + self.cb.c.unsqueeze(0).unsqueeze(0)  # [B, S, D]
            z_attn = F.linear(x_signal, self.W_router_attn)
        from hetero_fusion.core.router import SparseRouterSTE
        alpha_attn = F.softmax(z_attn, dim=-1)
        ahat_attn = SparseRouterSTE.apply(alpha_attn, 1)
        # attn expert 池
        attn_sum = torch.zeros_like(x_shared)
        attn_outs = []
        for m in range(self.num_experts_attn):
            a_out = self.attn_pools[m](x_shared)
            attn_outs.append(a_out.detach())
            attn_sum = attn_sum + ahat_attn[..., m:m+1] * 0.05 * a_out
        # 3. ffn 路由 (可能含中枢)
        z_ffn = F.linear(x_det, self.W_router_ffn)
        z_ffn = self.cb.augment_router_logits(z_ffn, position="ffn")
        alpha_ffn = F.softmax(z_ffn, dim=-1)
        ahat_ffn = SparseRouterSTE.apply(alpha_ffn, 1)
        ffn_sum = torch.zeros_like(x_shared)
        from hetero_fusion.core.quant import FakeQuantSTE
        for m in range(self.num_experts_ffn):
            h_a = x_det * self.gammas[m] + self.betas[m]
            f_m = F.linear(F.silu(F.linear(h_a, self.w_gates[m])) *
                           F.linear(h_a, self.w_ups[m]), self.w_downs[m])
            f_q = FakeQuantSTE.apply(f_m, 4, 128)
            ffn_sum = ffn_sum + ahat_ffn[..., m:m+1] * self.alphas[m] * f_q
        # 中枢 EMA
        self.cb.ema_update(attn_outs)
        return x_shared + attn_sum + ffn_sum


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


def run_seed(position, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] position={position} ...")
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)

    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    layer = PositionLayer(d_shared=D_SHARED, num_experts_attn=3, num_experts_ffn=3,
                            modal_dims=modal_dims, attn_pools=attn_pools, position=position)

    # 训练
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)

    # 训练 (用自己的训练循环, V8Trainer 需要 V_coop 等 v8 专属属性)
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y = layer(h_by_modal)
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        opt.step()

    # eval
    fuse_loss = 0.0
    with torch.no_grad():
        y = layer(h_by_modal)
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


# 位置枚举值 (与 BroadcastPosition 枚举对齐)
POSITIONS = ["none", "ffn", "attn", "both", "signal"]


def main():
    print("=" * 78)
    print("V20.0 — H5 假设: 中枢 broadcast 位置 (attn vs ffn)")
    print("=" * 78)
    print("5 变体 × 5 seeds = 25 run")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {p: [] for p in POSITIONS}

    print(f"\n每个 position × {SEEDS} seeds")
    for pos in POSITIONS:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(pos, seed, encoders, attn_pools)
                results[pos].append(fuse)
                print(f"  [{pos:>16} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{pos:>16} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[pos].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'position':>18} | {'fuse MSE':>9} | {'vs none':>11}")
    print("-" * 78)
    none_mean = statistics.mean([m for m in results["none"] if not np.isnan(m)])
    for pos in POSITIONS:
        vals = [x for x in results[pos] if not np.isnan(x)]
        if not vals:
            print(f"{pos:>18} | {'N/A':>9} | {'N/A':>11}")
            continue
        m_mean = statistics.mean(vals)
        gain = (none_mean - m_mean) / max(none_mean, 1e-9) * 100
        print(f"{pos:>18} | {m_mean:>9.4f} | {gain:>+10.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    for pos in POSITIONS:
        vals = [x for x in results[pos] if not np.isnan(x)]
        if vals:
            print(f"  {pos:>18}: {statistics.mean(vals):.4f}")


if __name__ == "__main__":
    main()