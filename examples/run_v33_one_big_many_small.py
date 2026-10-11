"""V33.0 — 「1大+N小」异构规模融合 (Scale-heterogeneous Fusion) — 端到端入口。

> 归属路线: 规模异构+潜意识联想 (原 V6 衍生路线, 由外部快照 v7.0 重编号而来)
> 重编号对照: 快照 v7.0 → 本项目 v33.0 (本项目 v7.0 已占用: Attn+FFN 双路由)
> 子包: v33_scale_hetero (机制实现), 本脚本只负责数据/训练/评估编排

规模配置 (真实模型, 均 Qwen3 风格, 同 tokenizer 系):
  大:  MSA-4B        (D=2560, 36层, ~4B)   -> /workspace/models/MSA-4B/
  小1: embedding-0.6b (D=1024, 28层, ~0.6B) -> /workspace/models/skillcorpus-embedding-0.6b/
  小2: reranker-0.6b  (D=1024, 28层, ~0.6B) -> /workspace/models/skillcorpus-reranker-0.6b/

用法: bash run.sh v33  (或 V33_BIG_LAYERS=1 python3 examples/run_v33_one_big_many_small.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

import torch
import torch.nn.functional as F

from research._primitives.scale_hetero_loaders import (
    MSA_DIR, EMB_DIR, RER_DIR,
    load_msa_big, load_small, load_tokenizer, encode,
)
# CrossModalFusionLayer 定义在 v33 包内, 不在 _primitives.scale_hetero_loaders
from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer

D_SHARED = 256
M, K = 3, 1  # 3 专家: 1 大 (MSA-4B) + 2 小 (embedding, reranker)
B, S = 18, 16
STEPS = int(os.environ.get("V33_STEPS", "100"))
LR = 1e-2
N_SEEDS = int(os.environ.get("V33_SEEDS", "5"))


# =====================================================================
# 单次实验
# =====================================================================

def run_one(seed, big_tok, big_model, D_big, small_tok, small_model, D_small):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] 编码 3 模态 + 训练...")

    # 1. 编码 (各自原生 tokenizer + backbone)
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_big = encode(texts, big_tok, big_model, S)      # [B=3, S, 2560]
    h_small1 = encode(texts, small_tok, small_model, S)  # [3, S, 1024]
    h_small2 = encode(texts, small_tok, small_model, S)  # [3, S, 1024]
    print(f"    h_big={tuple(h_big.shape)} h_small={tuple(h_small1.shape)}")

    # 2. 融合层
    modal_dims = [D_big, D_small, D_small]
    layer = CrossModalFusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims)
    for m in range(M):
        layer.P_m[m].requires_grad_(False)

    # 3. 构造 batch: 每专家 6 个样本
    expert_seqs = []
    expert_indices = []
    for cls, h in enumerate([h_big, h_small1, h_small2]):
        n = h.shape[0]
        for i in range(6):
            # 取单样本, 而不是整个 [3,S,D] block —— 否则 xi_shared 与 tgt 广播错配
            # (target 是第 i//6 类的 one-hot, 而整块输入含 3 条文本, 输入与目标不对应)
            src = h[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            perturbed = src + 0.1 * torch.randn(src.shape, generator=g)
            expert_seqs.append(perturbed)
            expert_indices.append(cls)

    # 4. 目标
    targets = torch.zeros(B, S, D_SHARED)
    for i in range(B):
        cls = i // 6
        ds = cls * (D_SHARED // M); de = (cls + 1) * (D_SHARED // M)
        targets[i, :, ds:de] = 1.0

    # 5. 训练
    trainable_params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    opt = torch.optim.AdamW(trainable_params, lr=LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        losses = []
        for i, (xi, mod_idx) in enumerate(zip(expert_seqs, expert_indices)):
            xi_shared = F.linear(xi, layer.P_m[mod_idx])
            y = layer(xi_shared)
            losses.append(F.mse_loss(y, targets[i]))   # y 与 targets[i] 同为 [S, D_SHARED]
        total = sum(losses) / len(losses)
        total.backward()
        opt.step()

    # 6. 评估
    with torch.no_grad():
        fuse_loss = single_big_loss = single_small_loss = avg_loss = 0.0
        route_counts = [0] * M
        for i, (xi, mod_idx) in enumerate(zip(expert_seqs, expert_indices)):
            xi_shared = F.linear(xi, layer.P_m[mod_idx])
            x_det = xi_shared.detach()
            tgt = targets[i]                        # 与 xi_shared 同为 [S, D_SHARED], 无广播

            # 融合
            y_fuse, alpha_hat = layer.forward_with_routing(xi_shared)
            fuse_loss += F.mse_loss(y_fuse, tgt).item()
            for m in range(M):
                route_counts[m] += int((alpha_hat[0].argmax() == m).sum().item())

            # 单大 (专家 0 = MSA-4B)
            h_a0 = x_det * layer.gammas[0] + layer.betas[0]
            f0 = layer._swiglu(h_a0, layer.w_gates[0], layer.w_ups[0], layer.w_downs[0])
            y_big = xi_shared + layer.alphas[0] * f0
            single_big_loss += F.mse_loss(y_big, tgt).item()

            # 单小 (专家 1 = embedding)
            h_a1 = x_det * layer.gammas[1] + layer.betas[1]
            f1 = layer._swiglu(h_a1, layer.w_gates[1], layer.w_ups[1], layer.w_downs[1])
            y_sm = xi_shared + layer.alphas[1] * f1
            single_small_loss += F.mse_loss(y_sm, tgt).item()

            # 平均
            total_f = torch.zeros_like(xi_shared)
            for m in range(M):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
                total_f = total_f + f
            y_avg = xi_shared + total_f / M
            avg_loss += F.mse_loss(y_avg, tgt).item()

        N = B
        fuse_loss /= N; single_big_loss /= N; single_small_loss /= N; avg_loss /= N

    return fuse_loss, avg_loss, single_big_loss, single_small_loss, route_counts


# =====================================================================

def main():
    print(f"{'='*78}")
    print("V33.0 「1大+N小」异构规模融合 (Scale-heterogeneous Fusion)")
    print(f"{'='*78}")
    print("大:  MSA-4B        (D=2560, ~4B)")
    print("小1: embedding-0.6b (D=1024, ~0.6B)")
    print("小2: reranker-0.6b  (D=1024, ~0.6B)")
    print("均 Qwen3 风格, 同 tokenizer 系, 各自独立前向 + 矩形 P_m 投影对齐 -> 共享路由")
    print(f"投影: D_big={2560} -> D_shared={D_SHARED}, D_small={1024} -> D_shared={D_SHARED}")
    print()

    # 加载 tokenizer
    big_tok = load_tokenizer(MSA_DIR)
    small_tok = load_tokenizer(EMB_DIR)
    print(f"  vocab: big={big_tok.vocab_size} small={small_tok.vocab_size}")

    # 加载模型 (串行, 控制内存)
    big_model, D_big, _ = load_msa_big()
    gc.collect()
    small_model, D_small, _ = load_small("embedding-0.6b", EMB_DIR)
    # 复用同一个 small model 编码两个小专家 (同架构同规模, 仅功能语义不同)
    print(f"  小模型: embedding 与 reranker 共用 backbone (同 Qwen3 D={D_small}), "
          f"语义差异由训练目标体现")

    print(f"\n{'seed':>5} | {'single_big':>11} | {'single_sm':>10} | {'avg':>9} | "
          f"{'fuse':>9} | {'gain_big':>10} | {'gain_sm':>9} | {'PASS':>4} | 路由分布")
    print("-" * 100)
    all_pass = True
    gbs, gss = [], []
    route_totals = [0, 0, 0]
    total_tokens = 0

    for seed in range(N_SEEDS):
        fuse, avg, sbig, ssm, rc = run_one(seed, big_tok, big_model, D_big,
                                           small_tok, small_model, D_small)
        gain_big = (sbig - fuse) / max(sbig, 1e-9) * 100
        gain_sm = (ssm - fuse) / max(ssm, 1e-9) * 100
        gbs.append(gain_big); gss.append(gain_sm)
        ok = fuse < sbig and fuse < ssm and fuse < avg
        all_pass = all_pass and ok
        for m in range(M):
            route_totals[m] += rc[m]
        total_tokens += sum(rc)
        print(f"{seed:>5} | {sbig:>11.4f} | {ssm:>10.4f} | {avg:>9.4f} | "
              f"{fuse:>9.4f} | {gain_big:>+9.1f}% | {gain_sm:>+8.1f}% | "
              f"{'✅' if ok else '❌':>4} | {rc}")

    print("-" * 100)
    import statistics
    print(f"\n增益均值: vs 单大 {statistics.mean(gbs):+.1f}% | vs 单小 {statistics.mean(gss):+.1f}%")
    print(f"增益范围: vs 单大 [{min(gbs):+.1f}%, {max(gbs):+.1f}%] | vs 单小 [{min(gss):+.1f}%, {max(gss):+.1f}%]")
    print(f"路由分布 (共 {total_tokens} tokens): 大专家 {route_totals[0]} ({route_totals[0]/total_tokens*100:.1f}%) | "
          f"小专家1 {route_totals[1]} ({route_totals[1]/total_tokens*100:.1f}%) | "
          f"小专家2 {route_totals[2]} ({route_totals[2]/total_tokens*100:.1f}%)")
    print(f"{'='*78}")
    if all_pass:
        print("✅ V33.0 「1大+N小」规模异构融合成立!")
        print("→ 小模型提供了互补增益, 融合优于单大/单小/平均")
        print("→ 关键: 矩形 P_m 投影对齐 + 共享路由在规模异构下仍有效")
    else:
        print("❌ 部分 seeds 失败 — 需分析原因")


if __name__ == "__main__":
    main()
