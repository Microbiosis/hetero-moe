"""V7.0 同架构多任务双路由 — v7-B。

隔离'架构异构'变量, 单独测'功能异构'贡献:
    - 三个底座全是 TinyBERT (D=312), 但通过不同种子扰动 Q/K/V/O 模拟'不同任务微调'
    - 路由面对的不是架构差异, 而是'功能差异' (attn 模式不同)

控制变量: D_m 全是 312, P_m 全是 312x256 方阵 (退化为 v6 风格)
独立变量: AttnPool 的 Q/K/V/O 来自不同的随机扰动, 模拟三个不同任务的底座

用法: bash run.sh v7b
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.routing_evolution.v7_attention import (
    AttnPool, CrossArchAttnFFNFusionLayer, V7Trainer,
)

D_SHARED = 256
D_BERT = 312
N_CLS = 3
N_EXPERTS = 3
B, S, STEPS, LR = 18, 16, 100, 1e-2
SEEDS = 5


def build_three_attn_pools(task_seeds):
    """三个底座都是 TinyBERT (D=312), 但 Q/K/V/O 用不同随机种子扰动。
    模拟'同一架构、不同任务微调'的差异。"""
    pools = []
    for s in task_seeds:
        g = torch.Generator().manual_seed(s)
        # 真实 TinyBERT 的 Q/K/V 是 [312,312], 标准差约 0.04
        scale = 0.04
        w_q = torch.randn(D_BERT, D_BERT, generator=g) * scale
        w_k = torch.randn(D_BERT, D_BERT, generator=g) * scale
        w_v = torch.randn(D_BERT, D_BERT, generator=g) * scale
        w_o = torch.randn(D_BERT, D_BERT, generator=g) * scale
        pools.append(AttnPool(D_BERT, D_SHARED, num_heads=12,
                              w_q=w_q, w_k=w_k, w_v=w_v, w_o_native=w_o))
    return pools


def make_targets(seed):
    """每模态一个 class direction"""
    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED)
        t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    return targets


def make_samples(seed):
    """每模态 6 个样本, 总 B=18
    输入是 D_shared 空间的随机向量 (不再走 HF 编码, 因为是同架构虚拟实验)
    """
    samples = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        for i in range(6):
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            # 输入向量略偏置到该模态对应的 class direction
            x = 0.5 * F.one_hot(torch.tensor(list(range(ds, de))), D_SHARED).float().mean(0)
            x = x.unsqueeze(0).expand(S, -1).clone()
            x = x + 0.3 * torch.randn(S, D_SHARED, generator=g)
            samples.append(x)
    return samples


def run_one(seed):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] 构建同架构多任务双路由层 + 训练...")

    # 三个 AttnPool (都是 TinyBERT 维度, 但 Q/K/V/O 不同)
    task_seeds = [1001, 2002, 3003]
    attn_pools = build_three_attn_pools(task_seeds)
    layer = CrossArchAttnFFNFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools,
        modal_dims=[D_BERT, D_BERT, D_BERT],   # 同架构 -> 同 D_m
        top_k_attn=1, top_k_ffn=1,
    )

    samples = make_samples(seed)
    targets = make_targets(seed)

    # 训练
    trainer = V7Trainer(layer, lr=LR, phase=2)
    x_list = [s.unsqueeze(0) for s in samples]  # 每条 [1, S, D_shared]
    t_list = [t.unsqueeze(0) for t in targets]  # 每条 [1, S, D_shared]
    for step in range(STEPS):
        trainer.step(x_list, t_list)

    # 评估 (4 组)
    with torch.no_grad():
        fuse_loss, attn_only_loss, ffn_only_loss, avg_loss = 0.0, 0.0, 0.0, 0.0
        for x, t in zip(x_list, t_list):
            # 7.1 fuse
            y_fuse = layer(x)
            fuse_loss += F.mse_loss(y_fuse, t).item()
            # 7.2 attn_only
            attn_sum, _ = layer._attn_path(x)
            y_attn = x + attn_sum
            attn_only_loss += F.mse_loss(y_attn, t).item()
            # 7.3 ffn_only
            ffn_sum, _ = layer._ffn_path(x)
            y_ffn = x + ffn_sum
            ffn_only_loss += F.mse_loss(y_ffn, t).item()
            # 7.4 avg
            x_det = x.detach()
            total_f = torch.zeros_like(x)
            for m in range(layer.num_experts_ffn):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                gate = F.linear(h_a, layer.w_gates[m])
                up   = F.linear(h_a, layer.w_ups[m])
                f    = F.linear(F.silu(gate) * up, layer.w_downs[m])
                total_f = total_f + f
            y_avg = x + total_f / layer.num_experts_ffn
            avg_loss += F.mse_loss(y_avg, t).item()
        fuse_loss /= B; attn_only_loss /= B; ffn_only_loss /= B; avg_loss /= B

    return fuse_loss, attn_only_loss, ffn_only_loss, avg_loss


def main():
    print("=" * 78)
    print("V7.0 同架构多任务双路由 — v7-B 功能异构基准")
    print("=" * 78)
    print("3 × TinyBERT (D=312), 但 Q/K/V/O 用不同随机种子扰动")
    print("→ 隔离'架构异构'变量, 单独测'功能异构'贡献")
    print()

    print(f"\n{'seed':>5} | {'fuse MSE':>9} | {'attn_only':>10} | {'ffn_only':>9} | "
          f"{'avg MSE':>9} | {'vs avg':>8} | {'PASS':>4}")
    print("-" * 78)
    all_pass = True
    gs = []
    ao_sum = fo_sum = 0.0
    for seed in range(SEEDS):
        fuse, attn_only, ffn_only, avg = run_one(seed)
        ao_sum += attn_only
        fo_sum += ffn_only
        gain = (avg - fuse) / max(avg, 1e-9) * 100
        gs.append(gain)
        ok = fuse < avg
        all_pass = all_pass and ok
        print(f"{seed:>5} | {fuse:>9.4f} | {attn_only:>10.4f} | {ffn_only:>9.4f} | "
              f"{avg:>9.4f} | {gain:>+7.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 78)
    print(f"\n增益均值 vs avg: {statistics.mean(gs):+.1f}%  "
          f"(范围 [{min(gs):+.1f}%, {max(gs):+.1f}%])")
    print(f"attn_only 均值: {ao_sum/SEEDS:.4f} | ffn_only 均值: {fo_sum/SEEDS:.4f}")
    print("=" * 78)
    if all_pass:
        print("✅ V7-B 功能异构成立: 即使架构相同, 不同 attn 模式的双路由融合仍优于等权平均")
        print("→ 关键: 异构贡献来自 attn 通路的功能差异, 不是架构差异")
    else:
        print("❌ 部分 seeds 失败 — 需分析")


if __name__ == "__main__":
    main()
