"""H' 粒度匹配原理验证: per-token 回归任务, token vs sequence 路由。

H' (修正后, V-Model 审查产出):
  路由粒度应匹配损失粒度。
  - 池化目标 (1 信号): sequence (1 决策) 匹配; token (S 决策) 过参数化
  - per-token 目标 (S 信号): token (S 决策) 匹配; sequence (1 决策) 欠参数化 → 更高 loss 下界

实验设计 (V-Model 预检四条全满足):
  1. 专家输出不同: 冻结 FFN 用 realistic init (各专家权重独立) → f_m 各异
  2. 正确路由可达 loss→0: target_t = x_t + f_{m*(t)}_noisy, 正确路由 (K=1, α=1) → y=target → loss 0
  3. 损失 per-token 非池化: F.mse_loss(y, target) 逐 token (不 pool over S)
  4. 序列内需多专家: m*(b,s)=argmax(x·W*) 线性标签, 随机 x → 一序列内覆盖多专家

关键隔离 (排除 v5.1 B' 的适配器补偿干扰):
  冻结 P_m=I, γ=1, β=0, α=1; 仅训 W_router。损失差异完全归因于路由能力。

预期: token loss→低 (粒度匹配, 可逐 token 专化); sequence loss→更高下界 (1 决策服务 S 目标)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from hetero_fusion.core.fusion import swiglu_forward
from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.router import SparseRouterSTE
from archive.pre_baseline.v5_alpha.seq_router import SeqSparseRouterSTE

D, D_FF, M, K = 32, 64, 4, 1  # K=1: 硬路由, 避免 softmax scale 干扰
B, S = 4, 16
STEPS = 60
LR = 1e-2  # 仅 W_router, 用较高 LR 快速收敛 (测能力非守规范保守 LR)


def build_layer(granularity, seed=7):
    torch.manual_seed(seed)
    from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity=granularity)
    realistic_ffn_init(layer, D, D_FF)
    # 隔离: 冻结全部适配器, 仅留 W_router 可训
    with torch.no_grad():
        for m in range(M):
            layer.P_m[m].copy_(torch.eye(D))      # 投影 = 恒等
            layer.gammas[m].fill_(1.0)             # γ=1
            layer.betas[m].fill_(0.0)             # β=0
            layer.alphas[m].fill_(1.0)            # α=1 (无需增长)
    for m in range(M):
        layer.P_m[m].requires_grad_(False)
        layer.gammas[m].requires_grad_(False)
        layer.betas[m].requires_grad_(False)
        layer.alphas[m].requires_grad_(False)
    return layer


def compute_expert_outputs(layer, x):
    """返回 {m: f_m_noisy(x)} [B,S,D], 与层内前向一致 (P_m=I,γ=1,β=0 → h_aligned=x_det)."""
    out = {}
    x_det = x.detach()
    with torch.no_grad():
        for m in range(M):
            f = swiglu_forward(x_det, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
            out[m] = FakeQuantSTE.apply(f, layer.quant_bits, layer.quant_group_size)
    return out


def make_task(layer, x, seed_star=123):
    """构造 per-token 线性标签任务: m*(b,s)=argmax(x·W*), target=x+f_{m*}."""
    g = torch.Generator().manual_seed(seed_star)
    W_star = torch.randn(M, D, generator=g) * 0.1  # 真实路由权重
    z_star = F.linear(x, W_star)  # [B,S,M]
    m_star = z_star.argmax(dim=-1)  # [B,S] 每 token 的正确专家
    f_noisy = compute_expert_outputs(layer, x)
    # target[b,s] = x[b,s] + f_{m*(b,s)}[b,s]
    target = x.clone()
    for b in range(B):
        for s in range(S):
            mm = m_star[b, s].item()
            target[b, s] = x[b, s] + f_noisy[mm][b, s]
    return target, m_star


def forward_loss(layer, x, target):
    """前向 + per-token MSE (非池化)."""
    y, z, ah, ahs = layer(x)
    loss = F.mse_loss(y, target)  # 逐元素 (含 token 维), 非池化
    return loss, y, m_star_routing_accuracy(ah, ahs)


def m_star_routing_accuracy(ah, ahs):
    """路由正确率: 选中的专家 == m*(b,s) 的比例."""
    if ahs is not None:  # sequence: 广播, 每 token 同一选择
        picked = ahs.argmax(dim=-1)  # [B]
        # 广播到 [B,S]
        picked = picked.unsqueeze(1).expand(-1, S)
    else:  # token
        picked = ah.argmax(dim=-1)  # [B,S]
    return picked


def run(granularity, x, target, m_star):
    layer = build_layer(granularity)
    opt = torch.optim.AdamW([layer.W_router], lr=LR)
    picked = None
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss, y, picked = forward_loss(layer, x, target)
        loss.backward()
        opt.step()
    # 终态路由正确率
    correct = (picked == m_star).float().mean().item()
    return loss.item(), correct


def main():
    torch.manual_seed(0)
    x = torch.randn(B, S, D)
    # 用 token 粒度的层来构造任务 (任意都行, 任务与粒度无关)
    task_layer = build_layer("token", seed=7)
    target, m_star = make_task(task_layer, x)

    # 验证任务可达: 若用"完美路由" (直接选 m*), loss 应为 0
    with torch.no_grad():
        f_noisy = compute_expert_outputs(task_layer, x)
        perfect_y = x.clone()
        for b in range(B):
            for s in range(S):
                perfect_y[b, s] = x[b, s] + f_noisy[m_star[b, s].item()][b, s]
        floor = F.mse_loss(perfect_y, target).item()
    print(f"任务可达性验证: 完美路由 loss = {floor:.2e} (应为 ~0)")

    print(f"\n{'='*64}\nH' 粒度匹配原理: per-token 回归 (K=1, 仅训 W_router)\n{'='*64}")
    print(f"配置: B={B}, S={S}, M={M}, K=1, D={D}, {STEPS} 步")
    # 序列内专家分布 (验证条件 4: 序列内需多专家)
    uniq_per_seq = [len(set(m_star[b].tolist())) for b in range(B)]
    print(f"每序列覆盖专家数: {uniq_per_seq} (条件4: 序列内需多专家 → sequence 1决策无法覆盖)")

    results = {}
    for gran in ("token", "sequence"):
        loss, acc = run(gran, x, target, m_star)
        results[gran] = (loss, acc)
        print(f"\n--- {gran} ---")
        print(f"  最终 per-token MSE: {loss:.6f}")
        print(f"  路由正确率 (picked==m*): {acc*100:.1f}%")

    print(f"\n{'='*64}\n对照分析\n{'='*64}")
    lt, ls = results["token"][0], results["sequence"][0]
    at, asc = results["token"][1], results["sequence"][1]
    ratio = ls / max(lt, 1e-12)
    print(f"  {'metric':<24} | {'token':>12} | {'sequence':>12} | {'seq/tok':>8}")
    print("  " + "-" * 64)
    print(f"  {'最终 per-token MSE':<24} | {lt:>12.6f} | {ls:>12.6f} | {ratio:>8.2f}")
    print(f"  {'路由正确率':<24} | {at*100:>11.1f}% | {asc*100:>11.1f}% | {asc/max(at,1e-9):>8.2f}")

    print(f"\n判定门 (H' 粒度匹配原理):")
    h_pass = lt < ls * 0.5  # token loss 应显著低于 sequence (粒度匹配)
    a_pass = at > asc + 0.1  # token 路由正确率显著更高
    print(f"  token MSE < sequence×0.5: {'PASS' if h_pass else 'FAIL'} ({lt:.6f} vs {ls*0.5:.6f})")
    print(f"  token 正确率 > sequence+10%: {'PASS' if a_pass else 'FAIL'} ({at*100:.1f}% vs {asc*100:.1f}%)")
    if h_pass and a_pass:
        print("  → H' PASS: per-token 任务上 token routing 显著更优 (粒度匹配原理证实)")
        print("  → 路由粒度应匹配损失粒度: 池化→sequence, per-token→token")
    else:
        print("  → H' FAIL/PARTIAL: 见分析 (可能需更多步数或不同任务难度)")


if __name__ == "__main__":
    main()
