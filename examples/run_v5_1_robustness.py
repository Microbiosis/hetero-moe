"""H' 多种子鲁棒性验证 (诚信相关性最高: 7.5× 是 robust 效应还是种子运气)。

5 个种子, 每种子 token vs sequence 对照。判定: 所有种子均 token < sequence → robust; 任一翻转 → Baselined ✅ 不成立。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from hetero_fusion.core.fusion import swiglu_forward
from hetero_fusion.core.quant import FakeQuantSTE
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init

D, D_FF, M, K = 32, 64, 4, 1
B, S, STEPS, LR = 4, 16, 60, 1e-2


def run_one(gran, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, S, D, generator=g)
    torch.manual_seed(seed)
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity=gran)
    realistic_ffn_init(layer, D, D_FF)
    with torch.no_grad():
        for m in range(M):
            layer.P_m[m].copy_(torch.eye(D))
            layer.gammas[m].fill_(1.0); layer.betas[m].fill_(0.0); layer.alphas[m].fill_(1.0)
    for m in range(M):
        for p in (layer.P_m[m], layer.gammas[m], layer.betas[m], layer.alphas[m]):
            p.requires_grad_(False)
    g2 = torch.Generator().manual_seed(seed + 1000)
    W_star = torch.randn(M, D, generator=g2) * 0.1
    m_star = F.linear(x, W_star).argmax(-1)
    f_noisy = {}
    with torch.no_grad():
        for m in range(M):
            f = swiglu_forward(x, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
            f_noisy[m] = FakeQuantSTE.apply(f, layer.quant_bits, layer.quant_group_size)
    target = x.clone()
    for b in range(B):
        for s in range(S):
            target[b, s] = x[b, s] + f_noisy[m_star[b, s].item()][b, s]
    opt = torch.optim.AdamW([layer.W_router], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y, _, _, _ = layer(x)
        loss = F.mse_loss(y, target)
        loss.backward(); opt.step()
    with torch.no_grad():
        y, _, ah, ahs = layer(x)
        loss = F.mse_loss(y, target).item()
        picked = (ahs.argmax(-1).unsqueeze(1).expand(-1, S)
                  if ahs is not None else ah.argmax(-1))
        acc = (picked == m_star).float().mean().item()
    return loss, acc


def main():
    seeds = [0, 1, 2, 3, 4]
    print(f"{'='*70}\nH' 多种子鲁棒性 (5 seeds, token vs sequence)\n{'='*70}")
    print(f"{'seed':>5} | {'token MSE':>10} | {'seq MSE':>10} | {'ratio seq/tok':>13} | "
          f"{'tok acc':>7} | {'seq acc':>7} | {'token更优?':>10}")
    print("-" * 70)
    all_token_better = True
    ratios = []
    for seed in seeds:
        lt, at = run_one("token", seed)
        ls, asc = run_one("sequence", seed)
        r = ls / max(lt, 1e-12)
        ratios.append(r)
        better = lt < ls and at > asc
        all_token_better = all_token_better and better
        print(f"{seed:>5} | {lt:>10.4f} | {ls:>10.4f} | {r:>13.2f} | "
              f"{at*100:>6.1f}% | {asc*100:>6.1f}% | {'✅' if better else '❌':>10}")
    print("-" * 70)
    import statistics
    print(f"\n种子数: {len(seeds)} | ratio 范围 [{min(ratios):.2f}, {max(ratios):.2f}] | "
          f"ratio 均值 {statistics.mean(ratios):.2f}")
    print(f"全部种子 token<sequence: {'✅ 是 → robust, 7.5× 非运气' if all_token_better else '❌ 否 → 种子依赖, ✅ 不成立'}")
    if all_token_better:
        print("\n→ H' 效应 robust: 多种子下 token 始终优于 sequence, Baselined ✅ 经鲁棒性加固。")


if __name__ == "__main__":
    main()
