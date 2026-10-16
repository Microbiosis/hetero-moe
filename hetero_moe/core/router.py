"""§2.3 / §5 — 稀疏门控路由与容量感知推理路由。

包含两个算子:
    1. SparseRouterSTE  (§2.3)  —— 训练期硬 Top-K 路由 + STE，防止非 Top-K 梯度泄漏。
    2. capacity_route   (§5)   —— 推理期容量因子约束 + 溢出降级 + 动态重归一化。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


class SparseRouterSTE(torch.autograd.Function):
    """带 Mask 的 Top-K 稀疏路由 (§2.3)。

    前向硬截断: 仅保留 Top-K 位置的 alpha，其余置 0。
    反向 STE:    仅 Top-K 位置透传梯度 (mask 乘以 grad_output)，非 Top-K 梯度为 0，
                 杜绝梯度泄漏。

    Args:
        alpha: [*, M] softmax 路由概率。
        k:     Top-K (int)。

    Returns:
        [*, M] 稀疏化的路由权重 (前向)，反向梯度按 mask 透传。
    """

    @staticmethod
    def forward(ctx, alpha, k):
        _, idx = torch.topk(alpha, k, dim=-1)
        mask = torch.zeros_like(alpha).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(mask)
        return alpha * mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None


@torch.no_grad()
def capacity_route(
    alpha: torch.Tensor,
    top_k: int,
    capacity_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """§5.1 / §5.2 容量感知路由。

    流程 (§5.1 → §5.2):
        1. 容量上限:   Capacity_m = C · (B·S) / M  (§5.1)，向下取整，最小 1。
        2. 逐专家接纳: 对每个专家 m，在其 Top-K 候选 token 中按亲和度降序
                       接纳至多 Capacity_m 个；超出部分对 m 溢出 (§5.2 降级)。
        3. 重归一化:   对每个 token 的"实际激活专家集合" Valid 做归一化
                       alpha_tilde[m] = alpha[m] / sum_{m' in Valid} alpha[m']，
                       防止残差幅值被压缩导致分布偏移 (§5.2)。
        4. 全溢出跳过: 若 Valid = ∅，该 token 跳过 FFN，y_t = x_t (§5.2)。

    Args:
        alpha:            [B, S, M] softmax 路由概率 (训练期由 Router 计算)。
        top_k:           每个 token 考虑的专家数 K。
        capacity_factor: C >= 1 (§5.1 推荐 1.25)。

    Returns:
        alpha_tilde: [B, S, M] 重归一化后的路由权重 (溢出专家位置为 0)。
        valid_mask:  [B, S, M] bool，标记每个 token 实际激活的专家。
        stats:       溢出统计字典 (供 §5.2 "持续溢出>5% 需提升 C" 监控)。
    """
    B, S, M = alpha.shape
    N = B * S
    cap = max(1, int(capacity_factor * N / M))  # §5.1

    flat_alpha = alpha.reshape(N, M).contiguous()
    # 每个 token 的 Top-K 专家索引（按亲和度降序）
    _, topk_idx = flat_alpha.topk(top_k, dim=-1)  # [N, K]

    valid = torch.zeros(N, M, dtype=torch.bool, device=alpha.device)
    requested = torch.zeros(M, dtype=torch.long, device=alpha.device)
    overflow = torch.zeros(M, dtype=torch.long, device=alpha.device)

    # 逐专家接纳（M 为小规模，如 §6.1 的 M=4；大规模可改用向量化桶排）。
    for m in range(M):
        in_topk = (topk_idx == m).any(dim=-1)  # [N] 哪些 token 把 m 排进 Top-K
        requested[m] = in_topk.sum()
        cand = in_topk.nonzero(as_tuple=True)[0]  # 候选 token 全局索引
        if cand.numel() == 0:
            continue
        aff = flat_alpha[cand, m]  # 这些 token 对 m 的亲和度
        n_acc = min(cap, cand.numel())
        # 取亲和度最高的 n_acc 个 token 被 m 接纳；其余对 m 溢出
        _, local_sel = aff.topk(n_acc)
        valid[cand[local_sel], m] = True
        overflow[m] = cand.numel() - n_acc

    # ---- §5.2 重归一化 ----
    alpha_t = flat_alpha * valid.float()
    denom = alpha_t.sum(dim=-1, keepdim=True)
    skip = denom <= 0  # Valid = ∅ 的 token
    alpha_t = torch.where(
        denom > 0,
        alpha_t / denom.clamp(min=1e-12),
        torch.zeros_like(alpha_t),
    )

    stats = {
        "capacity_per_expert": cap,
        "per_expert_requested": requested.tolist(),
        "per_expert_overflow": overflow.tolist(),
        # 溢出率 = 溢出 token / 请求 token（§5.2 阈值 5% 监控）
        "per_expert_overflow_rate": (
            overflow.float() / requested.float().clamp(min=1.0)
        ).tolist(),
        "skip_tokens": int(skip.sum().item()),  # 全溢出跳过数
        "total_tokens": N,
        "skip_rate": float(skip.sum().item() / N),
    }
    return alpha_t.view(B, S, M), valid.view(B, S, M), stats
