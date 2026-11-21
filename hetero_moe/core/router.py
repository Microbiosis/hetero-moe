"""§2.3 / §5 — 稀疏门控路由、容量感知推理路由与训练期容量一致性 STE (v5.1)。

包含三个算子:
    1. SparseRouterSTE   (§2.3)  —— 训练期硬 Top-K 路由 + STE，防止非 Top-K 梯度泄漏。
    2. capacity_route    (§5)    —— 推理期容量因子约束 + 溢出降级 + 动态重归一化。
    3. CapacityRouterSTE (v5.1)  —— 训练期容量感知路由 STE，使训练与推理共享相同的容量上界
                                    Capacity_m = floor(C · N / M)，消除训练(C=∞)与推理(C=1.25)分布偏移。
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
    def forward(ctx, alpha, k, renorm: bool = False):
        _, idx = torch.topk(alpha, k, dim=-1)
        mask = torch.zeros_like(alpha).scatter_(-1, idx, 1.0)
        sparse = alpha * mask
        ctx.renorm = bool(renorm)
        ctx.k = int(k)
        if ctx.renorm:
            denom = sparse.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            out = sparse / denom
            ctx.save_for_backward(mask, out, denom)
            return out
        ctx.save_for_backward(mask)
        return sparse

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, "renorm", False):
            mask, out, denom = ctx.saved_tensors
            if getattr(ctx, "k", 2) <= 1:
                # K=1 时前向 sparse/denom 恒为 1.0，若直接套商法则 (grad - dot) 会使梯度恒等于 0；
                # 采用 STE 重缩放直通 (denom 视为常数因子)，由外层 softmax(z) 的 Jacobian 自然产生
                # 跨专家竞争梯度，确保 top_k=1 且 renorm_topk=True 时 W_router 正常更新。
                return grad_output * mask / denom, None, None
            dot = (grad_output * out).sum(dim=-1, keepdim=True)
            grad_alpha = (grad_output - dot) * mask / denom
            return grad_alpha, None, None
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None, None


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
    """
    B, S, M = alpha.shape
    N = B * S
    cap = max(1, int(capacity_factor * N / M))  # §5.1

    flat_alpha = alpha.reshape(N, M).contiguous()
    _, topk_idx = flat_alpha.topk(top_k, dim=-1)  # [N, K]

    valid = torch.zeros(N, M, dtype=torch.bool, device=alpha.device)
    requested = torch.zeros(M, dtype=torch.long, device=alpha.device)
    overflow = torch.zeros(M, dtype=torch.long, device=alpha.device)

    for m in range(M):
        in_topk = (topk_idx == m).any(dim=-1)
        requested[m] = in_topk.sum()
        cand = in_topk.nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            continue
        aff = flat_alpha[cand, m]
        n_acc = min(cap, cand.numel())
        _, local_sel = aff.topk(n_acc)
        valid[cand[local_sel], m] = True
        overflow[m] = cand.numel() - n_acc

    alpha_t = flat_alpha * valid.float()
    denom = alpha_t.sum(dim=-1, keepdim=True)
    skip = denom <= 0
    alpha_t = torch.where(
        denom > 0,
        alpha_t / denom.clamp(min=1e-12),
        torch.zeros_like(alpha_t),
    )

    stats = {
        "capacity_per_expert": cap,
        "per_expert_requested": requested.tolist(),
        "per_expert_overflow": overflow.tolist(),
        "per_expert_overflow_rate": (
            overflow.float() / requested.float().clamp(min=1.0)
        ).tolist(),
        "skip_tokens": int(skip.sum().item()),
        "total_tokens": N,
        "skip_rate": float(skip.sum().item() / N),
    }
    return alpha_t.view(B, S, M), valid.view(B, S, M), stats


class CapacityRouterSTE(torch.autograd.Function):
    """训练期容量感知路由 STE (v5.1): 前向执行 capacity_route 截断，反向按接纳掩码透传梯度。"""

    @staticmethod
    def forward(ctx, alpha: torch.Tensor, top_k: int, capacity_factor: float, renorm: bool = True):
        alpha_t, valid_mask, _ = capacity_route(alpha.detach(), int(top_k), float(capacity_factor))
        mask = valid_mask.to(alpha.dtype)
        sparse = alpha * mask
        ctx.renorm = bool(renorm)
        ctx.k = int(top_k)
        if ctx.renorm:
            denom = sparse.sum(dim=-1, keepdim=True)
            out = torch.where(denom > 0, sparse / denom.clamp(min=1e-12), torch.zeros_like(sparse))
            ctx.save_for_backward(mask, out, denom.clamp(min=1e-12))
            return out
        ctx.save_for_backward(mask)
        return sparse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if getattr(ctx, "renorm", False):
            mask, out, denom = ctx.saved_tensors
            if getattr(ctx, "k", 2) <= 1:
                return grad_output * mask / denom, None, None, None
            dot = (grad_output * out).sum(dim=-1, keepdim=True)
            grad_alpha = (grad_output - dot) * mask / denom
            return grad_alpha, None, None, None
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None, None, None


def capacity_route_ste(
    alpha: torch.Tensor,
    top_k: int,
    capacity_factor: float,
    renorm: bool = True,
) -> torch.Tensor:
    """函数式封装: 训练期容量感知路由 STE。"""
    return CapacityRouterSTE.apply(alpha, int(top_k), float(capacity_factor), bool(renorm))
