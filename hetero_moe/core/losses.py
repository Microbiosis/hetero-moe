"""§3 — 优化目标与损失函数。

L_total = L_LM + λ1·L_balance + λ2·L_z   (§3)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """§3.1 语言建模损失 (交叉熵)。

        L_LM = -1/(B·S) Σ_{b,s} log P(y_{b,s}^{true} | y_{b,s})

    Args:
        logits:  [B, S, V] 输出投影后的 logits (y 经冻结 LM Head 得到)。
        targets: [B, S]    真实 token id。
        ignore_index: padding 等忽略索引。

    Returns:
        标量损失 (对有效 token 数取平均)。
    """
    V = logits.size(-1)
    # F.cross_entropy 期望 [N, V] / [N]
    return F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1).long(),
        ignore_index=ignore_index,
        reduction="mean",
    )


def balance_loss(alpha: torch.Tensor, alpha_hat: torch.Tensor, top_k: int) -> torch.Tensor:
    """§3.2 负载均衡损失。

        f_m = mean_{b,s} α̂_{b,s,m}                    (实际路由权重均值)
        P_m = mean_{b,s} 𝟙[m ∈ TopK(α_{b,s})]        (token 把 m 排进 Top-K 的比例)
        L_balance = M · Σ_m f_m · P_m

    Args:
        alpha:     [B, S, M] 路由 softmax (未稀疏化，用于计算 P_m 的 TopK 归属)。
        alpha_hat: [B, S, M] 实际稀疏路由权重 (用于 f_m)。
        top_k:     Top-K 值。

    Returns:
        标量损失。
    """
    M = alpha.size(-1)
    # f_m: 实际激活权重的均值
    f = alpha_hat.mean(dim=(0, 1))  # [M]
    # P_m: 每个 token 把 m 排进 Top-K 的比例 (基于未稀疏 alpha)
    topk_idx = alpha.topk(top_k, dim=-1).indices  # [B, S, K]
    # 𝟙[m ∈ TopK(α)]
    in_topk = (topk_idx.unsqueeze(-1) == torch.arange(M, device=alpha.device)).any(dim=2).float()
    P = in_topk.mean(dim=(0, 1))  # [M]
    return M * (f * P).sum()


def z_loss(z: torch.Tensor) -> torch.Tensor:
    """§3.3 Router Z-Loss。

        L_z = 1/(B·S) Σ_{b,s} ( log Σ_m exp(z_{b,s,m}) )²

    作用: 约束 Router logits 数值范围，防止路由置信度爆炸。
    """
    return (torch.logsumexp(z, dim=-1) ** 2).mean()


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor,
    alpha_hat: torch.Tensor,
    z: torch.Tensor,
    top_k: int,
    lam_balance: float = 0.01,
    lam_z: float = 0.001,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """§3 总损失 L_total = L_LM + λ1·L_balance + λ2·L_z。

    Returns:
        dict: {total, lm, balance, z} 各分量 (便于日志/可追溯性)。
    """
    l_lm = lm_loss(logits, targets, ignore_index=ignore_index)
    l_bal = balance_loss(alpha, alpha_hat, top_k)
    l_z = z_loss(z)
    total = l_lm + lam_balance * l_bal + lam_z * l_z
    return {"total": total, "lm": l_lm, "balance": l_bal, "z": l_z}
