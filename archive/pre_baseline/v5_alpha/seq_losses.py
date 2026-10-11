"""§3.2 / §3.3 序列级损失 (D1 闭合)。

v4.0 的 token 级 f_m/P_m 在 G=sequence 下语义失效, 此处重推导为序列级。
梯度路径完全在序列级, 不触碰 token 级残差。
"""
from __future__ import annotations

import torch


def balance_loss_seq(alpha_seq: torch.Tensor, alpha_hat_seq: torch.Tensor, top_k: int) -> torch.Tensor:
    """§3.2 序列级负载均衡损失。

        f_m^(seq) = (1/B) Σ_b α̂_{b,m}^(seq)                       (序列级权重分数)
        P_m^(seq) = (1/B) Σ_b 𝟙[m ∈ TopK(α_b^(seq))]             (序列级选择概率)
        L_balance^(seq) = M · Σ_m f_m^(seq) · P_m^(seq)

    Args:
        alpha_seq:     [B, M] softmax 路由概率 (未稀疏化, 用于 P_m 的 TopK 归属)
        alpha_hat_seq:  [B, M] 稀疏化后软权重 (用于 f_m)
        top_k:         Top-K
    """
    M = alpha_seq.size(-1)
    f = alpha_hat_seq.mean(dim=0)  # [M]
    topk_idx = alpha_seq.topk(top_k, dim=-1).indices  # [B, K]
    in_topk = (topk_idx.unsqueeze(-1) == torch.arange(M, device=alpha_seq.device)).any(dim=1).float()
    P = in_topk.mean(dim=0)  # [M]
    return M * (f * P).sum()


def z_loss_seq(z_seq: torch.Tensor) -> torch.Tensor:
    """§3.3 序列级 Router Z-Loss。

        L_z^(seq) = (1/B) Σ_b ( log Σ_m exp(z_{b,m}) )²
    """
    return (torch.logsumexp(z_seq, dim=-1) ** 2).mean()
