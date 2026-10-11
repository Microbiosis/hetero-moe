"""§5.1 句级 InfoNCE 损失 (替代 v4.0 的 L_LM)。

Embedding 的训练目标是句级对比损失, 非 next-token prediction。
采用 SimCLR 风格 in-batch InfoNCE, 配对正样本 (0,1),(2,3),...
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(e: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """句级 InfoNCE。

    Args:
        e: [B, D] 句向量 (B 必须为偶数; 配对 (0,1),(2,3),...)
        temperature: 温度系数

    Returns:
        标量损失。随机初始化时 ≈ log(B-1) (均匀分布下界)。
    """
    B = e.size(0)
    assert B % 2 == 0 and B >= 2, "B 必须为偶数且 >= 2"
    e = F.normalize(e, dim=-1)
    sim = e @ e.T / temperature  # [B, B]
    # 屏蔽自相似 (对角线), 避免自己当正样本
    self_mask = torch.eye(B, device=e.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    # 正样本索引: i 的正样本是 i^1 (XOR 1)
    pos_idx = torch.arange(B, device=e.device) ^ 1
    # log-softmax over all j != i, 取 -log sim[i, pos[i]]
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    loss = -log_prob[torch.arange(B, device=e.device), pos_idx].mean()
    return loss
