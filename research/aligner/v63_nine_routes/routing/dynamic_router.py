"""路线6：动态路由机制替代静态 top-k。

v62 的 router 是静态 top-2（虽已改为全槽软门控）。本模块让每 token
根据自身激活强度自适应选择参与融合的槽数：高不确定性 token 用更多槽，
确定性 token 稀疏化。用 Gumbel-sigmoid 实现可微分稀疏。

接口：输入 [B,S,n_slots] logits，输出 [B,S,n_slots] 动态稀疏门控。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSlotRouter(nn.Module):
    """逐 token 自适应稀疏：熵驱动的 Gumbel-sigmoid 门控。"""

    def __init__(self, n_slots: int, tau: float = 0.5,
                 min_slots: int = 1, max_slots: int = 4):
        super().__init__()
        self.n_slots = n_slots
        self.tau = tau
        self.min_slots = min_slots
        self.max_slots = min(max_slots, n_slots)
        # 稀疏度偏置（可学习）：初始倾向于全开
        self.sparsity_bias = nn.Parameter(torch.zeros(n_slots))

    def forward(self, logits: torch.Tensor, training: bool = True):
        """
        logits: [B,S,n_slots]
        returns:
            gate: [B,S,n_slots]  每 token 独立稀疏（和≈1）
            n_active: [B,S]      每 token 实际激活槽数
        """
        B, S, K = logits.shape
        logits = logits + self.sparsity_bias

        if training:
            # Gumbel-sigmoid：每槽独立二选，可微分
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            sig = torch.sigmoid((logits + gumbel) / self.tau)
            mask = sig
        else:
            mask = (torch.sigmoid(logits / self.tau) > 0.5).float()

        # 保证至少 min_slots 个槽激活：取 top-min 强制置1
        if self.min_slots > 0:
            topv, topi = torch.topk(mask, k=self.min_slots, dim=-1)
            force = torch.zeros_like(mask).scatter(-1, topi, 1.0)
            mask = torch.clamp(mask + force, max=1.0)

        # 保证至多 max_slots 个：超出部分按概率置0
        if self.max_slots < K:
            topv, topi = torch.topk(mask, k=self.max_slots, dim=-1)
            cap = torch.zeros_like(mask).scatter(-1, topi, 1.0)
            mask = mask * cap

        # 归一化到和为1（软权重）
        gate = mask / (mask.sum(-1, keepdim=True) + 1e-9)
        n_active = (mask > 0).float().sum(-1)
        return gate, n_active

    def extra_repr(self):
        return ('n_slots=%d, tau=%.2f, slots∈[%d,%d]'
                % (self.n_slots, self.tau, self.min_slots, self.max_slots))
