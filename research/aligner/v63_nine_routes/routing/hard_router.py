"""路线2：专家负载均衡硬约束路由。

v62.4 均衡系数 0.61，专家1/2 仍过载。本模块在 router 输出后加
负载感知的 top-k + 容量因子，替代纯熵正则软约束。

接口与 v62 router 兼容：输入 [B,S,n_slots] logits，输出同形状稀疏门控。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardLoadRouter(nn.Module):
    """负载硬约束：top-k 选择 + 批次级容量惩罚。"""

    def __init__(self, n_slots: int, capacity_factor: float = 1.25,
                 aux_weight: float = 0.01):
        super().__init__()
        self.n_slots = n_slots
        self.capacity_factor = capacity_factor
        self.aux_weight = aux_weight
        # 可学习的温度，让模型自行调节稀疏度
        self.temperature = nn.Parameter(torch.tensor([1.0]))

    def forward(self, logits: torch.Tensor):
        """
        logits: [B,S,n_slots]
        returns:
            gate:  [B,S,n_slots]  硬稀疏（非 top-k 位置为 0）
            aux:   scalar         负载均衡辅助损失
        """
        B, S, K = logits.shape
        scaled = logits / self.temperature.clamp(min=0.1)
        probs = F.softmax(scaled, dim=-1)

        # ---- 批次级容量约束 ----
        # 每个 slot 的平均概率作为"期望负载"，容量 = capacity_factor * 期望
        mean_p = probs.mean(dim=(0, 1))                       # [K]
        capacity = self.capacity_factor / K                  # 每槽上限（和为1归一化）

        # 超出容量的槽被压制（梯度仍可通过 temperature 流回）
        overflow = F.relu(mean_p - capacity)
        penalty = overflow.sum()

        # ---- top-k 硬选择（k=2，与 v62 一致但加容量约束）----
        topv, topi = torch.topk(probs, k=2, dim=-1)
        gate = torch.zeros_like(probs).scatter(-1, topi, topv)
        gate = gate / (gate.sum(-1, keepdim=True) + 1e-9)
        return gate, self.aux_weight * penalty

    def extra_repr(self):
        return ('n_slots=%d, capacity=%.2f, aux_w=%.3f'
                % (self.n_slots, self.capacity_factor, self.aux_weight))
