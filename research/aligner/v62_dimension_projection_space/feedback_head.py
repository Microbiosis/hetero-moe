"""路线9：跨层递归反馈（Recursive Fusion Feedback）。

上层融合层根据融合结果的置信度，生成 top-down 信号微调下层投影参数。

结构：
  fusion 输出 y [B,S,1024] → feedback_head → ΔP_slot [B,S,n_slots,d,d]
  通过残差连接回写：P_slot ← P_slot + λ·ΔP_slot

实现：feedback_head 输出低秩 delta（rank-4），避免全量 256×256。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveFeedbackHead(nn.Module):
    """从融合输出生成投影矩阵的残差修正。"""

    def __init__(self, d_fused: int = 1024, n_slots: int = 4,
                 slot_dim: int = 256, rank: int = 4):
        super().__init__()
        self.d_fused = d_fused
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.rank = rank

        # 置信度评估：融合输出 → 每槽一个标量置信度
        self.conf = nn.Sequential(
            nn.Linear(d_fused, 128), nn.ReLU(),
            nn.Linear(128, n_slots), nn.Sigmoid()
        )
        # 低秩 delta 生成器：融合输出 → 每槽 (U,V)
        out_dim = n_slots * 2 * slot_dim * rank
        self.gen = nn.Sequential(
            nn.Linear(d_fused, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.lambda_ = nn.Parameter(torch.tensor([0.01]))

    def forward(self, y: torch.Tensor):
        """
        y: [B,S,d_fused] 融合层输出
        returns:
            confidence: [B,S,n_slots]
            deltas: list of n_slots tensors, each [slot_dim,slot_dim]（batch 级共享）
        """
        conf = self.conf(y)                              # [B,S,n_slots]
        mu = y.detach().mean(dim=(0, 1))                 # [d_fused] 全局均值
        raw = self.gen(mu)                               # [out_dim]

        deltas = []
        per_slot = 2 * self.slot_dim * self.rank
        for i in range(self.n_slots):
            chunk = raw[i * per_slot:(i + 1) * per_slot]
            U = chunk[:self.slot_dim * self.rank].view(self.slot_dim, self.rank)
            V = chunk[self.slot_dim * self.rank:].view(self.rank, self.slot_dim)
            deltas.append(torch.matmul(U, V))            # [d,d]
        return conf, deltas

    def apply_feedback(self, proj_weights: list, y: torch.Tensor,
                       slot_idx_map: dict):
        """把 feedback 应用到投影权重（in-place，调用方控制梯度）。

        proj_weights: 各槽的投影矩阵参数列表（nn.Parameter）
        """
        conf, deltas = self.forward(y)
        with torch.no_grad():
            for slot_name, idx in slot_idx_map.items():
                w = proj_weights[idx]
                d = deltas[list(slot_idx_map).index(slot_name)]
                w.add_(self.lambda_ * d.mean(dim=(0, 1)))
        return conf

    def extra_repr(self):
        return ('fused=%d, slots=%d, dim=%d, rank=%d'
                % (self.d_fused, self.n_slots, self.slot_dim, self.rank))
