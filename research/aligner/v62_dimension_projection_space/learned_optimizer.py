"""路线5：学习式权重优化器（Learned Weight Optimizer）。

用一个小型网络 O(W, g) 替代 SGD/Adam 更新投影矩阵：
  输入：权重的奇异值谱（top-k）+ 梯度范数
  输出：低秩更新 ΔW = U V^T（rank-4）

元训练：让 O 学会在 alignment_loss / consistency_loss 下稳定更新。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedWeightOptimizer(nn.Module):
    """学习式优化器：预测低秩权重更新。"""

    def __init__(self, d: int = 256, rank: int = 4, sv_topk: int = 8):
        super().__init__()
        self.d = d
        self.rank = rank
        self.sv_topk = sv_topk
        # 输入：sv_topk 个奇异值 + 1 个梯度范数 + 1 个 loss 值
        in_dim = sv_topk + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2 * d * rank)      # U_flat + V_flat
        )
        self.scale = nn.Parameter(torch.tensor([0.01]))

    def predict_delta(self, W: torch.Tensor, grad: torch.Tensor,
                      loss_val: float = 0.0) -> torch.Tensor:
        """
        W: [d_out, d_in], grad: 同形状
        returns: ΔW [d_out, d_in] 低秩
        """
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W.detach(), full_matrices=False)
            sv = S[:self.sv_topk]
            if sv.shape[0] < self.sv_topk:
                sv = F.pad(sv, (0, self.sv_topk - sv.shape[0]))
            gnorm = grad.detach().norm().view(1)
            lval = torch.tensor([loss_val], device=W.device)
            inp = torch.cat([sv, gnorm, lval]).unsqueeze(0).float()
        out = self.net(inp).squeeze(0)
        Uf = out[:self.d * self.rank].view(self.d, self.rank)
        Vf = out[self.d * self.rank:].view(self.rank, W.shape[1])
        return self.scale * (Uf @ Vf)

    def step(self, W: torch.Tensor, grad: torch.Tensor,
             loss_val: float = 0.0) -> torch.Tensor:
        """返回更新后的权重（in-place 由调用方决定）。"""
        delta = self.predict_delta(W, grad, loss_val)
        return W + delta

    def extra_repr(self):
        return ('d=%d, rank=%d, sv_topk=%d' % (self.d, self.rank, self.sv_topk))
