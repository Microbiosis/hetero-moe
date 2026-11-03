"""路线 1：元投影网络（HyperProjection Slot）。

v62 的 SlotSubspace 使用固定正交投影 P，权重在训练后定型。
本模块让投影矩阵**随输入分布自适应**：

  W_eff[b] = W_slow + U(x_stats) @ V(x_stats)^T * scale

其中 W_slow 冻结（保持正交初始化），U/V 由一个轻量 hypernetwork
根据当前 batch 的 hidden state 统计（mean/std）生成，低秩 rank=8。

设计约束：
- W_slow requires_grad=False，只训练 hyper 参数（约 33K/槽）
- 推理路径与 nn.Linear 接口兼容：forward(x) 内部自动计算 stats
- 默认不启用（use_hyper=False），保证 v62.3 行为不变
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HyperLinear(nn.Module):
    """元投影线性层：slow weight + hypernetwork 低秩 delta。"""

    def __init__(self, d_in: int, d_out: int, rank: int = 8,
                 slow_trainable: bool = False):
        super().__init__()
        self.d_in, self.d_out, self.rank = d_in, d_out, rank

        # slow weight：等维时正交（保距），否则 Kaiming；默认冻结
        self.weight = nn.Parameter(torch.empty(d_out, d_in),
                                   requires_grad=slow_trainable)
        if d_out == d_in:
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # hypernetwork：stats(2*d_in) → U_flat(d_out*rank) + V_flat(d_in*rank)
        self.hyper = nn.Sequential(
            nn.Linear(2 * d_in, 128),
            nn.ReLU(),
            nn.Linear(128, (d_out + d_in) * rank)
        )
        self.scale = nn.Parameter(torch.tensor([0.05]))

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,S,d_in] → [B, 2*d_in]（沿序列维度求 mean/std）。"""
        if x.dim() == 3:
            mu = x.mean(dim=1)
            sd = x.std(dim=1)
        elif x.dim() == 2:
            mu = x.mean(dim=0, keepdim=True).expand(x.shape[0], -1)
            sd = x.std(dim=0, keepdim=True).expand(x.shape[0], -1)
        else:
            raise ValueError('x 应为 [B,S,d] 或 [N,d]，得到 %s' % tuple(x.shape))
        return torch.cat([mu, sd], dim=-1)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """生成 [B, d_out, d_in] 的低秩权重修正。"""
        s = self._stats(x)
        h = self.hyper(s)
        B = h.shape[0]
        U = h[:, :self.d_out * self.rank].view(B, self.d_out, self.rank)
        V = h[:, self.d_out * self.rank:].view(B, self.d_in, self.rank)
        return torch.bmm(U, V.transpose(1, 2)) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,S,d_in] → [B,S,d_out]。接口与 nn.Linear 兼容。"""
        B, S, D = x.shape
        d = self.delta(x)                                   # [B, d_out, d_in]
        W_eff = self.weight.unsqueeze(0) + d                # [B, d_out, d_in]
        return torch.bmm(x, W_eff.transpose(1, 2))          # [B,S,d_out]

    def describe(self) -> str:
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_slow = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return ('HyperLinear %d→%d rank=%d | 可训 %d (hyper) | 冻结 %d (slow)'
                % (self.d_in, self.d_out, self.rank, n_train, n_slow))
