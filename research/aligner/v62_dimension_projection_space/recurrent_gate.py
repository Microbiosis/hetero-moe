"""路线2：递归槽门控（Recurrent Slot Gate）。

把 v62 的静态 slot_gate（nn.Parameter [n_slots]）改为 GRU cell，
在 token 序列上传递 hidden state，逐 token 生成四槽贡献权重。

接口：forward(x_seq) -> gate [B,S,n_slots]，兼容原 slot_gate 用法。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentSlotGate(nn.Module):
    """GRU 驱动的逐 token 槽门控。

    输入：当前 token 的 canonical 特征 [B,S,d]
    输出：该 token 的四槽门控 [B,S,n_slots]（softmax 归一化）
    """

    def __init__(self, d_in: int = 1024, n_slots: int = 4, hidden: int = 64):
        super().__init__()
        self.n_slots = n_slots
        self.hidden = hidden
        # 序列状态：每样本独立 GRU
        self.cell = nn.GRUCell(d_in, hidden)
        self.head = nn.Linear(hidden, n_slots)
        # 初始 hidden 为可学习参数
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,S,d_in]
        returns: [B,S,n_slots] 逐 token 门控（沿 slot 维 softmax）
        """
        B, S, D = x.shape
        h = self.h0.unsqueeze(0).expand(B, -1)          # [B,hidden]
        gates = []
        for t in range(S):
            h = self.cell(x[:, t, :], h)                # [B,hidden]
            g = self.head(h)                            # [B,n_slots]
            gates.append(g)
        gates = torch.stack(gates, dim=1)               # [B,S,n_slots]
        return F.softmax(gates, dim=-1)

    def extra_repr(self):
        return ('d_in→hidden=%d, n_slots=%d (GRU 序列门控)' % (self.hidden, self.n_slots))
