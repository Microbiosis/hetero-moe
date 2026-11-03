"""路线8：记忆增强路由（Memory-Augmented Router）。

维护一组可修改的"记忆权重" M ∈ R^{N×d}，通过 attention-like 读写
存储历史路由决策，并反馈到当前 router。

写：M ← M + α·outer(q, v)（当前 batch 的路由 query/value）
读：router 输入 = [current_features, retrieved_memory]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAugmentedRouter(nn.Module):
    """带外部记忆的路由器。"""

    def __init__(self, d_in: int, n_slots: int, n_memory: int = 64,
                 d_mem: int = 64, alpha: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.n_slots = n_slots
        self.n_memory = n_memory
        self.d_mem = d_mem
        self.alpha = alpha

        # 记忆槽：key 和 value
        self.register_buffer('mem_key', torch.randn(n_memory, d_mem) * 0.02)
        self.register_buffer('mem_val', torch.zeros(n_memory, n_slots))
        self.register_buffer('mem_age', torch.zeros(n_memory))

        # 投影：features → query
        self.q_proj = nn.Linear(d_in, d_mem)
        # 路由头：features + retrieved → slot logits
        self.router = nn.Linear(d_in + n_slots, n_slots)

    @torch.no_grad()
    def write(self, q: torch.Tensor, v: torch.Tensor):
        """q: [B,d_mem] 当前 query；v: [B,n_slots] 当前路由值。"""
        # 找最旧的记忆槽替换（或累加）
        idx = torch.argmin(self.mem_age).item()
        self.mem_key[idx] = q.mean(0)
        self.mem_val[idx] = v.mean(0)
        self.mem_age[idx] = 0
        self.mem_age += 1

    @torch.no_grad()
    def read(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B,d_mem] → retrieved: [B,n_slots]。"""
        scores = q @ self.mem_key.T                     # [B,N]
        w = F.softmax(scores / (self.d_mem ** 0.5), dim=-1)
        return w @ self.mem_val                          # [B,n_slots]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,S,d_in]
        returns: [B,S,n_slots] 路由 logits
        """
        B, S, D = x.shape
        xin = x.reshape(B * S, D)
        q = self.q_proj(xin)                             # [B*S,d_mem]
        retrieved = self.read(q)                         # [B*S,n_slots]
        logits = self.router(torch.cat([xin, retrieved], dim=-1))
        logits = logits.view(B, S, self.n_slots)

        # 写记忆（用当前 logits 的 softmax 作为 value）
        if self.training:
            with torch.no_grad():
                v = F.softmax(logits.reshape(B * S, -1), dim=-1)
                self.write(q, v)
        return logits

    def reset_memory(self):
        self.mem_key.normal_(0, 0.02)
        self.mem_val.zero_()
        self.mem_age.zero_()

    def extra_repr(self):
        return ('d_in=%d, n_slots=%d, memory=%d×%d'
                % (self.d_in, self.n_slots, self.n_memory, self.d_mem))
