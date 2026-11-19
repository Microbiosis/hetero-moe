"""路线8：记忆增强路由（Memory-Augmented Router）—— 负反馈修复版。

v62.6 初版失败根因：只有正反馈——write 存 softmax(logits)，read 用
memory 强化 router，被选中的槽自我强化，最终 V 槽独占（熵 0.076）。

本版加三层负反馈：
  1. 记忆衰减（decay）：每次 write 前旧记忆整体淡出，防止单条记忆永久主导
  2. 过载减压（relief）：某槽在 memory 中累积超过阈值时，全局缩放该列
  3. 检索居中（centering）：read 返回相对 memory 均值的偏差，打破同向累积
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAugmentedRouter(nn.Module):
    """带外部记忆 + 负反馈均衡的路由器。"""

    def __init__(self, d_in: int, n_slots: int, n_memory: int = 64,
                 d_mem: int = 64, alpha: float = 0.1,
                 decay: float = 0.95, relief: float = 0.9,
                 overload_thresh: float = 1.5):
        super().__init__()
        self.d_in = d_in
        self.n_slots = n_slots
        self.n_memory = n_memory
        self.d_mem = d_mem
        self.alpha = alpha
        self.decay = decay
        self.relief = relief
        self.overload_thresh = overload_thresh

        self.register_buffer('mem_key', torch.randn(n_memory, d_mem) * 0.02)
        self.register_buffer('mem_val', torch.zeros(n_memory, n_slots))
        # 初始化为递增正年龄，保证首次写入依次填满 0..n_memory-1 全部槽位后再按 LRU 驱逐最老槽
        self.register_buffer('mem_age', torch.arange(n_memory, 0, -1, dtype=torch.float32))

        self.q_proj = nn.Linear(d_in, d_mem)
        self.router = nn.Linear(d_in + n_slots, n_slots)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def write(self, q: torch.Tensor, v: torch.Tensor):
        """q: [B,d_mem]；v: [B,n_slots]。带负反馈与真实 LRU 驱逐的写入。"""
        # 负反馈1：旧记忆衰减，越老的记忆影响越小
        self.mem_val.mul_(self.decay)
        self.mem_key.mul_(self.decay ** 0.5)

        # LRU 驱逐：选择年龄最大的槽位（修复原 argmin 永久覆盖 0 号槽的 bug）
        idx = int(torch.argmax(self.mem_age).item())
        self.mem_key[idx] = q.detach().mean(0)
        self.mem_val[idx] = v.detach().mean(0)
        self.mem_age += 1
        self.mem_age[idx] = 0

        # 负反馈2：过载减压——某槽累积超过全槽均值 × 阈值时，缩放该列
        col_sum = self.mem_val.sum(dim=0)
        mean_sum = col_sum.mean()
        if mean_sum > 1e-9:
            overload = col_sum > mean_sum * self.overload_thresh
            if overload.any():
                self.mem_val[:, overload] *= self.relief

    def read(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B,d_mem] → retrieved [B,n_slots]，保留 q_proj 梯度通路并带居中负反馈。"""
        # clone 隔离同前向内后续 write() 的原地 buffer 更新，避免 autograd 版本冲突
        mem_k = self.mem_key.detach().clone()
        mem_v = self.mem_val.detach().clone()
        scores = q @ mem_k.T
        w = F.softmax(scores / (self.d_mem ** 0.5), dim=-1)
        retrieved = w @ mem_v
        # 负反馈3：减去 memory 全局趋势，只返回"相对偏差"，并用 alpha 控制注入幅度
        retrieved = (retrieved - mem_v.mean(dim=0, keepdim=True)) * self.alpha
        return retrieved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,S,d_in] → logits [B,S,n_slots]。"""
        B, S, D = x.shape
        xin = x.reshape(B * S, D)
        q = self.q_proj(xin)
        retrieved = self.read(q)
        logits = self.router(torch.cat([xin, retrieved], dim=-1))
        logits = logits.view(B, S, self.n_slots)

        if self.training:
            with torch.no_grad():
                v = F.softmax(logits.reshape(B * S, -1), dim=-1)
                self.write(q, v)
        return logits

    def reset_memory(self):
        self.mem_key.normal_(0, 0.02)
        self.mem_val.zero_()
        self.mem_age.copy_(
            torch.arange(self.n_memory, 0, -1, device=self.mem_age.device, dtype=self.mem_age.dtype)
        )

    def extra_repr(self):
        return ('d_in=%d, n_slots=%d, memory=%d×%d, decay=%.2f, '
                'relief=%.2f, thresh=%.1f'
                % (self.d_in, self.n_slots, self.n_memory, self.d_mem,
                   self.decay, self.relief, self.overload_thresh))
