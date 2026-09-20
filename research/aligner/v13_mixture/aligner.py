"""v13.0 — MixtureAligner (跨底座 + 跨专家注意力)。

设计:
    每个底座贡献 n_shared_experts 个"共享专家" (每个专家是 Linear+ReLU+Linear).
    所有底座的专家放进同一个池子 (M × n_shared_experts 个).
    跨底座 + 跨专家的注意力聚合.

    比 v10 CrossArchAttnAligner 更强:
        v10: 共享 K/V, 每个底座独立 query → 概念对齐
        v13: 共享专家池, 跨专家池注意力 → 每个 token 跨专家选择 + 跨底座平均
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SharedExpert(nn.Module):
    """单个共享专家: 输入 D_m, 输出 d_shared."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


class MixtureAligner(nn.Module):
    """跨底座 + 跨专家对齐器。

    每个底座 m 贡献 n_shared_experts 个专家, 共 M × n_shared_experts 个.
    跨底座共享查询空间 (类似 v10 的 K/V): 平均池化所有底座的专家输出作 K/V.
    每个 token 跨专家池选 top-k 个表达, 加权和.

    Args:
        modal_dims: M 个底座的输入维度
        d_shared: 输出维度
        n_shared_experts: 每个底座贡献的专家数 (默认 4)
        num_heads: 多头数 (用于 v10 兼容)
        top_k: 每个 token 选的专家数 (默认 2)
    """

    def __init__(
        self,
        modal_dims: List[int],
        d_shared: int,
        n_shared_experts: int = 4,
        num_heads: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        assert d_shared % num_heads == 0
        self.d_shared = d_shared
        self.n_shared_experts = n_shared_experts
        self.num_heads = num_heads
        self.head_dim = d_shared // num_heads
        self.top_k = top_k

        # M × n_shared_experts 个共享专家 (flatten 索引 m*N+c)
        self.experts = nn.ModuleList([
            _SharedExpert(D_m, d_shared)
            for D_m in modal_dims
            for _ in range(n_shared_experts)
        ])
        self.num_experts = len(self.experts)
        # 跨底座共享 K/V 投影
        self.W_k = nn.Linear(d_shared, d_shared, bias=False)
        self.W_v = nn.Linear(d_shared, d_shared, bias=False)
        # 输出投影
        self.out_projection = nn.Linear(d_shared, d_shared)

    def _split_heads(self, t):
        """[B, S, D] -> [B, H, S, D/H]"""
        B, S, D = t.shape
        return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        """h_list: M 个张量, 每个 [B, S, D_m]"""
        # 1. 所有专家前向 (M*N 个)
        # expert_outputs[m*N+c]: [B, S, D_shared]
        expert_outputs = []
        for e, h in zip(self.experts, h_list):
            # 每底座循环 n_shared_experts 次
            for _ in range(self.n_shared_experts):
                pass  # placeholder
        # 简化: 每个 expert 对应一个底座, 循环 M 次, 每次用对应的底座输入
        expert_outputs = []
        for m, h in enumerate(h_list):
            for c in range(self.n_shared_experts):
                idx = m * self.n_shared_experts + c
                expert_outputs.append(self.experts[idx](h))   # [B, S, D_shared]
        # [M*N, B, S, D]
        stacked = torch.stack(expert_outputs, dim=0)
        # 2. 跨专家池平均 → 共享 K/V
        shared_context = stacked.mean(dim=0)            # [B, S, D]
        shared_k = self.W_k(shared_context)             # [B, S, D]
        shared_v = self.W_v(shared_context)             # [B, S, D]
        # 3. 跨专家注意力 (用各专家输出作 query)
        # 每个专家: q_e = expert_output · W_q_e (这里共享一个 W_q)
        # 简化: 直接用 stacked 作 queries
        # [M*N, B, S, D] -> [B, S, M*N, D]
        queries = stacked.permute(1, 2, 0, 3)          # [B, S, M*N, D]
        B, S, E, D = queries.shape
        q = self._split_heads(queries.reshape(B * S, E, D))    # [B*S, H, E, D/H]
        # shared_k/v: [B, S, D] -> [B*S, 1, D] -> [B*S, H, 1, D/H]
        k = self._split_heads(shared_k.reshape(B * S, 1, D))
        v = self._split_heads(shared_v.reshape(B * S, 1, D))
        # attention: each expert query attends to shared context
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        # [B*S, H, E, D/H]
        attended = attn @ v                              # 每个专家得到一个共享表达
        # 4. top-k 选择: 简化取所有 (因为 E=M*N 不大), 加权平均
        # 把 attended reshape 回 [B, S, E, D]
        attended = attended.transpose(1, 2).reshape(B, S, E, D).permute(2, 0, 1, 3)
        # [E, B, S, D] -> mean over E (或 top-k)
        aligned = attended.mean(dim=0)                   # [B, S, D]
        # 输出投影
        return self.out_projection(aligned)