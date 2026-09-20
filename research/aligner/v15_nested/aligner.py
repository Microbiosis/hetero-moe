"""v15.0 — MoMAligner (Mixture of Mixture 嵌套专家池).

设计:
    Level 1 (底座级): 每底座贡献 n_inner 个"原始专家" (Linear-ReLU-Linear).
                       所有底座的原始专家放在一起 (M × n_inner 个),
                       跨专家 attn -> 共享表示 1.
    Level 2 (池级): 共享表示 1 通过 n_outer 个"高层专家"再加工.
                    跨高层专家 attn -> 最终统一表示.

    总参数量: M × n_inner × expert_size + n_outer × expert_size
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    """单个专家 (Linear + ReLU + Linear)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


class MoMAligner(nn.Module):
    """两级嵌套专家池对齐器.

    Args:
        modal_dims: M 个底座的输入维度
        d_shared: 输出维度
        n_inner: 每底座的原始专家数 (默认 4)
        n_outer: 高层专家数 (默认 4)
        num_heads: 多头数 (默认 4)
    """

    def __init__(
        self,
        modal_dims: List[int],
        d_shared: int,
        n_inner: int = 4,
        n_outer: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        assert d_shared % num_heads == 0
        self.d_shared = d_shared
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.num_heads = num_heads
        self.head_dim = d_shared // num_heads

        # Level 1: 原始专家 (M × n_inner)
        self.inner_experts = nn.ModuleList([
            _Expert(D_m, d_shared)
            for D_m in modal_dims
            for _ in range(n_inner)
        ])
        self.n_inner_total = len(self.inner_experts)
        # Level 1 共享 K/V (基于原始专家输出平均)
        self.W_k1 = nn.Linear(d_shared, d_shared, bias=False)
        self.W_v1 = nn.Linear(d_shared, d_shared, bias=False)
        # Level 1 输出投影
        self.out_proj_1 = nn.Linear(d_shared, d_shared)

        # Level 2: 高层专家 (n_outer 个, 输入是 d_shared)
        self.outer_experts = nn.ModuleList([
            _Expert(d_shared, d_shared)
            for _ in range(n_outer)
        ])
        self.n_outer_total = len(self.outer_experts)
        # Level 2 共享 K/V
        self.W_k2 = nn.Linear(d_shared, d_shared, bias=False)
        self.W_v2 = nn.Linear(d_shared, d_shared, bias=False)
        # 最终输出投影
        self.out_proj_final = nn.Linear(d_shared, d_shared)

    def _split_heads(self, t):
        """[B, S, D] -> [B, H, S, D/H]"""
        B, S, D = t.shape
        return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        """h_list: M 个张量, 每个 [B, S, D_m]"""
        # ===== Level 1: 底座级 (跨底座 + 跨原始专家) =====
        # 1a. 每个原始专家前向
        inner_outputs = []
        for m, h in enumerate(h_list):
            for c in range(self.n_inner):
                idx = m * self.n_inner + c
                inner_outputs.append(self.inner_experts[idx](h))
        # [M*n_inner, B, S, D]
        inner_stack = torch.stack(inner_outputs, dim=0)
        # 1b. 跨原始专家池平均 → 共享 K/V
        l1_context = inner_stack.mean(dim=0)             # [B, S, D]
        shared_k1 = self.W_k1(l1_context)
        shared_v1 = self.W_v1(l1_context)
        # 1c. 跨原始专家 attn (每个专家输出作 query)
        B, S, D = l1_context.shape
        # [M*n_inner, B, S, D] -> [B*S, M*n_inner, D]
        inner_q = inner_stack.permute(1, 2, 0, 3).reshape(B * S, self.n_inner_total, D)
        q = self._split_heads(inner_q)
        k = self._split_heads(shared_k1.reshape(B * S, 1, D))
        v = self._split_heads(shared_v1.reshape(B * S, 1, D))
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        attended_l1 = (attn @ v).transpose(1, 2).reshape(B, S, self.n_inner_total, D)
        # [B, S, M*n_inner, D] -> mean over experts -> [B, S, D]
        l1_output = attended_l1.mean(dim=2)             # [B, S, D]
        l1_output = self.out_proj_1(l1_output)

        # ===== Level 2: 池级 (跨高层专家) =====
        # 2a. 每个高层专家前向 (输入是 l1_output)
        outer_outputs = [expert(l1_output) for expert in self.outer_experts]
        # [n_outer, B, S, D]
        outer_stack = torch.stack(outer_outputs, dim=0)
        # 2b. 跨高层专家池平均 → 共享 K/V
        l2_context = outer_stack.mean(dim=0)              # [B, S, D]
        shared_k2 = self.W_k2(l2_context)
        shared_v2 = self.W_v2(l2_context)
        # 2c. 跨高层专家 attn
        outer_q = outer_stack.permute(1, 2, 0, 3).reshape(B * S, self.n_outer_total, D)
        q2 = self._split_heads(outer_q)
        k2 = self._split_heads(shared_k2.reshape(B * S, 1, D))
        v2 = self._split_heads(shared_v2.reshape(B * S, 1, D))
        attn2 = F.softmax(q2 @ k2.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        attended_l2 = (attn2 @ v2).transpose(1, 2).reshape(B, S, self.n_outer_total, D)
        # [B, S, n_outer, D] -> mean -> [B, S, D]
        l2_output = attended_l2.mean(dim=2)              # [B, S, D]
        return self.out_proj_final(l2_output)