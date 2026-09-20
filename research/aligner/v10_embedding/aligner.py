"""v10.0 — 语义对齐器 (Semantic Aligner)。

设计:
    把 v6/v7/v8/v9 的线性投影 P_m (D_shared × D_m) 替换为跨架构对齐机制.
    每个 token 查询"共享概念空间", 不同底座的同一概念在 attention 后自然对齐.

基线 (v9 的 P_m 线性投影):
    x_shared = h_m · P_m^T    (D_m → D_shared, 线性)

升级 (v10 的 SemanticAligner):
    1. 各底座先各自降维到 d_aligned (= D_shared // 2)
    2. 跨底座共享 K/V 空间: shared_k, shared_v
    3. 每个 token 用 q_m 查询 shared_k, 加权 shared_v → 概念向量
    4. 概念向量与各底座的降维输出拼接 → 最终统一表示

class CrossArchMeanAligner:
    简化版: 跨底座均值对齐, 不引入共享 K/V
class CrossArchAttnAligner:
    完整版: 共享 K/V 注意力
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAligner(nn.Module):
    """语义对齐器基类。

    输入: h_list = [h_A, h_B, h_C], 每个 [B, S, D_m]
    输出: h_aligned [B, S, D_shared] (统一空间)
    """

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class CrossArchMeanAligner(SemanticAligner):
    """简化版: 各底座各自降维 + 跨底座均值。

    工具 A 的具体实现:
        h_aligned_m = h_m · W_down_m            (D_m → D_shared)
        h_aligned = mean(h_aligned_A, h_aligned_B, h_aligned_C)
    """

    def __init__(self, modal_dims: List[int], d_shared: int):
        super().__init__()
        self.d_shared = d_shared
        self.down_projections = nn.ModuleList(
            [nn.Linear(D_m, d_shared) for D_m in modal_dims]
        )

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        aligned = [proj(h) for proj, h in zip(self.down_projections, h_list)]
        return torch.stack(aligned).mean(dim=0)


class CrossArchAttnAligner(SemanticAligner):
    """完整版: 共享 K/V 的跨底座注意力。

    工具 B 的具体实现:
        1. 各底座降维: z_m = h_m · W_down_m              (D_m → d_aligned)
        2. 共享 K/V:   shared_k = mean(z_A, z_B, z_C) · W_k
                       shared_v = mean(z_A, z_B, z_C) · W_v
        3. 各底座 attention:
              q_m = z_m · W_q
              α_m = softmax(q_m · shared_k^T / √D)
              concept_m = α_m · shared_v
        4. 拼接:    h_aligned_m = z_m + concept_m (残差)
        5. 跨底座均值: h_aligned = mean(aligned_A, aligned_B, aligned_C)
    """

    def __init__(self, modal_dims: List[int], d_shared: int, num_heads: int = 4):
        super().__init__()
        assert d_shared % num_heads == 0
        self.d_shared = d_shared
        self.d_aligned = d_shared  # 中间维度
        self.num_heads = num_heads
        self.head_dim = d_shared // num_heads

        # 各底座降维到 d_shared
        self.down_projections = nn.ModuleList(
            [nn.Linear(D_m, d_shared) for D_m in modal_dims]
        )
        # 共享 K/V 投影 (跨底座共用)
        self.W_k = nn.Linear(d_shared, d_shared, bias=False)
        self.W_v = nn.Linear(d_shared, d_shared, bias=False)
        # 各底座 Q 投影 (可训练, 让不同底座学不同 query)
        self.q_projections = nn.ModuleList(
            [nn.Linear(d_shared, d_shared) for _ in modal_dims]
        )
        # 输出投影
        self.out_projection = nn.Linear(d_shared, d_shared)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, S, D] -> [B, H, S, D/H]"""
        B, S, D = t.shape
        return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        # 1. 各底座降维
        z_list = [proj(h) for proj, h in zip(self.down_projections, h_list)]
        z_stack = torch.stack(z_list, dim=0)        # [M, B, S, D]
        # 2. 跨底座平均 → 共享 K/V 上下文
        z_mean = z_stack.mean(dim=0)                # [B, S, D]
        shared_k = self.W_k(z_mean)                 # [B, S, D]
        shared_v = self.W_v(z_mean)                 # [B, S, D]
        # 3. 各底座 attention (共享 K/V, 各自 Q)
        aligned_list = []
        for m, z_m in enumerate(z_list):
            q_m = self.q_projections[m](z_m)        # [B, S, D]
            # multi-head attention
            q = self._split_heads(q_m)
            k = self._split_heads(shared_k)
            v = self._split_heads(shared_v)
            attn = F.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
            concept = (attn @ v).transpose(1, 2).contiguous().view(q_m.shape)
            # 4. 残差
            aligned_list.append(z_m + concept)
        # 5. 跨底座均值
        aligned = torch.stack(aligned_list).mean(dim=0)
        # 输出投影
        return self.out_projection(aligned)