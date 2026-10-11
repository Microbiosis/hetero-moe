"""research._primitives.aligners — 跨架构对齐器的**提升后实现**.

原实现位于 ``research/aligner/v10_embedding/aligner.py``。2026-10-11 提升到
``_primitives``, 原因: ``corpus_scale`` 线的 v25 / v28 需要 ``CrossArchAttnAligner``,
而 ``research.aligner.v10_embedding`` 属于另一条研究线 —— 跨线 import 会让
``check_self_containment.py --strict`` 判失败。项目现有的 ``ALLOWED_INTRA_LINE``
三条**全部是同线复用**, 从未批准跨线, 因此正解是提升而不是加豁免。

v10 的 ``aligner.py`` 保留为转发壳, 因此 ``research.aligner.v10_embedding`` 的
所有既有 import(含 ``examples/`` 下 20 余处)均不受影响。
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