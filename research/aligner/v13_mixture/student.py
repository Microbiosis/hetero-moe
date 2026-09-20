"""v13.0 — StudentModel (学生底座 + LoRA)。

设计:
    单底座 (TinyBERT 简化版, 只用 hidden_states 接口) + LoRA 增量.
    学生可独立推理, 不需要其他底座.

    与教师的关系:
        教师: v10/v13 多底座协同 (跨架构, 高参数量)
        学生: 单底座 + LoRA (低参数量)
        蒸馏: 学生 logits / hidden_states 对齐教师 logits / hidden_states
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """带 LoRA 的 Linear 层: W → W + ΔW, ΔW = B · A (rank=r)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        # 冻结的"预训练"权重 (用正交初始化模拟)
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.W)
        self.W.requires_grad = False
        # 可训练的 LoRA 增量
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scaling = alpha / rank

    def forward(self, x):
        # x: [..., in_features]
        # 输出 = (W + B @ A · scaling) · x
        delta_w = (self.B @ self.A) * self.scaling        # [out, in]
        return F.linear(x, self.W + delta_w)


class StudentModel(nn.Module):
    """学生单底座: D_m → D_shared, 用 LoRA 微调.

    只用 Linear (简化版底座), 模拟 TinyBERT 单底座能力.
    推理时只需这一个模块, 无需其他底座.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        lora_rank: int = 8,
        hidden: int = 128,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # 模拟底座: 2 层 FFN
        self.down = LoRALinear(d_in, hidden, rank=lora_rank)
        self.up = LoRALinear(hidden, hidden, rank=lora_rank)
        self.head = LoRALinear(hidden, d_out, rank=lora_rank)
        # 输出归一化
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        """x: [B, S, d_in] → y: [B, S, d_out]"""
        h = F.relu(self.down(x))
        h = F.relu(self.up(h))
        y = self.head(h)
        return self.norm(y)

    def trainable_param_count(self) -> int:
        """只算 LoRA 可训练参数 (底座冻结)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())