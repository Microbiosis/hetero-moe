"""v19.0 — C-1 根因补全诊断工具.

H1: U 训练扰动 (v8.1 已确认, v9.0 修正 A 已修复)
H2: EMA 太慢 (v8.1 弱信号, v9.0 修正 C 已修复)
H3: 训练步数太少 (v19 新增)
H4: 学生容量太小 (v19 新增)
H5: 中枢广播位置不对 (v19 新增: 加到 attn 而非 ffn)

本模块提供 3 个诊断工具, 在 examples/run_v19_*.py 中调用:
    CapacityComparison: 不同学生容量下 C-1 单独表现
    LongTrainingComparison: 100 步 vs 1000 步下 C-1 单独表现
    DiagnosticRunner: 整合所有诊断
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CapacityComparison(nn.Module):
    """H4 诊断: 不同学生容量下的 C-1 单独表现.

    Args:
        D_shared: 共享维度
        num_experts: 路由专家数
        student_capacity: 学生层数 (e.g., [64, 32] 表示 2 层 MLP)
    """

    def __init__(self, d_shared: int, num_experts: int, student_capacity: List[int]):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        # 学生 (可训练)
        layers = []
        in_dim = d_shared
        for h in student_capacity:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, d_shared))
        self.student = nn.Sequential(*layers)
        # 中枢 (c + U)
        self.c = nn.Parameter(torch.zeros(d_shared))
        self.U = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        # 路由器 (恒等映射测试)
        self.W = nn.Parameter(torch.eye(num_experts, d_shared))

    def forward(self, x):
        """x: [B, S, D] -> (y, alpha)"""
        # 学生预测残差
        s = self.student(x)
        # 中枢 + U bias 加到路由器
        z = F.linear(x, self.W) + self.U @ self.c
        alpha = F.softmax(z, dim=-1)
        y = s  # 简化: 学生输出即融合输出
        return y, alpha


class LongTrainingComparison(nn.Module):
    """H3 诊断: 不同训练步数下的 C-1 表现."""

    def __init__(self, d_shared: int, num_experts: int):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        # 简化学生
        self.student = nn.Sequential(
            nn.Linear(d_shared, d_shared),
            nn.ReLU(),
            nn.Linear(d_shared, d_shared),
        )
        # 中枢 (c + U)
        self.c = nn.Parameter(torch.zeros(d_shared))
        self.U = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        # 路由器
        self.W = nn.Parameter(torch.eye(num_experts, d_shared))


class DiagnosticRunner:
    """整合所有 v19 诊断, 提供统一运行接口."""

    def __init__(self):
        self.results = {}

    def record(self, hypothesis: str, condition: str, fuse_mse: float):
        """记录诊断结果."""
        if hypothesis not in self.results:
            self.results[hypothesis] = []
        self.results[hypothesis].append({
            "condition": condition,
            "fuse_mse": fuse_mse,
        })

    def summary(self) -> str:
        lines = ["=== v19 C-1 根因诊断汇总 ==="]
        for hyp, entries in self.results.items():
            lines.append(f"\n[{hyp}]")
            for e in entries:
                lines.append(f"  {e['condition']}: fuse={e['fuse_mse']:.4f}")
        return "\n".join(lines)