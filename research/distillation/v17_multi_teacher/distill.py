"""v17.0 — MultiTeacherDistiller (多教师蒸馏损失).

设计:
    输入:
        student_outputs: dict[teacher_name -> hidden_state]  (学生预测)
        teacher_outputs: dict[teacher_name -> hidden_state]  (教师真值, detach)
        weights: dict[teacher_name -> float]  (各教师权重, 默认等权)
    输出:
        total_loss = sum_k (weights[k] · MSE(student_outputs[k], teacher_outputs[k]))

    关键: 每个教师对齐的是同一学生的同一前向输出, 不是不同的学生.
    这样所有教师都在"指导同一个学生收敛".
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTeacherDistiller(nn.Module):
    """多教师蒸馏损失聚合器.

    Args:
        teacher_names: 教师名字列表, e.g. ['v10_attn', 'v13_mixture', 'v15_mom']
        weights: 各教师权重 (默认等权)
    """

    def __init__(self, teacher_names, weights=None):
        super().__init__()
        self.teacher_names = list(teacher_names)
        if weights is None:
            weights = {n: 1.0 / len(teacher_names) for n in self.teacher_names}
        self.weights = weights

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """返回 total loss + 每个教师的独立 loss (用于诊断)."""
        losses = {}
        total = 0.0
        for name in self.teacher_names:
            s = student_outputs[name]
            t = teacher_outputs[name].detach()
            l = F.mse_loss(s, t)
            losses[f"loss_{name}"] = l
            total = total + self.weights[name] * l
        losses["total"] = total
        return losses