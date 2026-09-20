"""v13.0 — 蒸馏损失 (学生对齐教师)。

设计:
    教师 (v10/v13 多底座协同) 输出 y_teacher ∈ [B, S, D_shared]
    学生 (单底座 + LoRA) 输出 y_student ∈ [B, S, D_shared]
    蒸馏损失: L_distill = α · MSE(y_student, y_teacher.detach())
            + β · MSE(student_logit, teacher_logit)

    学生只看一个底座的输入 (BERT 模态), 但要预测教师融合 3 模态后的输出.
    这是真正的"嵌合"——学生学会"何时调用其他模态的思考方式".
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def distill_loss(
    y_student: torch.Tensor,
    y_teacher: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    temperature: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """组合蒸馏损失。

    Args:
        y_student / y_teacher: hidden states, [B, S, D]
        student_logits / teacher_logits: logits, [B, S, V]
        alpha: hidden state loss 权重
        beta: logits distillation loss 权重
        temperature: softmax 温度

    Returns:
        dict: total, hidden, logit
    """
    # 1. Hidden state distillation (MSE)
    hidden_loss = F.mse_loss(y_student, y_teacher.detach())
    # 2. Logit distillation (KL 散度)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    logit_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    total = alpha * hidden_loss + beta * logit_loss
    return {"total": total, "hidden": hidden_loss, "logit": logit_loss}