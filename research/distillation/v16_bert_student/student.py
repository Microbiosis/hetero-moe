"""v16.0 — 真实 BERT 学生 + LoRA.

设计:
    BERTStudent = AutoModel (冻结) + LoRA(hidden_states) + 输出投影
    推理时只需 BERTStudent, 无需其他底座.

与 v13 的 StudentModel (MLP) 对比:
    v13: 简化 MLP (随机初始化), 不能复用任何预训练知识
    v16: 真实 BERT (预训练), 加 LoRA 保持泛化能力
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class LoRAAdapter(nn.Module):
    """LoRA 加在 hidden_states 上: h -> h + (h · A^T · B^T · scaling)."""

    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(rank, hidden_size))
        self.B = nn.Parameter(torch.zeros(hidden_size, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scaling = alpha / rank

    def forward(self, h):
        # h: [B, S, D] -> [B, S, D]
        delta = (h @ self.A.t()) @ self.B.t() * self.scaling
        return h + delta


class BERTStudent(nn.Module):
    """真实 BERT 学生: AutoModel + LoRA + 输出投影到 D_shared.

    推理时只需这一个模型, 无需其他底座.

    Args:
        model_name: HF 模型名 (默认 TinyBERT)
        d_out: 输出维度 (D_shared)
        lora_rank: LoRA rank
        cache_dir: HF 模型缓存目录
    """

    def __init__(
        self,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        d_out: int = 256,
        lora_rank: int = 8,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.d_out = d_out
        # 加载预训练 BERT (冻结)
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        for p in self.bert.parameters():
            p.requires_grad = False
        # 获取 hidden_size
        hidden_size = self.bert.config.hidden_size
        self.hidden_size = hidden_size
        # LoRA on hidden_states (可训练)
        self.lora = LoRAAdapter(hidden_size, rank=lora_rank)
        # 输出投影 (hidden_size -> d_out)
        self.head = nn.Linear(hidden_size, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, input_ids, attention_mask=None):
        """input_ids: [B, S] -> y: [B, S, d_out]"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state                    # [B, S, D_bert]
        h = self.lora(h)                                 # LoRA 增量
        y = self.head(h)                                 # [B, S, d_out]
        return self.norm(y)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bert_param_count(self) -> int:
        return sum(p.numel() for p in self.bert.parameters())