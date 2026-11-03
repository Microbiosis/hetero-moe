"""路线6：模态条件权重生成器（Modality-Conditioned Generator）。

用模态 embedding 作为条件，通过轻量 generator 生成 LoRA-style adapter，
插入 SlotSubspace 和 fusion 层，使同一套基础权重按模态自修改行为。

注入规则（依据 v62 实测的槽-模态关系）：
  文本模态（Qwen）→ 注入 Q/O 槽 adapter
  视觉模态（ViT）  → 注入 K/V 槽 adapter
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAdapter(nn.Module):
    """模态条件 LoRA adapter 生成器。"""

    def __init__(self, d_in: int, d_out: int, rank: int = 8,
                 n_modalities: int = 3):
        super().__init__()
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.n_modalities = n_modalities
        # 每个模态一个低秩生成器：modal_emb → (U, V)
        self.modal_emb = nn.Embedding(n_modalities, 16)
        self.gen = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, (d_in + d_out) * rank)
        )
        self.scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, modality_id: int) -> tuple:
        """
        modality_id: 0=Qwen(text), 1=BERT(text), 2=ViT(vision)
        returns: (A [d_out,rank], B [rank,d_in])
        """
        emb = self.modal_emb(torch.tensor([modality_id],
                                           device=self.modal_emb.weight.device))
        out = self.gen(emb).squeeze(0)
        A = out[:self.d_out * self.rank].view(self.d_out, self.rank)
        B = out[self.d_out * self.rank:].view(self.rank, self.d_in)
        return A, B

    def adapt(self, W: torch.Tensor, modality_id: int) -> torch.Tensor:
        """返回 W + scale * A@B（LoRA 形式）。"""
        A, B = self(modality_id)
        return W + self.scale * (A @ B)

    def extra_repr(self):
        return ('%d→%d rank=%d modalities=%d'
                % (self.d_in, self.d_out, self.rank, self.n_modalities))


class ModalityAdapterBank(nn.Module):
    """管理多个槽/层的 adapter 集合。"""

    def __init__(self, slots=('Q', 'K', 'V', 'O'), d: int = 256, rank: int = 8):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for s in slots:
            self.adapters[s] = ModalityAdapter(d, d, rank=rank)

    # 槽-模态注入表（v62 实测：文本→Q/O，视觉→K/V）
    INJECTION = {
        'Qwen': ['Q', 'O'],
        'BERT': ['K', 'V'],
        'ViT':  ['K', 'V'],
    }

    def get_adapter(self, slot: str, modality: str):
        return self.adapters[slot]
