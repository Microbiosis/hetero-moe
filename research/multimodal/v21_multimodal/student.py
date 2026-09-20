"""v21.0 — MultiModalStudent (BERT-text + ViT-image + 模态路由器 + LoRA).

设计:
    学生 = [BERT (冻结) + LoRA] (text) + [ViT (冻结) + LoRA] (image) + 模态路由器
    模态路由器 W_modality: D_shared → [text_weight, image_weight]  (gate-style)
    学生预测: y = alpha * text_out + (1 - alpha) * image_out

    训练目标:
        - 蒸馏: 学生模态路由器权重对齐教师融合权重
        - 任务: 学生输出对齐真实 task target (与 v18 一致)

与 v16 关系:
    v16: 单 BERT 学生 + LoRA
    v21: BERT + ViT 学生 + LoRA + 模态路由器 (多模态)

与 v6/v7 关系:
    v6/v7 用 attn_pool + cross-arch aligner 协同多底座
    v21 学生端用更简单的"模态路由器 + LoRA"实现多模态融合
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, ViTModel


class LoRAHiddenAdapter(nn.Module):
    """LoRA 加在 hidden_states 上 (v16 复用).

    h -> h + (h @ A^T @ B^T * scaling)
    """

    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(rank, hidden_size))
        self.B = nn.Parameter(torch.zeros(hidden_size, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scaling = alpha / rank

    def forward(self, h):
        delta = (h @ self.A.t()) @ self.B.t() * self.scaling
        return h + delta


class MultiModalStudent(nn.Module):
    """多模态学生: BERT-text + ViT-image + 模态路由器.

    Args:
        text_model: HF 模型名 (默认 TinyBERT)
        image_model: HF 模型名 (默认 ViT-tiny)
        d_out: 输出维度 (D_shared, 默认 256)
        lora_rank: LoRA rank
        gate_alpha: 模态路由器 gate-style alpha (默认 0.1)
    """

    def __init__(
        self,
        text_model: str = "huawei-noah/TinyBERT_General_4L_312D",
        image_model: str = "WinKawaks/vit-tiny-patch16-224",
        d_out: int = 256,
        lora_rank: int = 8,
        gate_alpha: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.d_out = d_out
        self.gate_alpha = gate_alpha

        # 文本底座 (冻结)
        self.text_bert = AutoModel.from_pretrained(text_model, cache_dir=cache_dir)
        for p in self.text_bert.parameters():
            p.requires_grad = False
        text_hidden = self.text_bert.config.hidden_size
        self.text_lora = LoRAHiddenAdapter(text_hidden, rank=lora_rank)
        self.text_head = nn.Linear(text_hidden, d_out)

        # 图像底座 (冻结)
        self.image_vit = ViTModel.from_pretrained(image_model, cache_dir=cache_dir)
        for p in self.image_vit.parameters():
            p.requires_grad = False
        image_hidden = self.image_vit.config.hidden_size
        self.image_lora = LoRAHiddenAdapter(image_hidden, rank=lora_rank)
        self.image_head = nn.Linear(image_hidden, d_out)

        # 模态路由器 (gate-style)
        # 用输入 text/image 的"概要"决定权重 (这里简化: 用 router token)
        self.W_modality = nn.Parameter(torch.zeros(d_out, 2))  # [D, 2]
        # 模态选择用 sigmoid 温度
        self.b_modality = nn.Parameter(torch.zeros(2))

        self.norm_text = nn.LayerNorm(d_out)
        self.norm_image = nn.LayerNorm(d_out)

    def encode_text(self, input_ids, attention_mask=None):
        out = self.text_bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, S, D_text]
        h = self.text_lora(h)
        y = self.text_head(h)
        return self.norm_text(y)

    def encode_image(self, pixel_values):
        out = self.image_vit(pixel_values=pixel_values)
        h = out.last_hidden_state  # [B, S, D_image]
        # ViT 输出包含 CLS token, 我们用 patch tokens (去 CLS)
        h = h[:, 1:, :]  # [B, S-1, D_image]
        h = self.image_lora(h)
        y = self.image_head(h)
        return self.norm_image(y)

    def compute_modality_weights(self, x_shared):
        """x_shared: [B, S, D_shared] -> weights: [B, S, 2]"""
        # 简化: 用 mean-pool 的 x_shared 算权重
        pooled = x_shared.mean(dim=1)  # [B, D]
        logits = pooled @ self.W_modality + self.b_modality
        # gate-style: 用温度调节 (alpha 控制强度)
        temperature = 1.0 + self.gate_alpha
        weights = F.softmax(logits / temperature, dim=-1)
        # 扩展到 [B, S, 2]
        return weights.unsqueeze(1).expand(-1, x_shared.size(1), -1)

    def forward(self, text_inputs, image_pixels, x_shared=None):
        """前向.

        Args:
            text_inputs: dict with 'input_ids' and optional 'attention_mask'
            image_pixels: [B, C, H, W] tensor
            x_shared: 可选的已对齐输入 (用于教师输入)

        Returns:
            y: [B, S_text, D_shared] 学生预测
            weights: [B, S_text, 2] 模态权重 (text vs image)
        """
        # 1. 编码
        text_out = self.encode_text(
            text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
        )   # [B, S_text, D]
        image_out = self.encode_image(image_pixels)  # [B, S_image, D]
        # 对齐序列长度 (取较短)
        S = min(text_out.size(1), image_out.size(1))
        text_out = text_out[:, :S, :]
        image_out = image_out[:, :S, :]

        # 2. 模态路由器 (用 text_out 的 mean 作 query)
        weights = self.compute_modality_weights(text_out)  # [B, S, 2]

        # 3. 加权融合
        y = weights[..., 0:1] * text_out + weights[..., 1:2] * image_out
        return y, weights

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())