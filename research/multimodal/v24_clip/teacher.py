"""v24.0 — CLIPTeacher (CLIP 预训练 image-text 对齐教师).

设计:
    加载 CLIP (ViT-B/32), 冻结全部参数
    提供: encode_image(pixels) -> [B, D_clip]
           encode_text(input_ids, attention_mask) -> [B, D_clip]
           fusion_score(image, text) -> [B, B] (cosine similarity)
    用于 v24 蒸馏: 学生 hidden → 对齐 CLIP image+text 融合空间
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPTeacher(nn.Module):
    """CLIP 预训练教师 (frozen).

    Args:
        model_name: HF 模型名 (默认 openai/clip-vit-base-patch32, 151M)
        cache_dir: HF 模型缓存目录
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        for p in self.model.parameters():
            p.requires_grad = False
        # 隐藏维度
        self.text_dim = self.model.config.projection_dim   # 512
        self.vision_dim = self.model.config.vision_config.hidden_size   # 768

    @torch.no_grad()
    def encode_image(self, pixel_values):
        """pixel_values: [B, 3, H, W] (CLIP processor 输出) -> [B, D_clip]"""
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        return self.model.visual_projection(vision_out.pooler_output)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        """input_ids: [B, L], attention_mask: [B, L] -> [B, D_clip]"""
        text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.model.text_projection(text_out.pooler_output)

    @torch.no_grad()
    def fused_target(self, pixel_values, input_ids, attention_mask, alpha=0.5):
        """融合 image + text 特征作为蒸馏目标.

        Args:
            alpha: text 权重 (0=纯 image, 1=纯 text, 0.5=平均)

        Returns:
            target: [B, D_clip] (text 投影到 D_clip 后与 image 融合)
        """
        img_feat = self.encode_image(pixel_values)        # [B, 512]
        txt_feat = self.encode_text(input_ids, attention_mask)  # [B, 512]
        # 归一化 (CLIP 标准)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        return alpha * txt_feat + (1 - alpha) * img_feat