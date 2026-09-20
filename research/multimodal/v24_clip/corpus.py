"""v24.0 — Mini image-text corpus (合成数据, 不依赖 torchvision/PIL).

设计:
    - 11 个基础颜色 → (R, G, B) ∈ [0, 1]
    - 每个样本: 纯色 [3, H, W] 图像 + 含颜色名的英文短语
    - train/eval 划分 3:1, 颜色分布均匀

相对 v23b:
    v23b 用 PIL 生成随机噪点图 + 24 样本, 模型从零学 image-text 对齐 → 失败
    v24 用纯色图 + 12 样本, CLIP 已有 image-text 空间, 学生只需学"匹配 CLIP"
"""
from __future__ import annotations

from typing import Dict, List

import torch


# 11 个颜色 → (R, G, B) 通道 ∈ [0, 1]
COLOR_RGB: Dict[str, tuple] = {
    "red":     (1.0, 0.0, 0.0),
    "green":   (0.0, 0.6, 0.0),
    "blue":    (0.0, 0.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0),
    "black":   (0.0, 0.0, 0.0),
    "white":   (1.0, 1.0, 1.0),
    "purple":  (0.5, 0.0, 0.5),
    "orange":  (1.0, 0.5, 0.0),
    "brown":   (0.6, 0.3, 0.0),
    "pink":    (1.0, 0.75, 0.8),
    "gray":    (0.5, 0.5, 0.5),
}


# 每个颜色对应的若干英文短语 (CLIP 训练分布接近自然语言)
_COLOR_TEMPLATES: Dict[str, List[str]] = {
    "red":    ["a red square", "the color red", "a photo of something red",
               "pure red color", "bright red", "deep red"],
    "green":  ["a green square", "the color green", "a photo of something green",
               "pure green color", "bright green", "dark green"],
    "blue":   ["a blue square", "the color blue", "a photo of something blue",
               "pure blue color", "bright blue", "navy blue"],
    "yellow": ["a yellow square", "the color yellow", "a photo of something yellow",
               "pure yellow color", "bright yellow", "lemon yellow"],
    "black":  ["a black square", "the color black", "a photo of something black",
               "pure black color", "dark black", "jet black"],
    "white":  ["a white square", "the color white", "a photo of something white",
               "pure white color", "bright white", "snow white"],
    "purple": ["a purple square", "the color purple", "a photo of something purple",
               "pure purple color", "violet purple", "dark purple"],
    "orange": ["an orange square", "the color orange", "a photo of something orange",
               "pure orange color", "bright orange", "tangerine"],
    "brown":  ["a brown square", "the color brown", "a photo of something brown",
               "pure brown color", "earth brown", "chocolate brown"],
    "pink":   ["a pink square", "the color pink", "a photo of something pink",
               "pure pink color", "light pink", "rose pink"],
    "gray":   ["a gray square", "the color gray", "a photo of something gray",
               "pure gray color", "medium gray", "slate gray"],
}


def generate_color_image(color: str, image_size: int = 224) -> torch.Tensor:
    """生成 [3, H, W] 纯色张量.

    Args:
        color: 颜色名 (必须在 COLOR_RGB 中)
        image_size: H = W

    Returns:
        [3, image_size, image_size] float32 tensor, 值 ∈ [0, 1]
    """
    if color not in COLOR_RGB:
        raise ValueError(f"unknown color {color!r}; expected one of {list(COLOR_RGB)}")
    r, g, b = COLOR_RGB[color]
    img = torch.zeros(3, image_size, image_size, dtype=torch.float32)
    img[0] = r  # R 通道
    img[1] = g  # G 通道
    img[2] = b  # B 通道
    return img


class MiniImageTextCorpus:
    """小型 image-text 语料, 纯色图 + 颜色短语.

    Args:
        train_size: 训练样本数 (默认 12, 颜色循环)
        eval_size:  评测样本数 (默认 4)
        image_size: 图像 H=W (默认 224)
        seed:       随机种子 (文本模板索引)
    """

    def __init__(
        self,
        train_size: int = 12,
        eval_size: int = 4,
        image_size: int = 224,
        seed: int = 0,
    ):
        self.train_size = train_size
        self.eval_size = eval_size
        self.image_size = image_size
        self.seed = seed
        self.colors = list(COLOR_RGB.keys())

        self.train_samples = self._build_samples(train_size, "train")
        self.eval_samples = self._build_samples(eval_size, "eval")

    def _build_samples(self, n: int, split: str) -> List[Dict]:
        """循环颜色, 模板按 (split, seed, idx) 取."""
        out: List[Dict] = []
        # 不同 split 用不同的 seed 偏移避免完全重复
        offset = 0 if split == "train" else 7
        for i in range(n):
            color = self.colors[(i + offset) % len(self.colors)]
            templates = _COLOR_TEMPLATES[color]
            # 用 i 取模板 (train 与 eval 选不同模板)
            text = templates[(i + offset) % len(templates)]
            out.append({
                "image": generate_color_image(color, self.image_size),
                "text":  text,
                "color": color,
            })
        return out

    def all_train(self) -> List[Dict]:
        return self.train_samples

    def all_eval(self) -> List[Dict]:
        return self.eval_samples

    def __repr__(self) -> str:
        return (f"MiniImageTextCorpus(train={len(self.train_samples)}, "
                f"eval={len(self.eval_samples)}, image={self.image_size}x{self.image_size})")
