"""v26.0 — Wikipedia 主题结构图语料 (替代 v25 合成 np.random 噪声).

设计:
    内置 16 个 Wikipedia 主题 → 颜色 + 形状 + 位置 + 大小 标签.
    训练和 held-out 使用不同主题, 互不重叠.
    用 numpy 直接生成 224x224x3 uint8 图像, 不依赖 PIL/network.
    接口对齐 v8 encode_image (HxWx3 uint8).

可追溯性:
    MiniImageCorpus       -> v18 MiniCorpus 接口克隆 (numpy → [B, H, W, 3] uint8)
    _WIKI_THEMES          -> 16 个 Wikipedia 主题 (主题 → 颜色+形状+位置+大小)
    _HELD_OUT_THEMES      -> 6 个 held-out 主题 (与 train 不重叠)
    generate_wiki_image   -> 单主题 → numpy 224x224x3 uint8 图像

Wikipedia 主题 → 图像语义:
    主题          颜色       形状        位置       大小      背景色
    sun          red       circle     center    large     light_yellow
    ocean        blue      rect       bottom    wide      light_cyan
    forest       green     rect       left      medium    light_green
    mountain     gray      triangle   center    large     light_gray
    desert       yellow    rect       spread    wide      light_yellow
    sky          cyan      circle     top       small     light_blue
    volcano      red       triangle   center    large     light_gray
    river        blue      rect       bottom    long      light_blue
    meadow       green     rect       spread    wide      light_green
    coral        orange    circle     left      medium    light_blue
    glacier      cyan      triangle   right     large     light_cyan
    canyon       brown     rect       center    large     light_orange
    prairie      yellow    rect       spread    wide      light_yellow
    swamp        green     rect       bottom    wide      dark_green
    tundra       gray      rect       spread    wide      light_gray
    savanna      yellow    rect       center    wide      light_yellow

Held-out 主题 (与 train 不同):
    galaxy       purple    circle     center    large     dark_purple
    aurora       green     wave       top       wide      dark_blue
    crystal      cyan      triangle   left      medium    light_cyan
    rainforest   dark_green rect      spread    wide      dark_green
    iceberg      cyan      triangle   right     large     light_blue
    meteor       orange    circle     center    small     dark_blue
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np


# 主题 → (颜色 RGB, 形状, 位置, 大小, 背景色 RGB)
WIKI_COLORS: dict = {
    "red":         (220, 50, 50),
    "blue":        (50, 80, 220),
    "green":       (50, 180, 80),
    "yellow":      (240, 220, 50),
    "cyan":        (50, 200, 220),
    "gray":        (130, 130, 130),
    "orange":      (240, 140, 50),
    "brown":       (150, 90, 50),
    "purple":      (160, 50, 200),
    "dark_green":  (20, 100, 40),
    "dark_blue":   (20, 30, 80),
    "dark_purple": (60, 20, 100),
    "light_yellow":(255, 250, 200),
    "light_cyan":  (220, 250, 250),
    "light_green": (220, 250, 220),
    "light_gray":  (235, 235, 235),
    "light_blue":  (220, 240, 255),
    "light_orange":(255, 235, 200),
}

# 16 个训练主题
_WIKI_THEMES: List[Tuple[str, str, str, str, str]] = [
    # (主题, 颜色, 形状, 位置, 大小)
    ("sun",        "red",      "circle",   "center", "large"),
    ("ocean",      "blue",     "rect",     "bottom", "wide"),
    ("forest",     "green",    "rect",     "left",   "medium"),
    ("mountain",   "gray",     "triangle", "center", "large"),
    ("desert",     "yellow",   "rect",     "spread", "wide"),
    ("sky",        "cyan",     "circle",   "top",    "small"),
    ("volcano",    "red",      "triangle", "center", "large"),
    ("river",      "blue",     "rect",     "bottom", "long"),
    ("meadow",     "green",    "rect",     "spread", "wide"),
    ("coral",      "orange",   "circle",   "left",   "medium"),
    ("glacier",    "cyan",     "triangle", "right",  "large"),
    ("canyon",     "brown",    "rect",     "center", "large"),
    ("prairie",    "yellow",   "rect",     "spread", "wide"),
    ("swamp",      "dark_green","rect",    "bottom", "wide"),
    ("tundra",     "gray",     "rect",     "spread", "wide"),
    ("savanna",    "yellow",   "rect",     "center", "wide"),
]

# 6 个 held-out 主题 (与 train 不同)
_HELD_OUT_THEMES: List[Tuple[str, str, str, str, str]] = [
    ("galaxy",     "purple",      "circle",   "center", "large"),
    ("aurora",     "green",       "wave",     "top",    "wide"),
    ("crystal",    "cyan",        "triangle", "left",   "medium"),
    ("rainforest", "dark_green",  "rect",     "spread", "wide"),
    ("iceberg",    "cyan",        "triangle", "right",  "large"),
    ("meteor",     "orange",      "circle",   "center", "small"),
]

# 主题 → 背景色 (与主色形成对比)
_THEME_BACKGROUND: dict = {
    "sun":         "light_yellow",
    "ocean":       "light_cyan",
    "forest":      "light_green",
    "mountain":    "light_gray",
    "desert":      "light_yellow",
    "sky":         "light_blue",
    "volcano":     "light_gray",
    "river":       "light_blue",
    "meadow":      "light_green",
    "coral":       "light_blue",
    "glacier":     "light_cyan",
    "canyon":      "light_orange",
    "prairie":     "light_yellow",
    "swamp":       "dark_green",
    "tundra":      "light_gray",
    "savanna":     "light_yellow",
    "galaxy":      "dark_purple",
    "aurora":      "dark_blue",
    "crystal":     "light_cyan",
    "rainforest":  "dark_green",
    "iceberg":     "light_blue",
    "meteor":      "dark_blue",
}


def generate_wiki_image(theme: Tuple[str, str, str, str, str], image_size: int = 224) -> np.ndarray:
    """单主题 → numpy [H, W, 3] uint8 图像.

    Args:
        theme: (主题名, 颜色, 形状, 位置, 大小)
        image_size: 224 (与 ViT processor 一致)

    Returns:
        np.ndarray[uint8, H, W, 3]
    """
    name, color, shape, position, size = theme
    H = W = image_size

    # 背景色
    bg_name = _THEME_BACKGROUND.get(name, "light_gray")
    bg = WIKI_COLORS[bg_name]
    img = np.full((H, W, 3), bg, dtype=np.uint8)

    # 主色
    fg = np.array(WIKI_COLORS[color], dtype=np.uint8)

    # 形状大小
    if size == "large":
        s = 80
        s_h, s_w = 80, 80
    elif size == "medium":
        s = 60
        s_h, s_w = 60, 60
    elif size == "wide":
        s = 60
        s_h, s_w = 40, 130
    elif size == "long":
        s = 60
        s_h, s_w = 30, 160
    elif size == "small":
        s = 35
        s_h, s_w = 35, 35
    else:
        s = 60
        s_h, s_w = 60, 60

    # 位置
    if position == "center":
        cy, cx = H // 2, W // 2
    elif position == "top":
        cy, cx = H // 4, W // 2
    elif position == "bottom":
        cy, cx = 3 * H // 4, W // 2
    elif position == "left":
        cy, cx = H // 2, W // 4
    elif position == "right":
        cy, cx = H // 2, 3 * W // 4
    elif position == "spread":
        cy, cx = H // 2, W // 2
    else:
        cy, cx = H // 2, W // 2

    # 形状
    if shape == "circle":
        r = s // 2
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[mask] = fg
    elif shape == "rect":
        h, w = s_h, s_w
        y0, y1 = max(0, cy - h // 2), min(H, cy + h // 2)
        x0, x1 = max(0, cx - w // 2), min(W, cx + w // 2)
        img[y0:y1, x0:x1] = fg
    elif shape == "triangle":
        # 等腰三角形, 顶点向上, 底边向下
        h = s
        w = int(s * 0.866)  # 等边
        y0 = max(0, cy - h // 2)
        y1 = min(H, cy + h // 2)
        for y in range(y0, y1):
            progress = (y - y0) / max(1, (y1 - y0))
            half_w = int(w * progress / 2)
            x0 = max(0, cx - half_w)
            x1 = min(W, cx + half_w)
            if x1 > x0:
                img[y, x0:x1] = fg
    elif shape == "wave":
        # 横向波动
        amplitude = s // 2
        for x in range(W):
            y = cy + int(amplitude * np.sin(2 * np.pi * x / 60))
            if 0 <= y < H:
                img[max(0, y - 2):min(H, y + 3), x] = fg
    else:
        # 默认圆形
        r = s // 2
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[mask] = fg

    return img


class MiniImageCorpus:
    """Wikipedia 主题结构图语料 (16 训练 + 6 held-out).

    接口对齐 v18 MiniCorpus, 但输出 numpy [B, H, W, 3] uint8 (非 tokenized),
    供 encode_image (HxWx3 uint8) 调用.

    与 v18 MiniCorpus 区别: 不调 tokenizer, 而是直接 numpy 生成图像.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.train_themes = _WIKI_THEMES
        self.held_out = _HELD_OUT_THEMES

    def train_batch(self, batch_size: int = 4) -> np.ndarray:
        """随机采样 batch_size 个 Wikipedia 主题 → numpy [B, H, W, 3] uint8."""
        idx = np.random.randint(0, len(self.train_themes), size=batch_size)
        themes = [self.train_themes[i] for i in idx]
        return np.stack([generate_wiki_image(t, self.image_size) for t in themes], axis=0)

    def eval_batch(self, batch_size: int = 6) -> np.ndarray:
        """返回 held-out 主题 → numpy [B, H, W, 3] uint8."""
        themes = self.held_out[:batch_size]
        return np.stack([generate_wiki_image(t, self.image_size) for t in themes], axis=0)

    def all_eval(self) -> np.ndarray:
        """返回所有 held-out 主题 → numpy [B, H, W, 3] uint8."""
        return self.eval_batch(len(self.held_out))

    def train_images(self, n: int = None) -> List[np.ndarray]:
        """返回训练图像列表 (供 encode_image 逐张调用, 非 batched)."""
        themes = self.train_themes if n is None else self.train_themes[:n]
        return [generate_wiki_image(t, self.image_size) for t in themes]

    def eval_images(self, n: int = None) -> List[np.ndarray]:
        """返回 held-out 图像列表."""
        themes = self.held_out if n is None else self.held_out[:n]
        return [generate_wiki_image(t, self.image_size) for t in themes]


def cycle_batches(corpus: MiniImageCorpus, batch_size: int = 4) -> Iterator[np.ndarray]:
    """无限循环训练 batch."""
    while True:
        yield corpus.train_batch(batch_size)
