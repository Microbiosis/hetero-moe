"""v29.0 — 大型 Wikipedia 主题结构图语料 (50 train + 16 held-out, 3× v26).

设计:
    v26 MiniImageCorpus 只有 16 train + 6 held-out (4 shapes: circle/rect/triangle/wave).
    v29 LargeImageCorpus 扩展到 50 train + 16 held-out (3×), 新增 2 种形状:
        - hexagon (六边形)
        - star (五角星)

    接口对齐 v26 MiniImageCorpus:
        - train_themes: List[Tuple[str, str, str, str, str]]
        - held_out: List[Tuple[str, str, str, str, str]]
        - train_batch(batch_size): numpy [B, 224, 224, 3] uint8
        - eval_batch(batch_size): held-out numpy
        - all_eval(): all held-out
        - train_images(n): List[np.ndarray]
        - generate_wiki_image(theme): single image

    内容覆盖 50 个 Wikipedia 主题, 6 种形状, 14 种颜色, 5 种位置, 5 种大小.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np


# 14 种颜色 (RGB 0-255), 含 v26 的 12 种 + dark_red, gold
WIKI_COLORS_LARGE: dict = {
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
    "dark_red":    (150, 20, 20),
    "gold":        (220, 180, 50),
    # 背景色
    "light_yellow": (255, 250, 200),
    "light_cyan":   (220, 250, 250),
    "light_green":  (220, 250, 220),
    "light_gray":   (235, 235, 235),
    "light_blue":   (220, 240, 255),
    "light_orange": (255, 235, 200),
    "dark_purple":  (60, 20, 100),
}


# 50 个 Wikipedia 主题 → (主题, 颜色, 形状, 位置, 大小)
_LARGE_WIKI_THEMES: List[Tuple[str, str, str, str, str]] = [
    # === Nature 主题 (10 个) ===
    ("sun",        "red",      "circle",   "center", "large"),
    ("moon",       "gray",     "circle",   "top",    "small"),
    ("ocean",      "blue",     "wave",     "bottom", "wide"),
    ("forest",     "green",    "rect",     "left",   "medium"),
    ("mountain",   "gray",     "triangle", "center", "large"),
    ("desert",     "yellow",   "rect",     "spread", "wide"),
    ("sky",        "cyan",     "circle",   "top",    "small"),
    ("volcano",    "red",      "triangle", "center", "large"),
    ("river",      "blue",     "rect",     "bottom", "long"),
    ("meadow",     "green",    "rect",     "spread", "wide"),
    # === Geography 主题 (10 个) ===
    ("coral",      "orange",   "circle",   "left",   "medium"),
    ("glacier",    "cyan",     "triangle", "right",  "large"),
    ("canyon",     "brown",    "rect",     "center", "large"),
    ("prairie",    "yellow",   "rect",     "spread", "wide"),
    ("swamp",      "dark_green","rect",     "bottom", "wide"),
    ("tundra",     "gray",     "rect",     "spread", "wide"),
    ("savanna",    "yellow",   "rect",     "center", "wide"),
    ("beach",      "yellow",   "wave",     "bottom", "wide"),
    ("valley",     "green",    "rect",     "center", "wide"),
    ("cliff",      "gray",     "rect",     "right",  "large"),
    # === Man-made 主题 (10 个) ===
    ("pyramid",    "yellow",   "triangle", "center", "large"),
    ("castle",     "gray",     "rect",     "center", "large"),
    ("tower",      "gray",     "rect",     "top",    "small"),
    ("bridge",     "brown",    "rect",     "bottom", "long"),
    ("wall",       "gray",     "rect",     "center", "wide"),
    ("mosque",     "gold",     "triangle", "center", "medium"),
    ("temple",     "yellow",   "triangle", "center", "medium"),
    ("lighthouse", "yellow",   "rect",     "right",  "small"),
    ("windmill",   "brown",    "triangle", "center", "medium"),
    ("lighthouse2","red",      "circle",   "right",  "small"),
    # === Astronomical 主题 (10 个) ===
    ("planet",     "orange",   "circle",   "center", "medium"),
    ("star",       "gold",     "star",     "center", "large"),
    ("comet",      "cyan",     "triangle", "right",  "medium"),
    ("nebula",     "purple",   "circle",   "center", "large"),
    ("moon_crescent","gray",   "circle",   "left",   "small"),
    ("sunset",     "red",      "circle",   "bottom", "medium"),
    ("sunrise",    "yellow",   "circle",   "top",    "medium"),
    ("aurora",     "green",    "wave",     "top",    "wide"),
    ("galaxy",     "purple",   "hexagon",  "center", "large"),
    ("blackhole",  "dark_blue","circle",   "center", "small"),
    # === Living 主题 (10 个) ===
    ("tree",       "green",    "triangle", "center", "large"),
    ("flower",     "red",      "circle",   "center", "small"),
    ("bird",       "blue",     "triangle", "top",    "small"),
    ("fish",       "cyan",     "wave",     "left",   "medium"),
    ("leaf",       "green",    "triangle", "left",   "small"),
    ("butterfly",  "purple",   "star",     "center", "small"),
    ("snake",      "green",    "wave",     "right",  "medium"),
    ("mushroom",   "brown",    "circle",   "left",   "small"),
    ("shell",      "orange",   "circle",   "bottom", "small"),
    ("feather",    "yellow",   "triangle", "right",  "small"),
]


# 16 个 held-out 主题 (与 train 不重叠)
_LARGE_HELD_OUT_THEMES: List[Tuple[str, str, str, str, str]] = [
    ("rainbow",      "purple",      "wave",     "center", "wide"),
    ("iceberg",      "cyan",        "triangle", "right",  "large"),
    ("meteor",       "orange",      "circle",   "center", "small"),
    ("rainforest",   "dark_green",  "rect",     "spread", "wide"),
    ("crystal",      "cyan",        "hexagon",  "left",   "medium"),
    ("crown",        "gold",        "star",     "center", "medium"),
    ("lantern",      "red",         "circle",   "top",    "small"),
    ("compass",      "gray",        "circle",   "center", "medium"),
    ("diamond",      "cyan",        "hexagon",  "center", "small"),
    ("phoenix",      "red",         "star",     "center", "large"),
    ("snowflake",    "cyan",        "star",     "center", "medium"),
    ("lightning",    "yellow",      "triangle", "right",  "medium"),
    ("kelp",         "green",       "wave",     "left",   "wide"),
    ("amethyst",     "purple",      "hexagon",  "right",  "medium"),
    ("ruby",         "dark_red",    "hexagon",  "center", "small"),
    ("emerald",      "green",       "hexagon",  "bottom", "medium"),
]


# 主题 → 背景色 (与主色形成对比)
_THEME_BACKGROUND_LARGE: dict = {
    "sun":         "light_yellow",
    "moon":        "light_gray",
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
    "beach":       "light_blue",
    "valley":      "light_green",
    "cliff":       "light_gray",
    "pyramid":     "light_yellow",
    "castle":      "light_gray",
    "tower":       "light_blue",
    "bridge":      "light_blue",
    "wall":        "light_gray",
    "mosque":      "light_orange",
    "temple":      "light_orange",
    "lighthouse":  "light_blue",
    "windmill":    "light_yellow",
    "lighthouse2": "light_blue",
    "planet":      "dark_purple",
    "star":        "dark_blue",
    "comet":       "dark_blue",
    "nebula":      "dark_purple",
    "moon_crescent":"light_gray",
    "sunset":      "light_orange",
    "sunrise":     "light_yellow",
    "aurora":      "dark_blue",
    "galaxy":      "dark_purple",
    "blackhole":   "dark_blue",
    "tree":        "light_green",
    "flower":      "light_yellow",
    "bird":        "light_blue",
    "fish":        "light_blue",
    "leaf":        "light_green",
    "butterfly":   "light_yellow",
    "snake":       "light_green",
    "mushroom":    "light_orange",
    "shell":       "light_blue",
    "feather":     "light_yellow",
    "rainbow":     "light_blue",
    "iceberg":     "light_blue",
    "meteor":      "dark_blue",
    "rainforest":  "dark_green",
    "crystal":     "light_cyan",
    "crown":       "light_orange",
    "lantern":     "light_yellow",
    "compass":     "light_gray",
    "diamond":     "light_cyan",
    "phoenix":     "light_orange",
    "snowflake":   "light_cyan",
    "lightning":   "light_gray",
    "kelp":        "light_blue",
    "amethyst":    "light_orange",
    "ruby":        "light_orange",
    "emerald":     "light_green",
}


def generate_wiki_image(theme: Tuple[str, str, str, str, str], image_size: int = 224) -> np.ndarray:
    """单主题 → numpy [H, W, 3] uint8 图像.

    支持 6 种形状: circle / rect / triangle / wave / hexagon / star.
    """
    name, color, shape, position, size = theme
    H = W = image_size

    # 背景色
    bg_name = _THEME_BACKGROUND_LARGE.get(name, "light_gray")
    bg = WIKI_COLORS_LARGE[bg_name]
    img = np.full((H, W, 3), bg, dtype=np.uint8)

    # 主色
    fg = np.array(WIKI_COLORS_LARGE[color], dtype=np.uint8)

    # 形状大小
    if size == "large":
        s = 80; s_h, s_w = 80, 80
    elif size == "medium":
        s = 60; s_h, s_w = 60, 60
    elif size == "wide":
        s = 60; s_h, s_w = 40, 130
    elif size == "long":
        s = 60; s_h, s_w = 30, 160
    elif size == "small":
        s = 35; s_h, s_w = 35, 35
    else:
        s = 60; s_h, s_w = 60, 60

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
        h = s
        w = int(s * 0.866)
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
        amplitude = s // 2
        for x in range(W):
            y = cy + int(amplitude * np.sin(2 * np.pi * x / 60))
            if 0 <= y < H:
                img[max(0, y - 2):min(H, y + 3), x] = fg
    elif shape == "hexagon":
        # 六边形 (顶点上下, 共 6 个顶点)
        r = s // 2
        vertices = []
        for i in range(6):
            angle = np.pi / 6 + i * np.pi / 3  # 起始角度, 每 60° 一个顶点
            vy = cy + int(r * np.sin(angle))
            vx = cx + int(r * np.cos(angle))
            vertices.append((vy, vx))
        # 用扫描线填充
        for y in range(max(0, cy - r), min(H, cy + r + 1)):
            xs = []
            for i in range(6):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 6]
                if (v1[0] <= y < v2[0]) or (v2[0] <= y < v1[0]):
                    if v1[0] != v2[0]:
                        t = (y - v1[0]) / (v2[0] - v1[0])
                        xs.append(int(v1[1] + t * (v2[1] - v1[1])))
            if len(xs) >= 2:
                xs.sort()
                x0, x1 = max(0, xs[0]), min(W, xs[-1])
                if x1 > x0:
                    img[y, x0:x1] = fg
    elif shape == "star":
        # 五角星 (5 个外顶点 + 5 个内顶点)
        r_outer = s // 2
        r_inner = r_outer * 0.4
        vertices = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5  # 起始 -90°, 每 36° 一个顶点
            r = r_outer if i % 2 == 0 else r_inner
            vy = cy + int(r * np.sin(angle))
            vx = cx + int(r * np.cos(angle))
            vertices.append((vy, vx))
        # 用扫描线填充
        for y in range(max(0, cy - r_outer), min(H, cy + r_outer + 1)):
            xs = []
            for i in range(10):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 10]
                if (v1[0] <= y < v2[0]) or (v2[0] <= y < v1[0]):
                    if v1[0] != v2[0]:
                        t = (y - v1[0]) / (v2[0] - v1[0])
                        xs.append(int(v1[1] + t * (v2[1] - v1[1])))
            if len(xs) >= 2:
                xs.sort()
                x0, x1 = max(0, xs[0]), min(W, xs[-1])
                if x1 > x0:
                    img[y, x0:x1] = fg
    else:
        # 默认圆形
        r = s // 2
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[mask] = fg

    return img


class LargeImageCorpus:
    """大型 Wikipedia 主题结构图语料 (50 train + 16 held-out)."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.train_themes = _LARGE_WIKI_THEMES
        self.held_out = _LARGE_HELD_OUT_THEMES

    def train_batch(self, batch_size: int = 4) -> np.ndarray:
        idx = np.random.randint(0, len(self.train_themes), size=batch_size)
        themes = [self.train_themes[i] for i in idx]
        return np.stack([generate_wiki_image(t, self.image_size) for t in themes], axis=0)

    def eval_batch(self, batch_size: int = 6) -> np.ndarray:
        themes = self.held_out[:batch_size]
        return np.stack([generate_wiki_image(t, self.image_size) for t in themes], axis=0)

    def all_eval(self) -> np.ndarray:
        return self.eval_batch(len(self.held_out))

    def train_images(self, n: int = None) -> List[np.ndarray]:
        themes = self.train_themes if n is None else self.train_themes[:n]
        return [generate_wiki_image(t, self.image_size) for t in themes]

    def eval_images(self, n: int = None) -> List[np.ndarray]:
        themes = self.held_out if n is None else self.held_out[:n]
        return [generate_wiki_image(t, self.image_size) for t in themes]


def cycle_batches(corpus: LargeImageCorpus, batch_size: int = 4) -> Iterator[np.ndarray]:
    """无限循环训练 batch."""
    while True:
        yield corpus.train_batch(batch_size)
