"""V49.0 — 组合式数据生成器 (让"样本量"成为真正的自变量)。

为什么不用 v48 的手工文本: v48 只有 3 条文本 (3 类), 样本量只能靠"加扰动"
往上堆 —— 同一句话的噪声版本, 几乎不提供新信息, 所以 B 臂必然学成记忆器
(过拟合间隙 95%-99.9%)。要分辨"记忆"与"真学到对齐", 必须有**内容不同**的
样本, 而且要能**线性放大**。

本模块用**槽位组合**生成文本:
    每类 = 2 个槽位族, 各 16 个取值 → 每类 256 种不同组合
    训练池 = 每类前 192 种, 留出池 = 每类后 64 种 (**组合完全不相交**)
于是"留出"测的是"对没见过的组合的泛化", 而不是"对同一句话的噪声的泛化"。

样本量自变量: 每类取 n_per_class ∈ {4, 32, 128} → 每底座 {32, 256, 1024} 条。
"""
from __future__ import annotations

__all__ = ["CLASSES", "make_texts", "Split", "build_split", "SAMPLE_SIZES"]

# 8 个语义类, 每类两个槽位族 (16×16 = 256 种组合)
CLASSES = {
    "animal": (["small", "big", "wild", "cute", "old", "young", "red", "blue",
                "fuzzy", "tiny", "huge", "calm", "loud", "soft", "sharp", "rare"],
               ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "goat",
                "mouse", "bear", "fox", "wolf", "owl", "deer", "pig", "duck"]),
    "vehicle": (["big", "small", "old", "new", "fast", "slow", "red", "blue",
                 "heavy", "light", "long", "short", "black", "white", "green", "yellow"],
                ["truck", "boat", "plane", "bus", "train", "bike", "taxi", "van",
                 "tram", "ferry", "scooter", "tractor", "jeep", "minibus", "ship", "rocket"]),
    "fruit": (["ripe", "green", "sweet", "sour", "small", "big", "red", "yellow",
               "dry", "fresh", "wild", "tall", "round", "flat", "cold", "warm"],
              ["apple", "pear", "peach", "plum", "mango", "lemon", "lime", "melon",
               "cherry", "grape", "berry", "fig", "kiwi", "papaya", "apricot", "date"]),
    "color": (["deep", "light", "dark", "pale", "bright", "dull", "warm", "cool",
               "soft", "sharp", "pure", "mixed", "clear", "faded", "rich", "olive2"],
              ["red", "blue", "green", "black", "white", "brown", "gray", "pink",
               "gold", "silver", "copper", "beige", "violet", "orange", "cream", "olive"]),
    "place": (["old", "new", "north", "south", "east", "west", "high", "low",
               "near", "far", "big", "small", "quiet", "busy", "cold", "warm"],
              ["city", "forest", "desert", "harbor", "village", "campus", "market", "valley",
               "tunnel", "bridge", "island", "meadow", "canyon", "temple", "station", "bakery"]),
    "tool": (["heavy", "light", "sharp", "dull", "old", "new", "long", "short",
              "big", "small", "flat", "round", "clean", "rusty", "wooden", "metal"],
             ["hammer", "wrench", "pliers", "shovel", "drill", "saw", "axe", "rake",
              "chisel", "clamp", "mallet", "crowbar", "sickle", "trowel", "file", "vise"]),
    "drink": (["hot", "cold", "sweet", "bitter", "fresh", "old", "strong", "weak",
               "clear", "dark", "large", "small", "iced", "warm", "plain", "mixed"],
              ["coffee", "tea", "juice", "water", "milk", "soda", "cider", "broth",
               "lemonade", "kefir", "smoothie", "espresso", "cocoa", "ale", "tonic", "yogurt"]),
    "cloth": (["thick", "thin", "old", "new", "long", "short", "heavy", "light",
               "warm", "cool", "clean", "dirty", "dark", "bright", "soft", "rough"],
              ["shirt", "coat", "glove", "scarf", "boot", "hat", "sock", "apron",
               "sweater", "skirt", "belt", "cape", "hood", "vest", "robe", "sandal"]),
}

# 每类的组合数上限 (16×16)
COMBOS_PER_CLASS = 256
TRAIN_COMBOS = 128          # 训练池: 每类前 128 种
HELDOUT_COMBOS = 128        # 留出池: 每类后 128 种 (与训练池**不相交**, 装得下 n_per_class=128)

# 样本量自变量: 每类每底座取多少条 → 每底座总数 = 8 × n_per_class
SAMPLE_SIZES = {"S32": 4, "S256": 32, "S1024": 128}


class Split:
    """一份"协议": 训练文本 + 留出文本 (按类分组, 组合不相交)。"""

    def __init__(self, n_per_class: int, seed: int = 0):
        self.n_per_class = n_per_class
        self.train: list = []      # [(class_idx, text)]
        self.heldout: list = []
        for ci, (adj, nouns) in enumerate(CLASSES.values()):
            tr, ho = [], []
            for i in range(TRAIN_COMBOS):
                a = adj[i % len(adj)]
                n = nouns[(i // len(adj)) % len(nouns)]
                tr.append(f"{a} {n}")
            for j in range(TRAIN_COMBOS, COMBOS_PER_CLASS):
                a = adj[j % len(adj)]
                n = nouns[(j // len(adj)) % len(nouns)]
                ho.append(f"{a} {n}")
            # 每类取 n_per_class 条: 用**固定置换**取前 n 条 → 各档**嵌套**
            # (S32 ⊂ S256 ⊂ S1024), 这样只需编码一次最大档, 小的直接切片。
            perm = _perm(TRAIN_COMBOS, seed * 1000 + ci)
            self.train += [(ci, tr[perm[i]]) for i in range(n_per_class)]
            perm2 = _perm(HELDOUT_COMBOS, seed * 1000 + ci + 7717)
            self.heldout += [(ci, ho[perm2[i]]) for i in range(n_per_class)]

    @property
    def n_classes(self) -> int:
        return len(CLASSES)

    def texts(self, split: str) -> list:
        return [t for _, t in getattr(self, split)]

    def labels(self, split: str) -> list:
        return [c for c, _ in getattr(self, split)]

    def __repr__(self):
        return (f"Split(n_per_class={self.n_per_class}, train={len(self.train)}, "
                f"heldout={len(self.heldout)}, classes={self.n_classes})")


def _perm(n: int, seed: int):
    """确定性置换 (numpy Generator, 与 torch 无关, 便于跨机复现)。"""
    import numpy as _np
    return _np.random.default_rng(seed).permutation(n).tolist()


def build_split(n_per_class: int, seed: int = 0) -> Split:
    return Split(n_per_class, seed)


def make_texts(n_per_class: int, seed: int = 0):
    """便捷入口: 返回 (train_texts, train_labels, heldout_texts, heldout_labels)。"""
    sp = build_split(n_per_class, seed)
    return sp.texts("train"), sp.labels("train"), sp.texts("heldout"), sp.labels("heldout")
