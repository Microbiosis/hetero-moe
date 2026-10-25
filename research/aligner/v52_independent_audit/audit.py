"""V52.0 — 独立复算 v50/v51 的探针结论 (审计包).

为什么要独立复算
----------------
v50 (规模异构线) 与 v51 (架构异构线) 的结论是:
    "『融合优于单专家』的 margin 主要是**识别机制**的产物, 不是语义融合;
     换一个有意义的空间后 margin 塌掉。"
这两条结论目前是**单点实现、单一种子、单一个配置**得出的。审计包必须能在
**不复用原实现**的前提下把它们算出来, 并且**预先登记**什么情况下它们算被推翻。

本包刻意**不 import** v49/v50/v51 的任何代码: 数据生成器、抬升、探针、训练循环
全部重写, 只保留两样必须相同的东西: (a) 被审计的底座与融合层本身 (那是主张的对象),
(b) 数据分布的"形状" (三类 + 槽位组合), 但具体词表、网格划分与种子都换了。

三个检查
--------
C1 判决性复算: 三个读数在**新种子 + 未见组合留出**上重算
    (a) 冻结随机空间: 源身份可解码性 ≫ 目标类别可解码性
    (b) 目标置换后: margin 变化 ≤ 10 个百分点
    (c) 有意义空间 (非线性可训练): margin ≤ 冻结随机臂的一半
C2 探针的**反向**对照 (v50/v51 只做了正向自检):
    目标被**逐样本随机化**的臂 → 模型不可能解出 → 类别可解码必须掉到随机水平。
    若不掉 → 探针测的不是"输出里有目标信息", 而是别的东西 → v50/v51 的全部
    探针读数作废 → 审计失败。
C3 配置稳健性: D_shared ∈ {128, 256, 512} 与不同类别数上重跑关键两臂。

预登记判决 (写在 SPEC 里, 事后不得改)
--------------------------------------
V1 审计通过: C1 的 (a)(b)(c) 三条在 ≥4/5 seeds 上全部满足。
V2 审计失败: C2 的随机目标臂里类别可解码仍 ≥ 0.7 (探针无效)。
任一条不满足 → 相应主张被推翻, 论文必须回退相应表述。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CLASSES", "make_split", "build_targets", "RidgeProbe",
           "MLPLift", "ARMS", "PREDICTIONS"]

# ----------------------------------------------------------------------
# 1) 独立的数据生成器 (与 v49 的词表/网格/种子都不同)
# ----------------------------------------------------------------------

CLASSES = {
    "tool":   (["brass", "steel", "wooden", "plastic", "sharp", "blunt", "tiny", "huge",
                "left", "right", "new", "old", "dry", "wet", "heavy", "light"],
               ["hammer", "chisel", "mallet", "spanner", "plier", "file", "rake", "hoe",
                "shovel", "awl", "chock", "crowbar", "clamp", "anvil", "punch", "drift"]),
    "bird":   (["pale", "dark", "bright", "dull", "young", "old", "small", "large",
                "silent", "noisy", "fast", "slow", "curly", "sleek", "bold", "shy"],
               ["robin", "sparrow", "falcon", "heron", "lark", "kite", "owl", "raven",
                "finch", "crane", "eagle", "swan", "tern", "wren", "magpie", "gull"]),
    "metal":  (["pure", "mixed", "rusty", "polished", "heavy", "light", "hard", "soft",
                "ancient", "modern", "flat", "curved", "coarse", "fine", "hot", "cold"],
               ["iron", "copper", "brass", "zinc", "nickel", "silver", "pewter", "bronze",
                "tin", "lead", "chrome", "brass2", "steel2", "alloy", "billet", "ingot"]),
    "drink":  (["hot", "iced", "fizzy", "flat", "strong", "weak", "fresh", "stale",
                "clear", "dark", "large", "small", "sweet", "dry", "creamy", "fruity"],
               ["coffee", "tea", "cider", "water", "milk", "soda", "kombucha", "kefir",
                "smoothie", "lemonade", "espresso", "matcha", "chai", "cocoa", "tisane", "fizzy"]),
    "road":   (["north", "south", "east", "west", "high", "low", "long", "short",
                "busy", "quiet", "rough", "smooth", "old", "new", "wide", "narrow"],
               ["avenue", "lane", "bridge", "tunnel", "alley", "bypass", "loop", "causeway",
                "boulevard", "corridor", "crossing", "roundabout", "sidewalk", "passage", "junction", "ramp"]),
    "plant":  (["tall", "short", "dry", "wet", "young", "old", "bitter", "sweet",
                "spiky", "soft", "pale", "dark", "climbing", "ground", "potted", "wild"],
               ["fern", "moss", "reed", "vine", "shrub", "orchid", "clover", "thistle",
                "willow", "maple", "birch", "aloe", "bamboo", "sedge", "poppy", "laurel"]),
    "fabric": (["coarse", "smooth", "thick", "thin", "stiff", "soft", "rough", "fine",
                "bright", "dull", "new", "worn", "tight", "loose", "light", "heavy"],
               ["linen", "denim", "wool", "silk", "canvas", "suede", "velvet", "jersey",
                "tweed", "satin", "corduroy", "cashmere", "burlap", "chiffon", "felt", "lace"]),
    "stone":  (["rough", "smooth", "heavy", "light", "old", "new", "dark", "pale",
                "wet", "dry", "sharp", "round", "tall", "flat", "cold", "warm"],
               ["granite", "marble", "slate", "basalt", "chalk", "flint", "shale", "pumice",
                "obsidian", "gypsum", "quartz", "mica", "limestone", "sandstone", "alum", "jade"]),
}
N_CLASSES = len(CLASSES)
GRID = 16                     # 每类 16×16 = 256 种组合
TRAIN_PARITY = 0              # 训练用 (i1+i2)%2==0 的组合
HELDOUT_PARITY = 1            # 留出用 (i1+i2)%2==1 的组合 → 与训练**零交集**


def _pairs_for_class(adj, nouns, parity):
    """该类里满足奇偶性的全部组合 (i1, i2) → 文本。"""
    out = []
    for i1 in range(GRID):
        for i2 in range(GRID):
            if (i1 + i2) % 2 == parity:
                out.append(f"{adj[i1]} {nouns[i2]}")
    return out


def make_split(n_per_class: int, seed: int = 0):
    """返回 (train_pairs, heldout_pairs): 各自的 [(class_idx, text)]。

    训练与留出的**组合**完全不相交 (奇偶划分), 但每个槽位取值两边都出现 →
    留出测的是"对未见组合的泛化", 不是"对新词的泛化"。
    """
    train, held = [], []
    for ci, (adj, nouns) in enumerate(CLASSES.values()):
        tr = _pairs_for_class(adj, nouns, TRAIN_PARITY)
        ho = _pairs_for_class(adj, nouns, HELDOUT_PARITY)
        rng = np.random.default_rng(seed * 7919 + ci)
        tr_idx = rng.permutation(len(tr))[:n_per_class]
        ho_idx = rng.permutation(len(ho))[:n_per_class]
        for i in tr_idx:
            train.append((ci, tr[i]))
        for i in ho_idx:
            held.append((ci, ho[i]))
    return train, held


def build_targets(labels, d_shared: int, n_classes: int, device, randomized: bool = False,
                  seed: int = 0, n_tokens: int = 1):
    """目标块, 形状 [n, n_tokens, d_shared] (逐 token 目标)。

    randomized=True 时**逐样本随机**一个块 (用于 C2 的反向对照)。
    """
    blk = d_shared // n_classes
    tgt = torch.zeros(len(labels), n_tokens, d_shared)
    g = np.random.default_rng(seed)
    for i, c in enumerate(labels):
        b = int(g.integers(n_classes)) if randomized else c
        tgt[i, :, b * blk:(b + 1) * blk] = 1.0
    return tgt.to(device)


# ----------------------------------------------------------------------
# 2) 独立的探针: 岭回归 (闭式) —— 与 v50/v51 的 SGD 逻辑回归是不同估计量
# ----------------------------------------------------------------------

class RidgeProbe:
    """闭式岭回归探针: y ∈ R^{n_class} 多元, 用 one-hot 目标。

    选它是因为: (a) 没有随机性 (无初始化/无 minibatch), 估计量与 v50/v51 的
    SGD 逻辑回归**独立**; (b) 可以解析地算出"训练集上能拟合多少"作为上界参照。
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.W = None

    @staticmethod
    def _onehot(y, n_class):
        Y = torch.zeros(y.shape[0], n_class)
        Y[torch.arange(y.shape[0]), y] = 1.0
        return Y

    def fit(self, X, y, n_class):
        X = X.double()
        Y = self._onehot(y, n_class).double()
        Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
        A = Xb.T @ Xb + self.alpha * torch.eye(Xb.shape[1], dtype=torch.float64)
        self.W = torch.linalg.solve(A, Xb.T @ Y)
        return self

    def predict(self, X):
        X = X.double()
        Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
        return (Xb @ self.W).argmax(-1)


def probe_accuracy(X, y, n_class, alpha=1e-3, split=0.8, seed=0):
    """训练/测试划分后测准确率 (行随机置换 → 划分可复现)。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)
    n_tr = max(1, int(X.shape[0] * split))
    tr, te = idx[:n_tr], idx[n_tr:]
    if te.numel() == 0:
        te = tr
    p = RidgeProbe(alpha).fit(X[tr], y[tr], n_class)
    return float((p.predict(X[te]) == y[te]).double().mean())


# ----------------------------------------------------------------------
# 3) 独立的抬升 (与 v49 的 NonlinearProj 不同实现: 两层 + LayerNorm + GELU)
# ----------------------------------------------------------------------

class MLPLift(nn.Module):
    """非线性抬升: D_m → h → h → d_shared (GELU + LayerNorm), 参数量按线性臂对齐。"""

    def __init__(self, d_shared: int, D_m: int):
        super().__init__()
        # 目标参数量 = d_shared * D_m (线性臂)
        target = d_shared * D_m
        best_h, best_gap = 8, None
        for h in range(8, 2048):
            # LayerNorm(D_m)*2 + d1(D_m*h+h) + d2(h*h+h) + out(d_shared*h+d_shared)
            p = h * h + h * (D_m + d_shared + 2) + 2 * D_m + d_shared
            gap = abs(p - target)
            if best_gap is None or gap < best_gap:
                best_h, best_gap = h, gap
        h = best_h
        self.hidden = h
        self.ln = nn.LayerNorm(D_m)
        self.d1 = nn.Linear(D_m, h)
        self.d2 = nn.Linear(h, h)
        self.out = nn.Linear(h, d_shared)

    def forward(self, x, cond=None):
        z = self.ln(x)
        z = F.gelu(self.d1(z))
        z = F.gelu(self.d2(z))
        return self.out(z)


class LinearLift(nn.Module):
    """线性抬升 (与 v49 等价的基线, 但独立实现: 同样正交初始化)。"""

    def __init__(self, d_shared: int, D_m: int):
        super().__init__()
        self.P = nn.Parameter(torch.empty(d_shared, D_m))
        nn.init.orthogonal_(self.P)

    def forward(self, x, cond=None):
        return F.linear(x, self.P)


# ----------------------------------------------------------------------
# 4) 臂与预登记判决
# ----------------------------------------------------------------------

ARMS = {
    "frozen_random": ("linear", False),    # 论文原配置: 冻结随机正交
    "permuted_target": ("linear", True),   # 目标置换 (判定: margin 不变)
    "learned_lift": ("mlp", False),        # 有意义空间 (判定: margin 塌)
    "randomized_target": ("linear", False),  # C2 反向对照: 目标逐样本随机
}

PREDICTIONS = {
    "C1a": "冻结随机空间里 源可解码 − 类别可解码 ≥ 0.30 (≥4/5 seeds)",
    "C1b": "目标置换后 margin 变化 ≤ 10 个百分点 (≥4/5 seeds)",
    "C1c": "有意义空间下 margin ≤ 冻结随机臂的一半 (≥4/5 seeds)",
    "C2": "随机目标臂里 类别可解码 ≤ 0.50 (否则探针无效 → 审计失败)",
    "C3": "D_shared ∈ {128,256,512} 上 C1a/C1c 的结论方向一致",
}
