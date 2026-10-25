"""V53.0 — 第三审: 独立复算 v50/v51/v52 的结论 (第三套实现).

定位
----
审计链: v50 (审 v33 线) → v51 (审 v6/v7 线) → v52 (独立复算 v50/v51) →
**v53 (独立复算 v52)**。按"三审制", 这一代闭合审计链。

为什么要第三套实现而不是重跑 v52
------------------------------
重跑同一份代码只能查"是否可复现", 查不出"是否被实现细节带偏"。v53 因此把
**每个可变量都换一遍**, 并且加一个**正面量**:

  跨底座一致性 (cross-expert consistency)
      同一段文本, 分别过三个底座 (各自的编码 → 各自的投影 → 融合层) 之后,
      两个底座的输出有多一致? 一致性高 = 共享空间真的让异构底座"相遇"了。
      这是**正面证据**: 前两代只证明了"margin 是识别机制的产物"(负面),
      这一代要给出"有意义的空间确实做到了对齐"(正面)。

第三套实现的每一处都与前两代不同
------------------------------
| 维度 | v50/v51 | v52 | v53 (本代) |
|---|---|---|---|
| 数据划分 | 换扰动种子 | 奇偶划分 (未见组合) | **值域分割** (训练只见 slot1∈A∧slot2∈B, 留出只见 C∧D → 换域) |
| 探针 | SGD 逻辑回归 | 闭式岭回归 | **岭回归 + MLP 探针 + kNN 探针 (三族交叉)** |
| 抬升 | 两层 MLP | MLP+LayerNorm+GELU | **残差 MLP (残差连接 + 逐元素门控)** |
| 种子 | 0–4 | 1000–1004 | **2000–2004** |
| D_shared | 256 | 128/256/512 | **96 / 384** (与 v52 不重叠) |
| 正向量 | — | — | **跨底座一致性 (输入与输出)** |

预登记判决见 SPEC.md 与 PREDICTIONS (先写死, 事后不得改)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CLASSES", "make_split", "build_targets", "RidgeProbe", "MLPProbe",
           "KNNProbe", "probe_accuracy_all", "ResidualLift", "LinearLift",
           "ARMS", "PREDICTIONS", "consistency"]

# ----------------------------------------------------------------------
# 1) 第三套数据生成器: 值域分割 (训练域与留出域**不相交**)
# ----------------------------------------------------------------------

CLASSES = {
    "cavity": (["deep", "wide", "narrow", "round", "square", "sharp", "flat", "tall",
                "thin", "thick", "hollow", "solid", "open", "sealed", "soft", "hard"],
               ["crown", "neck", "ridge", "seam", "vent", "lumen", "edge", "bevel",
                "panel", "joint", "notch", "bore", "flange", "lattice", "socket", "rib"]),
    "pollen": (["dry", "wet", "light", "heavy", "round", "flat", "bright", "dull",
                "sticky", "loose", "packed", "fresh", "aged", "ripe", "raw", "fried"],
               ["grain", "cell", "pore", "husk", "wall", "coat", "stalk", "filament",
                "spike", "cluster", "theca", "anther", "awn", "pollen", "sac", "fringe"]),
    "cable": (["thin", "thick", "flat", "round", "braided", "coiled", "rigid", "flexible",
               "shielded", "bare", "double", "single", "split", "jointed", "stranded", "solid"],
              ["core", "sheath", "braid", "strand", "clamp", "bushing", "gland", "ferrule",
               "lacing", "spoke", "wrap", "loom", "lay", "splice", "cleat", "trunk"]),
    "bead":  (["small", "large", "bright", "dull", "smooth", "faceted", "round", "flat",
               "worn", "new", "carved", "moulded", "glazed", "polished", "raw", "cast"],
              ["loop", "bead", "spacer", "tube", "barrel", "cone", "disc", "seed",
               "bead2", "spindle", "toggle", "ring", "cube", "star", "oval", "cap"]),
    "timber": (["green", "dry", "hard", "soft", "split", "whole", "rough", "smooth",
                "curved", "straight", "seasoned", "fresh", "warped", "sanded", "aged", "new"],
               ["joist", "plank", "post", "lath", "sill", "beam", "stud", "rafter",
                "purlin", "header", "trim", "batten", "mullion", "tie", "gable", "crosstie"]),
    "tide":  (["high", "low", "rising", "falling", "spring", "neap", "calm", "rough",
               "brackish", "salty", "clear", "muddy", "warm", "cold", "ebb", "flood"],
              ["shoal", "bar", "lagoon", "rip", "swell", "surge", "eddy", "channel",
               "estuary", "sound", "cove", "trough", "gulf", "bay", "reef", "inlet"]),
}
N_CLASSES = len(CLASSES)
HALF = 8          # 每类 16 个槽位值 → 前 8 个 = 训练域, 后 8 个 = 留出域


def _combos(adj, nouns, slot1_ids, slot2_ids):
    return [f"{adj[i]} {nouns[j]}" for i in slot1_ids for j in slot2_ids]


def make_split(n_per_class: int, seed: int = 0):
    """训练域 (slot1∈前8 ∧ slot2∈前8) 与留出域 (slot1∈后8 ∧ slot2∈后8) —— 两域**零重叠**,
    且每个槽位取值也分属不同域 → 留出测的是**换域**泛化, 比"未见组合"更严。"""
    train, held = [], []
    for ci, (adj, nouns) in enumerate(CLASSES.values()):
        tr = _combos(adj, nouns, range(HALF), range(HALF))
        ho = _combos(adj, nouns, range(HALF, 16), range(HALF, 16))
        rng = np.random.default_rng(seed * 6151 + ci)
        ti = rng.permutation(len(tr))[:n_per_class]
        hi = rng.permutation(len(ho))[:n_per_class]
        train += [(ci, tr[i]) for i in ti]
        held += [(ci, ho[i]) for i in hi]
    return train, held


def build_targets(labels, d_shared: int, n_classes: int, device, randomized: bool = False,
                  seed: int = 0, n_tokens: int = 1):
    """目标块 [n, n_tokens, d_shared]; randomized=True 时逐样本随机 (C2 反向对照)。"""
    blk = d_shared // n_classes
    tgt = torch.zeros(len(labels), n_tokens, d_shared)
    g = np.random.default_rng(seed)
    for i, c in enumerate(labels):
        b = int(g.integers(n_classes)) if randomized else c
        tgt[i, :, b * blk:(b + 1) * blk] = 1.0
    return tgt.to(device)


# ----------------------------------------------------------------------
# 2) 三个探针族 (交叉验证: 结论不能依赖某一种探针)
# ----------------------------------------------------------------------

class RidgeProbe:
    """闭式岭回归 (与 v52 同族, 作为基准)。"""

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.W = None

    def fit(self, X, y, n_class):
        X = X.double()
        Y = torch.zeros(y.shape[0], n_class, dtype=torch.float64)
        Y[torch.arange(y.shape[0]), y] = 1.0
        Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
        A = Xb.T @ Xb + self.alpha * torch.eye(Xb.shape[1], dtype=torch.float64)
        self.W = torch.linalg.solve(A, Xb.T @ Y)
        return self

    def predict(self, X):
        X = X.double()
        Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
        return (Xb @ self.W).argmax(-1)


class MLPProbe:
    """非线性探针 (两层 MLP + GELU): 用来排除"线性探针看不见真结构"的反驳。"""

    def __init__(self, hidden: int = 64, epochs: int = 300, lr: float = 3e-3, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, X, y, n_class):
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, n_class)).double()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        Xd, yd = X.double(), y.long()
        n = Xd.shape[0]
        g = torch.Generator().manual_seed(self.seed + 3)
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, n, (min(64, n),), generator=g)
            loss = F.cross_entropy(self.net(Xd[idx]), yd[idx])
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        return self.net(X.double()).argmax(-1)


class KNNProbe:
    """k-最近邻探针 (无参数): 第三族, 与前两族无共同假设。"""

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X, y, n_class):
        self.X = X.double()
        self.y = y.long()
        return self

    def predict(self, X):
        X = X.double()
        d = torch.cdist(X, self.X)
        nn_idx = d.topk(self.k, largest=False).indices
        votes = torch.zeros(X.shape[0], int(self.y.max().item()) + 1, dtype=torch.float64)
        votes.scatter_add_(1, self.y[nn_idx], torch.ones(X.shape[0], self.k, dtype=torch.float64))
        return votes.argmax(-1)


PROBE_FAMILIES = {"ridge": RidgeProbe, "mlp": MLPProbe, "knn": KNNProbe}


def probe_accuracy_all(X, y, n_class, seed: int = 0, split: float = 0.8):
    """三个探针族各自的 (train, test) 准确率。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)
    n_tr = max(1, int(X.shape[0] * split))
    tr, te = idx[:n_tr], idx[n_tr:]
    if te.numel() == 0:
        te = tr
    out = {}
    for name, cls in PROBE_FAMILIES.items():
        p = cls().fit(X[tr], y[tr], n_class) if name != "mlp" else cls(seed=seed).fit(X[tr], y[tr], n_class)
        out[name] = (float((p.predict(X[te]) == y[te]).double().mean()),
                     float((p.predict(X[tr]) == y[tr]).double().mean()))
    return out


# ----------------------------------------------------------------------
# 3) 第三个抬升实现: 残差 MLP (残差连接 + 逐元素门控)
# ----------------------------------------------------------------------

class ResidualLift(nn.Module):
    """D_m → h → (h + 门控(MLP(h))) → d_shared。隐藏维按参数量反解, 与线性臂对齐。"""

    def __init__(self, d_shared: int, D_m: int):
        super().__init__()
        target = d_shared * D_m
        best_h, best_gap = 8, None
        for h in range(8, 2048):
            # ln(2*D_m) + inp(D_m*h+h) + res(h*h+h) + gate(h*h+h) + out(h*d+d)
            p = 2 * D_m + (D_m * h + h) + 2 * (h * h + h) + (h * d_shared + d_shared)
            gap = abs(p - target)
            if best_gap is None or gap < best_gap:
                best_h, best_gap = h, gap
        h = best_h
        self.hidden = h
        self.ln = nn.LayerNorm(D_m)
        self.inp = nn.Linear(D_m, h)
        self.res = nn.Linear(h, h)
        self.gate = nn.Linear(h, h)
        self.out = nn.Linear(h, d_shared)

    def forward(self, x, cond=None):
        z = F.gelu(self.inp(self.ln(x)))
        z = z + torch.tanh(self.gate(z)) * self.res(z)
        return self.out(z)


class LinearLift(nn.Module):
    """线性抬升 (正交初始化, 与论文原配置同构)。"""

    def __init__(self, d_shared: int, D_m: int):
        super().__init__()
        self.P = nn.Parameter(torch.empty(d_shared, D_m))
        nn.init.orthogonal_(self.P)

    def forward(self, x, cond=None):
        return F.linear(x, self.P)


# ----------------------------------------------------------------------
# 4) 跨底座一致性 (正面量)
# ----------------------------------------------------------------------

def consistency(outs_per_expert, labels_per_expert):
    """同一文本在三个底座上的输出 (或投影输入) 的两两余弦相似度均值。

    outs_per_expert: {expert: [n, ...]} (同一文本按行对齐)
    → 一致性高 = 共享空间真的让异构底座"相遇"。
    """
    experts = list(outs_per_expert)
    n = outs_per_expert[experts[0]].shape[0]
    vals = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            a = outs_per_expert[experts[i]].reshape(n, -1).float()
            b = outs_per_expert[experts[j]].reshape(n, -1).float()
            a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
            b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
            vals.append((a * b).sum(-1))
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean())


# ----------------------------------------------------------------------
# 5) 臂与预登记判决
# ----------------------------------------------------------------------

ARMS = {
    "frozen_random": ("linear", False),     # 论文原配置
    "permuted_target": ("linear", True),    # 目标置换对照
    "learned_lift": ("residual", False),    # 有意义空间
    "randomized_target": ("linear", False),  # 探针有效性反向对照
}

PREDICTIONS = {
    "D1a": "冻结随机空间里 源可解码 − 类别可解码 ≥ 0.30 (≥4/5 seeds), 且**三个探针族同向**",
    "D1b": "目标置换后 margin 变化 ≤ 10 个百分点 (≥4/5 seeds)",
    "D1c": "有意义空间下 margin ≤ 冻结随机臂的一半 (≥4/5 seeds)",
    "D2": "随机目标臂里 类别可解码 ≤ 0.50 (三个探针族都) → 否则探针无效, 审计失败",
    "D3": "有意义空间的**跨底座一致性**显著高于随机空间 (预登记: ≥ +0.15 余弦)",
    "D4": "D_shared ∈ {96, 384} 上 C1a/C1c 方向一致 (与 v52 的 {128,256,512} 不重叠)",
    "D5": "v52 的结论在本套实现下复现 (三审制闭合)",
}
