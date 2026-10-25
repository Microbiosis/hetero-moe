"""V49.0 — 投影空间的三轴机制: 非线性 / 条件依赖 / (容量对齐)。

三个轴与 v48 的关系
------------------
v48 结论: 线性可训练的 P_m 把绝对误差降 52%-77%, 但过拟合间隙 95%-99.9%,
且"融合 vs 单专家"相对增益归零 → 收益来自**拟合能力**。于是 v48 留下三个
悬而未决的可能, v49 一并做掉:

  轴 N (非线性): 冗余存在(CKA 0.48)但联合 PCA 抓不到 → 说明要靠**非线性**
      映射才能利用那份额外结构。v48 的 B 臂 P_m 是线性的, 所以它学不到。
  轴 C (条件依赖): 空间要"随着条件的改变而改变" → 投影由内容派生的条件
      调制 (超网络低秩调制), 而不是固定矩阵。
  轴 S (样本量): 18-36 样本 vs 1.18M 参数必然记忆 → 样本量做成自变量
      (32 / 256 / 1024, 由 data.py 的槽位组合生成器提供)。

容量对齐 (v41 的教训)
---------------------
  线性臂参数量 = d_shared × D_m
  非线性臂参数量 = D_m×h + h×d_shared + h, 其中 **h 按参数量反解**,
  使两臂参数量在 0.5% 内相等 → N 轴上的差异**不能**用"参数更多"解释。
  条件依赖臂额外多 V/U/超网络 (约 +7%); 该轴的判决性对照不是参数量而是
  **打乱条件 (shuffle-c)**: 若打乱后不退化, 条件就是装饰。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ARMS", "build_projector", "matched_hidden", "param_report", "permute_cond"]


def matched_hidden(d_shared: int, D_m: int) -> int:
    """求隐藏维 h, 使非线性臂参数量 ≈ 线性臂 (d_shared × D_m)。"""
    target = d_shared * D_m
    best_h, best_gap = 1, None
    for h in range(8, 1025):
        p = D_m * h + h * d_shared + h
        gap = abs(p - target)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


class LinearProj(nn.Module):
    """v33 同款: 一张矩阵 (D_m → d_shared), 可训练与否由外部决定。"""

    def __init__(self, d_shared: int, D_m: int):
        super().__init__()
        self.P = nn.Parameter(torch.empty(d_shared, D_m))
        nn.init.orthogonal_(self.P)

    def forward(self, x, cond=None):
        return F.linear(x, self.P)


class NonlinearProj(nn.Module):
    """两层 MLP 抬升: D_m → h → d_shared (h 按参数量对齐线性臂)。"""

    def __init__(self, d_shared: int, D_m: int, hidden: int | None = None):
        super().__init__()
        h = matched_hidden(d_shared, D_m) if hidden is None else hidden
        self.hidden = h
        self.down = nn.Linear(D_m, h)
        self.up = nn.Linear(h, d_shared)

    def forward(self, x, cond=None):
        return self.up(F.silu(self.down(x)))


class CondMixin(nn.Module):
    """条件依赖 (超网络低秩调制):

        y = P0 x  +  (x Vᵀ) ⊙ g(c) Uᵀ ,   g(c) = MLP(c) ∈ R^rank

    条件 c 由**输入自身**的均值池化派生 (内容派生, 不引入标签信息):
    换掉 c 就等于"用别的样本的条件来解释这条样本" → 打乱 c 的对照即
    "条件是否真在起作用"的判决性检验。
    """

    def __init__(self, d_shared: int, D_m: int, rank: int = 16, cond_dim: int = 32,
                 hidden: int = 32):
        # ⚠️ 这里**不能**再调 nn.Module.__init__(self): 兄弟基类
        # (LinearProj/NonlinearProj) 的 __init__ 已经初始化过模块, 再调一次会
        # **清空 _parameters** → 刚建好的 P 消失 (实测 'no attribute P')。
        # 同理也不能写 super().__init__(): MRO 的下一个类要求 (d_shared, D_m)。
        self.rank, self.cond_dim = rank, cond_dim
        self.V = nn.Linear(D_m, rank, bias=False)
        self.U = nn.Linear(rank, d_shared, bias=False)
        self.hyper = nn.Sequential(nn.Linear(cond_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, rank))
        nn.init.zeros_(self.U.weight)          # 初始时调制为 0 → 与静态臂起点一致
        nn.init.normal_(self.V.weight, std=0.01)

    def condition_from(self, x):
        """从输入派生每样本条件: [N, cond_dim]"""
        pooled = x.mean(dim=1)                                   # [N, D_m]
        return torch.tanh(pooled @ self.cond_pool)                # [N, cond_dim]

    def _ensure_pool(self, D_m):
        """条件池化矩阵 (buffer, 不训练): [D_m, cond_dim]"""
        if getattr(self, "cond_pool", None) is None or self.cond_pool.shape[0] != D_m:
            ref = next(self.parameters())
            self.register_buffer("cond_pool",
                                 torch.randn(D_m, self.cond_dim,
                                             device=ref.device, dtype=ref.dtype) * 0.01)

    def modulate(self, x, cond):
        g = self.hyper(cond)                                     # [N, rank]
        d = F.linear(x, self.V.weight) * g.unsqueeze(1)          # [N, S, rank]
        return F.linear(d, self.U.weight)


class LinearCondProj(CondMixin, LinearProj):
    """线性基座 + 条件调制"""

    def __init__(self, d_shared, D_m, rank=16, cond_dim=32):
        LinearProj.__init__(self, d_shared, D_m)
        CondMixin.__init__(self, d_shared, D_m, rank=rank, cond_dim=cond_dim)
        self._ensure_pool(D_m)

    def forward(self, x, cond=None):
        y = F.linear(x, self.P)
        if cond is None:
            cond = self.condition_from(x)
        return y + self.modulate(x, cond)


class NonlinearCondProj(CondMixin, NonlinearProj):
    """非线性基座 + 条件调制"""

    def __init__(self, d_shared, D_m, rank=16, cond_dim=32):
        NonlinearProj.__init__(self, d_shared, D_m)
        CondMixin.__init__(self, d_shared, D_m, rank=rank, cond_dim=cond_dim)
        self._ensure_pool(D_m)

    def forward(self, x, cond=None):
        y = self.up(F.silu(self.down(x)))
        if cond is None:
            cond = self.condition_from(x)
        return y + self.modulate(x, cond)


# 臂定义: (lift, conditional) —— 四格构成 N × C 的 2×2 析因
ARMS = {
    "L_static":   ("linear", False),      # N0C0
    "L_cond":     ("linear", True),       # N0C1
    "NL_static":  ("nonlinear", False),   # N1C0
    "NL_cond":    ("nonlinear", True),    # N1C1
    "L_static_frozen": ("linear", False), # 对照: 投影冻结 (v48 的 A 臂)
}
FROZEN_ARMS = {"L_static_frozen"}


def build_projector(arm: str, d_shared: int, D_m: int) -> nn.Module:
    lift, cond = ARMS[arm]
    if arm == "L_static_frozen":
        return LinearProj(d_shared, D_m)
    if lift == "linear":
        return LinearCondProj(d_shared, D_m) if cond else LinearProj(d_shared, D_m)
    return NonlinearCondProj(d_shared, D_m) if cond else NonlinearProj(d_shared, D_m)


def param_report(proj: nn.Module) -> dict:
    total = sum(p.numel() for p in proj.parameters())
    trainable = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "hidden": getattr(proj, "hidden", None)}


def permute_cond(cond: torch.Tensor, seed: int) -> torch.Tensor:
    """打乱条件 (跨样本置换) —— 条件轴的判决性对照。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(cond.shape[0], generator=g).to(cond.device)
    return cond[idx]
