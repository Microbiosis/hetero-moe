"""V57.0 — 自适应维度算子 (空间优先, 维度不是超参).

核心命题
--------
v48–v56 的全部争论都建立在一个前提上: `D_shared ∈ {96, 252, 256, 384}` 是**超参**。
于是每一代都在问「该多大」, 而这个问题本身没有答案 —— 一个固定的数不可能
同时适配秩为 3 和秩为 30 的数据。

本代把这个前提拆掉: **维度 r 是被观测出来的状态, 不是被设定的超参。**

    状态 = (V, s)          V ∈ R^{d×r} 正交基, s ∈ R^r 各方向能量
    维度 = r = V 的列数    —— 随数据出生/死亡, 双向

算子职责
--------
    collapse  某方向能量跌到噪声地板以下, 且持续 window 步 → 死亡
    expand    残差里出现超阈值的方向 → 出生

两件事都由**数据**决定, 不由人决定。这是「自适应」的可执行含义。

数学依据 (不是自创, 是拼装):
  · 奇异值硬阈值   Gavish & Donoho (2014) 的**最优**硬阈值
                   τ = ω(β)·σ̂,  ω(β) = 0.56β³ − 0.95β² + 1.82β + 1.43
                   σ̂ = median(s) / (√n · √median(MP_β))
                   这是「未知噪声下 Frobenius 最优」的解, 不是拍脑袋的阈值
  · 噪声地板       Marchenko–Pastur 分布中位数 (数值求)
  · 增量子空间     逐批正交化 + 生死判定 (类 GROUSE / 增量 SVD)
  · 梯度          子空间选择**不可导**, 但投影本身可导 →
                   自写 autograd.Function 做直通 (前向截断, 反向只传子空间内梯度)

必须诚实的一句
--------------
**扩张不是凭空造信息。** 「出生一个新方向」的精确含义是:
此前低于噪声地板的方向, 在新数据里浮出水面。
没有算子在 ambient 之外创造维度; ambient 是计算上界, r 是真实占用。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "mp_sqrt_median", "omega", "optimal_hard_threshold", "noise_sigma",
    "SpectralProject", "AdaptiveSubspace", "GrowthLog",
]


# ----------------------------------------------------------------------
# 1) 噪声地板: Marchenko–Pastur 的 sqrt(λ) 中位数
# ----------------------------------------------------------------------

_MP_CACHE: dict[float, float] = {}


def mp_sqrt_median(beta: float, n_mc: int = 1200, seed: int = 0) -> float:
    """c_β = median(sqrt(λ)), λ ~ MP(β), β = d/n ∈ (0,1].

    **用蒙特卡洛而不是解析式**: 解析密度在 β=1 的 λ→0 端点有 1/sqrt(λ) 奇性,
    数值积分会把中位数冲到 0 —— 实测得到 0.0036 (正确值 ≈0.8), 直接把阈值毁掉。
    生成一个 n×n 高斯矩阵、取奇异值中位数 /√n, 是同一个量且没有数值陷阱。
    固定 seed ⇒ 完全确定, 跨机器一致。
    """
    beta = float(min(max(beta, 1e-6), 1.0))
    key = round(beta, 6)
    if key in _MP_CACHE:
        return _MP_CACHE[key]

    g = torch.Generator().manual_seed(seed)
    n = int(n_mc)
    d = max(1, int(round(n * beta)))
    Z = torch.randn(n, d, generator=g, dtype=torch.float32)
    s = torch.linalg.svdvals(Z) / math.sqrt(n)   # ≈ sqrt(λ)
    out = float(s.median())
    _MP_CACHE[key] = out
    return out


def omega(beta: float) -> float:
    """Gavish–Donoho 最优硬阈值的系数 ω(β)。"""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def noise_sigma(s: torch.Tensor, n: int, d: int) -> float:
    """由奇异值中位数反解噪声标准差 σ̂。

    纯噪声 n×d 矩阵的奇异值 ≈ σ√max(n,d)·sqrt(λ), 所以
        σ̂ = median(s) / (√max(n,d) · c_β)
    """
    if s.numel() == 0:
        return 0.0
    beta = min(n, d) / max(n, d)
    med = float(s.median())
    return med / (math.sqrt(max(n, d)) * mp_sqrt_median(beta) + 1e-12)


def optimal_hard_threshold(s: torch.Tensor, n: int, d: int) -> float:
    """τ = ω(β)·σ̂·√max(n,d) —— 未知噪声下的最优奇异值硬阈值。

    量纲检查: 纯噪声的谱上边缘 ≈ σ(√n+√d), β=1 时为 2σ√n;
    ω(1)=2.86, 故 τ≈2.86σ√n **恰好高于**噪声明——这正是它该在的位置。
    """
    beta = min(n, d) / max(n, d)
    return omega(beta) * noise_sigma(s, n, d) * math.sqrt(max(n, d))


# ----------------------------------------------------------------------
# 2) 自写算子: 谱投影 (子空间选择不可导, 投影可导)
# ----------------------------------------------------------------------

class _SpectralProject(torch.autograd.Function):
    """前向: Z ← Z V Vᵀ  (投影到给定正交基张成的子空间)

    反向: 直通 —— 只回传子空间**内**的梯度, 不假装能对
    「选了哪几个方向」求导 (那个选择是离散的, 没有梯度)。

    这是「自写算子」的实质: 把可导与不可导的部分显式分开,
    而不是让 autograd 在一个伪梯度上蒙混过去。
    """

    @staticmethod
    def forward(ctx, Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        P = V @ V.transpose(0, 1) if V.numel() else torch.zeros(
            Z.shape[1], Z.shape[1], dtype=Z.dtype, device=Z.device)
        ctx.save_for_backward(P)
        return Z @ P

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (P,) = ctx.saved_tensors
        return g @ P, None


def SpectralProject(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return _SpectralProject.apply(Z, V)


# ----------------------------------------------------------------------
# 3) 空间: 状态是基, 维度是读数
# ----------------------------------------------------------------------

class GrowthLog:
    """维度轨迹: 每次观测记一条 (step, r, born, died)。"""

    def __init__(self):
        self.steps: list[int] = []
        self.r: list[int] = []
        self.born: list[int] = []
        self.died: list[int] = []
        self.captured: list[float] = []

    def add(self, step: int, r: int, born: int, died: int, captured: float):
        self.steps.append(step)
        self.r.append(r)
        self.born.append(born)
        self.died.append(died)
        self.captured.append(captured)

    def as_dict(self) -> dict:
        return {"steps": self.steps, "r": self.r, "born": self.born,
                "died": self.died, "captured": self.captured}

    def events(self) -> list[str]:
        out = []
        for s, b, d in zip(self.steps, self.born, self.died):
            if b or d:
                out.append(f"step {s}: r={self.r[len(out)] if False else ''} +{b}/-{d}")
        return out


class AdaptiveSubspace(nn.Module):
    """空间优先算子: 状态是正交基 V, 维度 r = V 的列数。

    用法
    ----
        space = AdaptiveSubspace(ambient_dim=2560)
        for batch in stream:
            coords = space(batch)        # [n, r] —— r 每次可能不同
            # 下游只用 coords, 从不设定维度

    参数
    ----
        ambient_dim   计算上界 d (不是维度; 维度是 r)
        tau_scale     阈值缩放 (1.0 = Gavish–Donoho 最优)
        death_frac    方向能量 < death_frac·σ̂² 视为噪声
        death_window  连续多少步低于地板才判死 (防抖动)
        max_rank      r 的硬上限 (默认 = ambient_dim)
    """

    def __init__(self, ambient_dim: int, tau_scale: float = 1.0,
                 death_frac: float = 1.0, death_window: int = 3,
                 max_rank: int | None = None):
        super().__init__()
        self.ambient_dim = int(ambient_dim)
        self.tau_scale = float(tau_scale)
        self.death_frac = float(death_frac)
        self.death_window = int(death_window)
        self.max_rank = int(max_rank or ambient_dim)
        self.register_buffer("V", torch.zeros(self.ambient_dim, 0))
        self.register_buffer("_below", torch.zeros(0, dtype=torch.long))
        self.log = GrowthLog()
        self._step = 0

    # -- 读数 --
    @property
    def r(self) -> int:
        return int(self.V.shape[1])

    @property
    def rank(self) -> int:
        return self.r

    def energy(self, Z: torch.Tensor) -> torch.Tensor:
        """各方向的逐样本方差 (能量)。"""
        if self.r == 0:
            return torch.zeros(0, device=Z.device)
        return (Z @ self.V).pow(2).mean(dim=0)

    def captured(self, Z: torch.Tensor) -> float:
        """子空间捕获的总能量占比。"""
        tot = float(Z.pow(2).sum())
        if tot <= 1e-12:
            return 1.0
        return float(SpectralProject(Z, self.V).pow(2).sum()) / tot

    # -- 前向 --
    def forward(self, Z: torch.Tensor, update: bool = True) -> torch.Tensor:
        """返回低维坐标 [n, r]; update=True 时同时做生死判定。"""
        Z = Z.detach().float()
        if update:
            self._collapse_and_expand(Z)
        if self.r == 0:
            return torch.zeros(Z.shape[0], 0, device=Z.device)
        return Z @ self.V

    # -- 生死 --
    @torch.no_grad()
    def _collapse_and_expand(self, Z: torch.Tensor):
        self._step += 1
        n, d = Z.shape
        n_born = n_died = 0

        # ---- 1) 先算残差与噪声地板 ----
        if self.r > 0:
            R = Z - SpectralProject(Z, self.V)
        else:
            R = Z
        # 用全矩阵奇异值估噪声 (含信号, 中位数仍稳健)
        s_all = torch.linalg.svdvals(R)
        tau = optimal_hard_threshold(s_all, n, d) * self.tau_scale
        sig = noise_sigma(s_all, n, d)

        # ---- 2) 死亡: 方向能量跌到噪声地板以下, 连续 window 步 ----
        if self.r > 0:
            e = self.energy(Z)
            floor = max(self.death_frac * sig ** 2, 1e-12)
            below = (e < floor)
            self._below = self._below + below.long()
            alive = self._below < self.death_window
            # 刚出生的方向给 window 步宽限 (不立刻判死)
            if not bool(alive.all()):
                keep = alive
                n_died = int((~keep).sum())
                self.V = self.V[:, keep].contiguous()
                self._below = self._below[keep]
                # 重新正交化 (删列后仍是正交的, 但保证数值干净)
                if self.r:
                    q, _ = torch.linalg.qr(self.V)
                    self.V = q[:, : self.V.shape[1]].contiguous()

        # ---- 3) 扩张: 残差里出现超阈值方向 ----
        if self.r < self.max_rank:
            if self.r > 0:
                R = Z - SpectralProject(Z, self.V)
            else:
                R = Z
            if n >= 2:
                Ur, Sr, Vrt = torch.linalg.svd(R)
                cand = []
                for i in range(min(Sr.numel(), self.max_rank - self.r)):
                    if float(Sr[i]) <= tau:
                        break
                    v = Vrt[i]
                    # 与已有基正交化 (针对浮点误差累积)
                    if self.r + len(cand) > 0:
                        B = torch.cat([self.V] + ([torch.stack(cand, 1)] if cand else []), 1)
                        if B.shape[1] > 0:
                            q, _ = torch.linalg.qr(B)
                            v = v - q @ (q.T @ v)
                    nrm = float(v.norm())
                    if nrm < 1e-6:
                        continue
                    cand.append(v / nrm)
                if cand:
                    newV = torch.stack(cand, 1)
                    self.V = torch.cat([self.V, newV], 1).contiguous()
                    self._below = torch.cat(
                        [self._below, torch.zeros(len(cand), dtype=torch.long)])
                    n_born = len(cand)

        self.log.add(self._step, self.r, n_born, n_died, self.captured(Z))
        return n_born, n_died

    # -- 轨迹 --
    def trajectory(self) -> dict:
        return self.log.as_dict()


# ----------------------------------------------------------------------
# 4) 空间优先 ⇒ 模型次之: 有效秩随空间自动变, 参数不重初始化
# ----------------------------------------------------------------------

class AdaptiveOperator(nn.Module):
    """把 W 定义在 environment 空间, 由空间决定它有多少被激活。

        y = P_V · W · P_V · x         P_V = V Vᵀ

    W ∈ R^{d×d} 形状**固定**, 但有效算子 P_V W P_V 的秩 ≤ r。
    于是:

      · r 长大 → 有效容量自动变大, **不需要重建模型 / 不重置参数**
      · r 收缩 → 被砍掉的方向自动不再参与, **灾难性遗忘被空间本身挡住**

    这是「空间优先, 然后才是模型」的可执行形式:
    维度由空间 (`AdaptiveSubspace`) 决定, 模型只是住在这个空间里的算子。
    参数总量不随 r 变 —— 变的是**有多少参数被用到**。
    """

    def __init__(self, ambient_dim: int, space: "AdaptiveSubspace | None" = None,
                 init_scale: float = 0.02):
        super().__init__()
        self.W = nn.Parameter(torch.randn(ambient_dim, ambient_dim) * init_scale / math.sqrt(ambient_dim))
        self.space = space

    def bind(self, space: "AdaptiveSubspace"):
        self.space = space
        return self

    @property
    def effective_rank(self) -> int:
        return self.space.r if self.space is not None else 0

    @property
    def effective_params(self) -> int:
        r = self.effective_rank
        return r * r + 2 * r * (self.W.shape[1] - r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.space is None or self.space.r == 0:
            return torch.zeros_like(x)
        V = self.space.V
        z = x @ V                      # [n, r]
        z = z @ (V.T @ self.W @ V)     # 有效算子, 秩 ≤ r
        return z @ V.T
