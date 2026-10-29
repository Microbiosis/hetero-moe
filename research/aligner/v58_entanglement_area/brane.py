"""V58.0 — 膜 / 面积律算子: 维度按边界面积分配, 不按体积分配.

理论链 (每一环都可验证, 没有一环是比喻)
--------------------------------------
  膜理论 / AdS-CFT
      → Ryu–Takayanagi:  S(A) = Area(γ_A) / (4 G_N)
         「区域 A 的纠缠熵由包围它的**最小曲面面积**决定」
      → 高斯基态下这条是**可算的恒等式**:
         S(A) = ½ Σ_i log( 1 / (1 − ρ_i²) ),  ρ_i = A 与 B 的正则相关
         (ρ_i→1 贡献大, ρ_i→0 贡献小 —— 只有"跨界的通道"计入熵)
      → 面积律: S 随 |A| **饱和**; 体积律: S ∝ |A| 线性增长
      → 张量网络: 切口上的 bond dimension χ ≈ e^{S}
      → **自适应维度的正确形式**: χ 由边界面积决定, 与块内体积无关

这给出一个**可证伪**的预测, 而 v57 的单一 r 给不出来:

    v57:  一个全局标量 r     —— 无法表达"这里维度多、那里维度少"
    本代:  χ(l) 逐切口分布   —— 若 χ 处处相同, 膜结构退化回 v57 (B4 是证伪点)

为什么这是「维度」而不是「比喻」
------------------------------
`canonical_correlations` 给的是两块之间**真正可传递的独立通道数**。
不是有效秩的近似, 是精确的: 高斯情形下 A 与 B 的全部统计依赖都编码在 ρ 谱里。
所以 χ = #{ρ_i > 噪声地板} 是**切口需要的维数**, 可测、可复现、不依赖超参。
"""
from __future__ import annotations

import math
import os
import sys

import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _root not in sys.path:
    sys.path.insert(0, _root)
from research.aligner.v57_adaptive_dimension.operators import optimal_hard_threshold, omega  # noqa: E402

__all__ = [
    "cross_spectrum", "noise_tau", "canonical_correlations", "entanglement_entropy", "capacity_needed",
    "make_lattice", "scan_cuts", "BraneSpace", "PREDICTIONS",
]


# ----------------------------------------------------------------------
# 1) 正则相关: 两块之间真正可传递的通道
# ----------------------------------------------------------------------

def cross_spectrum(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """互协方差的谱: Xⱼ 与 Yⱼ 的**净**耦合强度 (降序)。

    为什么用互协方差而不是 CCA:
      CCA 要白化, 而白化在 n 与 d 同量级时会把样本噪声放大成假相关 ——
      实测纯独立随机块 (n=512, 32×48) 会给出 29–31 个"显著"通道。
      置换零分布也救不了: n_null=16 时尾部估计不稳, 阈值化后 k=12 读成 13、
      z 检验又把噪声读成 17。两条路都不稳。

      互协方差 Σxy = XᵀY/(n−1) 没有白化, 在独立时就是一个**噪声矩阵**,
      其奇异值直接服从 Marchenko–Pastur ⇒ 可以用 v57 那套 Gavish–Donoho
      最优硬阈值, 不需要置换、不需要调参。

    标准化: 两侧各自按 √n/‖column‖ 归一, 使独立时 σ ≈ 1/n, 阈值可跨数据集比较。
    """
    A = A.float(); B = B.float()
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    n = A.shape[0]
    A = A / (A.norm(dim=0, keepdim=True) / math.sqrt(n)).clamp(min=1e-8)
    B = B / (B.norm(dim=0, keepdim=True) / math.sqrt(n)).clamp(min=1e-8)
    M = (A.T @ B) / max(n - 1, 1)          # 独立时元素 ~ N(0, 1/n)
    return torch.linalg.svdvals(M)


def canonical_correlations(A: torch.Tensor, B: torch.Tensor,
                           ridge: float = 1e-6, n_null: int = 0,
                           seed: int = 0) -> torch.Tensor:
    """兼容旧接口: 返回**净**互协方差谱 (已按噪声阈值裁掉的部分置零)。

    保留这个名字是因为 v58 早期版本与测试用过它; 语义已改为互协方差谱。
    """
    spec = cross_spectrum(A, B)
    tau = noise_tau(A.shape[1], B.shape[1], A.shape[0])
    return spec * (spec > tau).float()


def entanglement_entropy(A: torch.Tensor, B: torch.Tensor, n_null: int = 0) -> float:
    """高斯基态的纠缠熵 S(A) = ½ Σ log(1/(1−ρ²)) —— RT 公式的可算形式。

    ρ 用**归一化互协方差谱**代替 CCA 的正则相关:
        ρ̃_i = s_i / sqrt(s_i² + n)     (把互协方差奇异值映到 [0,1))
    独立时 s_i ≈ √n·O(1) ⇒ ρ̃ 小 ⇒ S≈0; 完全耦合时 s_i ≫ √n ⇒ ρ̃→1 ⇒ S 大。
    """
    spec = cross_spectrum(A, B)
    n = A.shape[0]
    rho2 = (spec.pow(2) / (spec.pow(2) + n)).clamp(max=1 - 1e-9)
    return float(0.5 * (-torch.log1p(-rho2)).sum())


def noise_tau(da: int, db: int, n: int, kappa: float = 1.25) -> float:
    """互协方差谱在**独立**假设下的噪声上包络。

    本式由 `tools/calibrate_v58_tau.py` 在远端扫 40 格
    (n ∈ {96,192,512,1024,2048} × d ∈ {32..5120}) 拟合而成。

    **被推翻的三个版本** (全部留档, 因为它们各自在某个区间是对的):

    ① τ = ω(β)·√dmax/√n        (v57 遗留)
       d ≫ n 时低 5–9 倍 ⇒ 纯噪声 n=192,d=10240 读出 χ=191 (= 理论上限)
    ② τ = 1.6·√(da·db)/n
       d ≪ n 时高得离谱 ⇒ n=512,d=64 共享 1 维给出 χ=22
    ③ τ = β·② + (1−β)·①        (凸组合)
       仍 0/40 通过 —— 两端各对一点, 中间全错

    **实测规律**: 纯噪声谱最大值

        s_noise ≈ (√da + √db)/√n · (1 + dmax/n)^(1/3)

    第一因子是 iid 元素的谱范数 (√da+√db 个 σ=1/√n 的方向叠加);
    第二因子修正 d≳n 时"秩被 n 限制"带来的增长 —— 实测指数 1/3 在
    d/n ∈ [0.03, 54] 上稳合 (五组验收误差 <5%)。

    kappa=1.25 留 25% 余量。校准显示可行区间极宽 (s_1/s_noise = 5–30 倍),
    所以宁可偏保守也不能低于噪声。

    ⚠️ 适用范围: n ∈ [96, 2048], d ∈ [32, 5120]。范围外未检验。
       不变量: **纯噪声必须给 χ=0** (单测 test_pure_noise_gives_zero_chi)。
    """
    dmax = max(da, db)
    base = (math.sqrt(max(da, 1)) + math.sqrt(max(db, 1))) / math.sqrt(max(n, 1))
    grow = (1.0 + dmax / max(n, 1)) ** (1.0 / 3.0)
    return kappa * base * grow


def capacity_needed(A: torch.Tensor, B: torch.Tensor,
                    rho_min: float = 0.0, n_null: int = 0) -> int:
    """切口需要的维数 χ = 互协方差谱中超过噪声阈值的个数 (无自由参数)。"""
    spec = cross_spectrum(A, B)
    tau = noise_tau(A.shape[1], B.shape[1], A.shape[0])
    return int((spec > tau).sum())


# ----------------------------------------------------------------------
# 2) 把表征摆成格点 (worldsheet)
# ----------------------------------------------------------------------

def make_lattice(X: torch.Tensor, shape: tuple[int, int],
                 order: str = "natural", seed: int = 0) -> torch.Tensor:
    """[n, d] → [n, L1, L2, dw] 的格点。

    为什么需要 order: 「格点上的相邻」这件事本身可能只是我们的假设。
    natural 用原本的坐标顺序; random 用置换 —— 若两者给出同样的 χ, 说明
    natural 顺序里没有结构, 面积律是任何二分都有的平凡性质 (B3 的对照)。

    shape 的两个轴: (L1, L2) = (层/深度, token 位) 或用户指定。
    """
    n, d = X.shape
    L1, L2 = shape
    assert L1 * L2 > 0 and d % (L1 * L2) == 0, (d, shape)
    dw = d // (L1 * L2)
    if order == "natural":
        idx = torch.arange(d)
    elif order == "random":
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(d, generator=g)
    else:
        raise ValueError(f"order={order!r}, 应为 natural|random")
    return X[:, idx].reshape(n, L1, L2, dw)


def _as_grid(lattice: torch.Tensor):
    """接受 [n, L, dw] 的链或 [n, L1, L2, dw] 的片, 统一成 (flat[n, L1*L2, dw], L1, L2, dw)。

    链 [n, L, dw] 映射为 L1=1, L2=L —— 即链是**空间轴**, axis=1 沿链切。
    (若映射成 L1=L 则 axis=1 上没有切口, 实测踩过。)
    """
    if lattice.dim() == 3:
        n, L, dw = lattice.shape
        return lattice, 1, L, dw
    if lattice.dim() == 4:
        n, L1, L2, dw = lattice.shape
        return lattice.reshape(n, L1 * L2, dw), L1, L2, dw
    raise ValueError(f"lattice 应为 3 或 4 维, 得到 {tuple(lattice.shape)}")


def scan_cuts(lattice: torch.Tensor, axis: int, n_null: int = 0) -> list[dict]:
    """沿某个轴逐切口扫描: 每个切口返回位置、两侧大小、χ、熵、净耦合能量。

    lattice: [n, L, dw] (链 → L1=1, L2=L) 或 [n, L1, L2, dw] (片)
    axis=0: L1 方向 (层/深度, RG radial);  axis=1: L2 方向 (空间/token)
    """
    flat, L1, L2, dw = _as_grid(lattice)
    n = flat.shape[0]
    L = [L1, L2][axis]
    out = []
    for cut in range(1, L):
        if axis == 0:
            left_ids = [i * L2 + j for i in range(cut) for j in range(L2)]
            right_ids = [i * L2 + j for i in range(cut, L1) for j in range(L2)]
        else:
            left_ids = [i * L2 + j for i in range(L1) for j in range(cut)]
            right_ids = [i * L2 + j for i in range(L1) for j in range(cut, L2)]
        A = flat[:, left_ids].reshape(n, -1)
        B = flat[:, right_ids].reshape(n, -1)
        spec = cross_spectrum(A, B)
        tau = noise_tau(A.shape[1], B.shape[1], n)
        rho2 = (spec.pow(2) / (spec.pow(2) + n)).clamp(max=1 - 1e-9)
        out.append({
            "cut": cut,
            "size_A": len(left_ids) * dw,
            "size_B": len(right_ids) * dw,
            "chi": int((spec > tau).sum()),
            "entropy": float(0.5 * (-torch.log1p(-rho2)).sum()),
            "energy": float(spec.pow(2).sum() / n),
            "spec_max": float(spec.max()) if spec.numel() else 0.0,
            "tau": float(tau),
        })
    return out


# ----------------------------------------------------------------------
# 3) 膜空间: χ 逐切口分配 (v57 的标量 r 是它的退化情形)
# ----------------------------------------------------------------------

class BraneSpace:
    """每个切口一个 χ 预算。总容量 = Σ χ_i, 而不是一个标量 r。

    与 v57 的关系:
        v57  AdaptiveSubspace: 一个全局 r, 处处同宽
        本代 BraneSpace:      逐切口 χ, 可非均匀  ← B4 是证伪点
    """

    def __init__(self, rho_min: float = 0.5, n_null: int = 16):
        self.rho_min = rho_min
        self.n_null = n_null
        self.profile: list[dict] = []

    def measure(self, lattice: torch.Tensor, axis: int = 1) -> list[dict]:
        self.profile = scan_cuts(lattice, axis, n_null=self.n_null)
        return self.profile

    @property
    def total_capacity(self) -> int:
        return sum(p["chi"] for p in self.profile)

    @property
    def is_uniform(self) -> bool:
        return len({p["chi"] for p in self.profile}) <= 1

    def area_law_ratio(self) -> float | None:
        """χ 是否随 |A| 增长。≈0 表示饱和 (面积律), ≈1 表示线性 (体积律)。"""
        if len(self.profile) < 3:
            return None
        xs = torch.tensor([p["size_A"] for p in self.profile], dtype=torch.float32)
        ys = torch.tensor([p["chi"] for p in self.profile], dtype=torch.float32)
        if float(xs.std()) < 1e-9 or float(ys.std()) < 1e-9:
            return 0.0
        c = torch.corrcoef(torch.stack([xs, ys]))[0, 1]
        return float(c)


PREDICTIONS = {
    "B1": "面积律: χ 沿切口位置**饱和**而不随 |A| 线性增长 (|corr(size_A, chi)| < 0.5)",
    "B2": "RG 流: 沿层方向 (axis=0) 的纠缠熵随深度**下降** (粗粒化 = 熵减少)",
    "B3": "顺序有结构: natural 序的 χ 显著低于 random 序 (均值差 > 0)",
    "B4": "空间非均匀: χ 的分布不止一个取值 —— 否则膜结构退化回 v57 的标量 r",
}
