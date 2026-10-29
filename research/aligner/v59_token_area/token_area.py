"""V59 — 面积律，但空间轴是 token 位置（非任意选择）。

## 为什么要有这一代

v58 把 2560 维隐表示切成 16 个"块"当空间轴。那个轴是**任意的**：
隐维没有天然次序，怎么切都行。v58 的 B3 两臂给出完全相同的
均值与总容量（17.60 / 264），据此判定"面积律是任何二分的平凡性质"。

要谈"空间上的面积律"，必须用一个**非任意**的轴。token 位置是的：
它是序列，相邻就是相邻。这一代只在 token 轴上做。

## 与 v58 的两个方法学改动

1. **阈值不再用拟合公式**，改为**每切口就地零分布**：
   对同一个 (n, d_a, d_b)，用纯噪声重复 R 次算谱最大值，取分位数作 τ。
   前四版公式全部被推翻（见 v58 brane.noise_tau），根因是"一个公式覆盖
   所有 (n,d)"这个想法本身不稳；就地测量把这个假设彻底去掉。

2. **谱计算的 Gram 技巧**：token 轴上 d 可到 38400，直接算 AᵀB 是
   O(d_a·d_b)。改用 AAᵀ / BBᵀ（n×n）求同一组奇异值，n=192 时快两个数量级。

## 判决（预登记，运行时不可改）

    N1 顺序有结构: 至少 3/15 个切口上，natural 的 χ 落在置换零分布
                   [2.5%, 97.5%] 之外
    N2 面积律:     corr(|A|, χ_natural) < 0.5
    N3 非均匀:     χ 剖面 ≥ 3 种取值
    N4 正控:       纯噪声在全部切口上 χ = 0

**N1 不过 ⇒ N2 不可解释**（v58 的教训：平凡性质也会"过"）。
"""
import math

import torch


# ---------------------------------------------------------------- 谱

def _standardize(X: torch.Tensor) -> torch.Tensor:
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    n = X.shape[0]
    return X / (X.norm(dim=0, keepdim=True) / math.sqrt(n)).clamp(min=1e-8)


def cross_spectrum(A: torch.Tensor, B: torch.Tensor,
                   gram_threshold: float = 4.0e6) -> torch.Tensor:
    """互协方差谱（降序）。d_a·d_b 大时走 Gram 技巧。

    直接路径: svdvals(AᵀB/(n−1))          内存 O(d_a·d_b)
    Gram 路径: 令 G_a=AAᵀ, G_b=BBᵀ (均 n×n)
               AᵀB 的非零奇异值 = sqrt(eig(G_a G_b))
               而 G_a G_b 相似于 Lᵀ G_b L（G_a=LLᵀ，L 为 Cholesky 因子）
               ⇒ 只需 n×n 的特征分解
    """
    A = _standardize(A)
    B = _standardize(B)
    n = A.shape[0]
    da, db = A.shape[1], B.shape[1]

    if float(da) * float(db) <= gram_threshold:
        M = (A.T @ B) / max(n - 1, 1)
        return torch.linalg.svdvals(M)

    G_a = A @ A.T
    G_b = B @ B.T
    # d < n 时 G_a 必然奇异 ⇒ 不能用 Cholesky。走特征分解取半正定平方根：
    #   G_a = U Λ Uᵀ,  G_a^{1/2} = U Λ^{1/2} Uᵀ
    #   G_a G_b 相似于 G_a^{1/2} G_b G_a^{1/2}（对称半正定）
    lam, U = torch.linalg.eigh(G_a)
    lam = lam.clamp(min=0.0)
    G_half = (U * torch.sqrt(lam)) @ U.T
    ev = torch.linalg.eigvalsh(G_half @ G_b @ G_half).clamp(min=0.0)
    spec = torch.sqrt(ev) / max(n - 1, 1)
    return torch.sort(spec, descending=True).values


def noise_ceiling(n: int, da: int, db: int, reps: int = 14,
                  seed: int = 0, safety: float = 1.25) -> float:
    """**就地**零分布：同一 (n, da, db) 的纯噪声谱最大值 × 安全系数。

    这是 v59 相对 v58 的核心方法学改动 —— 不再依赖任何拟合公式，
    而是"用同样的形状量一遍噪声"。

    为什么用 max × safety 而不是分位数: 分位数在 reps 少时挡不住尾部，
    实测 reps=6/q=0.95 时 7 个切口里有 3 个各漏进 1 个假通道；
    改成 reps=10/safety=1.2 后，样本打乱对照上仍偶发漏进 1 个。
    现用 reps=14/safety=1.25。
    代价是灵敏度略降 —— 但实测 s_1/s_noise 有 5–30 倍余量，降得起。
    """
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(reps):
        A = torch.randn(n, da, generator=g)
        B = torch.randn(n, db, generator=g)
        vals.append(float(cross_spectrum(A, B).max()))
    return max(vals) * safety


def chi_at(A: torch.Tensor, B: torch.Tensor, tau: float) -> int:
    return int((cross_spectrum(A, B) > tau).sum())


# ---------------------------------------------------------------- 切口

def cut_tokens(X: torch.Tensor, k: int, idx: torch.Tensor | None = None):
    """X: [n, S, D] → (A, B)，A 取前 k 个 site（或其 idx 指定的 k 个）。

    idx=None 表示 natural 序（取 0..k−1）。
    """
    n, S, D = X.shape
    if idx is None:
        idx = torch.arange(k)
    rest = torch.tensor([j for j in range(S) if j not in set(idx.tolist())])
    A = X[:, idx, :].reshape(n, -1)
    B = X[:, rest, :].reshape(n, -1)
    return A, B


def profile(X: torch.Tensor, taus: list[float], order: str = "natural",
            seed: int = 0) -> list[int]:
    """沿 token 轴逐切口算 χ。order='random' 时每次取随机 k 子集。"""
    n, S, D = X.shape
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(1, S):
        if order == "natural":
            A, B = cut_tokens(X, k)
        else:
            idx = torch.randperm(S, generator=g)[:k]
            A, B = cut_tokens(X, k, idx)
        out.append(chi_at(A, B, taus[k - 1]))
    return out


def center_over_sites(X: torch.Tensor) -> torch.Tensor:
    """去掉**逐样本的公共成分**：X[t] ← X[t] − mean_t X。

    为什么需要: 同一文本的所有 token 共享"这段文本的语义"，这是个
    共同因（common cause）。它会同时出现在任何二分的两侧，于是 χ 与
    "在哪里切"无关 —— 面积律就成了任何二分的平凡性质。

    去掉它之后，剩下的才是 token **特有**的偏离。v59 把这一臂作为主检验。
    """
    return X - X.mean(dim=1, keepdim=True)


def shuffle_rows(B: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """把 B 的**样本行**打乱：B 与 A 不再来自同一文本。

    这是比"置换 token 顺序"更直接的对照 —— 它检验的正是
    "两侧的相关性是不是来自同一个样本"。若样本打乱后 χ 仍高，
    说明测到的不是内容共享，而是特征本身的结构。
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B.shape[0], generator=g)
    return B[perm]


def matched_ceiling(A: torch.Tensor, B: torch.Tensor, reps: int = 12,
                    seed: int = 0, safety: float = 1.3) -> float:
    """**结构匹配**的零分布：打乱 B 的样本行，保留两侧各自的列相关结构。

    ⚠️ 为什么不能用 iid 噪声当零分布（实测, n=192, 128×128, 列高度相关）:
        iid 天花板下，**样本已打乱**的数据仍读出 χ=4。即 iid 零分布
        系统性低估谱的尾部约 4 个通道 —— 因为 σ=1/√(n−1) 这个尺度假设
        列之间独立，而当 A 的列共享一个强成分时（每个 token 都是 base+noise），
        有效独立列数远小于 d，谱被压缩、尾部变重。

    这不是本代的特例：v58 用的是**公式**阈值，同样是"假设列独立"的零分布，
    因此 v58 的 χ 读数整体被高估约这个量级。

    匹配零分布只破坏 A–B 的对应关系，保留其余一切：
        打乱 B 的行 ⇒ 每列的相关结构不变, 但 A[i] 不再配 B[i]。
    """
    vals = []
    for r in range(reps):
        Bp = shuffle_rows(B, seed=seed + r)
        vals.append(float(cross_spectrum(A, Bp).max()))
    return max(vals) * safety


def chi_matched(A: torch.Tensor, B: torch.Tensor, reps: int = 12,
                seed: int = 0) -> tuple[int, float]:
    """用匹配零分布数通道。返回 (χ, τ)。"""
    tau = matched_ceiling(A, B, reps=reps, seed=seed)
    return int((cross_spectrum(A, B) > tau).sum()), tau


def chi_iid(A: torch.Tensor, B: torch.Tensor, tau: float) -> int:
    """用 iid 噪声天花板数通道（仅作对照，已知高估）。"""
    return int((cross_spectrum(A, B) > tau).sum())


# ---------------------------------------------------------------------------
# 加速: token 级 Gram 预计算
# ---------------------------------------------------------------------------
# 为什么需要: 朴素做法对每个子集都算一遍 AᵀB 或 AAᵀ（O(n²·d)）。
# 本代要跑 15 切口 × (置换 30 × 6 reps + 匹配 10 + 打乱 10) × 2 臂 × 8 层
# ≈ 5 万次谱计算；d 最大 38400 时每次 ~0.5s ⇒ 需要数小时。
#
# 关键恒等式: 按列标准化后
#     A_std A_stdᵀ = Σ_{t∈idx} W_t W_tᵀ        （W_t = 该 token 的标准化列）
# 每个 token 的 M_t = W_t W_tᵀ 只需算**一次**（n×n），
# 之后任意子集只是 n×n 矩阵相加，谱分解也是 n×n。
# 于是单次谱计算从 O(n²d) 降到 O(n³)，且 d 完全不出现在热路径上。

class TokenGrams:
    """把一个 (n, S, D) 张量预编译成 S 个 n×n Gram 矩阵。

    之后任何 token 子集的标准化 Gram 都是这些矩阵的和。
    """

    def __init__(self, X: torch.Tensor):
        n, S, D = X.shape
        self.n, self.S, self.D = n, S, D
        Xc = X - X.mean(dim=0, keepdim=True)          # 逐列中心化（跨样本）
        s = Xc.norm(dim=0) / math.sqrt(n)             # [S, D] 每列标准差
        W = Xc / s.clamp(min=1e-8).unsqueeze(0)       # 每列缩放到 ‖·‖=√n
        self.W = W
        self.M = [W[:, t, :] @ W[:, t, :].T for t in range(S)]   # 各 [n, n]

    def G(self, idx) -> torch.Tensor:
        G = self.M[0].clone()
        first = True
        for t in idx:
            if first:
                G = self.M[int(t)].clone(); first = False
            else:
                G = G + self.M[int(t)]
        return G

    def G_complement(self, idx) -> torch.Tensor:
        rest = [t for t in range(self.S) if t not in set(int(i) for i in idx)]
        return self.G(rest)

    def M_rest_sum(self, idx) -> torch.Tensor:
        return self.G_complement(idx)


def spec_from_grams(G_a: torch.Tensor, G_b: torch.Tensor, n: int) -> torch.Tensor:
    """由两个 n×n Gram 矩阵直接得到互协方差谱（降序）。"""
    lam, U = torch.linalg.eigh(G_a)
    lam = lam.clamp(min=0.0)
    G_half = (U * torch.sqrt(lam)) @ U.T
    ev = torch.linalg.eigvalsh(G_half @ G_b @ G_half).clamp(min=0.0)
    return torch.sort(torch.sqrt(ev) / max(n - 1, 1), descending=True).values


def chi_from_grams(G_a, G_b, n, tau) -> int:
    return int((spec_from_grams(G_a, G_b, n) > tau).sum())


def matched_tau_from_grams(G_a, G_b, n, reps: int = 10, seed: int = 0,
                           safety: float = 1.3) -> float:
    """匹配零分布（打乱 B 的样本行）在 Gram 表示下 = 置换 G_b 的行列。"""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(reps):
        p = torch.randperm(n, generator=g)
        G_bp = G_b[p][:, p]
        vals.append(float(spec_from_grams(G_a, G_bp, n).max()))
    return max(vals) * safety
