"""V60 — 面积律的替代观测量：**逐 token 对的耦合强度 vs 距离**。

## 为什么换观测量

v58/v59 都用"切口谱 → 超阈通道数 χ"来刻画空间结构，两代都得到平凡结果。
v59 已经查出原因：

> χ 度量的是「两侧是否同源」，而同一个样本的所有 token 天然同源。
> 这个量对"空间怎么切"不敏感。

要问"token 轴上有没有空间结构"，观测量必须**对位置敏感**。
最小可用的定义是**局部性**（locality）：

    M[i, j]  = token i 与 token j 表征之间的耦合强度
    d        = |i − j|
    若存在空间结构 ⇒ M 随 d **衰减**

这不是面积律，但它才是"空间有几何"能被检验的最小陈述。面积律只是
局部性的一个推论；局部性不成立时，面积律必然平凡（正是 v58/v59 的处境）。

## 耦合强度

对 token i、j，各自按列标准化后得 W_i、W_j（[n, D]），
令 G_i = W_i W_iᵀ（n×n）。互协方差矩阵 W_iᵀ W_j 的非零奇异值
= sqrt(eig(G_i^{1/2} G_j G_i^{1/2}))。

取 **前 k 个奇异值之和** 作为耦合强度（k=1 即最强耦合方向，
k>1 更稳、抗单点噪声）。

## 判据（P1–P4，运行前固定）

    P0 正控:      纯噪声上 |corr(d, M̄(d))| < 0.3
    P1 距离衰减:  corr(d, M̄(d)) < −0.5
    P2 置换可分:  真实 corr 低于 40 次"打乱 token 顺序"零分布的 2.5% 分位
    P3 局部性比:  M̄(1) / M̄(S−1) > 1.5

**P0 不过 ⇒ 整表作废。P2 不过 ⇒ P1 可能只是单调趋势的副产品。**
P1 与 P2 同时过，才可说"token 轴存在局部结构"。
"""
import math

import torch


def _standardize_2d(X: torch.Tensor) -> torch.Tensor:
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    n = X.shape[0]
    return X / (X.norm(dim=0, keepdim=True) / math.sqrt(n)).clamp(min=1e-8)


def token_grams(X: torch.Tensor) -> torch.Tensor:
    """X: [n, S, D] → [S, n, n]，G[t] = W_t W_tᵀ（W_t 按列标准化）。"""
    n, S, D = X.shape
    W = _standardize_2d(X.reshape(n, S * D)).reshape(n, S, D)
    return torch.stack([W[:, t, :] @ W[:, t, :].T for t in range(S)], 0)


def pair_coupling(G_i: torch.Tensor, G_j: torch.Tensor, n: int,
                  topk: int = 3) -> float:
    """两个 token 的耦合强度 = 互协方差谱前 topk 项之和。"""
    lam, U = torch.linalg.eigh(G_i.clamp(min=0.0))
    lam = lam.clamp(min=0.0)
    G_half = (U * torch.sqrt(lam)) @ U.T
    ev = torch.linalg.eigvalsh(G_half @ G_j @ G_half).clamp(min=0.0)
    spec = torch.sqrt(ev) / max(n - 1, 1)
    vals = torch.sort(spec, descending=True).values
    return float(vals[:topk].sum())


def coupling_matrix(G: torch.Tensor, n: int, topk: int = 3) -> torch.Tensor:
    """[S, n, n] → [S, S] 对称耦合矩阵（对角为自耦合，分析时不用）。"""
    S = G.shape[0]
    M = torch.zeros(S, S)
    for i in range(S):
        M[i, i] = pair_coupling(G[i], G[i], n, topk)
        for j in range(i + 1, S):
            v = pair_coupling(G[i], G[j], n, topk)
            M[i, j] = v
            M[j, i] = v
    return M


def distance_profile(M: torch.Tensor):
    """[S,S] → (d 列表, M̄(d) 列表)，d = |i−j| = 1..S−1。排除对角。"""
    S = M.shape[0]
    ds, vals = [], []
    for d in range(1, S):
        pairs = [float(M[i, i + d]) for i in range(S - d)]
        ds.append(d)
        vals.append(sum(pairs) / len(pairs))
    return ds, vals


def corr(a, b) -> float:
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def permuted_profile(M: torch.Tensor, seed: int):
    """打乱 token 顺序后重算距离剖面。

    等价于同时重排 M 的行与列 —— 保持 M 的全部数值，只破坏
    "谁和谁相邻"这个信息。这是 P2 的零分布。
    """
    S = M.shape[0]
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(S, generator=g)
    Mp = M[p][:, p]
    ds, vals = distance_profile(Mp)
    return ds, vals


def center_over_sites(X: torch.Tensor) -> torch.Tensor:
    """去掉逐样本的公共成分：X[t] ← X[t] − mean_t X。"""
    return X - X.mean(dim=1, keepdim=True)
