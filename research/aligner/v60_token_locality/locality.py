"""V60 — 面积律的替代观测量：**同一文本内、位置对的表征相似度 vs 距离**。

## 为什么换观测量

v58/v59 用"切口谱 → χ"两代都平凡。v59 查明：χ 度量「两侧是否同源」，
而同一文本的所有 token 天然同源 ⇒ 对"怎么切"不敏感。

问题不在轴，在**观测量**。要问"token 轴上有没有空间结构"，
观测量必须对位置敏感。最小陈述是**局部性**：

    M[i, j] = 位置 i 与位置 j 的表征相似度
    有空间结构 ⇒ M 随 |i − j| 衰减

## 两个版本，第一个被实测枪毙

### ✗ v1: 互协方差谱范数（`pair_coupling`，保留作对照）

对 token i、j 各自的 `W_i` 按列标准化（‖col‖=√n）后取 `W_iᵀ W_j` 的前 k 谱之和。

**实测失败**：植入 `X[t] = base + t·direction` 后剖面**完全平**
（`[3.549, 3.545, 3.545, 3.546, ...]`）。原因：

    逐列标准化把每列缩放到 ‖·‖=√n ⇒ **幅度信息被抹掉**
    X[t] = base + t·direction 标准化后只剩 sign(t − 7.5)
    ⇒ 耦合强度只跟"在均值哪一侧"有关，与距离无关

更一般地：谱范数型耦合被**共享分量**完全主导。所有位置都含 `base`，
于是任意两位置的耦合强度都≈base 的强度，与它们在哪儿无关。
这就是 v58/v59 平凡的同一个机制，只是换了个说法。

### ✓ v2: 逐样本文本内的余弦相似度（`cos_profile`）

    M[i, j] = (1/n) Σ_s  cos( X[s, i, :], X[s, j, :] )

这个量的好处：

- **共享分量只抬高截距，不改变斜率**。文本语义让所有位置对都相似（截距高），
  但"相邻是否比远距更相似"（斜率）由位置结构决定。
- 因此不需要去均值 —— 而去均值会引入符号问题（见下）。

**为什么不去均值**：`X'[t] = X[t] − mean_t X = (t − 7.5)·direction + noise'`，
此时 `cos(X'[i], X'[j])` 的符号由 `(i−7.5)(j−7.5)` 决定 ——
只反映"在均值两侧与否"，同样丢失距离信息。共同因必须靠截距吸收，不能消掉。

## 判据（P0–P3，运行前固定）

    P0 正控:      纯噪声上 |corr(d, M̄(d))| < 0.3
    P1 距离衰减:  corr(d, M̄(d)) < −0.5
    P2 置换可分:  真实 corr < 40 次「打乱 token 顺序」零分布的 2.5% 分位
    P3 局部性比:  M̄(1) − M̄(S−1) > 0.01（余弦差，绝对值）

**P0 不过 ⇒ 整表作废。P2 不过 ⇒ P1 可能只是单调漂移的副产品。**

## 为什么 P2 必要

因果 LM 的 token 表征是**因果累积**的（位置 t 汇聚 0..t 的信息）。
因果累积产生的是随位置的**单调漂移**，不是局部性 ——
而单调漂移会被 `corr(d, M̄(d))` 误读成衰减。

P2 把 M 的行列同时重排（保持全部数值，只破坏"谁和谁相邻"），
重算剖面。若真实 corr 不显著低于这个零分布，"相邻"这个信息就没起作用。
"""
import math

import torch


# ---------------------------------------------------------------- v2 (主)

def cos_matrix(X: torch.Tensor) -> torch.Tensor:
    """X: [n, S, D] → M[i,j] = 跨样本平均的余弦相似度（[S, S]）。"""
    n, S, D = X.shape
    Xf = X.float()
    nrm = Xf.norm(dim=2, keepdim=True).clamp(min=1e-8)
    U = Xf / nrm                                    # [n, S, D] 逐向量单位化
    M = torch.einsum("nsd,ntd->st", U, U) / n
    return M


def distance_profile(M: torch.Tensor):
    """[S,S] → (d 列表 1..S−1, M̄(d) 列表)，排除对角。"""
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
    """打乱 token 顺序 → M 的行列同重排。保持数值，只破坏相邻关系。"""
    S = M.shape[0]
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(S, generator=g)
    return distance_profile(M[p][:, p])


def cos_profile(X: torch.Tensor, n_perm: int = 40, seed: int = 0):
    """一条臂的完整读数。"""
    M = cos_matrix(X)
    ds, prof = distance_profile(M)
    r = corr(ds, prof)
    null = sorted(corr(ds, permuted_profile(M, seed=seed + 100 + t)[1])
                  for t in range(n_perm))
    lo = null[int(0.025 * (len(null) - 1))]
    return {
        "profile": [round(float(v), 6) for v in prof],
        "corr": round(r, 5),
        "null_lo": round(lo, 5),
        "null_mean": round(sum(null) / len(null), 5),
        "gap": round(float(prof[0] - prof[-1]), 6),
        "verdict_P1": r < -0.5,
        "verdict_P2": r < lo,
        "verdict_P3": (prof[0] - prof[-1]) > 0.01,
    }


# ---------------------------------------------------------------- v1 (对照, 已证伪)

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
    """v1 的耦合强度 = 互协方差谱前 topk 项之和。

    ⚠️ **已证伪**：对位置结构完全不敏感（见模块 docstring）。
       保留此函数只为让 `test_v1_spectral_coupling_is_blind` 能复现该盲区。
    """
    lam, U = torch.linalg.eigh(G_i.clamp(min=0.0))
    lam = lam.clamp(min=0.0)
    G_half = (U * torch.sqrt(lam)) @ U.T
    ev = torch.linalg.eigvalsh(G_half @ G_j @ G_half).clamp(min=0.0)
    vals = torch.sort(torch.sqrt(ev) / max(n - 1, 1), descending=True).values
    return float(vals[:topk].sum())


def coupling_matrix(G: torch.Tensor, n: int, topk: int = 3) -> torch.Tensor:
    S = G.shape[0]
    M = torch.zeros(S, S)
    for i in range(S):
        M[i, i] = pair_coupling(G[i], G[i], n, topk)
        for j in range(i + 1, S):
            v = pair_coupling(G[i], G[j], n, topk)
            M[i, j] = v
            M[j, i] = v
    return M
