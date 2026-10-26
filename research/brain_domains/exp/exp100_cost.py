#!/usr/bin/env python3
"""exp100_cost.py — **只测成本结构，不出结果**：exp100 的单个 session 到底耗在哪。

背景：exp100 的分片（8 session/片、50 min 上限）**单 session 就要 ~18 min**
⇒ ⇒ 必须先知道是「联合 PCA 的 SVD」贵还是「可读性置换」贵，才知道该改哪。

测四项（**都在云端同一容器里跑** ⇒ 数字可比）：
1. `exp98_posneg_control.joint_pca`（= SVD(400 × (dA+dB))）—— 一次
2. **替代路线**：Gram 矩阵特征分解 `eigh(UcᵀUc)`（同样的 top-16 子空间，数值路线不同）
   ⇒ 并**对拍**两者的子空间差异（⇒ 若等价，就能换更快的路线）
3. `perm_p`（可读性置换，7 个带）
4. 整个「42 个有向对」的配对循环（按真实形状）
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/domains/scripts")
import exp98_posneg_control as E  # noqa: E402


def bench():
    out = {}
    rng = np.random.default_rng(0)
    n, dA, dB = 4130, 200, 250
    XA = rng.normal(size=(n, dA))
    XB = rng.normal(size=(n, dB))

    t0 = time.time()
    mu, W = E.joint_pca(XA, XB)
    t_svd = time.time() - t0
    print(f"[1] joint_pca (SVD) = {t_svd:.3f}s", flush=True)

    t0 = time.time()
    U = np.hstack([XA, XB])
    mu2 = U.mean(axis=0, keepdims=True)
    Uc = (U - mu2)[np.random.default_rng(0).choice(U.shape[0], 400, replace=False)]
    G = Uc.T @ Uc
    w, V = np.linalg.eigh(G)
    W2 = V[:, np.argsort(w)[::-1][:16]]
    t_eigh = time.time() - t0
    print(f"[2] Gram+eigh 路线 = {t_eigh:.3f}s", flush=True)

    # 对拍：两条路线的 top-16 子空间是否一致（用投影矩阵比较）
    m = min(W.shape[1], W2.shape[1])
    Q1, _ = np.linalg.qr(W[:, :m])
    Q2, _ = np.linalg.qr(W2[:, :m])
    # 子空间距离：||P1 - P2|| 的谱范数（0 = 完全一致）
    sub_diff = float(np.linalg.norm(Q1 @ Q1.T - Q2 @ Q2.T, 2))
    print(f"[3] 两条路线的 16 维子空间距离 ||P1-P2||_2 = {sub_diff:.3e}", flush=True)

    # perm_p（可读性）
    tr = np.zeros(n, dtype=bool)
    tr[: int(0.7 * n)] = True
    te = ~tr
    y = rng.integers(0, 118, n)
    Xtr, Xte = XA[tr], XA[te]
    t0 = time.time()
    E.perm_p(Xtr, y[tr], Xte, y[te], seed=0)
    t_perm = time.time() - t0
    print(f"[4] perm_p（一个带，d={dA}）= {t_perm:.3f}s", flush=True)

    # 整个配对循环（42 个有向对 ⇒ 84 次 joint_pca）
    Z = {"A": XA, "B": XB, "C": rng.normal(size=(n, 150))}
    bands = list(Z)
    t0 = time.time()
    n_pc = 0
    for A in bands:
        for B in bands:
            if A == B:
                continue
            mu, W = E.joint_pca(Z[A][: int(0.7 * n)], Z[B][: int(0.7 * n)])
            n_pc += 1
    t_pairs = time.time() - t0
    print(f"[5] 配对循环（{n_pc} 个有向对）= {t_pairs:.3f}s", flush=True)

    out = {"t_svd": t_svd, "t_eigh": t_eigh, "subspace_diff": sub_diff,
           "t_perm_one_band": t_perm, "t_pairs": t_pairs, "n_pairs": n_pc}
    print("===JSON===" + json.dumps(out), flush=True)
    print("===DONE===", flush=True)


if __name__ == "__main__":
    bench()
