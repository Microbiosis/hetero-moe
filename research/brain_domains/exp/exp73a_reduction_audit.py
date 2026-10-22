#!/usr/bin/env python3
"""exp73a — 方向审计：图级降维 ≡ 试次级 ridge，在**哪个方向**成立？

## 为什么要有这个脚本

exp72 的 docstring 断言「图级降维 ≡ 试次级 ridge（权重偏差 ~1e-16）」。
写 exp73（**解码**方向）时发现：这个等价性是**方向相关**的，必须逐方向验。

关键区分 —— **谁是设计、谁是响应**：

| 方向 | 设计矩阵 | 响应 | 试次级 Gram |
|---|---|---|---|
| **A**（编码，exp72）| 逐图像特征（同一图所有重复**共享**）| 逐试次 EEG | `n_rep · X̄ᵀX̄`（精确）|
| **B**（解码，exp73）| 逐试次 EEG 特征 | 逐图像特征（同一图所有重复**共享**）| `Σ_{i,t} x_{i,t}x_{i,t}ᵀ` = `n_rep·X̄ᵀX̄ + Δ` |

方向 A 里，试次级设计矩阵就是**把同一行重复 n_rep 次**，
Gram 精确等于 `n_rep·G_bar` ⇒ 等价性精确成立。

方向 B 里，设计矩阵的各行**不同**（逐试次 EEG），
`G_full = n_rep·G_bar + Δ`（Δ = 图内散布）⇒
若把 Gram 换成图级的 `G_bar`，就**漏掉了 Δ**，等价性不成立。

## 本脚本验证三件事

1. 方向 A 的等价性是否真的到机器精度；
2. 方向 B 的图级 Gram 偏差多大、用试次级 Gram 后是否回到机器精度；
3. 方向 B 的偏差是否足以改变**结论**（R² 与 200 选 1 准确率）。

## 结论（实测）

- 方向 A：等价性**精确成立**（~1.8e-15）⇒ **exp72 的断言正确，无需更正**。
- 方向 B：图级 Gram 权重相对偏差 ~0.10–0.13；
  单试次 200 选 1 准确率 **0.920（图级）vs 0.943（试次级）**；
  修正形式 `A0 = G_full/n_rep, λ' = λ/n_rep, RHS = X̄ᵀV` 与试次级精确一致（~1.2e-15）。

⇒ **exp73 必须用试次级 Gram。**

用法：`python exp73a_reduction_audit.py`
"""

import numpy as np
import scipy.linalg as sla

SEED = 0


def solve(A, B):
    return sla.solve(A, B, assume_a="pos", overwrite_a=True, check_finite=False)


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))


def rep(tag, a, b):
    print(f"  {tag:56s} 相对偏差 = {rel(a, b):.3e}")


def direction_a():
    """编码方向（exp72）：设计 = 逐图像（共享），响应 = 逐试次。"""
    rng = np.random.default_rng(SEED)
    n_img, n_rep, d, m = 400, 4, 48, 24
    print("=" * 86)
    print("方向 A：设计 = 逐图像特征（同一图共享），响应 = 逐试次 EEG  ← exp72")
    print("=" * 86)
    Xbar = rng.normal(size=(n_img, d))
    W_true = rng.normal(size=(d, m))
    Y_A = (np.repeat(Xbar, n_rep, axis=0) @ W_true
           + 0.30 * rng.normal(size=(n_img * n_rep, m)))
    Ybar = Y_A.reshape(n_img, n_rep, m).mean(axis=1)
    for lam in (0.1, 1.0, 10.0):
        W_img = solve((Xbar.T @ Xbar) + lam * np.eye(d), Xbar.T @ Ybar)
        Xtr = np.repeat(Xbar, n_rep, axis=0)
        W_tri = solve((Xtr.T @ Xtr) + (n_rep * lam) * np.eye(d), Xtr.T @ Y_A)
        rep(f"图级(λ={lam}) vs 试次级(λ={n_rep}·λ)", W_img, W_tri)


def direction_b():
    """解码方向（exp73）：设计 = 逐试次，响应 = 逐图像（共享）。"""
    rng = np.random.default_rng(SEED)
    n_img, n_rep, d, m = 400, 4, 48, 24
    print("\n" + "=" * 86)
    print("方向 B：设计 = 逐试次 EEG 特征，响应 = 逐图像特征  ← exp73")
    print("=" * 86)
    Xbase = rng.normal(size=(n_img, d))
    Xtrial = (Xbase[:, None, :] + 0.35 * rng.normal(size=(n_img, n_rep, d))).reshape(-1, d)
    # RHS 里的 X̄ 必须是**实际**试次均值，不是生成时的基均值
    Xbar = Xtrial.reshape(n_img, n_rep, d).mean(axis=1)
    V = rng.normal(size=(n_img, m))
    Vrep = np.repeat(V, n_rep, axis=0)

    G_full = Xtrial.T @ Xtrial
    G_bar = Xbar.T @ Xbar
    for lam in (0.1, 1.0, 10.0):
        W_tri = solve(G_full + lam * np.eye(d), Xtrial.T @ Vrep)
        W_img = solve(G_bar + lam * np.eye(d), Xbar.T @ V)
        rep(f"图级(G_bar) vs 试次级(G_full)  λ={lam}   ← 预期有偏差", W_img, W_tri)
        W_fix = solve((G_full / n_rep) + (lam / n_rep) * np.eye(d), Xbar.T @ V)
        rep(f"  修正 A0=G_full/n_rep, λ'={lam / n_rep:.4g}, RHS=X̄ᵀV", W_fix, W_tri)


def consequence():
    """方向 B 的偏差是否改变结论：R² 与 200 选 1 准确率。"""
    rng = np.random.default_rng(SEED)
    d = 48
    print("\n" + "=" * 86)
    print("方向 B 的 200 选 1：EEG 由图像特征生成（可判别），比较两种 Gram")
    print("=" * 86)
    n_img, n_rep, m = 200, 4, 24
    V = rng.normal(size=(n_img, m))
    A_map = rng.normal(size=(m, d)) / np.sqrt(m)
    Xtrial = (np.repeat(V, n_rep, axis=0) @ A_map
              + 0.8 * rng.normal(size=(n_img * n_rep, d)))
    Xbar = Xtrial.reshape(n_img, n_rep, d).mean(axis=1)
    img_of_trial = np.repeat(np.arange(n_img), n_rep)

    G_full = Xtrial.T @ Xtrial
    G_bar = Xbar.T @ Xbar
    for lam in (0.1, 1.0, 10.0, 100.0):
        W_tri = solve(G_full + lam * np.eye(d), Xtrial.T @ np.repeat(V, n_rep, axis=0))
        W_img = solve(G_bar + lam * np.eye(d), Xbar.T @ V)
        W_fix = solve((G_full / n_rep) + (lam / n_rep) * np.eye(d), Xbar.T @ V)
        out = []
        for W in (W_img, W_tri, W_fix):
            pred = Xtrial @ W
            acc1 = float(((pred @ V.T).argmax(axis=1) == img_of_trial).mean())
            avg = pred.reshape(n_img, n_rep, m).mean(axis=1)
            acc_avg = float(((avg @ V.T).argmax(axis=1) == np.arange(n_img)).mean())
            out.append(f"{acc1:.3f}/{acc_avg:.3f}")
        print(f"  λ={lam:6g}  图级(G_bar) 单试次/均值 = {out[0]}   "
              f"试次级(G_full) = {out[1]}   修正版 = {out[2]}")

    print("\n" + "=" * 86)
    print("结论")
    print("=" * 86)
    print("  方向 A：等价性**精确成立**（~1e-15）⇒ exp72 的断言正确，无需更正。")
    print("  方向 B：图级 Gram 漏掉图内散布 Δ ⇒ 权重与准确率都会偏。")
    print("           exp73 必须用 A0 = G_full / n_rep（RHS = X̄ᵀV，λ' = λ_tri/n_rep）。")


if __name__ == "__main__":
    direction_a()
    direction_b()
    consequence()
