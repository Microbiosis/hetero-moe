#!/usr/bin/env python3
"""exp98_bench.py — **只测性能，不出结果**：云端 numpy 的 BLAS/LAPACK 到底多快，
以及「可读性置换检验」这一步占了多少。

背景：8 个分片 Job 跑 21 分钟还没算完**一个 session** ⇒ 要先定位瓶颈，
再决定「改运行方式」还是「改实现」（后者是**出结果前的正当修订**，但必须留痕）。

测三件事：
1. `np.linalg.solve`（d×d，d×K 右端）× 100 —— 置换循环里的固定部分能不能复用；
2. `Xtr.T @ (Y - ym)`（d×n @ n×K）× 100 —— 能不能用「按类求和」替掉；
3. 优化版 vs 原版：**(a) 数值是否一致 (b) 快多少**。
"""
import json
import time

import numpy as np


def bench():
    rng = np.random.default_rng(0)
    n, d, K, n_perm = 4130, 280, 118, 100
    X = rng.normal(size=(n, d))
    y = rng.integers(0, K, size=n)
    print(f"numpy {np.__version__}  threads? cfg:", flush=True)
    try:
        print(json.dumps(np.__config__.show(mode="dicts")["Build Dependencies"].keys().__iter__().__next__()), flush=True)
    except Exception as e:
        print("cfg?", e, flush=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    t0 = time.time()
    A = Xc.T @ Xc + np.eye(d)
    t_gemm = time.time() - t0
    print(f"[1] Xc.T@Xc ({d}x{n}@{n}x{d}) = {t_gemm:.3f}s", flush=True)
    t0 = time.time()
    for _ in range(n_perm):
        Y = np.eye(K)[y]
        ym = Y.mean(axis=0, keepdims=True)
        B = Xc.T @ (Y - ym)
    t_orig = time.time() - t0
    print(f"[2] 原版置换 RHS × {n_perm} = {t_orig:.3f}s", flush=True)
    t0 = time.time()
    for _ in range(n_perm):
        W = np.linalg.solve(A, B)
    t_solve = time.time() - t0
    print(f"[3] np.linalg.solve × {n_perm} = {t_solve:.3f}s", flush=True)
    t0 = time.time()
    Ainv = np.linalg.inv(A)
    W2 = Ainv @ B
    t_inv = time.time() - t0
    print(f"[4] inv+solve-替代 × {n_perm}? = {t_inv:.3f}s（只算了一次 inv）", flush=True)
    # 优化版：按类求和 ⇒ 置换只换列
    t0 = time.time()
    M = np.zeros((n, K))
    M[np.arange(n), y] = 1.0
    S = Xc.T @ M
    T = Xc.T @ (M.sum(axis=0, keepdims=True).T / n)
    t_setup = time.time() - t0
    print(f"[5] 优化版一次setup = {t_setup:.3f}s", flush=True)
    rng2 = np.random.default_rng(0)
    t0 = time.time()
    diffs = []
    for _ in range(5):
        p = rng2.permutation(y)
        # 原版
        Y = np.eye(K)[p]
        ym = Y.mean(axis=0, keepdims=True)
        B_orig = Xc.T @ (Y - ym)
        W_orig = np.linalg.solve(A, B_orig)
        # 优化版：等价 = 列置换
        inv_p = np.argsort(p)
        B_opt = (S - T)[:, inv_p]
        W_opt = np.linalg.solve(A, B_opt)
        d = float(np.max(np.abs(W_orig - W_opt)))
        rel = d / max(1e-30, float(np.max(np.abs(W_orig))))
        diffs.append((d, rel))
    t_opt = time.time() - t0
    print(f"[6] 优化版 5 次置换 = {t_opt:.3f}s；与原版 max|W差| = {diffs}", flush=True)
    print(json.dumps({"t_gemm": t_gemm, "t_orig_rhs": t_orig, "t_solve100": t_solve,
                      "t_setup_opt": t_setup, "t_opt5": t_opt,
                      "max_abs_W_diff": max(x[0] for x in diffs),
                      "max_rel_W_diff": max(x[1] for x in diffs)}, ensure_ascii=False), flush=True)
    print("===DONE===", flush=True)


if __name__ == "__main__":
    bench()
