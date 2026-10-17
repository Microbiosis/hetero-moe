"""向量化选择性扫描 — 块内并行 (log-space relative cumsum) + 块间串行。

目标: 替换 StableDomainSSM.forward 的逐 token Python 循环。
     现状: 每训练步 ~8 万次微内核启动, 大配置 (B=256,L=320) 15 s/步。
     方案: 块内向量化 (2 个 cumsum + 逐元素 exp), 块间串行 (块数 = L/block)。

数学 (严格等价):
    h_t = dA_t · h_{t-1} + dBu_t,   dA_t = exp(log_dA_t) ∈ (0, 1)
    块内展开 (块首状态 x_b, 块内相对累积 L_t = Σ_{r≤t} log_dA_r):
        h_t = exp(L_t)·x_b + Σ_{j≤t} exp(L_t − L_j)·dBu_j
            = exp(L_t)·( x_b + Σ_{j≤t} dBu_j·exp(−L_j) )
    => 两次 cumsum + 一次广播乘即可得到整个块的 h 序列。

数值边界:
    单步 log_dA = δ·A_eff,  δ ∈ [1e-3, 1e-1] (clamp),  |A_eff| ≤ d_state·φ_max
    (φ_max = exp(C_BOUND) ≈ 1.35)。d_state=16 时单步下界 ≈ −2.16。
    块长 B 的 −L_t 上界 ≈ 2.16·B:
        B=16 → 34.6 | B=32 → 69.1 | B=64 → 138 (> float32 exp 上限 ~88.7 ✗)
    默认 block=16 留安全裕度; block=32 仍安全; 64 仅在小 d_state 下可用。
"""

from __future__ import annotations

import torch


def chunked_scan(log_dA, dBu, Cm, x0=None, block=16):
    """块并行扫描。全部输入均为 log 域/线性域张量, 不做 exp/log 往返。

    参数
    ----
    log_dA : (B, L, D, N)  ≤ 0  (即 δ·A_eff, A_eff<0, δ>0)
    dBu    : (B, L, D, N)
    Cm     : (B, L, N)
    x0     : (B, D, N) 或 None — 序列起始状态

    返回
    ----
    y      : (B, L, D)   扫描输出 (不含 D 跳连 / out_proj)
    x_last : (B, D, N)   末状态
    """
    B, L, D, N = dBu.shape
    device, dtype = dBu.device, dBu.dtype
    x = torch.zeros(B, D, N, device=device, dtype=dtype) if x0 is None else x0

    ys = []
    for s in range(0, L, block):
        e = min(s + block, L)
        lrel = torch.cumsum(log_dA[:, s:e], dim=1)            # (B, n, D, N) ≤ 0
        w = dBu[:, s:e] * torch.exp(-lrel)                     # 放大系数 ≤ exp(2.16·n)
        s_acc = torch.cumsum(w, dim=1)                         # 块内累积 (B, n, D, N)
        h = torch.exp(lrel) * (x.unsqueeze(1) + s_acc)         # (B, n, D, N)
        ys.append(torch.einsum("bldn,bln->bld", h, Cm[:, s:e]))
        x = h[:, -1]                                           # 块末状态 → 下一块

    return torch.cat(ys, dim=1), x


def scan_reference(log_dA, dBu, Cm, x0=None):
    """逐 token 参考实现 (与 cloud_bench 原版逐步对应), 供对拍。"""
    B, L, D, N = dBu.shape
    device, dtype = dBu.device, dBu.dtype
    h = torch.zeros(B, D, N, device=device, dtype=dtype) if x0 is None else x0.clone()
    y = torch.empty(B, L, D, device=device, dtype=dtype)
    for t in range(L):
        h = torch.exp(log_dA[:, t]) * h + dBu[:, t]
        y[:, t] = torch.einsum("bdn,bn->bd", h, Cm[:, t])
    return y, h


def verify_equivalence(dev, seeds=(0, 1, 2), sizes=((2, 128), (3, 257), (1, 512)),
                       blocks=(8, 16, 32)):
    """对拍: chunked_scan vs scan_reference, 覆盖非整除长度与多块边界。

    判据: max|diff| < 1e-5 (相对输出量级)。失败应终止而非继续。
    """
    print("\n=== 对拍: chunked_scan vs 逐 token 参考 ===")
    worst = 0.0
    for seed in seeds:
        torch.manual_seed(seed)
        for B, L in sizes:
            D, N = 64, 8
            # 模拟真实范围: 单步 log_dA ∈ [-2.16, 0)
            delta = torch.rand(B, L, D, device=dev) * (0.1 - 1e-3) + 1e-3
            a_eff = -(torch.rand(B, L, D, N, device=dev) * 15.0 + 1.0)   # [-16, -1]
            log_dA = delta.unsqueeze(-1) * a_eff
            dBu = torch.randn(B, L, D, N, device=dev) * 0.1
            Cm = torch.randn(B, L, N, device=dev) * 0.1
            x0 = torch.randn(B, D, N, device=dev) * 0.1

            y_ref, x_ref = scan_reference(log_dA, dBu, Cm, x0)
            for blk in blocks:
                y_new, x_new = chunked_scan(log_dA, dBu, Cm, x0, block=blk)
                d_y = (y_ref - y_new).abs().max().item()
                d_x = (x_ref - x_new).abs().max().item()
                worst = max(worst, d_y, d_x)
                print(f"  seed{seed} B={B} L={L:4d} block={blk:2d}  "
                      f"max|Δy|={d_y:.3e}  max|Δx|={d_x:.3e}")
    ok = worst < 1e-5
    print(f"  最差偏差 {worst:.3e}  —  {'PASS 等价' if ok else 'FAIL 不等价!!'}")
    return ok


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ok = verify_equivalence(dev)
    raise SystemExit(0 if ok else 1)
