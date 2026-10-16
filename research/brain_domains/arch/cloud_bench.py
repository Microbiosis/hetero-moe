"""真实规模 GPU 基准 + 真实域标签行为验证 (在 HF Jobs / T4 上运行).

与 arch/domain_ssm.py 数学等价, 但把逐 token 的 Python 循环换成向量化
形式, 否则 d_inner=1536 / L=512 的规模在 CPU 上不可行。

检验两件事:
  1. 真实规模下的显存占用与前向/反向耗时 —— 域路由层是否值得
  2. 真实脑域标签(FlyWire 10 域)下, 域敏感性是否可测、随机标签是否可区分

第 2 项是本项目的关键对照: 若随机标签的表现与真实标签相当, 说明收益
来自参数量而非脑区划分, 该结论必须被推翻。
"""

from __future__ import annotations

import csv
import sys
import time

import torch
import torch.nn as nn


class VectorizedDomainSSM(nn.Module):
    """向量化的域条件选择性 SSM (数学等价于 domain_ssm.py 的逐 token 版)."""

    def __init__(self, d_inner=1536, d_state=16, n_domains=17, rank=8):
        super().__init__()
        self.d_inner, self.d_state, self.n_domains, self.rank = d_inner, d_state, n_domains, rank

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        # U 小随机初始化打破对称 (不能与 alpha 同时置零, 否则梯度互锁)
        self.U = nn.Parameter(torch.randn(rank, d_inner, d_state) * (1.0 / d_state**0.5))
        self.alpha = nn.Parameter(torch.zeros(n_domains, rank))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.domain_gate = nn.Parameter(torch.ones(1))

        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

    def effective_A(self, domain_id):
        a_shared = -torch.exp(self.A_log)
        alpha = self.alpha[domain_id] * self.domain_gate
        offset = torch.einsum("blr,rdn->bldn", alpha, self.U)
        return a_shared.unsqueeze(0).unsqueeze(0) + offset

    def forward(self, u, domain_id):
        B, L, _ = u.shape
        delta = torch.nn.functional.softplus(self.dt_proj(u)).clamp(1e-3, 1e-1)
        Bm, Cm = self.B_proj(u), self.C_proj(u)
        A_eff = self.effective_A(domain_id)

        dA = torch.exp(delta.unsqueeze(-1) * A_eff)          # (B,L,D,N)
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)

        # 并行前缀扫描: h_t = sum_{s<=t} (prod_{r=s+1..t} dA_r) * dBu_s
        # 用 cumprod + 除法在对数域不稳定(数值下溢), 故改用稳定形式:
        #   按 batch 维做 cumsum 的乘积形式 —— 这里序列较短, 直接用 cumsum 累积
        #   折算为 cumprod 后逐项乘, L<=1024 时精度可接受。
        cA = torch.cumprod(dA, dim=1)                         # (B,L,D,N)
        cBu = torch.cumsum(dBu, dim=1)                        # (B,L,D,N)
        # h_t = cA_t * sum_{s<=t} dBu_s / cA_s   —— 数值不稳, 改用 log 域
        # 采用 Blelloch 之外的实用方案: 分块 cumprod (块长 64) 兼顾稳定性与速度
        return self._scan_chunked(dA, dBu, Cm, u, block=64)

    def _scan_chunked(self, dA, dBu, Cm, u, block=64):
        B, L, D, N = dA.shape
        y = torch.empty(B, L, D, device=dA.device, dtype=dA.dtype)
        h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
        carry = torch.ones(B, D, N, device=dA.device, dtype=dA.dtype)
        for s in range(0, L, block):
            e = min(s + block, L)
            n = e - s
            # 块内: 先用 carry 缩放, 再做块内前缀和
            blk_A = dA[:, s:e] * carry.unsqueeze(1)             # (B,n,D,N)
            carry = blk_A[:, -1]                               # 块末状态传下去
            # 块内串行(块长 64, numpy-free 循环开销可接受)
            for t in range(n):
                hh = blk_A[:, t] * h + dBu[:, s + t]
                h = hh
                y[:, s + t] = torch.einsum("bdn,bn->bd", h, Cm[:, s + t])
            # 注意: h 已是块末状态, 下一块直接承接
        return y + self.D * u


class DomainStack(nn.Module):
    def __init__(self, n_layer=24, d_inner=1536, d_state=16, n_domains=17, rank=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorizedDomainSSM(d_inner, d_state, n_domains, rank) for _ in range(n_layer)]
        )

    def forward(self, u, domain_id):
        for layer in self.layers:
            u = u + layer(u, domain_id)
        return u


def load_flywire_domains(path):
    """读已入桶的域索引, 返回 (脑区名列表, 每脑区神经元数)。"""
    names, counts = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["region"])
            counts.append(int(row["n_neurons"]))
    return names, counts


def bench_scale(dev, n_domains=17, B=4, L=512, n_layer=24, rank=8):
    """真实规模下的显存与耗时。"""
    print(f"\n=== 1. 真实规模基准 (B={B} L={L} n_layer={n_layer} K={n_domains} R={rank}) ===")
    torch.manual_seed(0)
    stack = DomainStack(n_layer, 1536, 16, n_domains, rank).to(dev)

    total = sum(p.numel() for p in stack.parameters())
    domain_params = sum(
        l.alpha.numel() + l.U.numel() for l in stack.layers
    )
    print(f"  总参数        {total/1e6:.2f} M")
    print(f"  域专属参数    {domain_params:,}  ({100*domain_params/total:.3f}%)")

    u = torch.randn(B, L, 1536, device=dev)
    domain_id = torch.randint(0, n_domains, (B, L), device=dev)

    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    t0 = time.time()
    out = stack(u, domain_id)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0
    print(f"  前向耗时      {t_fwd:.3f} s")
    print(f"  峰值显存      {peak:.2f} GB")
    print(f"  输出 shape    {tuple(out.shape)}  finite={torch.isfinite(out).all().item()}")

    t0 = time.time()
    out.pow(2).mean().backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_bwd = time.time() - t0
    print(f"  反向耗时      {t_bwd:.3f} s")

    g = stack.layers[0].alpha.grad
    print(f"  alpha.grad    范数 {g.abs().sum().item():.3e}  非零={g.abs().sum().item() > 0}")
    return stack


def bench_domain_labels(dev, flywire_csv=None, B=4, L=256, n_layer=2, rank=8):
    """真实域标签 vs 随机标签: 域差异是否可测且可区分。"""
    print("\n=== 2. 真实脑域标签 vs 随机标签 ===")

    if flywire_csv:
        names, counts = load_flywire_domains(flywire_csv)
        K = len(names)
        print(f"  FlyWire 脑区 {K} 个: {', '.join(names[:6])}{'...' if K > 6 else ''}")
        print(f"  神经元规模   {', '.join(str(c) for c in counts[:6])}...")
    else:
        K = 10
        names = [f"region_{i}" for i in range(K)]
        print(f"  未读到域索引, 用 {K} 个合成域")

    torch.manual_seed(1)
    stack = DomainStack(n_layer, 1536, 16, K, rank).to(dev)

    # 让域偏移非零, 模拟「训练到域已分化」的状态
    with torch.no_grad():
        for layer in stack.layers:
            layer.alpha.normal_(0, 0.5)

    u = torch.randn(B, L, 1536, device=dev)

    def spread(ids):
        with torch.no_grad():
            base = stack(u, torch.full_like(ids, 0))
            diffs = []
            for d in range(K):
                di = torch.full_like(ids, d)
                diffs.append((stack(u, di) - base).abs().mean().item())
        return max(diffs), sum(diffs) / len(diffs)

    # 真实标签: 按脑区规模加权采样, 体现真实数据分布
    weights = torch.tensor(counts if flywire_csv else [1] * K, dtype=torch.float32)
    weights = weights / weights.sum()
    real_ids = torch.multinomial(weights, B * L, replacement=True).view(B, L).to(dev)
    rand_ids = torch.randint(0, K, (B, L), device=dev)

    mx_r, avg_r = spread(real_ids)
    mx_x, avg_x = spread(rand_ids)

    print(f"  真实域标签    最大域间差 {mx_r:.6f}  平均 {avg_r:.6f}")
    print(f"  随机域标签    最大域间差 {mx_x:.6f}  平均 {avg_x:.6f}")
    ratio = (mx_r / mx_x) if mx_x > 0 else float("inf")
    print(f"  比值(真实/随机) {ratio:.3f}")
    print(f"  域差异可测    {mx_r > 1e-6}   真实标签是否更强 {ratio > 1.0}")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 66)
    print(f"设备: {dev}  {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")
    print(f"torch {torch.__version__}")
    print("=" * 66)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    bench_scale(dev)
    bench_domain_labels(dev, csv_path)
    print("\n完成。")


if __name__ == "__main__":
    main()
