"""真实规模 GPU 基准 + 域对照 (HF Jobs / T4) — v4 堆叠稳定性修复版.

=== 修复链 ===
v2 (df9dd6a 原版) 首跑暴露:
  (a) 内存 OOM — 全程物化 (B,L,D,N) 张量, 24 层 autograd 超 T4
  (b) 数值不等价 — carry 机制在块边界双重缩放 (数值推演偏差 1.69)

v3 修复 (a)(b) + 域偏移保号:
  - 分块扫描 + checkpoint → 峰值 0.72GB
  - 删除 carry, 块间直传 h → 与参考逐位一致 (max|diff|=0)
  - 域偏移改为保号参数化: A_d = -(exp(A_log)·φ), φ=exp(tanh(offset)·c)
    → A_d < 0 恒成立 (v2 的 A_d>0 曾致输出 1e19 + 梯度 nan)
  - 但 v3 重跑暴露 (c): 24 层堆叠在 alpha=0 (无域) 时仍 nan

v4 修复 (c) 堆叠稳定性 (纯数值推演定位 + 验证, 未占本地资源):
  - 根因: 堆叠 u = u + layer(u), 层输出含 D⊙u (D=1 跳连)
    → 每层近似翻倍 + 二次反馈加速 → ~12 层 NaN
    (推演: 无修复时层 9 达 6e18, 层 12 NaN — 与 T4 首跑一致)
  - 方案对比 (数值推演):
      裸堆叠: NaN | 仅 out_proj: 927 (仍增长) | out_proj+残差缩放: 0.40 ✓
  - 采用: ① 层输出经 out_proj (nn.Linear, bias=False, 默认初始化)
           —— 对应真实 mamba block 的 out_proj 收缩角色 (原基准缺失)
           ② 残差缩放 1/√(2·n_layer) (GPT-2 标准做法)
  - 附带: 参数从 63.19M → 119.8M (更接近真实 mamba-130m 量级)

=== 本版检验 ===
  1. 真实规模下的显存/耗时/可训练性 (B 扫描 8/4/2/1)
  2. 域敏感性 + 有界性 (init 阶段)
  注: 真实/随机标签的判别实验须在训练后进行 (init 阶段因对称性不可定义),
      本基准不产出该比较; 协议见 research/brain_domains/README.md 陷阱 #4。
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp


class StableDomainSSM(nn.Module):
    """保号域条件 SSM + out_proj (v4)。

    A_d = -(exp(A_log) * φ(offset)),  φ = exp(tanh(offset) * c),  A_d < 0 恒成立。
    层输出 = out_proj(ssm + D⊙u), 堆叠侧再做 1/√(2N) 残差缩放。
    """
    C_BOUND = float(os.environ.get("DOMAIN_C_BOUND", "0.3"))

    def __init__(self, d_inner=1536, d_state=16, n_domains=17, rank=8,
                 dt_min=1e-3, dt_max=1e-1, use_out_proj=True):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.n_domains, self.rank = n_domains, rank
        self.dt_min, self.dt_max = dt_min, dt_max
        self.use_out_proj = use_out_proj

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.U = nn.Parameter(torch.randn(rank, d_inner, d_state) * (1.0 / d_state**0.5))
        self.alpha = nn.Parameter(torch.zeros(n_domains, rank))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.domain_gate = nn.Parameter(torch.ones(1))

        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        # out_proj: 真实 mamba block 的收缩投影 (原基准缺失 → 堆叠翻倍爆炸)
        self.out_proj = nn.Linear(d_inner, d_inner, bias=False) if use_out_proj else None

    def effective_A(self, domain_id):
        """(B, L) -> (B, L, D, N); A_eff < 0 恒成立。"""
        a_shared = -torch.exp(self.A_log)
        alpha = self.alpha[domain_id] * self.domain_gate
        offset = torch.einsum("blr,rdn->bldn", alpha, self.U)
        phi = torch.exp(torch.tanh(offset) * self.C_BOUND)
        return a_shared.unsqueeze(0).unsqueeze(0) * phi

    def forward(self, u, domain_id, block=64):
        B, L, D = u.shape
        delta = F.softplus(self.dt_proj(u)).clamp(self.dt_min, self.dt_max)
        Bm, Cm = self.B_proj(u), self.C_proj(u)

        y = torch.empty(B, L, D, device=u.device, dtype=u.dtype)
        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)

        for s in range(0, L, block):
            e = min(s + block, L)
            A_blk = self.effective_A(domain_id[:, s:e])
            dA_blk = torch.exp(delta[:, s:e].unsqueeze(-1) * A_blk)
            dBu_blk = (delta[:, s:e].unsqueeze(-1)
                       * Bm[:, s:e].unsqueeze(2)
                       * u[:, s:e].unsqueeze(-1))
            for t in range(e - s):
                h = dA_blk[:, t] * h + dBu_blk[:, t]
                y[:, s + t] = torch.einsum("bdn,bn->bd", h, Cm[:, s + t])

        y_raw = y + self.D * u
        return self.out_proj(y_raw) if self.use_out_proj else y_raw


class DomainStack(nn.Module):
    def __init__(self, n_layer=24, cls=StableDomainSSM, **kw):
        super().__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([cls(**kw) for _ in range(n_layer)])
        # GPT-2 式残差缩放 (推演验证: 与 out_proj 组合后 24 层保持有界)
        self.residual_scale = 1.0 / math.sqrt(2 * n_layer)

    def forward(self, u, domain_id, use_checkpoint=True):
        for layer in self.layers:
            if use_checkpoint and torch.is_grad_enabled():
                u = u + self.residual_scale * cp(layer, u, domain_id, use_reentrant=False)
            else:
                u = u + self.residual_scale * layer(u, domain_id)
        return u


def load_flywire_domains(path):
    names, counts = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["region"])
            counts.append(int(row["n_neurons"]))
    return names, counts


def verify_equivalence(dev):
    """§0: (i) SSM 级等价性 — 关域+关 out_proj 时与参考逐位一致;
             (ii) 24 层堆叠稳定性 — 满配 (out_proj+缩放) 下有界。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from domain_ssm import DomainConditionedSSM

    print("\n=== 0. 对拍与稳定性 ===")
    torch.manual_seed(7)
    D_INNER, N, K, R = 64, 8, 5, 4
    B, L = 2, 32

    # (i) 等价性: 关 out_proj, alpha=0
    ref = DomainConditionedSSM(D_INNER, N, K, R).to(dev).eval()
    new = StableDomainSSM(D_INNER, N, K, R, use_out_proj=False).to(dev).eval()
    new.load_state_dict({k: v for k, v in ref.state_dict().items()
                         if k in new.state_dict()}, strict=False)

    u = torch.randn(B, L, D_INNER, device=dev) * 0.1
    domain_id = torch.randint(0, K, (B, L), device=dev)

    with torch.no_grad():
        y_ref = ref(u, domain_id)
        y_new0 = new(u, domain_id, block=8)
        d0 = (y_ref - y_new0).abs().max().item()
        # A_eff 全负检查
        A_eff = new.effective_A(domain_id)
        n_pos = (A_eff > 0).sum().item()
        # (ii) 24 层满配堆叠稳定性 (小尺寸)
        stack_small = DomainStack(24, cls=StableDomainSSM, d_inner=D_INNER, d_state=N,
                                  n_domains=K, rank=R).to(dev).eval()
        for layer in stack_small.layers:
            layer.alpha.normal_(0, 0.5)
        y_stack = stack_small(u, domain_id, use_checkpoint=False)
        m_stack = y_stack.abs().max().item()

    print(f"  SSM 级等价 (alpha=0)  max|diff| = {d0:.3e}  ({'PASS' if d0 < 1e-5 else 'FAIL'})")
    print(f"  A_eff > 0 元素数      {n_pos}  ({'PASS 全负' if n_pos == 0 else 'FAIL'})")
    print(f"  24层堆叠 (小尺寸)     |max| = {m_stack:.3e}  ({'PASS 有界' if m_stack < 1e2 else 'FAIL'})")
    return d0 < 1e-5 and n_pos == 0 and m_stack < 1e2


def _bench_one(dev, n_domains, B, L, n_layer, rank):
    torch.manual_seed(0)
    stack = DomainStack(n_layer, cls=StableDomainSSM, d_inner=1536, d_state=16,
                        n_domains=n_domains, rank=rank).to(dev)

    total = sum(p.numel() for p in stack.parameters())
    domain_params = sum(l.alpha.numel() + l.U.numel() for l in stack.layers)

    u = torch.randn(B, L, 1536, device=dev) * 0.1
    domain_id = torch.randint(0, n_domains, (B, L), device=dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = stack(u, domain_id)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.time() - t0
    out_max = out.abs().max().item()
    peak_fwd = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0

    t0 = time.time()
    out.pow(2).mean().backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_bwd = time.time() - t0
    peak_bwd = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0

    g = stack.layers[0].alpha.grad
    gn = g.abs().sum().item() if g is not None else 0.0

    ok = math.isfinite(out_max) and math.isfinite(gn) and gn > 0
    print(f"  [B={B} L={L}] 参数 {total/1e6:.2f}M (域 {100*domain_params/total:.3f}%) | "
          f"前向 {t_fwd:.2f}s 峰值 {peak_fwd:.2f}GB | 反向 {t_bwd:.2f}s 峰值 {peak_bwd:.2f}GB | "
          f"输出|max| {out_max:.3e} | alpha.grad {gn:.3e} {'OK' if ok else 'BAD!!'}")
    del stack, u, domain_id, out
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def bench_scale(dev, n_domains=17, L=512, n_layer=24, rank=8, batches=(8, 4, 2, 1)):
    print(f"\n=== 1. 真实规模基准 (L={L} n_layer={n_layer} K={n_domains} R={rank}) ===")
    for B in batches:
        try:
            _bench_one(dev, n_domains, B, L, n_layer, rank)
        except torch.cuda.OutOfMemoryError:
            print(f"  [B={B} L={L}] OOM — 超出显存上限")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [B={B} L={L}] 异常: {type(e).__name__}: {e}")


def bench_domain_sensitivity(dev, flywire_csv=None, B=4, L=256, n_layer=2, rank=8):
    """域敏感性与有界性 (init 阶段)。

    说明: 真实/随机标签的"表现"判别实验只能在训练后进行 — init 阶段模型对
    id 的响应由随机初始化的 alpha 决定, 与 id 序列是否符合真实分布无关
    (对称性), 强行比较会产出误导性数字。本函数只报告可定义的两项。
    """
    print("\n=== 2. 域敏感性与有界性 (init 阶段) ===")

    if flywire_csv:
        names, counts = load_flywire_domains(flywire_csv)
        K = len(names)
        print(f"  FlyWire 脑区 {K} 个: {', '.join(names[:6])}{'...' if K > 6 else ''}")
        print(f"  神经元规模   {', '.join(str(c) for c in counts[:6])}...")
    else:
        K = 10
        names, counts = [f"region_{i}" for i in range(K)], [1] * K
        print(f"  未读到域索引, 用 {K} 个合成域")

    torch.manual_seed(1)
    stack = DomainStack(n_layer, cls=StableDomainSSM, d_inner=1536, d_state=16,
                        n_domains=K, rank=rank).to(dev)

    with torch.no_grad():
        for layer in stack.layers:
            layer.alpha.normal_(0, 0.5)   # 模拟「训练到域已分化」的状态

    u = torch.randn(B, L, 1536, device=dev) * 0.1

    with torch.no_grad():
        base = stack(u, torch.zeros(B, L, dtype=torch.long, device=dev))
        diffs = []
        maxes = []
        for d in range(K):
            di = torch.full((B, L), d, dtype=torch.long, device=dev)
            yd = stack(u, di)
            diffs.append((yd - base).abs().mean().item())
            maxes.append(yd.abs().max().item())
        # 逐 token 混采序列 (体现真实标签是逐 token 变化的)
        mix_ids = torch.randint(0, K, (B, L), device=dev)
        y_mix = stack(u, mix_ids)
        d_mix = (y_mix - base).abs().mean().item()
        maxes.append(y_mix.abs().max().item())

    mx_all = max(maxes)
    print(f"  [敏感性] 单域偏移 (K={K} 遍历): 最大 |Δ输出| = {max(diffs):.3e}  平均 = {sum(diffs)/len(diffs):.3e}  可测: {max(diffs) > 1e-6}")
    print(f"  [敏感性] 混采序列 vs 全零:       |Δ输出| = {d_mix:.3e}  可测: {d_mix > 1e-6}")
    print(f"  [有界性] 全模式输出 |max| = {mx_all:.3e}  有限且有界: {math.isfinite(mx_all) and mx_all < 1e2}")
    print(f"  [对照说明] 真实 vs 随机标签判别须在训练后进行 (init 对称性不可定义);")
    print(f"             协议: 真实/打乱标签分别训练, 比较下游指标 (README 陷阱 #4)。")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 66)
    print(f"设备: {dev}  {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")
    print(f"torch {torch.__version__}  c_bound={StableDomainSSM.C_BOUND}  (v4: +out_proj +残差缩放)")
    print("=" * 66)

    if not verify_equivalence(dev):
        print("\n对拍/稳定性失败 — 终止。")
        sys.exit(1)

    bench_scale(dev)
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    bench_domain_sensitivity(dev, csv_path)
    print("\n完成。")


if __name__ == "__main__":
    main()
