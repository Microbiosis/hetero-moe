"""训练阶段域对照实验: 真实 vs 打乱 vs 无域 (HF Jobs / T4).

协议 (README 陷阱 #4 + v47 配对纪律):
  - 受控任务: K=8 个功能域的粘性切换 AR(1) 过程, 每域独立系数 (a_d, b_d)
  - 三变体 (同环境 / 同初始权重 / 同批序, 逐种子配对):
      A real     — 真实域标签进入路由
      B shuffled — 每序列固定置换的域标签 (保边际分布, 破坏与动力学的对齐)
      C none     — 域路由完全禁用 (gate=0, alpha 冻结)
  - 5 种子 × 3 变体 = 15 次训练; 固定预算; 留出集收尾评估
  - 报告: 逐种子配对差 (中位数 / 均值 ± 样本标准差 ddof=1) + 胜出数
  - 判据: 若 A ≈ B, 增益来自参数量而非域划分, 结论须推翻重来

任务设计:
  域链:    sticky=0.92 的马尔可夫链, K=8, L=80
  动力学:  x_t = a_{d(t)} * x_{t-1} + b_{d(t)} * u_t
           a ~ U[0.30, 0.98], b ~ U[0.30, 1.50]  (每种子一套环境)
  输入:    u_t (随机驱动) [模型只看 u; 域信息仅经路由进入 — 干净单变量]
  目标:    预测 x_t
  留出:    同环境独立生成的 512 条序列

模型: StableDomainSSM v4 (cloud_bench.py 修复版) — 输入投影 + 4 层 + 输出头;
      d_inner=256, d_state=16, rank=8, K=8; 残差缩放 + checkpoint 默认开启
预算:  900 步, batch=32, AdamW lr=1e-3 wd=0.01, grad clip 1.0
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_bench import StableDomainSSM, DomainStack  # noqa: E402

K = 8
STICKY = 0.92
L = 80
N_TRAIN = 2048
N_EVAL = 512
STEPS = 900
BATCH = 32
LR = 1e-3
EVAL_EVERY = 300
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["real", "shuffled", "none"]


class Model(nn.Module):
    def __init__(self, d_inner=256, n_layer=4, n_domains=K, rank=8, d_state=16):
        super().__init__()
        self.inp = nn.Linear(1, d_inner)
        self.stack = DomainStack(n_layer, cls=StableDomainSSM, d_inner=d_inner,
                                 d_state=d_state, n_domains=n_domains, rank=rank)
        self.head = nn.Linear(d_inner, 1)

    def forward(self, u, ids):
        h = self.inp(u.unsqueeze(-1))          # (B,L,1) -> (B,L,D)
        h = self.stack(h, ids)
        return self.head(h).squeeze(-1)        # (B,L)


def make_env(seed):
    g = torch.Generator().manual_seed(seed)
    a = 0.30 + (0.98 - 0.30) * torch.rand(K, generator=g)
    b = 0.30 + (1.50 - 0.30) * torch.rand(K, generator=g)
    return a, b


def gen_data(a, b, n_seq, seed, sticky=STICKY):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n_seq, L, dtype=torch.long)
    cur = torch.randint(0, K, (n_seq,), generator=g)
    for t in range(L):
        ids[:, t] = cur
        switch = torch.rand(n_seq, generator=g) < (1.0 - sticky)
        new = torch.randint(0, K, (n_seq,), generator=g)
        cur = torch.where(switch, new, cur)

    u = torch.randn(n_seq, L, generator=g)
    x = torch.zeros(n_seq, L)
    prev = torch.zeros(n_seq)
    for t in range(L):
        d = ids[:, t]
        prev = a[d] * prev + b[d] * u[:, t]
        x[:, t] = prev
    return ids, u, x


def eval_mse(model, u_e, x_e, ids_e, dev):
    model.eval()
    with torch.no_grad():
        tot = 0.0
        for s in range(0, u_e.shape[0], 128):
            e = min(s + 128, u_e.shape[0])
            pred = model(u_e[s:e], ids_e[s:e])
            tot += torch.nn.functional.mse_loss(pred, x_e[s:e], reduction="sum").item()
        return tot / (u_e.shape[0] * L)


def train_variant(seed, variant, dev):
    """单变体单种子训练; 返回 (final_eval_mse, 曲线)。"""
    t0 = time.time()
    torch.manual_seed(seed)
    model = Model().to(dev)
    n_par = sum(p.numel() for p in model.parameters())

    if variant == "none":
        for layer in model.stack.layers:
            layer.domain_gate.data.fill_(0.0)
            layer.domain_gate.requires_grad_(False)
            layer.alpha.requires_grad_(False)

    env_a, env_b = make_env(seed)
    ids_tr, u_tr, x_tr = gen_data(env_a, env_b, N_TRAIN, seed=10_000 + seed)
    ids_ev, u_ev, x_ev = gen_data(env_a, env_b, N_EVAL, seed=20_000 + seed)

    if variant == "shuffled":
        gp = torch.Generator().manual_seed(30_000 + seed)
        perm = torch.rand(N_TRAIN, L, generator=gp).argsort(dim=1)
        ids_tr = ids_tr.gather(1, perm)
        perm_e = torch.rand(N_EVAL, L, generator=gp).argsort(dim=1)
        ids_ev = ids_ev.gather(1, perm_e)
    elif variant == "none":
        ids_tr = torch.zeros_like(ids_tr)
        ids_ev = torch.zeros_like(ids_ev)

    ids_tr, u_tr, x_tr = ids_tr.to(dev), u_tr.to(dev), x_tr.to(dev)
    ids_ev, u_ev, x_ev = ids_ev.to(dev), u_ev.to(dev), x_ev.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    gb = torch.Generator().manual_seed(40_000 + seed)   # 同种子同批序 (跨变体配对)
    curve = []
    for step in range(1, STEPS + 1):
        model.train()
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=gb).to(dev)
        pred = model(u_tr[idx], ids_tr[idx])
        loss = torch.nn.functional.mse_loss(pred, x_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            m = eval_mse(model, u_ev, x_ev, ids_ev, dev)
            curve.append((step, round(m, 5)))
            print(f"  [seed {seed} | {variant:8s}] step {step:4d}  train {loss.item():.4f}  eval {m:.5f}")
    final = eval_mse(model, u_ev, x_ev, ids_ev, dev)
    print(f"  [seed {seed} | {variant:8s}] FINAL eval {final:.5f}  ({time.time()-t0:.0f}s, {n_par/1e3:.0f}k params)")
    del model, u_tr, x_tr, ids_tr
    torch.cuda.empty_cache()
    return final, curve


def paired(name1, name2, vals):
    """逐种子配对: Δ = vals[name1] - vals[name2] (负 = name1 更好)。"""
    d = [vals[name1][s] - vals[name2][s] for s in SEEDS]
    n_win = sum(1 for x in d if x < 0)
    med = statistics.median(d)
    mean = statistics.mean(d)
    std = statistics.stdev(d) if len(d) > 1 else 0.0
    line = (f"  {name1} vs {name2}: Δ中位数 {med:+.5f} | Δ均值 {mean:+.5f} ± {std:.5f} | "
            f"{name1} 胜 {n_win}/{len(d)}")
    print(line)
    return {"pair": f"{name1}-{name2}", "median": med, "mean": mean, "std": std, "wins": n_win}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"域对照训练实验  |  设备 {dev} {torch.cuda.get_device_name(0) if dev.type=='cuda' else ''}")
    print(f"K={K} sticky={STICKY} L={L} steps={STEPS} batch={BATCH} lr={LR} seeds={SEEDS}")
    print("=" * 70)

    t_all = time.time()
    results = {v: {} for v in VARIANTS}
    curves = {v: {} for v in VARIANTS}
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        for variant in VARIANTS:
            final, curve = train_variant(seed, variant, dev)
            results[variant][seed] = final
            curves[variant][seed] = curve

    print("\n" + "=" * 70)
    print("=== 汇总 (留出集 MSE, 越低越好) ===")
    print(f"{'seed':>4} | {'A real':>10} | {'B shuffled':>10} | {'C none':>10}")
    for s in SEEDS:
        print(f"{s:>4} | {results['real'][s]:>10.5f} | {results['shuffled'][s]:>10.5f} | {results['none'][s]:>10.5f}")

    print("\n--- 逐种子配对比较 (Δ = 前者 − 后者; 负 = 前者更好) ---")
    stats = [
        paired("real", "shuffled", results),
        paired("real", "none", results),
        paired("shuffled", "none", results),
    ]

    print("\n--- 判据 ---")
    ab = stats[0]
    if ab["median"] < 0 and ab["wins"] >= 4:
        verdict = "A 显著优于 B — 域对齐有增益, 域结构设计得到支持"
    elif abs(ab["median"]) < 1e-3 and ab["wins"] <= 3:
        verdict = "A ≈ B — 增益来自参数量而非域划分, 结论须推翻重来"
    else:
        verdict = "中间态 — 需更多种子/步数或加强任务域结构后再判"
    print(f"  {verdict}")

    print("\n--- JSON 记录 ---")
    print(json.dumps({
        "config": {"K": K, "sticky": STICKY, "L": L, "steps": STEPS, "batch": BATCH, "seeds": SEEDS},
        "final_mse": {v: {str(s): results[v][s] for s in SEEDS} for v in VARIANTS},
        "curves": {v: {str(s): curves[v][s] for s in SEEDS} for v in VARIANTS},
        "paired": stats,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_all, 1),
    }, ensure_ascii=False))
    print("\n完成。")


if __name__ == "__main__":
    main()
