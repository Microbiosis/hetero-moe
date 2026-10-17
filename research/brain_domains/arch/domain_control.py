"""域对照实验 — 长训练数值稳定修正版。

## 根因 (trace_nan.py 的诊断)
900 步训练中 `a_eff_min` 单调漂移: -16.0 (step1) → -25.7 (400) → -27.6 (600)。
`A_log` 是自由可训练参数, `exp(A_log)` 无上界 → 单步衰减 exp(-2.76) 看似无害,
但 block=32 时块内累积达 -88.3, exp(-88.3)=4.4e-39 已在 float32 次正规区,
`dBu/ea` 溢出 → NaN。eager 与 compile 同样中招 (465 vs 618 步), 证明与编译器无关。

## 修复
把状态转移矩阵的模长卡在物理上有意义的区间:
    A_shared = -exp(A_log).clamp(min=0.5, max=16)
下界 0.5: 最长记忆 ~2 个 token; 上界 16: 单步衰减 exp(-1.6)=0.2 (mamba 原始谱的上端)。
域调制 φ ∈ [exp(-c), exp(c)] = [0.74, 1.35]。
=> |a_eff| ≤ 16 × 1.35 = 21.6, 与训练无关地恒定有界。
=> block=32 时块内累积 ≥ -21.6 × 0.1 × 32 = -69.1, exp(-69.1) = 1.1e-30
   (float32 最小正规数 1.18e-38), 余 8 个数量级裕度。
block 保持 16 再翻倍裕度。

同时保留数值护栏: 每步检查 loss/grad 有限, 首次异常即报位置而非静默产出 NaN。
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from fast_scan import chunked_scan  # noqa: E402

K, STICKY, L = 8, 0.92, 80
N_TRAIN, N_EVAL, STEPS, BATCH, LR, EVAL_EVERY = 2048, 512, 900, 32, 1e-3, 300
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["real", "shuffled", "none"]
D_INNER, N_LAYER, RANK, D_STATE = 256, 4, 8, 16
BLOCK = 16
A_MIN, A_MAX = 0.5, 16.0


class FixedSSM(nn.Module):
    """关键改动: A_shared 钳制在 [A_MIN, A_MAX], 使 |a_eff| 恒有界。"""
    C_BOUND = 0.3
    LS_INIT = 1e-4

    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.log(
            torch.arange(1, D_STATE + 1, dtype=torch.float32).repeat(D_INNER, 1)))
        self.U = nn.Parameter(torch.randn(RANK, D_INNER, D_STATE) / math.sqrt(D_STATE))
        self.alpha = nn.Parameter(torch.zeros(K, RANK))
        self.D = nn.Parameter(torch.ones(D_INNER))
        self.domain_gate = nn.Parameter(torch.ones(1))
        self.dt_proj = nn.Linear(D_INNER, D_INNER)
        self.B_proj = nn.Linear(D_INNER, D_STATE, bias=False)
        self.C_proj = nn.Linear(D_INNER, D_STATE, bias=False)
        self.out_proj = nn.Linear(D_INNER, D_INNER, bias=False)
        self.pre_norm = nn.RMSNorm(D_INNER)
        self.gamma = nn.Parameter(torch.full((D_INNER,), self.LS_INIT))

    def a_shared(self):
        """有界: [A_MIN, A_MAX] 的负值。"""
        return -torch.exp(self.A_log).clamp(A_MIN, A_MAX)

    def forward(self, u, domain_id):
        B, L_, D = u.shape
        h_in = self.pre_norm(u)
        delta = F.softplus(self.dt_proj(h_in)).clamp(1e-3, 1e-1)
        Bm, Cm = self.B_proj(h_in), self.C_proj(h_in)

        alpha = self.alpha[domain_id] * self.domain_gate
        off = torch.einsum("blr,rdn->bldn", alpha, self.U)
        phi = torch.exp(torch.tanh(off) * self.C_BOUND)
        a_eff = self.a_shared().unsqueeze(0).unsqueeze(0) * phi
        log_dA = delta.unsqueeze(-1) * a_eff
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * h_in.unsqueeze(-1)

        y, _ = chunked_scan(log_dA, dBu, Cm, x0=None, block=BLOCK)
        return self.gamma * self.out_proj(y + self.D * h_in)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(1, D_INNER)
        self.layers = nn.ModuleList([FixedSSM() for _ in range(N_LAYER)])
        self.head = nn.Linear(D_INNER, 1)

    def forward(self, u, ids):
        h = self.inp(u.unsqueeze(-1))
        for layer in self.layers:
            h = h + layer(h, ids)
        return self.head(h).squeeze(-1)


def make_env(seed):
    g = torch.Generator().manual_seed(seed)
    return (0.30 + 0.68 * torch.rand(K, generator=g),
            0.30 + 1.20 * torch.rand(K, generator=g))


def gen_data(a, b, n_seq, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n_seq, L, dtype=torch.long)
    cur = torch.randint(0, K, (n_seq,), generator=g)
    for t in range(L):
        ids[:, t] = cur
        sw = torch.rand(n_seq, generator=g) < (1.0 - STICKY)
        cur = torch.where(sw, torch.randint(0, K, (n_seq,), generator=g), cur)
    u = torch.randn(n_seq, L, generator=g)
    x = torch.zeros(n_seq, L)
    prev = torch.zeros(n_seq)
    for t in range(L):
        d = ids[:, t]
        prev = a[d] * prev + b[d] * u[:, t]
        x[:, t] = prev
    return ids, u, x


def eval_mse(model, u_e, x_e, ids_e):
    model.eval()
    with torch.no_grad():
        tot = 0.0
        for s in range(0, u_e.shape[0], 128):
            e = min(s + 128, u_e.shape[0])
            tot += F.mse_loss(model(u_e[s:e], ids_e[s:e]), x_e[s:e],
                              reduction="sum").item()
        return tot / (u_e.shape[0] * L)


def train_one(seed, variant, dev, use_compile=True):
    t0 = time.time()
    torch.manual_seed(seed)
    model = Model().to(dev)
    if use_compile:
        model = torch.compile(model, mode="default")

    if variant == "none":
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        for layer in m.layers:
            layer.domain_gate.data.fill_(0.0)
            layer.domain_gate.requires_grad_(False)
            layer.alpha.requires_grad_(False)

    env_a, env_b = make_env(seed)
    ids_tr, u_tr, x_tr = gen_data(env_a, env_b, N_TRAIN, 10_000 + seed)
    ids_ev, u_ev, x_ev = gen_data(env_a, env_b, N_EVAL, 20_000 + seed)

    if variant == "shuffled":
        gp = torch.Generator().manual_seed(30_000 + seed)
        ids_tr = ids_tr.gather(1, torch.rand(N_TRAIN, L, generator=gp).argsort(dim=1))
        ids_ev = ids_ev.gather(1, torch.rand(N_EVAL, L, generator=gp).argsort(dim=1))
    elif variant == "none":
        ids_tr = torch.zeros_like(ids_tr)
        ids_ev = torch.zeros_like(ids_ev)

    ids_tr, u_tr, x_tr = ids_tr.to(dev), u_tr.to(dev), x_tr.to(dev)
    ids_ev, u_ev, x_ev = ids_ev.to(dev), u_ev.to(dev), x_ev.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    gb = torch.Generator().manual_seed(40_000 + seed)

    nan_at = None
    for step in range(1, STEPS + 1):
        model.train()
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=gb).to(dev)
        loss = F.mse_loss(model(u_tr[idx], ids_tr[idx]), x_tr[idx])
        if not torch.isfinite(loss):
            nan_at = step
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            nan_at = step
            break
        opt.step()

    final = eval_mse(model, u_ev, x_ev, ids_ev)
    tag = f"NaN@{nan_at}" if nan_at else "clean"
    print(f"  [seed {seed} | {variant:8s}] FINAL {final:.5f}  ({time.time()-t0:.0f}s, {tag})")
    return final, nan_at


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"域对照 (A 有界 + block=16) | {dev} "
          f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")
    print(f"A 钳制 [{A_MIN}, {A_MAX}] → |a_eff| ≤ {A_MAX * math.exp(FixedSSM.C_BOUND):.1f}, "
          f"block={BLOCK} 累积 ≥ {-A_MAX * math.exp(FixedSSM.C_BOUND) * 0.1 * BLOCK:.1f}")
    print(f"K={K} L={L} steps={STEPS} seeds={SEEDS}")
    print("=" * 70)

    t_all = time.time()
    res = {v: {} for v in VARIANTS}
    nans = {}
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        for v in VARIANTS:
            f, na = train_one(seed, v, dev)
            res[v][seed] = f
            if na:
                nans[f"{seed}_{v}"] = na

    print("\n" + "=" * 70)
    print("=== 汇总 (留出集 MSE) ===")
    print(f"{'seed':>4} | {'real':>9} | {'shuffled':>9} | {'none':>9}")
    for s in SEEDS:
        print(f"{s:>4} | {res['real'][s]:>9.5f} | {res['shuffled'][s]:>9.5f} | {res['none'][s]:>9.5f}")

    print("\n--- 配对比较 (Δ = 前者 − 后者) ---")
    stats = []
    valid = True
    for a, b in [("real", "shuffled"), ("real", "none"), ("shuffled", "none")]:
        d = [res[a][s] - res[b][s] for s in SEEDS]
        if not all(map(math.isfinite, d)):
            print(f"  {a} vs {b}: 含非有限值, 跳过")
            valid = False
            continue
        wins = sum(1 for x in d if x < 0)
        med, mean, std = statistics.median(d), statistics.mean(d), statistics.stdev(d)
        stats.append({"pair": f"{a}-{b}", "median": med, "mean": mean, "std": std, "wins": wins})
        print(f"  {a} vs {b}: 中位 {med:+.5f} | 均值 {mean:+.5f} ± {std:.5f} | 胜 {wins}/5")

    verdict = "未定 (有 NaN)" if nans else (
        "A 显著优于 B — 域对齐有增益" if stats and stats[0]["median"] < 0
        and stats[0]["wins"] >= 4 else "A ≈ B — 增益来自参数量")
    print(f"\n判据: {verdict}")
    print(f"NaN 记录: {nans if nans else '无'}")
    print(f"总耗时 {time.time() - t_all:.0f}s")

    print("\n--- JSON ---")
    print(json.dumps({"A_clamp": [A_MIN, A_MAX], "block": BLOCK,
                      "final_mse": {v: {str(s): res[v][s] for s in SEEDS} for v in VARIANTS},
                      "paired": stats, "nan": nans, "verdict": verdict,
                      "elapsed_s": round(time.time() - t_all, 1)}, ensure_ascii=False))
    print("完成。")


if __name__ == "__main__":
    main()
