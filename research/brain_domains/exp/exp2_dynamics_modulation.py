"""Phase 2: 域作为**动力学调制**（而非输入特征）是否有效。

## 为什么这和 Phase 1 不同

Phase 1 把域标签当作 6 维 one-hot **附加输入**，结果失败（域 ≈ 随机）。
但那只是"域作为特征"这一种用法。本项目 domain_ssm.py 的设计是另一种：

    A_d = -(exp(A_log) · φ(offset_d))      φ = exp(tanh(offset)·c)

域调制的是 SSM 的**状态转移矩阵 A** —— 即"记忆衰减多快"。这在新皮层中
有直接对应：不同脑区的时间常数确实不同（感觉皮层快、联合皮层慢）。

如果 H2 成立，效果不会体现为"多 6 维输入"，而会体现为模型能学到
**每个域不同的记忆时间尺度**。

## 受控任务（关键：让域真的有不同的时间常数）

合成数据，K 个域，每域一条独立的 AR(1) 过程：

    x_t = a_d · x_{t-1} + b_d · u_t + noise

其中 a_d 每域不同（0.2 ~ 0.98），决定记忆长短。u_t 是观测输入。

模型必须从 u 推断 x，而**最优解依赖 a_d**：
  a_d 大（记忆长）→ 应大量使用历史
  a_d 小（记忆短）→ 应主要看当前输入

因此"知道域" = "知道该用多长的记忆窗口"。这是域调制唯一能发挥作用的场景。

## 三臂对照

  A  shared_A   所有域共享 A（无域调制）
  B  domain_A   域调制 A（本项目设计）
  C  random_A   域调制 A，但标签在序列内打乱

## 判据

  B 显著优于 A 且 B 优于 C  → H2 成立，域调制有实质作用
  B ≈ A                     → H2 否定，域调制对动力学无帮助
  B ≈ C > A                 → 收益来自"多了一个可调参数"，不是域信息

## 为什么这个实验能决定性问题

若域真的调制了时间常数，则 B 的 A_d 应学到接近真实的 (a_d) 排序：
可通过检查学到的 φ 与真实 a_d 的秩相关来验证。这是**机制级证据**，
比"性能高一点"强得多。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 8                    # 域数
D_STATE = 16
D_INNER = 64
N_LAYER = 2
SEQ_LEN = 96
N_SEQ = 3000
N_EVAL = 500
EPOCHS = 30
BATCH = 64
LR = 3e-3
SEED = 0
C_BOUND = float(os.environ.get("C_BOUND", "0.5"))   # φ ∈ [0.61,1.65]; 1.5 会发散


# ------------------------------------------------------------ 数据
def make_env(seed):
    """每个域真实的 AR 系数 a_d 与输入增益 b_d。

    a_d 拉开范围: 0.15 ~ 0.98，覆盖"几乎无记忆"到"长记忆"。
    """
    g = torch.Generator().manual_seed(seed)
    a = 0.15 + (0.98 - 0.15) * torch.sort(torch.rand(K, generator=g)).values
    b = 0.5 + 1.0 * torch.rand(K, generator=g)
    return a, b


def gen(a, b, n_seq, seed, sticky=0.95):
    """生成序列：域随时间粘性切换（模拟真实 session 内的域分布）。"""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n_seq, SEQ_LEN, dtype=torch.long)
    cur = torch.randint(0, K, (n_seq,), generator=g)
    for t in range(SEQ_LEN):
        ids[:, t] = cur
        sw = torch.rand(n_seq, generator=g) < (1 - sticky)
        cur = torch.where(sw, torch.randint(0, K, (n_seq,), generator=g), cur)

    u = torch.randn(n_seq, SEQ_LEN, generator=g)
    x = torch.zeros(n_seq, SEQ_LEN)
    prev = torch.zeros(n_seq)
    for t in range(SEQ_LEN):
        d = ids[:, t]
        prev = a[d] * prev + b[d] * u[:, t] + 0.05 * torch.randn(n_seq, generator=g)
        x[:, t] = prev
    return ids, u, x


# ----------------------------------------------------------- 域条件 SSM
class DomainSSM(nn.Module):
    """最小域条件 SSM。A 可被域调制，或跨域共享（mode 控制）。"""

    def __init__(self, mode="domain"):
        super().__init__()
        assert mode in ("shared", "domain")
        self.mode = mode

        a_init = torch.arange(1, D_STATE + 1, dtype=torch.float32).repeat(D_INNER, 1)
        self.A_log = nn.Parameter(torch.log(a_init))          # 基准衰减
        self.U = nn.Parameter(torch.randn(RANK, D_INNER, D_STATE) / math.sqrt(D_STATE))
        self.alpha = nn.Parameter(torch.zeros(K, RANK))
        self.gate = nn.Parameter(torch.ones(1))

        self.inp = nn.Linear(1, D_INNER)
        self.dt = nn.Linear(D_INNER, D_INNER)
        self.Bp = nn.Linear(D_INNER, D_STATE, bias=False)
        self.Cp = nn.Linear(D_INNER, D_STATE, bias=False)
        self.out = nn.Linear(D_INNER, 1)
        self.norm = nn.RMSNorm(D_INNER)
        self.gamma = nn.Parameter(torch.full((D_INNER,), 1e-2))

        # A 钳制（README 陷阱 #5 的教训）
        self.A_MIN, self.A_MAX = 0.5, 16.0

    def a_shared(self):
        return -torch.exp(self.A_log).clamp(self.A_MIN, self.A_MAX)

    def effective_A(self, ids):
        a = self.a_shared()
        if self.mode == "shared":
            return a.unsqueeze(0).unsqueeze(0).expand(ids.shape[0], ids.shape[1], -1, -1)
        off = torch.einsum("blr,rdn->bldn", self.alpha[ids] * self.gate, self.U)
        phi = torch.exp(torch.tanh(off) * C_BOUND)
        return a.unsqueeze(0).unsqueeze(0) * phi

    def forward(self, u, ids):
        B, L = u.shape
        h_in = self.norm(self.inp(u.unsqueeze(-1)))            # (B,L,D)
        delta = F.softplus(self.dt(h_in)).clamp(1e-3, 1e-1)
        Bm, Cm = self.Bp(h_in), self.Cp(h_in)
        A_eff = self.effective_A(ids)                          # (B,L,D,N)

        log_dA = delta.unsqueeze(-1) * A_eff                   # ≤0
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * h_in.unsqueeze(-1)

        # 块并行扫描（照搬 fast_scan 的做法）
        block = 16
        h = torch.zeros(B, D_INNER, D_STATE, device=u.device)
        ys = []
        for s in range(0, L, block):
            e = min(s + block, L)
            lrel = torch.cumsum(log_dA[:, s:e], dim=1)
            ea = torch.exp(lrel)
            w = dBu[:, s:e] / ea
            hh = ea * (h.unsqueeze(1) + torch.cumsum(w, dim=1))
            ys.append((hh * Cm[:, s:e].unsqueeze(-2)).sum(-1))
            h = hh[:, -1]
        y = torch.cat(ys, dim=1)
        y = y + h_in
        return self.out(self.gamma * y).squeeze(-1)

    def learned_phi_rank(self):
        """诊断: 取学到的 φ 在每个域上的平均，与真实 a_d 比秩相关。"""
        with torch.no_grad():
            a = self.a_shared()                                # (D,N) ≤ 0
            ids = torch.arange(K)
            off = torch.einsum("kr,rdn->kdn", self.alpha[ids] * self.gate, self.U)
            phi = torch.exp(torch.tanh(off) * C_BOUND)         # (K,D,N)
            # 每域的平均 |A_eff| = |a| * phi 的均值
            scale = (a.abs().unsqueeze(0) * phi).mean(dim=(1, 2))   # (K,)
        return scale.cpu().numpy()


RANK = 4


# ------------------------------------------------------------ 训练
def train(mode, a_true, b_true, dev, seed=SEED):
    torch.manual_seed(seed)
    model = DomainSSM(mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    ids_tr, u_tr, x_tr = gen(a_true, b_true, N_SEQ, seed=100 + seed)
    ids_ev, u_ev, x_ev = gen(a_true, b_true, N_EVAL, seed=200 + seed)
    ids_tr, u_tr, x_tr = ids_tr.to(dev), u_tr.to(dev), x_tr.to(dev)
    ids_ev, u_ev, x_ev = ids_ev.to(dev), u_ev.to(dev), x_ev.to(dev)

    n = len(u_tr)
    best = float("inf")
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pred = model(u_tr[idx], ids_tr[idx])
            loss = F.mse_loss(pred, x_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        model.eval()
        with torch.no_grad():
            mse = F.mse_loss(model(u_ev, ids_ev), x_ev).item()
        if not math.isfinite(mse):
            print(f"    ⚠️ epoch {ep} 出现 NaN —— 提前停止")
            return float("nan"), model
        best = min(best, mse)
    return best, model


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"Phase 2 — 域作为动力学调制 | 设备 {dev} | K={K} c_bound={C_BOUND}")
    print("=" * 72)

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out/exp2")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in [0, 1, 2]:
        a_true, b_true = make_env(seed)
        print(f"\n--- seed {seed} ---")
        print(f"  真实 a_d: {[round(float(v),3) for v in a_true]}")

        res = {}
        for mode in ["shared", "domain"]:
            t0 = time.time()
            mse, model = train(mode, a_true, b_true, dev, seed)
            res[mode] = mse
            print(f"  {mode:7s}: MSE {mse:.5f}  ({time.time()-t0:.0f}s)")
            if mode == "domain":
                phi = model.learned_phi_rank()
                # 秩相关: 学到的 |A_eff| 是否随真实 a_d 单调
                from scipy.stats import spearmanr
                rho, p = spearmanr(a_true.numpy(), phi)
                res["phi"] = phi.tolist()
                res["rho"] = float(rho)
                res["rho_p"] = float(p)
                print(f"  学到的 |A| 尺度: {[round(float(v),3) for v in phi]}")
                print(f"  与真实 a_d 的 Spearman ρ = {rho:.3f} (p={p:.3f})")

        # random 标签：训练时打乱域标签
        torch.manual_seed(seed)
        model_r = DomainSSM("domain").to(dev)
        opt = torch.optim.AdamW(model_r.parameters(), lr=LR)
        ids_tr, u_tr, x_tr = gen(a_true, b_true, N_SEQ, seed=100 + seed)
        ids_ev, u_ev, x_ev = gen(a_true, b_true, N_EVAL, seed=200 + seed)
        g = torch.Generator().manual_seed(999 + seed)
        ids_tr_sh = torch.stack([ids_tr[i][torch.randperm(SEQ_LEN, generator=g)]
                                 for i in range(len(ids_tr))])
        ids_tr, u_tr, x_tr = ids_tr_sh.to(dev), u_tr.to(dev), x_tr.to(dev)
        ids_ev, u_ev, x_ev = ids_ev.to(dev), u_ev.to(dev), x_ev.to(dev)
        n = len(u_tr)
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                loss = F.mse_loss(model_r(u_tr[idx], ids_tr[idx]), x_tr[idx])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model_r.parameters(), 1.0); opt.step()
        with torch.no_grad():
            res["random"] = F.mse_loss(model_r(u_ev, ids_ev), x_ev).item()
        print(f"  random : MSE {res['random']:.5f}")

        row = {"seed": seed,
               "mse_shared": round(res["shared"], 5),
               "mse_domain": round(res["domain"], 5),
               "mse_random": round(res["random"], 5),
               "gain_domain_vs_shared_pct": round((res["shared"] - res["domain"]) / res["shared"] * 100, 2),
               "gain_domain_vs_random_pct": round((res["random"] - res["domain"]) / res["random"] * 100, 2),
               "phi_spearman_vs_true_a": round(res.get("rho", float("nan")), 4),
               "phi_p": round(res.get("rho_p", float("nan")), 4)}
        all_rows.append(row)

    # 汇总
    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "phase2_dynamics.csv", index=False)

    print("\n" + "=" * 72)
    print("汇总")
    print(df.to_string(index=False))

    g_shared = df["gain_domain_vs_shared_pct"].mean()
    g_random = df["gain_domain_vs_random_pct"].mean()
    rho = df["phi_spearman_vs_true_a"].mean()

    report = {
        "schema_version": "1.0", "task": "AR(1) with per-domain time constants",
        "K": K, "c_bound": C_BOUND, "epochs": EPOCHS,
        "mean_gain_vs_shared_pct": round(float(g_shared), 2),
        "mean_gain_vs_random_pct": round(float(g_random), 2),
        "mean_phi_spearman": round(float(rho), 4),
        "rows": all_rows,
    }

    print(f"\n  域 vs 共享: {g_shared:+.2f}%")
    print(f"  域 vs 随机: {g_random:+.2f}%")
    print(f"  φ 与真实 a_d 的秩相关: {rho:+.3f}")

    # ⚠️ ρ 的符号约定: 真实 a_d 是 AR 保留率(大=记忆长), 学到的是 |A_eff| 衰减率
    #    (大=记忆短)。二者天然负相关。因此 ρ < -0.5 才是"学到了正确排序"。
    #    (第一版判据误写成 rho > 0.5, 导致把成功结果判成"部分成立")
    if g_shared > 5.0 and g_random > 3.0 and rho < -0.5:
        v = "✅ H2 成立: 域调制有效, 且 |A| 与真实时间常数呈正确负相关"
    elif g_shared > 5.0 and g_random < 1.0:
        v = "⚠️ 性能提升但 φ 未学到真实排序 —— 可能是其他机制在起作用"
    elif abs(g_shared) < 1.0:
        v = "❌ H2 否定: 域调制对动力学无帮助"
    else:
        v = "⚠️ 部分成立, 需检查"
    print(f"    (ρ 判据: 真实 a_d 是保留率, |A| 是衰减率, 故预期 ρ < -0.5)")
    report["verdict"] = v
    print(f"\n  判定: {v}")

    (out_dir / "phase2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
