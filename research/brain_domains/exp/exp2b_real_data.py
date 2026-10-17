"""Phase 2b: 在真实 OmniBrain 数据上验证域调制。

## 与 exp2 的关系

exp2 在**合成 AR(1)** 上证明: 域调制 A 能把 MSE 从 0.23-0.34 降到 0.009-0.020
(+94.6%), 且 |A_eff| 与真实时间常数呈正确负相关 (ρ=-0.60)。

但那是我构造的任务 ——"每域不同的时间常数"是我设定的。需要回答:

    **真实神经数据里, 不同 L1 域的时间常数是否真的不同?**

若不是, 域调制在真实数据上不会有收益（因为没有可学的差异）。

## 设计

数据: OmniBrain spike_view, 每 session 内按时间 70/30 切分。

三臂:
  A  shared_A   所有域共享衰减
  B  domain_A   域调制衰减（本项目设计）
  C  random_A   打乱域标签

**额外诊断**: 对每个 session, 用 exp2 的方法检查学到的
|A_eff| 排序是否与"该域活动的自相关时间"排序一致。

真实数据的"域时间常数"可从数据本身估计:
对每个域, 取其窗口序列在各成分上的自相关衰减率,
作为该域的"经验时间常数"。

## 判据

  B > A > C  → 真实数据上也成立, 且随机对照排除参数量
  B ≈ C      → 尽管合成任务成立, 真实数据无可学的域差异

## 预期

由于 OmniBrain 的域标签是**窗内主导域**（activity-weighted vote），
相邻窗口常属于同一域（时间粘性），域切换本身可能就携带了
"动力学状态变化"的信息。这一点在合成数据中不存在，
可能让效果更强或更弱 —— 需实测。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

L1 = ["sensory", "motor", "memory", "association", "subcortical", "other"]
K = len(L1)
D_STATE, D_INNER, RANK = 16, 64, 4
EPOCHS = 25
BATCH = 32
LR = 2e-3
SEED = 0
C_BOUND = float(os.environ.get("C_BOUND", "0.5"))
TRAIN_FRAC = 0.7
MIN_SEQ = 200


class DomainSSMReal(nn.Module):
    """输入 64 维（OmniBrain 的 64 个成分），逐时间步建模。"""

    def __init__(self, mode="domain", d_in=64):
        super().__init__()
        self.mode = mode
        a_init = torch.arange(1, D_STATE + 1, dtype=torch.float32).repeat(d_in, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.U = nn.Parameter(torch.randn(RANK, d_in, D_STATE) / math.sqrt(D_STATE))
        self.alpha = nn.Parameter(torch.zeros(K, RANK))
        self.gate = nn.Parameter(torch.ones(1))

        self.inp = nn.Linear(d_in, d_in)
        self.dt = nn.Linear(d_in, d_in)
        self.Bp = nn.Linear(d_in, D_STATE, bias=False)
        self.Cp = nn.Linear(d_in, D_STATE, bias=False)
        self.out = nn.Linear(d_in, d_in)
        self.norm = nn.RMSNorm(d_in)
        self.gamma = nn.Parameter(torch.full((d_in,), 1e-2))
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

    def forward(self, x, ids):
        """x: (B, L, d_in)  ids: (B, L)"""
        B, L, _ = x.shape
        h_in = self.norm(self.inp(x))
        delta = F.softplus(self.dt(h_in)).clamp(1e-3, 1e-1)
        Bm, Cm = self.Bp(h_in), self.Cp(h_in)
        A_eff = self.effective_A(ids)
        log_dA = delta.unsqueeze(-1) * A_eff
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * h_in.unsqueeze(-1)

        block = 16
        h = torch.zeros(B, x.shape[-1], D_STATE, device=x.device)
        ys = []
        for s in range(0, L, block):
            e = min(s + block, L)
            lrel = torch.cumsum(log_dA[:, s:e], dim=1)
            ea = torch.exp(lrel)
            hh = ea * (h.unsqueeze(1) + torch.cumsum(dBu[:, s:e] / ea, dim=1))
            ys.append((hh * Cm[:, s:e].unsqueeze(-2)).sum(-1))
            h = hh[:, -1]
        y = torch.cat(ys, dim=1) + h_in
        return self.out(self.gamma * y)          # (B, L, d_in)

    def phi_scale(self):
        with torch.no_grad():
            a = self.a_shared()
            off = torch.einsum("kr,rdn->kdn", self.alpha * self.gate, self.U)
            phi = torch.exp(torch.tanh(off) * C_BOUND)
            return (a.abs().unsqueeze(0) * phi).mean(dim=(1, 2)).cpu().numpy()


def build_sequences(w, y, seq_len=32):
    """切定长子序列。任务是自监督的**下一步 64 维表示预测**。

    w: (N, 64, 8)  →  x: (N, 64) 每窗 8 bin 均值
    返回 X: (S, L, 64)  Ynext: (S, L-1, 64)  yseq: (S, L-1) 域标签

    注意长度: 用 x[t] 预测 x[t+1], 所以 L 个输入对应 L-1 个目标。
    """
    x = w.mean(axis=2).astype(np.float32)         # (N, 64)
    n_seq = len(x) // seq_len
    if n_seq < 1:
        return None, None, None
    xs = x[:n_seq * seq_len].reshape(n_seq, seq_len, -1)
    ys = y[:n_seq * seq_len].reshape(n_seq, seq_len)
    X = xs[:, :-1]                                 # (S, L-1, 64)
    Yn = xs[:, 1:]                                 # (S, L-1, 64) ← 下一时刻
    yseq = ys[:, :-1]                              # (S, L-1)
    return X, Yn, yseq


def empirical_tau(w, y, n_dom):
    """经验时间常数: 每域窗口序列的自相关衰减速度。

    取 1 阶自相关 r1，则时间常数 τ ≈ -1/ln(r1)（AR(1) 近似）。r1 越大 → τ 越大。
    """
    out = np.full(n_dom, np.nan)
    x = w.mean(axis=2)                            # (N, 64)
    for d in range(n_dom):
        m = y == d
        if m.sum() < 10:
            continue
        z = x[m].astype(np.float64)                # float64 防溢出
        # 逐成分归一化后再算 lag-1 自相关 (避免大数值相乘溢出)
        sd = z.std(axis=0) + 1e-8
        z = (z - z.mean(axis=0)) / sd
        a, b = z[:-1], z[1:]
        num = (a * b).mean(axis=0)
        den = np.sqrt((a**2).mean(axis=0) * (b**2).mean(axis=0)) + 1e-8
        r1 = float(np.clip((num / den).mean(), 1e-6, 0.9999))
        out[d] = -1.0 / np.log(r1)
    return out


def train_eval(mode, Xtr, Ytr, idtr, Xte, Yte, idte, dev, seed=SEED):
    torch.manual_seed(seed)
    model = DomainSSMReal(mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    Xt = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(Ytr).to(dev)
    It = torch.from_numpy(idtr.astype(np.int64)).to(dev)
    Xv = torch.from_numpy(Xte).to(dev); Yv = torch.from_numpy(Yte).to(dev)
    Iv = torch.from_numpy(idte.astype(np.int64)).to(dev)

    n = len(Xt)
    best = float("inf")
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.mse_loss(model(Xt[idx], It[idx]), Yt[idx])
            if not torch.isfinite(loss):
                return float("nan"), model
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        model.eval()
        with torch.no_grad():
            m = F.mse_loss(model(Xv, Iv), Yv).item()
        if not math.isfinite(m):
            return float("nan"), model
        best = min(best, m)
    return best, model


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp2b")
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"Phase 2b — 真实 OmniBrain 上的域调制 | 设备 {dev}")
    print(f"c_bound={C_BOUND} epochs={EPOCHS} seq_len=32")
    print("=" * 72)

    sp = root / "spike_view"
    meta = pd.read_parquet(sp / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}

    rows = []
    for sess in sorted(meta["session"].unique()):
        f = sp / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        m = meta[meta["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(w) < MIN_SEQ:
            continue
        y = np.array([dom2i[d] for d in m["domain_l1"]], dtype=np.int64)
        if len(np.unique(y)) < 2:
            continue

        X, Yn, yseq = build_sequences(w, y)
        if X is None:
            continue
        n_tr = int(len(X) * TRAIN_FRAC)
        if n_tr < 20 or len(X) - n_tr < 10:
            continue

        Xtr, Ytr, idtr = X[:n_tr], Yn[:n_tr], yseq[:n_tr]
        Xte, Yte, idte = X[n_tr:], Yn[n_tr:], yseq[n_tr:]

        rng = np.random.default_rng(SEED + sess)
        idtr_sh = idtr.copy()
        for i in range(len(idtr_sh)):
            rng.shuffle(idtr_sh[i])

        # 经验时间常数（用训练段估计）
        tau = empirical_tau(w[:n_tr * 31], y[:n_tr * 31], K)

        mse_s, model_s = train_eval("shared", Xtr, Ytr, idtr, Xte, Yte, idte, dev)
        mse_d, model_d = train_eval("domain", Xtr, Ytr, idtr, Xte, Yte, idte, dev)
        mse_r, _ = train_eval("domain", Xtr, Ytr, idtr_sh, Xte, Yte, idte, dev)

        if not all(map(math.isfinite, [mse_s, mse_d, mse_r])):
            print(f"  sess {sess:02d}: NaN, 跳过")
            continue

        phi = model_d.phi_scale()
        # φ 大 = |A| 大 = 衰减快 = 时间常数小 → 与 tau 负相关
        valid = np.isfinite(tau)
        rho = float("nan")
        if valid.sum() >= 3:
            from scipy.stats import spearmanr
            r, p = spearmanr(tau[valid], phi[valid])
            rho = float(r)

        rows.append({
            "session": int(sess), "n_seq": int(len(X)),
            "mse_shared": round(mse_s, 5), "mse_domain": round(mse_d, 5),
            "mse_random": round(mse_r, 5),
            "gain_vs_shared_pct": round((mse_s - mse_d) / mse_s * 100, 2),
            "gain_vs_random_pct": round((mse_r - mse_d) / mse_r * 100, 2),
            "phi_tau_spearman": round(rho, 4) if math.isfinite(rho) else None,
        })
        print(f"  sess {sess:02d}: shared {mse_s:.4f} | domain {mse_d:.4f} "
              f"({rows[-1]['gain_vs_shared_pct']:+.1f}%) | random {mse_r:.4f} "
              f"({rows[-1]['gain_vs_random_pct']:+.1f}%) | ρ(tau,|A|) {rho:+.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase2b_real.csv", index=False)
    if not len(df):
        print("无可用 session")
        return

    gs = df["gain_vs_shared_pct"].mean()
    gr = df["gain_vs_random_pct"].mean()
    rhos = df["phi_tau_spearman"].dropna()
    print("\n" + "=" * 72)
    print(f"  域 vs 共享: {gs:+.2f}%  (better on {(df['gain_vs_shared_pct']>0).sum()}/{len(df)})")
    print(f"  域 vs 随机: {gr:+.2f}%  (better on {(df['gain_vs_random_pct']>0).sum()}/{len(df)})")
    if len(rhos):
        print(f"  ρ(经验τ, |A|): 均值 {rhos.mean():+.3f}, "
              f"负值 {(rhos<0).sum()}/{len(rhos)}  ← 预期为负")

    verdict = ("✅ 真实数据上域调制有效" if gs > 2 and gr > 1
               else "❌ 真实数据上域调制无明显收益")
    print(f"\n  判定: {verdict}")

    (out / "phase2b_report.json").write_text(json.dumps({
        "schema_version": "1.0", "c_bound": C_BOUND, "epochs": EPOCHS,
        "mean_gain_vs_shared_pct": round(float(gs), 2),
        "mean_gain_vs_random_pct": round(float(gr), 2),
        "mean_rho_tau_vs_phi": round(float(rhos.mean()), 4) if len(rhos) else None,
        "n_sessions": int(len(df)), "verdict": verdict, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
