"""Phase 4: 域差异到底该调制哪个参数?

## 起因

Phase 3 发现域标签的效应量排序:
    mean_abs         ε² = 0.161   ← 活动幅度, 最强
    ac1              ε² = 0.089   ← 时间自相关
    std              ε² = 0.086
    hf_ratio         ε² = 0.079
    eff_dim          ε² = 0.036
    spectral_slope   ε² = 0.025
    sparsity         ε² = 0.000   ← 完全无差异

而 exp2b 调制的是 **A（状态转移 = 时间常数）**, 属于 ac1 那一类,
但 ac1 只是第二名, 且可能与 mean_abs 混淆。

**假设**: 域真正编码的是"这个窗有多活跃"(mean_abs), 那么域调制应该作用在
**输出增益** 上, 而不是时间常数。

## 实验 A: 偏相关分析（诊断）

控制 mean_abs 后, 其余统计量的域效应量还剩多少?

    ε²_resid(stat) = 用 mean_abs 分层匹配后的 ε²

做法: 对每个统计量, 按 mean_abs 分位数分成若干层, 在层内计算 ε²,
再按层大小加权平均。这样"相同活跃度下, 域是否还有别的差异"。

判据:
  ε² 下降 > 70%  → 该差异基本是 mean_abs 的副产物
  ε² 保持 > 60%  → 该差异独立于活跃度

## 实验 B: 三臂模型对比

同一个域条件 SSM, 只改"域调制哪个参数":

  A  mod_A       调制状态转移矩阵 A (时间常数)     ← exp2b 的做法
  B  mod_gain    调制输出增益 scale                 ← 对应 mean_abs
  C  mod_C       调制读出投影 C                     ← 另一条路径
  D  shared      无域调制                           ← 基线
  E  mod_gain_rand  调制增益但打乱标签              ← 对照

判据:
  若 B ≫ A 且 B ≫ D 且 B ≫ E → 域应调制增益, 验证 mean_abs 假说
  若 A ≈ B ≈ D              → 无论调什么参数都无用
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
from scipy.stats import kruskal

L1 = ["sensory", "motor", "memory", "association", "subcortical", "other"]
K = len(L1)
SEED = 0
MIN_PER_DOMAIN = 30
N_STRATA = 10          # 偏相关分析的分层数
EPOCHS = 25
BATCH = 32
LR = 2e-3
C_BOUND = 0.5


# ============================================================ 实验 A
def strata_eps2(v, y, n_strata=N_STRATA):
    """按 v 的分位数分层, 层内计算 ε², 加权平均。

    这是一个粗略的"控制 v 后看其他变量"的做法。v 本身若是混淆源,
    在层内它近似常数, 故层内 ε² 不能由 v 解释。
    """
    if not np.isfinite(v).all():
        return None
    q = np.quantile(v, np.linspace(0, 1, n_strata + 1))
    q = np.unique(q)
    if len(q) < 3:
        return None
    idx = np.clip(np.searchsorted(q, v, side="right") - 1, 0, len(q) - 2)

    num, den = 0.0, 0.0
    for s in range(len(q) - 1):
        m = idx == s
        if m.sum() < MIN_PER_DOMAIN * 2:
            continue
        groups = [v[m & (y == d)] for d in np.unique(y[m])]
        groups = [g for g in groups if len(g) >= MIN_PER_DOMAIN]
        if len(groups) < 2:
            continue
        e2, p = epsilon_sq_raw(groups)
        if e2 is None:
            continue
        w = m.sum()
        num += e2 * w
        den += w
    return num / den if den > 0 else None


def epsilon_sq_raw(groups):
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return None, None
    try:
        H, p = kruskal(*groups)
    except ValueError:
        return None, None
    return float(max(0.0, (H - k + 1) / (n - k))), float(p)


def run_partial(root, out):
    print("=" * 76)
    print("实验 A — 偏相关: 控制 mean_abs 后, 各统计量的域效应还剩多少?")
    print("=" * 76)

    from exp3_domain_fingerprint import window_stats
    sp = root / "spike_view"
    meta = pd.read_parquet(sp / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}
    stats_names = ["mean_abs", "std", "ac1", "spectral_slope",
                   "hf_ratio", "sparsity", "eff_dim"]

    rows = []
    for sess in sorted(meta["session"].unique()):
        f = sp / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        m = meta[meta["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(w) < 200:
            continue
        y = np.array([dom2i[d] for d in m["domain_l1"]], dtype=np.int64)
        st = window_stats(w)

        # 基准: 每个统计量单独的 ε²
        base = {}
        for name in stats_names:
            v = st[name]
            groups = [v[y == d] for d in np.unique(y)]
            groups = [g for g in groups if len(g) >= MIN_PER_DOMAIN]
            e2, _ = epsilon_sq_raw(groups) if len(groups) >= 2 else (None, None)
            base[name] = e2

        # 控制 mean_abs 后
        for name in stats_names:
            if name == "mean_abs" or base.get(name) is None:
                continue
            v = st[name]
            resid = strata_eps2(st["mean_abs"], y) if name != "mean_abs" else None
            # 真正要算的是: 控制 mean_abs 后, name 的域效应
            e2_resid = control_by(v, st["mean_abs"], y)
            if e2_resid is None:
                continue
            keep = e2_resid / base[name] * 100 if base[name] > 0 else 0
            rows.append({"session": int(sess), "stat": name,
                         "eps2_base": round(base[name], 4),
                         "eps2_ctrl_meanabs": round(e2_resid, 4),
                         "retention_pct": round(keep, 1)})
        print(f"  sess {sess:02d} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase4a_partial.csv", index=False)
    if not len(df):
        return None

    print("\n" + "=" * 76)
    print(f"{'统计量':<16} {'原始ε²':>9} {'控mean_abs后':>13} {'保留率':>9} {'判定':>16}")
    summary = {}
    for name in stats_names:
        sub = df[df["stat"] == name]
        if not len(sub):
            continue
        b = sub["eps2_base"].median()
        c = sub["eps2_ctrl_meanabs"].median()
        keep = sub["retention_pct"].median()
        if keep > 60:
            v = "✅ 独立于活跃度"
        elif keep > 30:
            v = "⚠️ 部分独立"
        else:
            v = "✗ 是 mean_abs 副产物"
        summary[name] = {"base": round(float(b), 4), "ctrl": round(float(c), 4),
                         "retention_pct": round(float(keep), 1), "verdict": v}
        print(f"{name:<16} {b:>9.4f} {c:>13.4f} {keep:>8.1f}% {v:>16}")
    return summary


def control_by(v, confound, y):
    """控制 confound 后, v 的域效应量（分层近似）。"""
    if not (np.isfinite(v).all() and np.isfinite(confound).all()):
        return None
    q = np.unique(np.quantile(confound, np.linspace(0, 1, N_STRATA + 1)))
    if len(q) < 3:
        return None
    idx = np.clip(np.searchsorted(q, confound, side="right") - 1, 0, len(q) - 2)
    num, den = 0.0, 0.0
    for s in range(len(q) - 1):
        m = idx == s
        if m.sum() < MIN_PER_DOMAIN * 2:
            continue
        groups = [v[m & (y == d)] for d in np.unique(y[m])]
        groups = [g for g in groups if len(g) >= MIN_PER_DOMAIN]
        if len(groups) < 2:
            continue
        e2, _ = epsilon_sq_raw(groups)
        if e2 is None:
            continue
        w = int(m.sum())
        num += e2 * w
        den += w
    return num / den if den else None


# ============================================================ 实验 B
class ModSSM(nn.Module):
    """可切换"域调制哪个参数"的 SSM。

    mode:
      shared    无域调制
      mod_A     调制状态转移 A (时间常数)      ← exp2b
      mod_gain  调制输出增益 scale             ← 新, 对应 mean_abs
      mod_C     调制读出投影 C                 ← 新
    """

    def __init__(self, mode="shared", d_in=64):
        super().__init__()
        self.mode = mode
        a_init = torch.arange(1, 16 + 1, dtype=torch.float32).repeat(d_in, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.U = nn.Parameter(torch.randn(4, d_in, 16) / 4.0)
        self.alpha = nn.Parameter(torch.zeros(K, 4))
        self.gate = nn.Parameter(torch.ones(1))

        self.inp = nn.Linear(d_in, d_in)
        self.dt = nn.Linear(d_in, d_in)
        self.Bp = nn.Linear(d_in, 16, bias=False)
        self.Cp = nn.Linear(d_in, 16, bias=False)
        self.out = nn.Linear(d_in, d_in)
        self.norm = nn.RMSNorm(d_in)
        self.gamma = nn.Parameter(torch.full((d_in,), 1e-2))
        self.A_MIN, self.A_MAX = 0.5, 16.0

    def domain_scale(self, ids):
        """返回每 token 的域调制系数 (B,L,1)，>0。"""
        off = torch.einsum("blr,r->bl", self.alpha[ids] * self.gate,
                           torch.ones(self.U.shape[0], device=ids.device))
        # 用 U 的均值做标量投影（gain/C 模式只需要一个标量系数）
        return torch.exp(torch.tanh(off) * C_BOUND)          # (B,L)

    def effective_A(self, ids):
        a = -torch.exp(self.A_log).clamp(self.A_MIN, self.A_MAX)
        if self.mode != "mod_A":
            return a.unsqueeze(0).unsqueeze(0).expand(ids.shape[0], ids.shape[1], -1, -1)
        off = torch.einsum("blr,rdn->bldn", self.alpha[ids] * self.gate, self.U)
        phi = torch.exp(torch.tanh(off) * C_BOUND)
        return a.unsqueeze(0).unsqueeze(0) * phi

    def forward(self, x, ids):
        B, L, _ = x.shape
        h_in = self.norm(self.inp(x))
        delta = F.softplus(self.dt(h_in)).clamp(1e-3, 1e-1)
        Bm, Cm = self.Bp(h_in), self.Cp(h_in)

        if self.mode == "mod_C":
            s = self.domain_scale(ids).unsqueeze(-1)          # (B,L,1)
            Cm = Cm * s

        A_eff = self.effective_A(ids)
        log_dA = delta.unsqueeze(-1) * A_eff
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * h_in.unsqueeze(-1)

        block = 16
        h = torch.zeros(B, x.shape[-1], 16, device=x.device)
        ys = []
        for t in range(0, L, block):
            e = min(t + block, L)
            lrel = torch.cumsum(log_dA[:, t:e], dim=1)
            ea = torch.exp(lrel)
            hh = ea * (h.unsqueeze(1) + torch.cumsum(dBu[:, t:e] / ea, dim=1))
            ys.append((hh * Cm[:, t:e].unsqueeze(-2)).sum(-1))
            h = hh[:, -1]
        y = torch.cat(ys, dim=1) + h_in

        if self.mode == "mod_gain":
            y = y * self.domain_scale(ids).unsqueeze(-1)      # 调制输出幅度

        return self.out(self.gamma * y)


def build_seq(w, y, seq_len=32):
    x = w.mean(axis=2).astype(np.float32)
    n = len(x) // seq_len
    if n < 2:
        return None, None, None
    xs = x[:n * seq_len].reshape(n, seq_len, -1)
    ys = y[:n * seq_len].reshape(n, seq_len)
    return xs[:, :-1], xs[:, 1:], ys[:, :-1]


def train_eval(mode, Xtr, Ytr, itr, Xte, Yte, ite, dev, seed=SEED):
    torch.manual_seed(seed)
    m = ModSSM(mode).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    Xt, Yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    It = torch.from_numpy(itr.astype(np.int64)).to(dev)
    Xv, Yv = torch.from_numpy(Xte).to(dev), torch.from_numpy(Yte).to(dev)
    Iv = torch.from_numpy(ite.astype(np.int64)).to(dev)

    n = len(Xt)
    best = float("inf")
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.mse_loss(m(Xt[idx], It[idx]), Yt[idx])
            if not torch.isfinite(loss):
                return float("nan")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
            opt.step()
        m.eval()
        with torch.no_grad():
            v = F.mse_loss(m(Xv, Iv), Yv).item()
        if not math.isfinite(v):
            return float("nan")
        best = min(best, v)
    return best


def run_modes(root, out):
    print("\n" + "=" * 76)
    print("实验 B — 域调制哪个参数最有效?")
    print("=" * 76)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = root / "spike_view"
    meta = pd.read_parquet(sp / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}
    rows = []
    t0 = time.time()

    for sess in sorted(meta["session"].unique()):
        f = sp / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        m = meta[meta["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(w) < 320:
            continue
        y = np.array([dom2i[d] for d in m["domain_l1"]], dtype=np.int64)
        X, Y, ys = build_seq(w, y)
        if X is None:
            continue
        n_tr = int(len(X) * 0.7)
        if n_tr < 15 or len(X) - n_tr < 8:
            continue

        Xtr, Ytr, itr = X[:n_tr], Y[:n_tr], ys[:n_tr]
        Xte, Yte, ite = X[n_tr:], Y[n_tr:], ys[n_tr:]

        rng = np.random.default_rng(SEED + sess)
        itr_sh = itr.copy()
        for i in range(len(itr_sh)):
            rng.shuffle(itr_sh[i])

        res = {}
        for mode in ["shared", "mod_A", "mod_gain", "mod_C"]:
            res[mode] = train_eval(mode, Xtr, Ytr, itr, Xte, Yte, ite, dev)
        res["mod_gain_rand"] = train_eval("mod_gain", Xtr, Ytr, itr_sh, Xte, Yte, ite, dev)

        if not all(math.isfinite(v) for v in res.values()):
            print(f"  sess {sess:02d}: 含 NaN, 跳过")
            continue

        base = res["shared"]
        row = {"session": int(sess)}
        for k_, v_ in res.items():
            row[f"mse_{k_}"] = round(v_, 5)
            row[f"gain_{k_}_pct"] = round((base - v_) / base * 100, 2)
        row["best_mode"] = min(res, key=res.get)
        rows.append(row)
        print(f"  sess {sess:02d}: shared {res['shared']:.4f} | "
              f"A {res['mod_A']:.4f} ({row['gain_mod_A_pct']:+.1f}%) | "
              f"gain {res['mod_gain']:.4f} ({row['gain_mod_gain_pct']:+.1f}%) | "
              f"C {res['mod_C']:.4f} ({row['gain_mod_C_pct']:+.1f}%) | "
              f"gain_rand ({row['gain_mod_gain_rand_pct']:+.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase4b_modes.csv", index=False)
    if not len(df):
        return None

    print("\n" + "=" * 76)
    print(f"跨 {len(df)} 个 session 的平均提升 (相对 shared):")
    summary = {}
    for mode in ["mod_A", "mod_gain", "mod_C", "mod_gain_rand"]:
        g = df[f"gain_{mode}_pct"].mean()
        w = (df[f"gain_{mode}_pct"] > 0).sum()
        summary[mode] = {"mean_gain_pct": round(float(g), 2), "wins": int(w)}
        print(f"  {mode:<14} {g:+7.2f}%   ({w}/{len(df)} 胜)")

    best = max(["mod_A", "mod_gain", "mod_C"], key=lambda k_: summary[k_]["mean_gain_pct"])
    print(f"\n  最佳调制目标: {best}")
    return summary


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp4")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rep = {"schema_version": "1.0"}

    if os.environ.get("SKIP_A") != "1":
        rep["partial"] = run_partial(root, out)
    if os.environ.get("SKIP_B") != "1":
        rep["modes"] = run_modes(root, out)

    rep["elapsed_s"] = round(time.time() - t0, 1)
    (out / "phase4_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n耗时 {rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
