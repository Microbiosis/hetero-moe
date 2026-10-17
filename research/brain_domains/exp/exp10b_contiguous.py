"""Phase 10b: 修正窗口抽样 —— 必须取连续段, 不能随机抽。

## 修什么

exp10 用 `rng.choice(len(w), MAX_WIN)` 从整段随机抽窗口再排序。
这破坏了**时间连续性**: 小鼠 session 有达 10,193 窗, 抽 800 后相邻样本
实际间隔约 1.3 秒, 而非 100ms。于是 "用前 8 bin 预测下一 bin" 的假设失效,
模型看到的是不连续的时间点拼接。

症状: 小鼠 R² = -0.64, 比它自己的噪声地板 (-0.34) 还差 0.30。
"比噪声还差" 就是分布不匹配的信号。

修正: 从 session **中段**取一段连续窗口（避开开头/结尾的适应期）。

## 同时修正: 跨模型的 R² 不可直接比较

不同模型有不同的"负 R² 地板"（取决于容量与正则）:
  Ridge (alpha=1000): 地板 ≈ -0.06
  MLP (512→128→128): 地板 ≈ -0.34

因此正确的比较是 **R² - 该模型自己的打乱地板**。本脚本对两种模型
都测打乱对照, 报告"净信号"。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
CTX = 8
HORIZONS = [1, 8]
WIN_LEN = 1200          # 连续取这么多个窗口
N_GROUPS = 3
EPOCHS = 30


def contiguous_slice(w, length=WIN_LEN):
    """从中间取连续一段。"""
    n = len(w)
    if n <= length:
        return w
    s = (n - length) // 2
    return w[s:s + length]


def make_pairs(x, k, ctx=CTX):
    n = x.shape[0]
    idx = np.arange(ctx - 1, n - k)
    if len(idx) < 60:
        return None, None
    X = np.stack([x[i - ctx + 1:i + 1].reshape(-1) for i in idx]).astype(np.float32)
    Y = x[idx + k].astype(np.float32)
    return X, Y


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1000.0):
    d = Xtr.shape[1]
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr_, Xte_ = (Xtr - xm) / xs, (Xte - xm) / xs
    ym = Ytr.mean(0, keepdims=True)
    A = Xtr_.T @ Xtr_ + alpha * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, Xtr_.T @ (Ytr - ym))
    pred = Xte_ @ W + ym
    return float(1.0 - ((Yte - pred) ** 2).sum() /
                 max(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(), 1e-12))


def mlp_r2(Xtr, Ytr, Xte, Yte, dev, seed=SEED):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, Ytr.shape[1]),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.MSELoss()
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xt = torch.from_numpy((Xtr - xm) / xs).to(dev)
    Xv = torch.from_numpy((Xte - xm) / xs).to(dev)
    Yt = torch.from_numpy(Ytr).to(dev)
    Yv = torch.from_numpy(Yte).to(dev)
    n, bs = len(Xt), 64
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            lossf(net(Xt[j]), Yt[j]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xv)
        return float(1.0 - ((Yv - pred) ** 2).sum().item() /
                     max(((Yv - Yv.mean(0, keepdim=True)) ** 2).sum().item(), 1e-12))


def shuffle_time(w, rng):
    return np.stack([rng.permutation(w[:, j]) for j in range(w.shape[1])], axis=1)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp10b")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print(f"Phase 10b — 连续窗口 + 各自的噪声地板 | {dev}")
    print(f"  每分组连续取 {WIN_LEN} 窗 | 视界 {HORIZONS}")
    print("=" * 80)

    specs = [("mouse", root / "spike_view", "session"),
             ("macaque", root / "spike_view_macaque", "file"),
             ("human", root / "spike_view_human", "file")]

    rows = []
    for species, d, gcol in specs:
        if not d.exists():
            continue
        meta = pd.read_parquet(d / "meta.parquet")
        groups = sorted(meta[gcol].unique())[:N_GROUPS]
        for gval in groups:
            f = d / (f"windows_sess{int(gval):03d}.npz" if gcol == "session" else str(gval))
            if not f.exists():
                continue
            w = np.load(f)["windows"].mean(axis=2)
            if len(w) < 400:
                continue
            w = contiguous_slice(w)
            ws = shuffle_time(w, rng)

            for k in HORIZONS:
                X, Y = make_pairs(w, k)
                Xs, Ys = make_pairs(ws, k)
                if X is None or Xs is None:
                    continue
                cut, cuts = int(len(X) * 0.7), int(len(Xs) * 0.7)
                if cut < 100 or cuts < 100:
                    continue

                r_lin = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
                r_lin_s = ridge_r2(Xs[:cuts], Ys[:cuts], Xs[cuts:], Ys[cuts:])
                r_mlp = mlp_r2(X[:cut], Y[:cut], X[cut:], Y[cut:], dev)
                r_mlp_s = mlp_r2(Xs[:cuts], Ys[:cuts], Xs[cuts:], Ys[cuts:], dev)

                rows.append({
                    "species": species, "group": str(gval), "k": k,
                    "n_win": int(len(w)),
                    "lin": round(r_lin, 4), "lin_shuf": round(r_lin_s, 4),
                    "mlp": round(r_mlp, 4), "mlp_shuf": round(r_mlp_s, 4),
                    "lin_net": round(r_lin - r_lin_s, 4),
                    "mlp_net": round(r_mlp - r_mlp_s, 4),
                })
                print(f"  {species:8s} {str(gval)[:20]:<20} k={k:<2d} "
                      f"lin {r_lin:+.3f}(净{r_lin-r_lin_s:+.3f}) "
                      f"mlp {r_mlp:+.3f}(净{r_mlp-r_mlp_s:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase10b.csv", index=False)

    print("\n" + "=" * 80)
    print("净信号（R² − 自身噪声地板）中位数")
    print("=" * 80)
    t = df.groupby(["species", "k"])[["lin", "lin_shuf", "lin_net", "mlp", "mlp_shuf", "mlp_net"]].median()
    print(t.round(4).to_string())

    print("\n" + "=" * 80)
    print("物种间净信号对比（k=1）")
    print("=" * 80)
    s1 = df[df.k == 1].groupby("species")[["lin_net", "mlp_net"]].median()
    print(s1.round(4).to_string())

    best = s1.max().to_dict()
    print(f"\n  最佳模型下的净信号排序: "
          + ", ".join(f"{sp}={v:.3f}" for sp, v in best.items()))

    (out / "phase10b_report.json").write_text(json.dumps({
        "schema_version": "1.0", "win_len": WIN_LEN, "horizons": HORIZONS,
        "table": {f"{s}|k{k}": {c: round(float(v), 4) for c, v in row.items()}
                  for (s, k), row in t.iterrows()},
        "k1_net_signal": {s: {c: round(float(v), 4) for c, v in row.items()}
                          for s, row in s1.iterrows()},
        "note": "窗口改为连续取中段，修复 exp10 的随机抽样破坏时间连续性问题",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
