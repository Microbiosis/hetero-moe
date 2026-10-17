"""Phase 10: 低可预测性是生物限制, 还是线性模型的限制?

## 问题

exp9 显示所有物种 R² 都很低（猕猴最好也只有 +0.19）。
但这用的是 **Ridge 线性回归**。若神经动力学是非线性的，
线性模型的 R² 反映的是**模型能力上限**，不是生物可预测性。

## 设计

同一批数据, 三种模型:
  A  Ridge (线性, alpha 已调好)
  B  小 MLP (1 隐层 128, 非线性)
  C  打乱时间顺序的 MLP（噪声对照）

若 MLP 显著优于 Ridge → 低 R² 是模型限制, 不是生物限制
若 MLP ≈ Ridge        → 线性已足够, 低 R² 反映真实可预测性上限

## 计算控制

- 每物种取 3 个代表分组（避免全量太慢）
- 每分组最多 800 窗
- MLP: 30 epoch, 早停
- 只用 k=1 和 k=8 两个视界
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
MAX_WIN = 800
N_GROUPS_PER_SPECIES = 3
EPOCHS = 30


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
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    net = nn.Sequential(
        nn.Linear(d_in, 128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, d_out),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.MSELoss()

    # 特征标准化
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xt = torch.from_numpy((Xtr - xm) / xs).to(dev)
    Xv = torch.from_numpy((Xte - xm) / xs).to(dev)
    Yt = torch.from_numpy(Ytr).to(dev)
    Yv = torch.from_numpy(Yte).to(dev)

    n = len(Xt)
    bs = 64
    best = float("inf")
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xt[j]), Yt[j])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(Xv), Yv).item()
        best = min(best, v)

    net.eval()
    with torch.no_grad():
        pred = net(Xv)
        ss_res = ((Yv - pred) ** 2).sum().item()
        ss_tot = ((Yv - Yv.mean(0, keepdim=True)) ** 2).sum().item()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp10")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"Phase 10 — 线性 vs 非线性可预测性 | 设备 {dev}")
    print("=" * 78)

    rng = np.random.default_rng(SEED)

    specs = [
        ("mouse", root / "spike_view", "session"),
        ("macaque", root / "spike_view_macaque", "file"),
        ("human", root / "spike_view_human", "file"),
    ]

    rows = []
    t0 = time.time()
    for species, d, gcol in specs:
        if not d.exists():
            continue
        meta = pd.read_parquet(d / "meta.parquet")
        groups = sorted(meta[gcol].unique())[:N_GROUPS_PER_SPECIES]

        for gval in groups:
            if gcol == "session":
                f = d / f"windows_sess{int(gval):03d}.npz"
            else:
                f = d / str(gval)
            if not f.exists():
                continue
            w = np.load(f)["windows"].mean(axis=2)      # (n_win, 64)
            if len(w) < 300:
                continue
            if len(w) > MAX_WIN:
                idx = rng.choice(len(w), MAX_WIN, replace=False)
                w = w[np.sort(idx)]

            for k in HORIZONS:
                X, Y = make_pairs(w, k)
                if X is None:
                    continue
                cut = int(len(X) * 0.7)
                if cut < 100:
                    continue

                r2_lin = ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
                r2_mlp = mlp_r2(X[:cut], Y[:cut], X[cut:], Y[cut:], dev)

                # 噪声对照: 打乱时间
                ws = np.stack([rng.permutation(w[:, j]) for j in range(w.shape[1])], axis=1)
                Xs, Ys = make_pairs(ws, k)
                c2 = int(len(Xs) * 0.7) if Xs is not None else 0
                r2_mlp_sh = mlp_r2(Xs[:c2], Ys[:c2], Xs[c2:], Ys[c2:], dev) if c2 > 100 else None

                rows.append({"species": species, "group": str(gval), "k": k,
                             "n_train": int(cut),
                             "r2_linear": round(r2_lin, 4),
                             "r2_mlp": round(r2_mlp, 4),
                             "r2_mlp_shuffled": round(r2_mlp_sh, 4) if r2_mlp_sh is not None else None})
                print(f"  {species:8s} {str(gval)[:22]:<22} k={k:<2d} "
                      f"linear {r2_lin:+.3f} | MLP {r2_mlp:+.3f} | MLP-shuf {r2_mlp_sh if r2_mlp_sh is None else round(r2_mlp_sh,3)}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase10.csv", index=False)

    print("\n" + "=" * 78)
    print("汇总（按物种 × 视界）")
    print("=" * 78)
    summ = df.groupby(["species", "k"])[["r2_linear", "r2_mlp", "r2_mlp_shuffled"]].median()
    print(summ.round(4).to_string())

    print("\n" + "=" * 78)
    print("MLP 相对线性的提升")
    print("=" * 78)
    df["gain_mlp_over_linear"] = df["r2_mlp"] - df["r2_linear"]
    g = df.groupby("species")["gain_mlp_over_linear"].agg(["median", "mean", "size"])
    print(g.round(4).to_string())

    med_gain = float(df["gain_mlp_over_linear"].median())
    print(f"\n  总体中位提升: {med_gain:+.4f}")
    if med_gain > 0.10:
        v = "✅ 非线性模型显著更好 → 低 R² 主要是线性模型的限制"
    elif med_gain > 0.03:
        v = "⚠️ 非线性略有提升，模型限制部分存在"
    else:
        v = "❌ 非线性无实质提升 → 低 R² 反映真实可预测性上限"
    print(f"  判定: {v}")

    (out / "phase10_report.json").write_text(json.dumps({
        "schema_version": "1.0", "horizons": HORIZONS, "ctx": CTX,
        "summary": {f"{s}|k{k}": {c: round(float(v), 4) for c, v in row.items()}
                    for (s, k), row in summ.iterrows()},
        "median_gain_mlp_over_linear": round(med_gain, 4),
        "verdict": v, "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
