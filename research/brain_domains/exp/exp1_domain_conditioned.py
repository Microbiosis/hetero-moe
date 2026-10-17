"""Phase 1: 域条件建模 —— 域标签作为输入是否真的有用?

Phase 0 已证明"窗口携带域信息"（域可被解码）。Phase 1 问的是更强的命题:

    把 domain_id 作为显式输入喂给模型, 能否提升**预测任务**的性能?

这两件事不同: Phase 0 是"域可解码"(分类问题), Phase 1 是"域有助于预测"
(条件建模)。一个信息可解码 ≠ 它对另一个任务有用。

## 任务设计: 下一步活动预测

对每个窗口, 用其 64×8 的表示预测**下一个窗口的帧级活动均值**。
这是自监督的, 无需外部标签, 且与 SSM 的时间动力学直接相关。

## 三臂对照（同数据同超参）

  A  no_domain   只吃 x
  B  with_domain 吃 x + domain one-hot (6 维)
  C  rand_domain 吃 x + 打乱的 domain one-hot  ← 关键: 排除"多了 6 个参数"的混淆

判据:
  B 显著优于 A 且 B 显著优于 C  → 域标签有实质信息
  B ≈ C 且都优于 A              → 收益只来自额外参数, 域标签无信息
  B ≈ A ≈ C                     → 域对预测无用（但 Phase 0 的"可解码"仍成立）

## 为什么用 MLP 而非 SSM

Phase 1 只问"域是否有用", 不涉及架构。用 MLP 单步预测排除架构混杂。
若 Phase 1 通过, Phase 2 再把域接进 SSM（这才是本项目的主线）。

## 数据划分

沿用 Phase 0 的 session 内时间切分（前 70% / 后 30%），
与前序实验可比。跨 session 的划分在 spike_view 上不成立（域集合不重叠）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

L1 = ["sensory", "motor", "memory", "association", "subcortical", "other"]
SEED = 0
TRAIN_FRAC = 0.7
MIN_WIN = 300          # 需要足够窗口才能切出"当前/下一窗口"对


def make_pairs(w: np.ndarray, y: np.ndarray):
    """把连续窗口序列转成 (x_t, x_{t+1}) 预测对。

    w: (N, 64, 8)   y: (N,) 域索引
    x_t  = w[t].reshape(-1)                    (512 维)
    tgt  = w[t+1].mean(axis=2)                 (64 维: 沿时间轴取均值, 保留通道维)
    """
    X = w[:-1].reshape(len(w) - 1, -1).astype(np.float32)
    Y = w[1:].mean(axis=2).astype(np.float32)
    d = y[:-1]
    return X, Y, d


def train_eval(Xtr, Ytr, dtr, Xte, Yte, dte, n_class, mode, dev=None):
    """训练一个 MLP, 返回测试 MSE / 相对提升。

    mode: 'none' | 'domain' | 'random'
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = Xtr.shape[1] + (n_class if mode != "none" else 0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 64),
            )

        def forward(self, x, d):
            if mode == "none":
                return self.net(x)
            oh = torch.nn.functional.one_hot(d, n_class).float()
            return self.net(torch.cat([x, oh], dim=1))

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.MSELoss()

    Xt = torch.from_numpy(Xtr).to(dev)
    Yt = torch.from_numpy(Ytr).to(dev)
    dt = torch.from_numpy(dtr.astype(np.int64)).to(dev)
    Xv = torch.from_numpy(Xte).to(dev)
    Yv = torch.from_numpy(Yte).to(dev)
    dv = torch.from_numpy(dte.astype(np.int64)).to(dev)

    n = len(Xt)
    bs = 512
    best = float("inf")
    for ep in range(30):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xt[idx], dt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            # 用训练集尾段做早停（不做超参搜索，只防过拟合）
            m = min(len(Xt), 4096)
            v = lossf(model(Xt[-m:], dt[-m:]), Yt[-m:]).item()
        if v < best:
            best = v
    model.eval()
    with torch.no_grad():
        mse = lossf(model(Xv, dv), Yv).item()
    return float(mse)


def run_session(sess, w, m, dom2i, dev):
    y = np.array([dom2i[d] for d in m["domain_l1"]], dtype=np.int64)
    X, Y, d = make_pairs(w, y)
    n_tr = int(len(X) * TRAIN_FRAC)
    if n_tr < MIN_WIN or len(X) - n_tr < MIN_WIN // 2:
        return None

    Xtr, Ytr, dtr = X[:n_tr], Y[:n_tr], d[:n_tr]
    Xte, Yte, dte = X[n_tr:], Y[n_tr:], d[n_tr:]

    rng = np.random.default_rng(SEED + sess)
    dtr_rand = dtr.copy(); rng.shuffle(dtr_rand)

    mse_none = train_eval(Xtr, Ytr, dtr, Xte, Yte, dte, len(L1), "none", dev)
    mse_dom = train_eval(Xtr, Ytr, dtr, Xte, Yte, dte, len(L1), "domain", dev)
    mse_rand = train_eval(Xtr, Ytr, dtr_rand, Xte, Yte, dte, len(L1), "random", dev)

    return {"session": int(sess), "n_tr": int(n_tr), "n_te": int(len(X) - n_tr),
            "mse_none": round(mse_none, 5), "mse_domain": round(mse_dom, 5),
            "mse_random": round(mse_rand, 5),
            "gain_domain": round((mse_none - mse_dom) / mse_none * 100, 2),
            "gain_random": round((mse_none - mse_rand) / mse_none * 100, 2),
            "domain_beats_random": bool(mse_dom < mse_rand)}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp1")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"Phase 1 — 域条件建模 | 设备 {dev}")
    print("=" * 72)

    sp = root / "spike_view"
    meta = pd.read_parquet(sp / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}

    t0 = time.time()
    rows = []
    for sess in sorted(meta["session"].unique()):
        f = sp / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]
        m = meta[meta["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(w) < MIN_WIN:
            continue
        if m["domain_l1"].nunique() < 2:
            continue
        r = run_session(sess, w, m, dom2i, dev)
        if r is None:
            continue
        rows.append(r)
        print(f"  sess {sess:02d}: none {r['mse_none']:.4f} | "
              f"domain {r['mse_domain']:.4f} ({r['gain_domain']:+.1f}%) | "
              f"random {r['mse_random']:.4f} ({r['gain_random']:+.1f}%) | "
              f"域胜随机 {'✓' if r['domain_beats_random'] else '✗'}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase1_spike.csv", index=False)

    print("\n" + "=" * 72)
    if not len(df):
        print("无可评估 session")
        return

    gd = df["gain_domain"].mean()
    gr = df["gain_random"].mean()
    wins = int(df["domain_beats_random"].sum())
    n = len(df)

    report = {
        "schema_version": "1.0", "task": "next-window activity prediction",
        "n_sessions": n,
        "mean_mse_none": round(float(df["mse_none"].mean()), 5),
        "mean_mse_domain": round(float(df["mse_domain"].mean()), 5),
        "mean_mse_random": round(float(df["mse_random"].mean()), 5),
        "mean_gain_domain_pct": round(float(gd), 2),
        "mean_gain_random_pct": round(float(gr), 2),
        "sessions_domain_beats_random": wins,
        "elapsed_s": round(time.time() - t0, 1),
    }

    print(f"  平均 MSE: none {report['mean_mse_none']:.4f} | "
          f"domain {report['mean_mse_domain']:.4f} ({gd:+.1f}%) | "
          f"random {report['mean_mse_random']:.4f} ({gr:+.1f}%)")
    print(f"  域胜随机: {wins}/{n} 个 session")

    if gd > 1.0 and wins >= n * 0.8:
        v = "✅ 域标签有实质信息 (域 > 随机, 且优于无域)"
    elif abs(gd - gr) < 0.5:
        v = "❌ 域标签无信息 (域 ≈ 随机, 收益只来自额外参数)"
    else:
        v = "⚠️ 弱/不一致结果, 需检查"
    report["verdict"] = v
    print(f"\n  判定: {v}")

    (out / "phase1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n耗时 {report['elapsed_s']}s")


if __name__ == "__main__":
    main()
