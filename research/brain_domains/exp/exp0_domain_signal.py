"""Phase 0 可行性: OmniBrain 窗口是否真的携带 L1 域信息?

这是所有下游实验的前提。若真实标签与打乱标签的表现相当，则"域结构"假说
在该数据集上不成立，后续训练全部无意义。

## 两个视图分开评估，因为它们的可用设计不同

### spike_view — session 内判别
每个 session 只有 1-4 个域（median 2），跨 session 的域集合几乎不重叠。
因此唯一干净的设计是 **同一 session 内**判别:
    train = 该 session 时间轴前 70% 的窗
    test  = 后 30%
完全控制 session 效应（同一电极/个体/行为），只问"能否区分哪个群体活跃"。

### eeg_view — 被试内判别
用官方切分: train/val 选超参，test 留出。同一被试内、跨 session
（train 是 session 0, test 是 session 3 且图像不同）—— 更强的泛化要求。

## 对照设计（关键）

**打乱标签必须在组内进行**:
- spike: 在 session 内打乱（保留该 session 的域分布）
- eeg:   在被试内打乱
若全局打乱，模型可以靠"哪个 session/被试"作为代理特征，人为抬高打乱基线。

## 判据

    gap = acc(real) − acc(shuffled)
    gap > 0.10 且 real 显著高于多数类基线  → 域信号存在，可进入 Phase 1
    gap < 0.03                              → 无域信号，假说否定，停止

同时报告 macro-F1（类别不均衡下比 accuracy 更诚实）。
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
# 时间切分比例 (spike 用)
TRAIN_FRAC = 0.7
# 每 session 最少窗口数才纳入评估
MIN_WIN = 100


# ------------------------------------------------------------- 特征
def featurize(w: np.ndarray) -> np.ndarray:
    """(N, C, T) → (N, 2C)：逐通道时间均值 + 标准差。

    刻意保持线性可分的简单特征: 若连这个都学不出域, 说明信号本身弱,
    而不是模型容量不够。

    注: 不做额外降维 —— 128 维 (64ch × 2) 对逻辑回归来说完全可控,
    真正拖慢的是 n_jobs 导致的进程 fork (见 eval_split 注释)。
    """
    return np.concatenate([w.mean(axis=2), w.std(axis=2)], axis=1).astype(np.float32)


# ------------------------------------------------------------- 评估
def eval_split(Xtr, ytr, Xte, yte, n_class, tag=""):
    """逻辑回归 + 小 MLP 两档，返回指标 dict。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    out = {}
    # 逻辑回归（线性基线）
    # ⚠️ n_jobs=-1 在 128 维 × 数千样本上会让 sklearn 反复 fork, 比单线程慢得多;
    #    max_iter=2000 且 multilabel 收敛更慢。78 次拟合 (39 session × 2) 下会卡死。
    #    改为单线程 + 300 迭代 (lbfgs 在此规模 50 步内即收敛)。
    lr = LogisticRegression(max_iter=300, C=1.0, n_jobs=1, tol=1e-3)
    lr.fit(Xtr, ytr)
    pred = lr.predict(Xte)
    out["lr_acc"] = float(accuracy_score(yte, pred))
    out["lr_f1"] = float(f1_score(yte, pred, average="macro", zero_division=0))

    # 小 MLP（非线性，检验是否有超出线性的结构）
    try:
        import torch
        import torch.nn as nn

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        mlp = nn.Sequential(
            nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_class),
        ).to(dev)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(Xtr).to(dev)
        yt = torch.from_numpy(ytr.astype(np.int64)).to(dev)
        Xv = torch.from_numpy(Xte).to(dev)
        n = len(Xt)
        bs = 512
        for ep in range(15):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = lossf(mlp(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        with torch.no_grad():
            p = mlp(Xv).argmax(1).cpu().numpy()
        out["mlp_acc"] = float(accuracy_score(yte, p))
        out["mlp_f1"] = float(f1_score(yte, p, average="macro", zero_division=0))
    except Exception as e:
        out["mlp_acc"] = None
        out["mlp_f1"] = None
        out["mlp_err"] = str(e)[:80]
    return out


def majority_baseline(y):
    vc = pd.Series(y).value_counts()
    return float(vc.iloc[0] / len(y))


# ---------------------------------------------------------- spike_view
def run_spike(root: Path, out: Path) -> dict:
    print("\n" + "=" * 70)
    print("spike_view — session 内判别 (时间切分 70/30)")
    print("=" * 70)
    meta = pd.read_parquet(root / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}

    rows = []
    for sess in sorted(meta["session"].unique()):
        f = root / f"windows_sess{sess:03d}.npz"
        if not f.exists():
            continue
        w = np.load(f)["windows"]                       # (n, 64, 8)
        m = meta[meta["session"] == sess].reset_index(drop=True)
        if len(m) != len(w) or len(w) < MIN_WIN:
            continue
        y = np.array([dom2i[d] for d in m["domain_l1"]], dtype=np.int64)
        # 只用该 session 内出现过的域
        present = np.unique(y)
        if len(present) < 2:
            continue

        n_tr = int(len(w) * TRAIN_FRAC)
        Xtr, ytr = featurize(w[:n_tr]), y[:n_tr]
        Xte, yte = featurize(w[n_tr:]), y[n_tr:]
        # 训练/测试都要至少含 2 类，否则无法评估
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        n_cls = len(present)
        real = eval_split(Xtr, ytr, Xte, yte, len(L1), f"sess{sess}")

        # 组内打乱: 只打乱训练标签（保留测试标签与分布）
        rng = np.random.default_rng(SEED + sess)
        ytr_sh = ytr.copy()
        rng.shuffle(ytr_sh)
        shuf = eval_split(Xtr, ytr_sh, Xte, yte, len(L1), f"sess{sess}-shuf")

        base = majority_baseline(yte)
        rows.append({
            "session": int(sess), "n_win": len(w), "n_domains": int(n_cls),
            "test_majority": round(base, 4),
            "real_lr": round(real["lr_acc"], 4), "shuf_lr": round(shuf["lr_acc"], 4),
            "real_mlp": round(real["mlp_acc"], 4) if real["mlp_acc"] is not None else None,
            "shuf_mlp": round(shuf["mlp_acc"], 4) if shuf["mlp_acc"] is not None else None,
            "real_f1": round(real["lr_f1"], 4),
        })
        print(f"  sess {sess:02d}: {len(w):5d} 窗, {n_cls} 域, 多数类 {base:.3f} | "
              f"real {real['lr_acc']:.3f} vs shuf {shuf['lr_acc']:.3f} "
              f"(Δ={real['lr_acc']-shuf['lr_acc']:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase0_spike.csv", index=False)

    if not len(df):
        return {"error": "无可用 session (均单域或样本不足)"}

    summary = {
        "n_sessions": int(len(df)),
        "mean_majority": round(float(df["test_majority"].mean()), 4),
        "mean_real_lr": round(float(df["real_lr"].mean()), 4),
        "mean_shuf_lr": round(float(df["shuf_lr"].mean()), 4),
        "gap_lr": round(float((df["real_lr"] - df["shuf_lr"]).mean()), 4),
        "mean_real_f1": round(float(df["real_f1"].mean()), 4),
        "sessions_real_gt_shuf": int((df["real_lr"] > df["shuf_lr"]).sum()),
        "sessions_real_gt_majority": int((df["real_lr"] > df["test_majority"]).sum()),
    }
    if df["real_mlp"].notna().any():
        summary["mean_real_mlp"] = round(float(df["real_mlp"].mean()), 4)
        summary["mean_shuf_mlp"] = round(float(df["shuf_mlp"].mean()), 4)
        summary["gap_mlp"] = round(float((df["real_mlp"] - df["shuf_mlp"]).mean()), 4)

    print(f"\n  汇总: real {summary['mean_real_lr']:.3f} vs shuf "
          f"{summary['mean_shuf_lr']:.3f} (Δ={summary['gap_lr']:+.3f}) | "
          f"多数类基线 {summary['mean_majority']:.3f}")
    print(f"  {summary['sessions_real_gt_shuf']}/{summary['n_sessions']} 个 session 上 real > shuf")
    return summary


# ------------------------------------------------------------ eeg_view
def run_eeg(root: Path, out: Path) -> dict:
    print("\n" + "=" * 70)
    print("eeg_view — 被试内判别 (train=session0 / test=session3 且图像不同)")
    print("=" * 70)
    w = np.load(root / "windows.npz")["windows"]
    meta = pd.read_parquet(root / "meta.parquet")
    dom2i = {d: i for i, d in enumerate(L1)}
    y_all = np.array([dom2i[d] for d in meta["domain_l1"]], dtype=np.int64)

    rows = []
    for sub in sorted(meta["subject"].unique()):
        m_tr = (meta["subject"] == sub) & (meta["split"] == "train")
        m_te = (meta["subject"] == sub) & (meta["split"] == "test")
        if m_tr.sum() < MIN_WIN or m_te.sum() < 10:
            continue
        Xtr, ytr = featurize(w[m_tr.to_numpy()]), y_all[m_tr.to_numpy()]
        Xte, yte = featurize(w[m_te.to_numpy()]), y_all[m_te.to_numpy()]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        real = eval_split(Xtr, ytr, Xte, yte, len(L1), sub)
        rng = np.random.default_rng(SEED + abs(hash(sub)) % 10000)
        ytr_sh = ytr.copy(); rng.shuffle(ytr_sh)
        shuf = eval_split(Xtr, ytr_sh, Xte, yte, len(L1), sub + "-shuf")
        base = majority_baseline(yte)

        rows.append({"subject": sub, "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                     "test_majority": round(base, 4),
                     "real_lr": round(real["lr_acc"], 4), "shuf_lr": round(shuf["lr_acc"], 4),
                     "real_mlp": round(real["mlp_acc"], 4) if real["mlp_acc"] is not None else None,
                     "shuf_mlp": round(shuf["mlp_acc"], 4) if shuf["mlp_acc"] is not None else None,
                     "real_f1": round(real["lr_f1"], 4)})
        print(f"  {sub}: {int(m_tr.sum())}→{int(m_te.sum())} 窗, 多数类 {base:.3f} | "
              f"real {real['lr_acc']:.3f} vs shuf {shuf['lr_acc']:.3f} "
              f"(Δ={real['lr_acc']-shuf['lr_acc']:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase0_eeg.csv", index=False)
    if not len(df):
        return {"error": "无可评估被试"}
    return {
        "n_subjects": int(len(df)),
        "mean_majority": round(float(df["test_majority"].mean()), 4),
        "mean_real_lr": round(float(df["real_lr"].mean()), 4),
        "mean_shuf_lr": round(float(df["shuf_lr"].mean()), 4),
        "gap_lr": round(float((df["real_lr"] - df["shuf_lr"]).mean()), 4),
        "mean_real_f1": round(float(df["real_f1"].mean()), 4),
        "subjects_real_gt_shuf": int((df["real_lr"] > df["shuf_lr"]).sum()),
        "subjects_real_gt_majority": int((df["real_lr"] > df["test_majority"]).sum()),
    }


def verdict(s: dict, kind: str) -> str:
    if "error" in s:
        return f"{kind}: {s['error']}"
    gap, real, base = s["gap_lr"], s["mean_real_lr"], s["mean_majority"]
    if gap > 0.10 and real > base + 0.05:
        return (f"{kind}: ✅ 域信号存在 (real {real:.3f} vs shuf "
                f"{s['mean_shuf_lr']:.3f}, Δ={gap:+.3f}, 多数类 {base:.3f})")
    if gap < 0.03:
        return (f"{kind}: ❌ 无域信号 (Δ={gap:+.3f} < 0.03) —— 域结构假说在该视图上不成立")
    return f"{kind}: ⚠️ 弱信号 (Δ={gap:+.3f}) —— 需更多数据或更强特征"


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp0")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 72)
    print("Phase 0 — OmniBrain 域信号可行性验证")
    print("=" * 72)

    report = {"schema_version": "1.0", "method": "linear+MLP, 组内打乱标签对照",
              "train_frac_spike": TRAIN_FRAC, "seed": SEED}

    if os.environ.get("SKIP_SPIKE") != "1":
        report["spike_view"] = run_spike(root / "spike_view", out)
    if os.environ.get("SKIP_EEG") != "1":
        report["eeg_view"] = run_eeg(root / "eeg_view", out)

    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "phase0_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("判定")
    print("=" * 72)
    if "spike_view" in report:
        print("  " + verdict(report["spike_view"], "spike_view"))
    if "eeg_view" in report:
        print("  " + verdict(report["eeg_view"], "eeg_view"))
    print(f"\n耗时 {report['elapsed_s']}s")


if __name__ == "__main__":
    main()
