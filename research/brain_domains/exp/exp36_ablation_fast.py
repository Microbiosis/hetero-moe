"""Phase 36: 轴消融（快速版）—— 用闭式岭回归替代迭代逻辑回归。

## 为什么重写

exp35 用 200 类逻辑回归，要 720 次拟合 × 300 迭代，单被试就十几分钟。
跑了一小时还没完，只能取消。

## 替代方案：闭式岭分类器（Ridge Classifier）

把标签做 one-hot，闭式解：

    W = (XᵀX + λI)⁻¹ Xᵀ Y

一次矩阵求解，无迭代。对 630 维 × 16000 样本，几秒钟完成。

这个方法在神经解码文献里是标准做法（"least-squares classifier"），
性能与 LR 接近，速度快 100 倍以上。

## 同时缩减规模

  - 被试数 6 → 4
  - 每个变体的重复 8 → 5
  - k 只保留 200（最关键的那个）

## 保留全部五个消融变体

  A_full          63 电极 × 10 时间段 = 630 维
  B_one_time      单时间窗 63 维
  C_one_channel   单电极 10 维
  D_band_only     5 频带 × 63 电极 = 315 维
  E_2rep          全特征但只用 2 次重复训练（测 repetition 轴）
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
FS = 250.0
N_SUBJECTS = 4
WIN_MS = (50, 500)
N_TIME_BINS = 10
K = 200
N_TRIALS = 5
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
LAMBDA = 1.0


def ridge_classify(Xtr, ytr, Xte, n_class, lam=LAMBDA):
    """闭式岭分类器。Xtr: (n,d), ytr: (n,), 返回 (n_te,) 预测。"""
    n, d = Xtr.shape
    Y = np.zeros((n, n_class), dtype=np.float32)
    Y[np.arange(n), ytr] = 1.0
    # 中心化
    xm = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - xm
    A = Xc.T @ Xc
    A.flat[:: d + 1] += lam * np.trace(A) / d
    try:
        W = np.linalg.solve(A, Xc.T @ Y)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, Xc.T @ Y, rcond=None)[0]
    scores = (Xte - xm) @ W
    return scores.argmax(axis=1)


def base_correct(eeg):
    a = int(WIN_MS[0] / 1000 * FS)
    base = eeg[:, :, :, :max(a, 1)].mean(axis=3, keepdims=True)
    return eeg.astype(np.float32) - base


def to_time_bins(X, n_bins=N_TIME_BINS):
    T = X.shape[-1]
    seg = T // n_bins
    if seg < 1:
        return None
    return X[:, :, :, :seg * n_bins].reshape(
        X.shape[0], X.shape[1], X.shape[2], n_bins, seg).mean(axis=4)


def band_power(X, fs=FS):
    T = X.shape[-1]
    win = np.hanning(T).astype(np.float32)
    P = np.abs(np.fft.rfft(X * win, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    out = np.zeros(X.shape[:-1] + (len(BANDS),), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        if m.sum():
            out[..., i] = P[..., m].mean(axis=-1)
    return out


def decode_fast(Xf, k, rng, n_trials=N_TRIALS, n_rep_use=None):
    """Xf: (n_img, n_rep, feat)。按 repetition 切分。闭式岭。"""
    n_img, n_rep, feat = Xf.shape
    if k > n_img:
        return None
    n_tr = (n_rep // 2) if n_rep_use is None else min(n_rep_use, n_rep // 2)
    if n_tr < 2:
        return None

    accs = []
    for _ in range(n_trials):
        cls = rng.choice(n_img, k, replace=False)
        Xs = Xf[cls].reshape(-1, feat)
        y = np.repeat(np.arange(k), n_rep)
        tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_tr) for i in range(k)])
        te = np.concatenate([np.arange(i * n_rep + n_rep - n_tr, (i + 1) * n_rep)
                             for i in range(k)])
        Xtr, ytr = Xs[tr], y[tr]
        Xte, yte = Xs[te], y[te]
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-6
        pred = ridge_classify((Xtr - mu) / sd, ytr, (Xte - mu) / sd, k)
        accs.append(float((pred == yte).mean()))
    return float(np.mean(accs)) if accs else None


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp36")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 84, flush=True)
    print("Phase 36 — 轴消融（闭式岭分类器，快速版）", flush=True)
    print("=" * 84, flush=True)
    print(f"  k={K} | 被试 {N_SUBJECTS} | 每变体 {N_TRIALS} 次重复", flush=True)

    rows = []
    t0 = time.time()

    for i in range(1, N_SUBJECTS + 1):
        sid = f"sub-{i:02d}"
        f = src / f"{sid}.npz"
        if not f.exists():
            continue
        print(f"\n  --- {sid} ---", flush=True)
        eeg = np.load(f)["eeg"]
        X = base_correct(eeg)
        a = int(WIN_MS[0] / 1000 * FS)
        b = min(int(WIN_MS[1] / 1000 * FS), X.shape[-1])
        Xw = X[:, :, :, a:b]
        n_img, n_rep, n_ch, T = Xw.shape
        Tb = to_time_bins(Xw)
        print(f"    {Xw.shape} → bins {Tb.shape}", flush=True)

        variants = {
            "A_full": Tb.reshape(n_img, n_rep, n_ch * N_TIME_BINS),
            "B_one_time": Tb[:, :, :, N_TIME_BINS // 2:N_TIME_BINS // 2 + 1].reshape(
                n_img, n_rep, n_ch),
        }
        ch_var = Tb.reshape(n_img * n_rep, n_ch, N_TIME_BINS).var(axis=(0, 2))
        bc = int(np.argmax(ch_var))
        variants["C_one_channel"] = Tb[:, :, bc:bc + 1, :].reshape(n_img, n_rep, N_TIME_BINS)

        bp = band_power(Xw)
        variants["D_band_only"] = bp.reshape(n_img, n_rep, n_ch * len(BANDS))
        variants["E_2rep"] = variants["A_full"]

        for name, Xf in variants.items():
            n_use = 2 if name == "E_2rep" else None
            acc = decode_fast(Xf, K, rng, n_rep_use=n_use)
            if acc is None:
                continue
            rows.append({"subject": sid, "variant": name,
                         "n_feat": int(Xf.shape[-1]),
                         "acc": round(acc, 5), "ratio": round(acc * K, 2)})
            print(f"    {name:<16} feat={Xf.shape[-1]:>4} acc={acc:.4f} "
                  f"({acc*K:.1f}x)", flush=True)
        del eeg, X, Xw, Tb
        print(f"    [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase36.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    if not len(df):
        return

    print("\n" + "=" * 84, flush=True)
    print("消融结果（中位，k=200）", flush=True)
    print("=" * 84, flush=True)
    g = df.groupby("variant").agg(n=("subject", "size"), feat=("n_feat", "median"),
                                  acc=("acc", "median"),
                                  ratio=("ratio", "median")).sort_values(
        "ratio", ascending=False)
    print(g.round(4).to_string(), flush=True)

    full = float(g.loc["A_full", "ratio"]) if "A_full" in g.index else np.nan
    print("\n" + "=" * 84, flush=True)
    print("各轴贡献（相对 full 的损失）", flush=True)
    print("=" * 84, flush=True)
    contrib = {}
    for name in g.index:
        if name == "A_full":
            continue
        r = float(g.loc[name, "ratio"])
        drop = (full - r) / full * 100 if full > 0 else np.nan
        contrib[name] = {"ratio": round(r, 2), "drop_pct": round(drop, 1)}
        print(f"  {name:<16} {r:>6.1f}x   损失 {drop:>5.1f}%", flush=True)

    axis_map = {
        "B_one_time": "time 轴（时间窗数）",
        "C_one_channel": "channel 轴（电极数）",
        "D_band_only": "time 轴（压成频带）",
        "E_2rep": "repetition 轴（试次数）",
    }
    print("\n" + "=" * 84, flush=True)
    print("判定", flush=True)
    print("=" * 84, flush=True)
    verdicts = {}
    for name, axis in axis_map.items():
        if name not in contrib:
            continue
        d = contrib[name]["drop_pct"]
        lvl = "决定性" if d > 50 else "重要" if d > 20 else "次要" if d > 5 else "冗余"
        verdicts[name] = {"axis": axis, **contrib[name], "level": lvl}
        print(f"  {axis:<26} 损失 {d:>5.1f}%  → {lvl}", flush=True)

    (out / "phase36_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "classifier": "closed-form ridge (least-squares) classifier",
        "why_rewritten": "exp35 used 200-class logistic regression (720 fits x 300 iters), too slow",
        "k": K, "n_subjects": N_SUBJECTS, "n_trials": N_TRIALS,
        "window_ms": list(WIN_MS), "lambda": LAMBDA,
        "variants": {k: {"n_feat": int(g.loc[k, "feat"]),
                         "acc": round(float(g.loc[k, "acc"]), 5),
                         "ratio": round(float(g.loc[k, "ratio"]), 2)}
                     for k in g.index},
        "contributions": verdicts, "full_ratio": round(full, 2),
        "per_subject": rows, "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
