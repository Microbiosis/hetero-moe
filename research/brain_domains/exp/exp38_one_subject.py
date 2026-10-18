"""Phase 38: 单被试轴消融 —— 一个进程只做一个被试，靠多 job 并行。

## 为什么放弃进程内并行

exp37 用 ProcessPoolExecutor + fork，结果 CPU 0%、RAM 1/32 GB、完全挂死。

根因：**fork 之后 BLAS 死锁**。
父进程 `import numpy` 时初始化了 OpenBLAS 线程池（内部有 mutex）。
`fork()` 出的子进程继承了这个**已被锁住**的 mutex 状态，
子进程第一次调用 BLAS（矩阵乘/求逆）就永久阻塞。

即使设了 OMP_NUM_THREADS=1 也没用——OpenBLAS 的初始化锁在 import 时就已经拿了。

出路有两条：
  a) 用 `spawn` 而非 fork（子进程重新 import，无继承锁）—— 但每次 spawn 要
     重新加载 numpy + 读取 485MB 的 npz，开销大
  b) **任务级并行**：起 N 个独立的 HF Job，每个 job 一个进程只做一个被试

本脚本采用 (b)。它接受一个 subject 参数，只做该被试的 5 个变体，
单进程顺序执行，无任何 fork。4 个被试 = 4 个并行 job。

## CPU 使用

单进程内用满 1 核（BLAS 多线程在 630 维上收益有限）。
真正的并行来自 4 个 job 同时跑：总 CPU = 4 核。

## 为什么这次不会挂

没有 fork、没有 ProcessPoolExecutor、没有共享状态。
纯 numpy 顺序计算，与 exp34 相同的执行模式（exp34 跑通了）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FS = 250.0
WIN_MS = (50, 500)
N_TIME_BINS = 10
K = 200
N_TRIALS = 5
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
LAMBDA = 1.0
SEED = 0


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


def decode_batched(Xf, k, seed, n_trials=N_TRIALS, n_rep_use=None):
    """5 次随机抽类堆成一个大块对角问题，一次求解。"""
    n_img, n_rep, feat = Xf.shape
    if k > n_img:
        return None, feat

    rng = np.random.default_rng(seed)
    n_tr = (n_rep // 2) if n_rep_use is None else min(n_rep_use, n_rep // 2)
    if n_tr < 2:
        return None, feat

    Xtr_b, ytr_b, Xte_b, yte_b = [], [], [], []
    offset = 0
    for _ in range(n_trials):
        cls = rng.choice(n_img, k, replace=False)
        Xs = Xf[cls].reshape(-1, feat)
        y = np.repeat(np.arange(k), n_rep) + offset
        tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_tr) for i in range(k)])
        te = np.concatenate([np.arange(i * n_rep + n_rep - n_tr, (i + 1) * n_rep)
                             for i in range(k)])
        Xtr_b.append(Xs[tr]); ytr_b.append(y[tr])
        Xte_b.append(Xs[te]); yte_b.append(y[te])
        offset += k

    Xtr = np.vstack(Xtr_b); ytr = np.concatenate(ytr_b)
    Xte = np.vstack(Xte_b); yte = np.concatenate(yte_b)
    n_class = offset

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    n, d = Xtr.shape
    Y = np.zeros((n, n_class), dtype=np.float32)
    Y[np.arange(n), ytr] = 1.0
    A = Xtr.T @ Xtr
    A.flat[:: d + 1] += LAMBDA * np.trace(A) / d
    B = Xtr.T @ Y
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, B, rcond=None)[0]

    pred = (Xte @ W).argmax(axis=1)
    return float((pred == yte).mean()), feat


def build_variants(eeg):
    X = base_correct(eeg)
    a = int(WIN_MS[0] / 1000 * FS)
    b = min(int(WIN_MS[1] / 1000 * FS), X.shape[-1])
    Xw = X[:, :, :, a:b]
    n_img, n_rep, n_ch, _ = Xw.shape

    Tb = to_time_bins(Xw)
    out = {
        "A_full": Tb.reshape(n_img, n_rep, n_ch * N_TIME_BINS),
        "B_one_time": Tb[:, :, :, N_TIME_BINS // 2:N_TIME_BINS // 2 + 1].reshape(
            n_img, n_rep, n_ch),
    }
    ch_var = Tb.reshape(n_img * n_rep, n_ch, N_TIME_BINS).var(axis=(0, 2))
    bc = int(np.argmax(ch_var))
    out["C_one_channel"] = Tb[:, :, bc:bc + 1, :].reshape(n_img, n_rep, N_TIME_BINS)

    bp = band_power(Xw)
    out["D_band_only"] = bp.reshape(n_img, n_rep, n_ch * len(BANDS))
    out["E_2rep"] = out["A_full"]
    return out


def main():
    # 用法: python exp38_one_subject.py <eeg_dir> <out_dir> <subject_idx>
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp38")
    sidx = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out.mkdir(parents=True, exist_ok=True)

    sid = f"sub-{sidx:02d}"
    f = src / f"{sid}.npz"
    if not f.exists():
        print(f"❌ 缺 {f}", flush=True)
        return

    print("=" * 78, flush=True)
    print(f"Phase 38 — {sid} 轴消融（单进程，无 fork）", flush=True)
    print("=" * 78, flush=True)
    print(f"  k={K} | 变体 5 | 每变体 {N_TRIALS} 次重复", flush=True)

    t0 = time.time()
    eeg = np.load(f)["eeg"]
    print(f"  载入 {eeg.shape} ({time.time()-t0:.1f}s)", flush=True)

    variants = build_variants(eeg)
    print(f"  特征构造完成 ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for name, Xf in variants.items():
        n_use = 2 if name == "E_2rep" else None
        t1 = time.time()
        acc, feat = decode_batched(Xf, K, SEED + sidx, n_rep_use=n_use)
        if acc is None:
            continue
        rows.append({"subject": sid, "variant": name, "n_feat": int(feat),
                     "acc": round(acc, 5), "ratio": round(acc * K, 2),
                     "sec": round(time.time() - t1, 1)})
        print(f"  {name:<16} feat={feat:>4} acc={acc:.4f} ({acc*K:.1f}x) "
              f"[{time.time()-t1:.1f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{sid}_ablation.csv", index=False)

    print("\n" + "-" * 78, flush=True)
    print(df.to_string(index=False), flush=True)
    full = df[df.variant == "A_full"]["ratio"]
    if len(full):
        base = float(full.iloc[0])
        print(f"\n  相对 A_full ({base:.1f}x) 的损失:", flush=True)
        for _, r in df.iterrows():
            if r["variant"] == "A_full":
                continue
            print(f"    {r['variant']:<16} {r['ratio']:>6.1f}x  "
                  f"损失 {(base-r['ratio'])/base*100:>5.1f}%", flush=True)

    (out / f"{sid}_report.json").write_text(json.dumps({
        "subject": sid, "k": K, "n_trials": N_TRIALS,
        "variants": rows, "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  完成，耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
