"""Phase 37: 真正吃满 CPU 的轴消融 —— 多进程 + 单次大矩阵求解。

## 为什么之前慢（CPU 利用率低的三个原因）

1. **单线程 Python 循环**
   exp36 在被试和变体上全是 for 循环，8 个 vCPU 只用了 1 个。

2. **小矩阵无法喂饱 BLAS**
   每次求解是 (1600 × 630) 的 XᵀX → 630×630 求逆。
   单次矩阵乘 ~6e8 flops，对多线程 BLAS 来说太小，线程调度开销 > 收益。

3. **重复了 5 次相同计算**
   5 次随机抽类只是换了子集，本可以合并成一次张量运算。

## 本版三个改动

### 改动 1: multiprocessing 并行
被试 × 变体 = 4 × 5 = 20 个完全独立的任务，用 8 个 worker 并行。
理论加速 8×。

### 改动 2: 单次大矩阵求解
不再循环 5 次。把 5 次随机抽类的样本**堆成一个大矩阵**：

    X_all ∈ R^{(5 × 1600) × 630}
    Y_all ∈ R^{(5 × 1600) × (5 × 200)}  块对角 one-hot

一次求解 (5×200)=1000 类的 W。矩阵乘 3e9 flops，BLAS 能吃满。

### 改动 3: 显式监控 CPU 利用率
后台起一个线程读 /proc/stat，每 2 秒记录一次 CPU 使用率，
结束时报告均值与峰值。这样"是否吃满"可验证，不是猜的。

## 判据

  CPU 均值 > 70%  → 并行有效
  总耗时 < 5 分钟 → 可接受
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # 单任务内单线程，靠多进程并行

SEED = 0
FS = 250.0
N_SUBJECTS = 4
WIN_MS = (50, 500)
N_TIME_BINS = 10
K = 200
N_TRIALS = 5
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
LAMBDA = 1.0
N_WORKERS = 8


# ------------------------------------------------------------ CPU 监控
class CpuMonitor:
    """后台线程读 /proc/stat，记录 CPU 使用率。"""

    def __init__(self, interval=1.0):
        import threading
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _read(self):
        with open("/proc/stat") as f:
            line = f.readline()
        parts = [int(x) for x in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        return total, idle

    def _run(self):
        t0, i0 = self._read()
        while not self._stop.wait(self.interval):
            t1, i1 = self._read()
            dt, di = t1 - t0, i1 - i0
            if dt > 0:
                self.samples.append((1.0 - di / dt) * 100.0)
            t0, i0 = t1, i1

    def stop(self):
        self._stop.set()
        self._th.join(timeout=3)
        return self.stats()

    def stats(self):
        if not self.samples:
            return {"mean": None, "max": None, "n": 0}
        a = np.array(self.samples)
        return {"mean": round(float(a.mean()), 1),
                "max": round(float(a.max()), 1),
                "n": int(len(a))}


# ------------------------------------------------------------ 特征
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


def build_variants(eeg):
    """返回 {name: (n_img, n_rep, feat)}"""
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


# ------------------------------------------------------------ 一次性大矩阵求解
def decode_batched(Xf, k, seed, n_trials=N_TRIALS, n_rep_use=None):
    """把 n_trials 次随机抽类堆成一个大块对角问题，一次求解。

    返回 (k 类准确率, 特征维度)
    """
    n_img, n_rep, feat = Xf.shape
    if k > n_img:
        return None, feat

    rng = np.random.default_rng(seed)
    n_tr = (n_rep // 2) if n_rep_use is None else min(n_rep_use, n_rep // 2)
    if n_tr < 2:
        return None, feat

    # ---- 构造大矩阵
    Xtr_blocks, ytr_blocks = [], []
    Xte_blocks, yte_blocks = [], []
    offset = 0
    for t in range(n_trials):
        cls = rng.choice(n_img, k, replace=False)
        Xs = Xf[cls].reshape(-1, feat)
        y = np.repeat(np.arange(k), n_rep) + offset
        tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_tr) for i in range(k)])
        te = np.concatenate([np.arange(i * n_rep + n_rep - n_tr, (i + 1) * n_rep)
                             for i in range(k)])
        Xtr_blocks.append(Xs[tr])
        ytr_blocks.append(y[tr])
        Xte_blocks.append(Xs[te])
        yte_blocks.append(y[te])
        offset += k

    Xtr = np.vstack(Xtr_blocks)
    ytr = np.concatenate(ytr_blocks)
    Xte = np.vstack(Xte_blocks)
    yte = np.concatenate(yte_blocks)
    n_class = offset

    # ---- 标准化（用训练集统计）
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    # ---- 闭式岭：一次求解 (n_class) 类
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
    acc = float((pred == yte).mean())
    return acc, feat


# ------------------------------------------------------------ worker
def run_one(task):
    """单个 (被试, 变体) 任务。在子进程里跑。"""
    sid, variant, path = task
    t0 = time.time()
    eeg = np.load(path)["eeg"]
    variants = build_variants(eeg)
    if variant not in variants:
        return None
    Xf = variants[variant]
    n_use = 2 if variant == "E_2rep" else None
    acc, feat = decode_batched(Xf, K, SEED + hash(sid) % 10000, n_rep_use=n_use)
    if acc is None:
        return None
    return {
        "subject": sid, "variant": variant, "n_feat": int(feat),
        "acc": round(acc, 5), "ratio": round(acc * K, 2),
        "sec": round(time.time() - t0, 1),
    }


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp37")
    out.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    print("=" * 84, flush=True)
    print("Phase 37 — 并行轴消融（多进程 + 批量求解）", flush=True)
    print("=" * 84, flush=True)
    print(f"  workers={N_WORKERS} | k={K} | 被试 {N_SUBJECTS} | 变体 5", flush=True)

    # 起 CPU 监控
    mon = CpuMonitor(interval=1.0)

    files = [(f"sub-{i:02d}", src / f"sub-{i:02d}.npz") for i in range(1, N_SUBJECTS + 1)]
    files = [(s, p) for s, p in files if p.exists()]
    variants = ["A_full", "B_one_time", "C_one_channel", "D_band_only", "E_2rep"]
    tasks = [(s, v, str(p)) for s, p in files for v in variants]

    print(f"  任务数 {len(tasks)}", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)
                print(f"    {r['subject']} {r['variant']:<16} "
                      f"feat={r['n_feat']:>4} acc={r['acc']:.4f} "
                      f"({r['ratio']:.1f}x) [{r['sec']}s]", flush=True)
    elapsed = time.time() - t0

    cpu = mon.stop()
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase37.csv", index=False)

    print("\n" + "=" * 84, flush=True)
    print(f"总耗时 {elapsed:.0f}s | CPU 均值 {cpu['mean']}% 峰值 {cpu['max']}%",
          flush=True)
    print("=" * 84, flush=True)

    if not len(df):
        return

    print("\n消融结果（中位，k=200）", flush=True)
    print("-" * 84, flush=True)
    g = df.groupby("variant").agg(n=("subject", "size"), feat=("n_feat", "median"),
                                  acc=("acc", "median"),
                                  ratio=("ratio", "median")).sort_values(
        "ratio", ascending=False)
    print(g.round(4).to_string(), flush=True)

    full = float(g.loc["A_full", "ratio"]) if "A_full" in g.index else np.nan
    print("\n各轴贡献（相对 full）", flush=True)
    print("-" * 84, flush=True)
    contrib = {}
    for name in g.index:
        if name == "A_full":
            continue
        r = float(g.loc[name, "ratio"])
        drop = (full - r) / full * 100 if full > 0 else np.nan
        contrib[name] = {"ratio": round(r, 2), "drop_pct": round(drop, 1)}
        print(f"  {name:<16} {r:>6.1f}x   损失 {drop:>5.1f}%", flush=True)

    axis_map = {"B_one_time": "time（时间窗数）",
                "C_one_channel": "channel（电极数）",
                "D_band_only": "time（压成频带）",
                "E_2rep": "repetition（试次数）"}
    print("\n判定", flush=True)
    print("-" * 84, flush=True)
    verdicts = {}
    for name, axis in axis_map.items():
        if name not in contrib:
            continue
        d = contrib[name]["drop_pct"]
        lvl = "决定性" if d > 50 else "重要" if d > 20 else "次要" if d > 5 else "冗余"
        verdicts[name] = {"axis": axis, **contrib[name], "level": lvl}
        print(f"  {axis:<24} 损失 {d:>5.1f}%  → {lvl}", flush=True)

    (out / "phase37_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "optimizations": ["multiprocessing 8 workers", "batched ridge (single solve for 5x200 classes)",
                          "OMP_NUM_THREADS=1 per worker to avoid oversubscription"],
        "k": K, "n_subjects": N_SUBJECTS, "n_trials": N_TRIALS,
        "n_workers": N_WORKERS, "lambda": LAMBDA,
        "elapsed_s": round(elapsed, 1),
        "cpu": cpu,
        "variants": {kk: {"n_feat": int(g.loc[kk, "feat"]),
                          "acc": round(float(g.loc[kk, "acc"]), 5),
                          "ratio": round(float(g.loc[kk, "ratio"]), 2)}
                     for kk in g.index},
        "contributions": verdicts, "full_ratio": round(full, 2),
        "per_subject": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
