"""Phase 35: 用已验证的解码器做轴消融 —— 每个轴贡献多少信息？

## 核心思路

exp34 证明 condition 轴有效（200 类 7.6-12.5%，15-25x 随机）。
现在把这个解码器当成**测量仪器**，逐个去掉一个轴，看解码性能掉多少。

    去掉时间轴（只用一个时间窗）  → 掉多少？
    去掉频率轴（只用宽带功率）    → 掉多少？
    去掉电极轴（只用一个电极）    → 掉多少？
    去掉试次轴（用 1 次 vs 平均 40 次）→ 掉多少？

这比"每个轴单独算个统计量"强得多，因为**都在同一个任务上测量**。

## 五个消融实验

    A  full            (63 电极 × 10 时间段) = 630 维
    B  time=1 段       只用 150-250ms 一段 = 63 维
    C  单电极          只用最优 1 个电极 × 10 段 = 10 维
    D  只用频带功率    5 个频带 × 63 电极 = 315 维（时间信息被压掉）
    E  只用试次平均    1 次 vs 40 次（信噪比曲线）

## 预期

    B 掉最多  → 时间轴最重要（有锁时成分）
    D 掉较少  → 频带功率也携带信息，但少于时间波形
    C 掉很多  → 空间分布重要
    E 单调    → 单试次信息量

## 判据

  某轴消融后准确率降到随机水平的 <5x → 该轴是主要贡献者
  降到 >15x 不变                        → 该轴冗余
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
N_SUBJECTS = 6
WIN_MS = (50, 500)
N_TIME_BINS = 10
KS = [2, 10, 200]
N_TRIALS = 8
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def base_correct(eeg):
    """减 0-50ms 基线。"""
    a = int(WIN_MS[0] / 1000 * FS)
    base = eeg[:, :, :, :max(a, 1)].mean(axis=3, keepdims=True)
    return eeg.astype(np.float32) - base


def to_time_bins(X, n_bins=N_TIME_BINS):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch, n_bins)"""
    T = X.shape[-1]
    seg = T // n_bins
    if seg < 1:
        return None
    return X[:, :, :, :seg * n_bins].reshape(
        X.shape[0], X.shape[1], X.shape[2], n_bins, seg).mean(axis=4)


def band_power(X, fs=FS):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch, n_band)"""
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


def decode(Xf, k, rng, n_trials=N_TRIALS, n_rep_use=None):
    """Xf: (n_img, n_rep, feat) 已构好特征。按 repetition 切分。"""
    from sklearn.linear_model import LogisticRegression

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
        try:
            clf = LogisticRegression(max_iter=300, C=0.1, solver="lbfgs")
            clf.fit((Xtr - mu) / sd, ytr)
            accs.append(float((clf.predict((Xte - mu) / sd) == yte).mean()))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else None


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp35")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 88, flush=True)
    print("Phase 35 — 轴消融：每个轴对 200 类解码贡献多少", flush=True)
    print("=" * 88, flush=True)

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
        Xw = X[:, :, :, a:b]                       # 50-500ms
        n_img, n_rep, n_ch, T = Xw.shape

        Tb = to_time_bins(Xw)                      # (img, rep, ch, n_bins)
        print(f"    shape {Xw.shape} → time bins {Tb.shape}", flush=True)

        variants = {}

        # A) full: ch × time_bins
        variants["A_full"] = Tb.reshape(n_img, n_rep, n_ch * N_TIME_BINS)

        # B) 单时间窗（150-250ms，中间段）
        mid = N_TIME_BINS // 2
        variants["B_one_time"] = Tb[:, :, :, mid:mid + 1].reshape(n_img, n_rep, n_ch)

        # C) 单电极（选方差最大的通道）
        ch_var = Tb.reshape(n_img * n_rep, n_ch, N_TIME_BINS).var(axis=(0, 2))
        best_ch = int(np.argmax(ch_var))
        variants["C_one_channel"] = Tb[:, :, best_ch:best_ch + 1, :].reshape(
            n_img, n_rep, N_TIME_BINS)

        # D) 只用频带功率（时间压掉）
        bp = band_power(Xw)
        variants["D_band_only"] = bp.reshape(n_img, n_rep, n_ch * len(BANDS))

        # E) 只用时间波形，无频带（等价 A，但作为对照）
        #    以及 E2: 单试次（n_rep_use=2）
        variants["E_full_2rep"] = variants["A_full"]

        for name, Xf in variants.items():
            n_use = 2 if name == "E_full_2rep" else None
            accs = {}
            for k in KS:
                acc = decode(Xf, k, rng, n_rep_use=n_use)
                if acc is not None:
                    accs[k] = acc
            if not accs:
                continue
            r200 = accs.get(200)
            rows.append({
                "subject": sid, "variant": name, "n_feat": int(Xf.shape[-1]),
                "acc_k2": round(accs.get(2, np.nan), 4),
                "acc_k10": round(accs.get(10, np.nan), 4),
                "acc_k200": round(r200, 4) if r200 else None,
                "ratio_k200": round(r200 * 200, 2) if r200 else None,
            })
            print(f"    {name:<18} feat={Xf.shape[-1]:>4} "
                  f"k2={accs.get(2, float('nan')):.3f} "
                  f"k10={accs.get(10, float('nan')):.3f} "
                  f"k200={r200 if r200 is None else round(r200,4)} "
                  f"({r200*200 if r200 else 0:.1f}x)", flush=True)
        del eeg, X, Xw, Tb

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase35.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    if not len(df):
        return

    print("\n" + "=" * 88, flush=True)
    print("轴消融结果（中位，k=200）", flush=True)
    print("=" * 88, flush=True)
    g = df.groupby("variant").agg(
        n=("subject", "size"), feat=("n_feat", "median"),
        acc200=("acc_k200", "median"), ratio=("ratio_k200", "median")).sort_values(
        "ratio", ascending=False)
    print(g.round(3).to_string(), flush=True)

    print("\n" + "=" * 88, flush=True)
    print("各轴贡献（相对 full 的损失）", flush=True)
    print("=" * 88, flush=True)
    full = g.loc["A_full", "ratio"] if "A_full" in g.index else np.nan
    contrib = {}
    for name in g.index:
        if name == "A_full":
            continue
        r = g.loc[name, "ratio"]
        drop = (full - r) / full * 100 if full > 0 else np.nan
        contrib[name] = {"ratio": round(float(r), 2), "drop_pct": round(float(drop), 1)}
        print(f"  {name:<18} {r:>6.1f}x  损失 {drop:>5.1f}%", flush=True)

    # 判定
    print("\n" + "=" * 88, flush=True)
    print("判定", flush=True)
    print("=" * 88, flush=True)
    axis_map = {
        "B_one_time": "time 轴（时间窗数）",
        "C_one_channel": "channel 轴（电极数）",
        "D_band_only": "time 轴（被压成频带）",
        "E_full_2rep": "repetition 轴（试次数）",
    }
    verdicts = {}
    for name, axis in axis_map.items():
        if name not in contrib:
            continue
        d = contrib[name]["drop_pct"]
        if d > 50:
            lvl = "决定性"
        elif d > 20:
            lvl = "重要"
        elif d > 5:
            lvl = "次要"
        else:
            lvl = "冗余"
        verdicts[name] = {"axis": axis, **contrib[name], "level": lvl}
        print(f"  {axis:<26} 损失 {d:>5.1f}%  → {lvl}", flush=True)

    (out / "phase35_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "design": "ablate one axis at a time, measure 200-class decoding",
        "ks": KS, "window_ms": list(WIN_MS), "n_time_bins": N_TIME_BINS,
        "variants": {k: {"n_feat": int(g.loc[k, "feat"]),
                         "acc200": round(float(g.loc[k, "acc200"]), 4),
                         "ratio": round(float(g.loc[k, "ratio"]), 2)}
                     for k in g.index},
        "contributions": verdicts,
        "full_ratio": round(float(full), 2) if np.isfinite(full) else None,
        "per_subject": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
