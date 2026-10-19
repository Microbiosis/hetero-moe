"""Phase 39: τ 的分辨率不变性检验（Space 内跑）。

## 为什么必须做

Space 的 run_tau 给出 sub-01 τ = 145 ms（IQR 62 ms）。
但 exp19 的教训是：**如果 τ 随采样率变化，它就不是真实时间常数**。

exp19 当时发现 τ 固定在 3.5 bins（随 bin 宽度缩放），
根因是"窗口内取平均"这个平滑操作。本任务的 VAR(1) 是在**逐样本**上做的，
没有窗口平均，理论上应该能得到真实 τ。

但仍须验证：把 250 Hz 降到 125 / 62.5 / 31.25 Hz，τ(ms) 是否一致？

    若一致 → 真实时间常数
    若随降采样线性缩放 → 仍是分辨率伪影

## 判据

  三种采样率下 τ(ms) 的比值都在 [0.75, 1.35] → 通过
  否则 → 不可信

## 同时测多个被试

  sub-01 ~ sub-05，各 3 种采样率

> ## ⚠️ 估计量已弃用（2026-10-19）
>
> 本脚本用 `np.median(acf_fft(P), axis=1)` 估 τ，这是**基依赖**的：
> 换一组正交基 τ 就变。实测 20 ms spike 上旋转 5 次变化 1.65 倍。
>
> 正交不变的替代是 `tr C(τ) / tr C(0)`，见 `lib/tau.py` 与
> `FINDINGS_TAU_GAUGE.md`。用不变量重跑的结果：`exp51_fix.py`、`exp52_fix43.py`。
>
> 本文件保留以记录原始流程，结论以 FINDINGS_TAU_GAUGE.md 为准。
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
EEG_DIR = Path("/domains/omnibrain4d/eeg")
OUT = Path("/domains/runs/tau-resolution")
N_PCS = 32
MAX_IMAGES = 30
DECIMATIONS = [1, 2, 4]        # 250 / 125 / 62.5 Hz
SUBJECTS = [1, 2, 3, 4, 5]


def estimate_tau(eeg, target_fs, n_pcs=N_PCS, max_images=MAX_IMAGES):
    """在指定采样率下估 τ（ms）。"""
    n_img, n_rep, n_ch, T = eeg.shape
    dec = int(round(FS / target_fs))
    if dec < 1:
        return None

    d = min(n_pcs, n_ch)
    taus = []
    n_use = min(n_img, max_images)

    for i in range(n_use):
        seg = eeg[i].astype(np.float32)                 # (n_rep, n_ch, T)
        if dec > 1:
            seg = seg[:, :, ::dec]                       # 降采样
        Ti = seg.shape[-1]
        X = seg.transpose(1, 0, 2).reshape(n_ch, n_rep * Ti)

        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-6
        Z = ((X - mu) / sd).T                            # (time, ch)

        Zc = Z - Z.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(Zc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        P = U[:, :d] * S[:d]                             # (time, d)

        A_, B_ = P[:-1], P[1:]
        G = A_.T @ A_
        G.flat[:: d + 1] += 1e-2 * np.trace(G) / d
        try:
            W = np.linalg.solve(G, A_.T @ B_)
        except np.linalg.LinAlgError:
            continue
        ev = np.abs(np.linalg.eigvals(W))
        ev = ev[(ev > 1e-4) & (ev < 1.0)]
        if len(ev) == 0:
            continue
        lam = float(np.sort(ev)[::-1][0])
        dt_ms = 1000.0 / target_fs                     # 每步的毫秒数
        taus.append(-1.0 / np.log(lam) * dt_ms)

    if not taus:
        return None
    return {
        "tau_ms": float(np.median(taus)),
        "iqr": float(np.percentile(taus, 75) - np.percentile(taus, 25)),
        "n": len(taus),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 39 — τ 的分辨率不变性检验", flush=True)
    print("=" * 78, flush=True)
    print(f"  被试 {SUBJECTS} | 采样率 {[int(FS/d) for d in DECIMATIONS]} Hz "
          f"| {N_PCS} PCA 维 | {MAX_IMAGES} 图像", flush=True)

    rows = []
    t0 = time.time()

    for sid in SUBJECTS:
        f = EEG_DIR / f"sub-{sid:02d}.npz"
        if not f.exists():
            print(f"  [skip] sub-{sid:02d}", flush=True)
            continue
        print(f"\n  --- sub-{sid:02d} ---", flush=True)
        eeg = np.load(f)["eeg"]

        for dec in DECIMATIONS:
            fs = FS / dec
            t1 = time.time()
            r = estimate_tau(eeg, fs)
            if r is None:
                continue
            rows.append({
                "subject": f"sub-{sid:02d}", "fs_hz": round(fs, 2),
                "decimation": dec, "tau_ms": round(r["tau_ms"], 1),
                "iqr": round(r["iqr"], 1), "n_images": r["n"],
            })
            print(f"    {fs:>6.2f} Hz: τ={r['tau_ms']:>7.1f} ms "
                  f"(IQR {r['iqr']:>5.1f}) [{time.time()-t1:.1f}s]", flush=True)
        del eeg

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    if not len(df):
        return

    print("\n" + "=" * 78, flush=True)
    print("分辨率不变性判定", flush=True)
    print("=" * 78, flush=True)
    print(f"{'被试':<8} {'250Hz':>9} {'125Hz':>9} {'62.5Hz':>9} "
          f"{'比值2x':>8} {'比值4x':>8} {'判定':>8}", flush=True)

    verdicts = {}
    for sid, g in df.groupby("subject"):
        m = g.set_index("fs_hz")["tau_ms"]
        base = m.get(250.0)
        if base is None:
            continue
        r2 = m.get(125.0, np.nan) / base
        r4 = m.get(62.5, np.nan) / base
        ok = (0.75 <= r2 <= 1.35) and (0.75 <= r4 <= 1.35)
        verdicts[sid] = {"base_250": round(float(base), 1),
                         "ratio_125": round(float(r2), 3),
                         "ratio_62": round(float(r4), 3),
                         "consistent": bool(ok)}
        print(f"{sid:<8} {base:>9.1f} {m.get(125.0, np.nan):>9.1f} "
              f"{m.get(62.5, np.nan):>9.1f} {r2:>8.3f} {r4:>8.3f} "
              f"{'✅' if ok else '❌':>8}", flush=True)

    n_ok = sum(1 for v in verdicts.values() if v["consistent"])
    n_total = len(verdicts)
    print(f"\n  一致: {n_ok}/{n_total}", flush=True)

    if n_total and n_ok == n_total:
        verdict = ("✅ τ 通过分辨率不变性检验 —— 是真实时间常数，"
                   f"中位 {df[df.fs_hz==250.0]['tau_ms'].median():.0f} ms")
    elif n_ok == 0:
        verdict = ("❌ τ 随采样率系统变化 —— 仍是分辨率伪影，"
                   "不能作为生物时间常数报告")
    else:
        verdict = f"⚠️ {n_ok}/{n_total} 通过，证据不充分"
    print(f"\n  {verdict}", flush=True)

    (OUT / "report.json").write_text(json.dumps({
        "task": "tau_resolution_invariance",
        "fs_list": [FS / d for d in DECIMATIONS],
        "n_pcs": N_PCS, "max_images": MAX_IMAGES,
        "rows": rows, "verdicts": verdicts,
        "n_consistent": n_ok, "n_total": n_total,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
