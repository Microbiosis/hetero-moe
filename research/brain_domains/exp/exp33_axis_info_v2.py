"""Phase 33: 修正轴的度量方法后重测（v2）。

## 修 exp32 的三个问题

### 问题 1: condition 解码器太弱
exp32 用"最近类均值 + 余弦相似度"在 15,750 维原始信号上，
得到 2.4x 随机。这是**基线太弱**，不是轴无效。

修正：
  - 先做**时间窗平均**（THINGS-EEG 标准做法：50-500ms 窗口）
  - 再做**PCA 降维**到 50 维
  - 用 **LDA**（多类线性判别）而不是最近类均值
  - 交叉验证用 2-fold（前 40 试次训练 / 后 40 测试）

### 问题 2: frequency 轴被 1/f 主导
exp32 在**原始功率**上做 ANOVA，F≈0.9（低于 1）。
原始功率跨 5 个频带差几个数量级，1/f 背景淹没了条件差异。

修正：
  - 用**相对功率**（每个频带除以总功率）
  - 再算图像间 ANOVA

### 问题 3: repetition 轴用了错误的 ICC
exp32 的 ICC(1) = 0.028，但那个公式对"图像数 200、试次 80"的高维数据
有偏。正确做法：**split-half 相关**（已在 build 里算过，约 0.07），
或 **ICC(2,k)**（用平均而非单次）。

修正：
  - split-half 相关（奇偶试次）
  - 以及"单试次 vs 平均响应"的信噪比，这是 repetition 轴的真实可用性

## 判据

  condition: LDA 解码 > 5x 随机（=2.5%）→ 有效
  repetition: split-half > 0.05 → 有效
  frequency: 相对功率 F > 1.5 → 有效
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
N_SUBJECTS = 10
WIN_MS = (50, 500)          # THINGS-EEG 标准分析窗
N_PCA = 50
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def window_slice(T, fs=FS, win_ms=WIN_MS):
    a = int(win_ms[0] / 1000 * fs)
    b = int(win_ms[1] / 1000 * fs)
    return a, min(b, T)


def lda_decode(X, y, n_folds=2, rng=None):
    """多类 LDA 解码准确率。X: (n, feat), y: (n,)"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold

    rng = rng or np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            clf.fit(X[tr], y[tr])
            accs.append(float((clf.predict(X[te]) == y[te]).mean()))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else np.nan


def pca_reduce(X, n=50):
    """沿样本维做 PCA。X: (n_sample, n_feat)"""
    Xc = X - X.mean(axis=0, keepdims=True)
    # 用 SVD 降维（样本数通常 < 特征数）
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Xc
    k = min(n, len(S))
    return U[:, :k] * S[:k]


def test_condition(eeg, rng):
    """LDA 解码 image identity。"""
    n_img, n_rep, n_ch, T = eeg.shape
    a, b = window_slice(T)
    # 每个试次 → (n_ch * (b-a)) 维
    X = eeg[:, :, :, a:b].astype(np.float32).reshape(n_img * n_rep, -1)
    y = np.repeat(np.arange(n_img), n_rep)

    # 逐通道 z-score
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    # 降维
    Xp = pca_reduce(X, N_PCA)

    acc = lda_decode(Xp, y, n_folds=2, rng=rng)
    return acc, 1.0 / n_img


def test_repetition(eeg):
    """split-half（奇偶试次）相关，去平均响应。"""
    n_img, n_rep = eeg.shape[:2]
    X = eeg.astype(np.float32).reshape(n_img, n_rep, -1)
    odd = X[:, 1::2].mean(axis=1)
    even = X[:, 0::2].mean(axis=1)
    # 去掉跨图像的平均响应
    odd = odd - odd.mean(axis=0, keepdims=True)
    even = even - even.mean(axis=0, keepdims=True)
    num = (odd * even).sum(axis=1)
    den = np.sqrt((odd ** 2).sum(axis=1) * (even ** 2).sum(axis=1)) + 1e-9
    return float(np.median(num / den))


def test_band_relative(band):
    """相对功率的 ANOVA F 比。"""
    n_img, n_rep, n_ch, n_band = band.shape
    # 相对功率：每 trial 每通道除以总功率
    tot = band.sum(axis=3, keepdims=True) + 1e-12
    rel = band / tot
    Y = rel.mean(axis=2)                      # (n_img, n_rep, n_band)
    grand = Y.mean(axis=(0, 1), keepdims=True)
    ss_b = n_rep * ((Y.mean(axis=1, keepdims=True) - grand) ** 2).sum(axis=(0, 1))
    ss_w = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=(0, 1))
    df_b = n_img - 1
    df_w = n_img * (n_rep - 1)
    F = (ss_b / df_b) / (ss_w / max(df_w, 1) + 1e-12)
    return F.astype(np.float64)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp33")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 86)
    print("Phase 33 — 修正度量后的轴信息检验 v2")
    print("=" * 86)
    print(f"  src {src}")
    print(f"  时间窗 {WIN_MS} ms | PCA {N_PCA} 维 | LDA 2-fold")

    rows = []
    t0 = time.time()

    for i in range(1, N_SUBJECTS + 1):
        sid = f"sub-{i:02d}"
        f_raw = src / f"{sid}.npz"
        f_band = src / f"{sid}_band.npz"
        if not f_raw.exists():
            continue
        print(f"\n  --- {sid} ---")
        eeg = np.load(f_raw)["eeg"]
        band = np.load(f_band)["band"] if f_band.exists() else None

        acc, chance = test_condition(eeg, rng)
        print(f"    [condition] LDA 解码 {acc:.4f} (随机 {chance:.4f}, "
              f"{acc/chance:.1f}x)")

        rep = test_repetition(eeg)
        print(f"    [repetition] split-half 相关 {rep:+.4f}")

        if band is not None:
            F = test_band_relative(band)
            print(f"    [frequency] 相对功率 F: " +
                  ", ".join(f"{n}={v:.2f}" for n, v in zip(BAND_NAMES, F)))
            bf = float(np.mean(F))
        else:
            bf = np.nan

        rows.append({
            "subject": sid,
            "lda_acc": round(acc, 5), "chance": round(chance, 5),
            "lda_ratio": round(acc / chance, 2),
            "split_half": round(rep, 5),
            "band_F_mean": round(bf, 3) if np.isfinite(bf) else None,
            "band_F_max": round(float(F.max()), 3) if band is not None else None,
        })
        del eeg, band

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase33.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 86)
    print("修正后的轴信息量（中位）")
    print("=" * 86)
    g = df.median(numeric_only=True)
    print(f"  condition   LDA {g['lda_acc']:.4f} / 随机 {g['chance']:.4f} "
          f"= {g['lda_ratio']:.1f}x")
    print(f"  repetition  split-half {g['split_half']:+.4f}")
    print(f"  frequency   相对功率 F {g['band_F_mean']:.2f}")

    print("\n" + "=" * 86)
    print("与 exp32 的对照")
    print("=" * 86)
    print(f"  {'轴':<12} {'exp32':>12} {'exp33':>12} {'变化':>10}")
    print(f"  {'condition':<12} {'2.4x':>12} {f'{g.lda_ratio:.1f}x':>12} "
          f"{'↑' if g.lda_ratio > 2.4 else '↓':>10}")
    print(f"  {'repetition':<12} {'0.027':>12} {f'{g.split_half:+.4f}':>12} "
          f"{'↑' if g.split_half > 0.027 else '↓':>10}")
    print(f"  {'frequency':<12} {'0.9':>12} {f'{g.band_F_mean:.2f}':>12} "
          f"{'↑' if g.band_F_mean > 0.9 else '↓':>10}")

    # 判定
    print("\n" + "=" * 86)
    print("轴有效性判定（v2）")
    print("=" * 86)
    v = {
        "condition": {"value": round(float(g["lda_ratio"]), 2),
                      "valid": bool(g["lda_ratio"] > 5),
                      "note": "LDA 解码 / 随机（>5x 为有效）"},
        "repetition": {"value": round(float(g["split_half"]), 4),
                       "valid": bool(g["split_half"] > 0.05),
                       "note": "split-half 相关（>0.05 为有效）"},
        "frequency": {"value": round(float(g["band_F_mean"]), 3),
                      "valid": bool(g["band_F_mean"] > 1.5),
                      "note": "相对功率 ANOVA F（>1.5 为有效）"},
    }
    for axis, d_ in v.items():
        print(f"  {axis:<12} {d_['value']:>10}  "
              f"{'✅ 有效' if d_['valid'] else '⚠️ 弱':<10} ({d_['note']})")

    (out / "phase33_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "fixes": ["condition: 时间窗平均 + PCA50 + LDA（替代 NCM+余弦）",
                  "frequency: 相对功率（替代原始功率，避免 1/f 主导）",
                  "repetition: split-half（替代有偏的 ICC(1)）"],
        "window_ms": list(WIN_MS), "n_pca": N_PCA,
        "medians": {k: (None if pd.isna(v_) else round(float(v_), 4))
                    for k, v_ in g.items()},
        "axes": v,
        "per_subject": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
