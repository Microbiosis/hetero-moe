"""Phase 34: 用正确方法测 condition 轴（2 / 10 / 200 类递进）。

## 为什么 exp33 的 LDA 失败了

exp33 结果 1.7x 随机，比 exp32 的朴素基线（2.4x）还差。三个错误：

1. **200 类 LDA 不适用**
   LDA 判别方向上限 = n_classes-1 = 199，但每折每类仅 40 样本。
   shrinkage 在这么稀疏的条件下过度收缩，判别力被抹平。

2. **PCA 在全部样本上做（含测试集）**
   既是泄漏，也让方向偏向"重建整体方差"而非"区分类别"。

3. **特征未做基线校正 + 未降采样**
   63 × 200 时间点 = 15,750 维，噪声主导。

## 本实验三层递进

### L1: 2 类解码（最可靠起点）
随机 = 50%。若 2 类都做不到 → 流程有问题。

### L2: 10 类
随机 = 10%。

### L3: 200 类
随机 = 0.5%。文献用更强流程可达 20-40%。

严格交叉验证：训练/测试按 repetition 切（前 40 / 后 40），
同一张图的不同重复不跨折。

## 判据

  L1 > 60%  → 流程正确，condition 轴有信号
  L2 > 20%  → 更强证据
  L3 > 2%   → 与文献同量级
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
N_PERM = 30


def prep_features(eeg, win_ms=WIN_MS, n_bins=N_TIME_BINS):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch*n_bins)"""
    n_img, n_rep, n_ch, T = eeg.shape
    a = int(win_ms[0] / 1000 * FS)
    b = min(int(win_ms[1] / 1000 * FS), T)
    if b - a < n_bins:
        return None

    # 基线校正：减 0-50ms 均值
    base = eeg[:, :, :, :max(a, 1)].mean(axis=3, keepdims=True)
    X = eeg[:, :, :, a:b].astype(np.float32) - base

    # 时间降采样
    seg = (b - a) // n_bins
    X = X[:, :, :, :seg * n_bins].reshape(n_img, n_rep, n_ch, n_bins, seg)
    X = X.mean(axis=4)
    return X.reshape(n_img, n_rep, n_ch * n_bins)


def decode_k(X, k, rng, n_perm=N_PERM, n_trials=15):
    """从 n_img 类抽 k 类，按 repetition 切分，逻辑回归解码。"""
    from sklearn.linear_model import LogisticRegression

    n_img, n_rep, feat = X.shape
    if k > n_img:
        return None, None

    n_train = n_rep // 2
    accs, null = [], []

    for _ in range(n_trials):
        cls = rng.choice(n_img, k, replace=False)
        Xs = X[cls].reshape(-1, feat)
        y = np.repeat(np.arange(k), n_rep)

        tr = np.concatenate([np.arange(i * n_rep, i * n_rep + n_train)
                             for i in range(k)])
        te = np.concatenate([np.arange(i * n_rep + n_train, (i + 1) * n_rep)
                             for i in range(k)])

        Xtr, ytr = Xs[tr], y[tr]
        Xte, yte = Xs[te], y[te]

        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

        try:
            clf = LogisticRegression(max_iter=300, C=0.1, solver="lbfgs")
            clf.fit(Xtr, ytr)
            accs.append(float((clf.predict(Xte) == yte).mean()))
        except Exception:
            continue

        if len(null) < n_perm:
            try:
                clf2 = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs")
                clf2.fit(Xtr, rng.permutation(ytr))
                null.append(float((clf2.predict(Xte) == yte).mean()))
            except Exception:
                pass

    if not accs:
        return None, None
    return float(np.mean(accs)), (float(np.mean(null)) if null else None)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp34")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 86, flush=True)
    print("Phase 34 — 正确的 condition 解码（2 / 10 / 200 类递进）", flush=True)
    print("=" * 86, flush=True)
    print(f"  窗口 {WIN_MS} ms | 时间降采样 {N_TIME_BINS} 段 | LR C=0.1", flush=True)
    print(f"  切分：repetition 前 40 / 后 40（同类不跨折）", flush=True)

    rows = []
    t0 = time.time()

    for i in range(1, N_SUBJECTS + 1):
        sid = f"sub-{i:02d}"
        f = src / f"{sid}.npz"
        if not f.exists():
            continue
        print(f"\n  --- {sid} ---", flush=True)
        eeg = np.load(f)["eeg"]
        X = prep_features(eeg)
        if X is None:
            print("    特征构造失败", flush=True)
            continue
        print(f"    features {X.shape}", flush=True)

        res = {"subject": sid}
        for k in [2, 10, 200]:
            acc, base = decode_k(X, k, rng)
            if acc is None:
                continue
            ch = 1.0 / k
            res[f"acc_k{k}"] = round(acc, 4)
            res[f"chance_k{k}"] = round(ch, 4)
            res[f"ratio_k{k}"] = round(acc / ch, 2)
            res[f"null_k{k}"] = round(base, 4) if base is not None else None
            print(f"    k={k:>3}: acc {acc:.4f} (随机 {ch:.4f}, {acc/ch:.1f}x) "
                  f"| 打乱 {base if base is None else round(base, 4)}", flush=True)
        rows.append(res)
        del eeg, X

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase34.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s", flush=True)

    if not len(df):
        return

    print("\n" + "=" * 86, flush=True)
    print("三层递进结果（中位）", flush=True)
    print("=" * 86, flush=True)
    for k in [2, 10, 200]:
        c = f"acc_k{k}"
        if c not in df.columns:
            continue
        m = float(df[c].median())
        ch = 1.0 / k
        n = int(df[c].notna().sum())
        print(f"  k={k:>3}: 准确率 {m:.4f} | 随机 {ch:.4f} | {m/ch:.1f}x  (n={n})",
              flush=True)

    print("\n" + "=" * 86, flush=True)
    print("判定", flush=True)
    print("=" * 86, flush=True)
    v = {}
    for k, thresh in [(2, 0.60), (10, 0.20), (200, 0.02)]:
        c = f"acc_k{k}"
        if c not in df.columns or df[c].notna().sum() == 0:
            continue
        m = float(df[c].median())
        v[f"k{k}"] = {"acc": round(m, 4), "chance": round(1.0 / k, 4),
                      "ratio": round(m * k, 2), "threshold": thresh,
                      "pass": bool(m > thresh)}
        print(f"  k={k:>3}: {m:.4f} vs 阈值 {thresh} → "
              f"{'✅ 通过' if m > thresh else '❌ 未通过'}", flush=True)

    k2 = v.get("k2", {})
    if k2.get("pass"):
        verdict = (f"✅ condition 轴有效 — 2 类解码 {k2['acc']:.1%}（随机 50%），"
                   "流程正确，轴携带图像身份信息")
    elif k2:
        verdict = (f"❌ 2 类解码仅 {k2['acc']:.1%}（随机 50%）— "
                   "即使最简单的二分类也做不到，流程或数据有问题")
    else:
        verdict = "无法判定"
    print(f"\n  {verdict}", flush=True)

    (out / "phase34_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "why_exp33_failed": [
            "200 类 LDA：判别方向上限 199，每折每类仅 40 样本，shrinkage 过度收缩",
            "PCA 在全部样本上做（含测试集），既是泄漏又偏向重建而非判别",
            "未做基线校正、未降采样，特征 15750 维噪声主导",
        ],
        "fixes": ["2/10/200 类递进", "逻辑回归替代 LDA",
                  "基线校正 + 时间降采样 10 段", "训练集统计的标准化",
                  "置换检验基线"],
        "window_ms": list(WIN_MS), "n_time_bins": N_TIME_BINS,
        "levels": v, "verdict": verdict, "per_subject": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
