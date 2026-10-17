"""Phase 13: τ 能识别记录位点吗?

## 为什么做这个

exp12 发现 τ 主要由脑区解释（区域 ω²=0.643 > 物种 0.543），
且 h2 的 37 个片段 τ 标准差仅 0.13 —— 提示 τ 是记录位点的稳定属性。

若如此，则 τ **应该能用来分类记录位点**。这是一个干净的验证：

    若 τ 能把 M1/PMd、DMFC、Area2、小鼠各域、人类两个位点分开，
    说明它是真实的功能指纹，不是噪声。

这是本项目少有的**可做正向验证**的方向。

## 设计

特征（三种，逐步增强）:
  F1  τ 单独（1 维）
  F2  R²(k) 曲线（8 维，k=1..16）
  F3  R²(k) 曲线 + 噪声地板 + n_win（10 维）

任务: 多分类，预测 region
  - 只用在 exp12 已算出 τ 的分组（157 个）

评估:
  - 分层 5 折交叉验证（每折尽量保留每类样本）
  - 报告 balanced accuracy（类别不平衡）
  - 对照: 打乱标签后的 balanced accuracy

判据:
  balanced accuracy 显著高于打乱 → τ 确实是位点指纹
  τ 单独就能达到 > 50% → τ 是强特征
  只有 F3 才能分 → τ 本身不够，需结合其他信息

## 注意

类别极不平衡（M1_PMd 67 vs Area2 1），因此:
  - 只保留样本数 >= 5 的类别
  - 用 balanced accuracy 而非普通准确率
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
N_FOLD = 5
N_PERM = 200
MIN_CLASS = 5


def load_features(csv_path: Path):
    """读 exp12 的 τ 表 + exp11full 的 R²(k) 曲线，按分组对齐。"""
    t = pd.read_csv(csv_path)
    print(f"  exp12 记录: {len(t)}, 区域 {t['region'].nunique()}")
    return t


def balanced_acc(y_true, y_pred):
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        m = y_true == c
        if m.sum() == 0:
            continue
        accs.append((y_pred[m] == c).mean())
    return float(np.mean(accs)) if accs else 0.0


def stratified_folds(y, n_fold, rng):
    """分层折。"""
    idx = np.arange(len(y))
    folds = [[] for _ in range(n_fold)]
    for c in np.unique(y):
        ci = idx[y == c]
        rng.shuffle(ci)
        for j, i in enumerate(ci):
            folds[j % n_fold].append(i)
    return [np.array(sorted(f), dtype=int) for f in folds if len(f)]


def knn_predict(Xtr, ytr, Xte, k=5):
    """标准化 + kNN（小样本下比复杂模型稳健）。"""
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    out = []
    for row in nn:
        vals, cnts = np.unique(ytr[row], return_counts=True)
        out.append(vals[np.argmax(cnts)])
    return np.array(out)


def cv_score(X, y, rng, n_fold=N_FOLD):
    folds = stratified_folds(y, n_fold, rng)
    pred = np.empty(len(y), dtype=y.dtype)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        if len(tr) < 5:
            pred[f] = y[tr][0] if len(tr) else y[0]
            continue
        pred[f] = knn_predict(X[tr], y[tr], X[f])
    return balanced_acc(y, pred), pred


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out/exp12/phase12.csv")
    src2 = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp11full/phase11.csv")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/out/exp13")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("Phase 13 — τ 能识别记录位点吗?")
    print("=" * 78)

    t = load_features(src)
    t = t.dropna(subset=["tau"])

    # 合并 R²(k) 曲线（exp11_full 的输出）
    if src2.exists():
        r = pd.read_csv(src2)
        r["group"] = r["group"].astype(str)
        t["group"] = t["group"].astype(str)
        rcols = [c for c in r.columns if c.startswith("r2_")]
        r2 = r[["species", "group"] + rcols].rename(
            columns={c: f"curve_{c}" for c in rcols})
        t = t.merge(r2, on=["species", "group"], how="left")
        hit = t["curve_r2_1"].notna().sum() if "curve_r2_1" in t.columns else 0
        print(f"  合并 R²(k): 命中 {hit}/{len(t)}，曲线列 {len([c for c in t.columns if c.startswith('curve_')])} 个")

    # 只保留样本足够多的类别
    cnt = t["region"].value_counts()
    keep = cnt[cnt >= MIN_CLASS].index
    t = t[t["region"].isin(keep)].reset_index(drop=True)
    print(f"\n  保留类别（n>={MIN_CLASS}）: {len(keep)} 个, 共 {len(t)} 组")
    print(t["region"].value_counts().to_string())

    if t["region"].nunique() < 2:
        print("类别不足，退出")
        return

    y = t["region"].to_numpy()

    # 特征集
    r2cols = [c for c in t.columns if c.startswith("curve_r2_")]
    r2cols = [c for c in r2cols if t[c].notna().all()]
    print(f"  可用曲线特征: {len(r2cols)} 维")

    feats = {}
    if t["tau"].notna().all():
        feats["F1_tau_only"] = t[["tau"]].to_numpy(dtype=float)
    if r2cols:
        feats["F2_curve"] = t[r2cols].to_numpy(dtype=float)
        extra = [c for c in ["floor", "n_win"] if c in t.columns and t[c].notna().all()]
        if extra:
            feats["F3_curve_meta"] = np.column_stack(
                [t[r2cols].to_numpy(dtype=float)] +
                [t[[c]].to_numpy(dtype=float) for c in extra])

    if not feats:
        print("无明显可用特征")
        return

    print("\n" + "=" * 78)
    print(f"{'特征集':<18} {'维度':>5} {'balanced acc':>14} {'打乱基线':>10} {'提升':>8}")
    print("=" * 78)

    results = {}
    for name, X in feats.items():
        acc, _ = cv_score(X, y, rng)
        # 打乱基线
        null = []
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            a, _ = cv_score(X, yp, rng)
            null.append(a)
        null = np.array(null)
        base = float(null.mean())
        p = float((null >= acc).mean())
        results[name] = {"n_features": int(X.shape[1]), "balanced_acc": round(acc, 4),
                         "shuffled_mean": round(base, 4),
                         "lift": round(acc - base, 4), "p": p}
        print(f"{name:<18} {X.shape[1]:>5} {acc:>14.4f} {base:>10.4f} "
              f"{acc-base:>+8.4f}  (p={p:.3f})")

    # 混淆矩阵（最优特征）
    best = max(results, key=lambda k: results[k]["balanced_acc"])
    print(f"\n  最佳特征集: {best}")
    acc, pred = cv_score(feats[best], y, rng)
    print("\n  混淆矩阵（行=真实，列=预测）:")
    labels = sorted(np.unique(y))
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for a_, b_ in zip(y, pred):
        cm.loc[a_, b_] += 1
    print(cm.to_string())

    # ===== 附加对照: 只用标签明确的 3 个大位点 =====
    print("\n" + "=" * 78)
    print("附加对照: 仅 3 个标签明确的大位点（human_h1 / human_h2 / macaque_M1_PMd）")
    print("=" * 78)
    major = ["human_h1", "human_h2", "macaque_M1_PMd"]
    m = t[t["region"].isin(major)].reset_index(drop=True)
    print(f"  样本: {len(m)}  " + ", ".join(f"{k}={v}" for k, v in m['region'].value_counts().items()))
    ym = m["region"].to_numpy()
    major_res = {}
    for name, cols in [("tau_only", ["tau"]),
                       ("curve", r2cols),
                       ("curve+meta", r2cols + [c for c in ["floor", "n_win"] if c in m.columns])]:
        cols = [c for c in cols if c in m.columns]
        if not cols:
            continue
        X = m[cols].to_numpy(dtype=float)
        acc_m, _ = cv_score(X, ym, rng)
        null_m = []
        for _ in range(N_PERM):
            yp = rng.permutation(ym)
            a, _ = cv_score(X, yp, rng)
            null_m.append(a)
        bm = float(np.mean(null_m))
        major_res[name] = {"n_features": len(cols), "balanced_acc": round(acc_m, 4),
                           "shuffled": round(bm, 4), "lift": round(acc_m - bm, 4)}
        print(f"  {name:<14} dim={len(cols):2d}  acc={acc_m:.4f}  基线={bm:.4f}  lift={acc_m-bm:+.4f}")

    # 判读
    f1 = results.get("F1_tau_only", {})
    good = results[best]["balanced_acc"]
    print("\n" + "=" * 78)
    if good > 0.5 and results[best]["p"] < 0.01:
        v = "✅ τ 及衰减曲线可有效识别记录位点 —— 是真实的功能指纹"
    elif good > 0.3 and results[best]["p"] < 0.05:
        v = "⚠️ 有一定识别力，但不够强"
    else:
        v = "❌ 无法识别位点 —— τ 不是稳定的指纹"
    print(f"  判定: {v}")
    if f1:
        print(f"  （仅 τ 单独: balanced acc = {f1['balanced_acc']:.3f}, "
              f"lift = {f1['lift']:+.3f}）")

    (out / "phase13_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_groups": int(len(t)), "n_classes": int(len(labels)),
        "classes": labels, "n_fold": N_FOLD, "n_perm": N_PERM,
        "results": results, "best": best, "verdict": v,
        "major_sites": major_res,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
