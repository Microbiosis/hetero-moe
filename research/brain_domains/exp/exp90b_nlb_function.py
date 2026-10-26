#!/usr/bin/env python3
"""exp90b_nlb_function.py — NLB 线的**功能判据**（③-CENT 运动 / ③-FRONT 决策时程）。

## 它回答什么

`BRAIN_STRUCTURE_SPEC.md` §4.4：③（前部自己的功能）**单开一条线**（形状共享、权重不共享）：

| | 数据 | 判据 | 诚实边界 |
|---|---|---|---|
| **③-CENT**（运动）| 猕猴 **M1/PMd**（churchland，6 session，1588+ trial/session）| **伸手方向解码**（8 类）跨 trial 泛化 ⇒ 显著高于同 session 随机基线 | 经典 BCI 目标解码；**不是**「运动皮层是什么」 |
| **③-FRONT**（执行）| 猕猴 **DMFC**（dmfc-rsg，2 session，283 trial/session）| **决策时程**（`go_offset = go_time − start_time`）可预测性 | ⚠️ 只有 2–3 个 session ⇒ **只报，不作判决**；**不是**「执行控制」 |

## 为什么在 NLB 上做（而不是继续在 THINGS-EEG 上）

exp90 的实测结论（`FINDINGS_TIER0_RSA.md` §三十三）：在 THINGS-EEG 上，
- 形态判据①（3×3 转分化）**测不出来**（前部两原语没有可读输出 ⇒ 3×3 退化成 1×1）；
- 自监督目标**会塌缩**（没有带负样本的目标）。

⇒ NLB 上这两条**同时消失**：
- 有**真实行为标签**（伸手方向 / 决策时程）⇒ 监督目标带负样本 ⇒ **不会塌缩**；
- 每个原语都有**可读的功能输出**⇒ 「这个区的活动能不能读出行为」这个问题**可测**。

⇒ ⇒ 本实验是**功能判据（③）**，不是形态判据（①）：NLB 的两个区**不在同一记录里**
（churchland 与 dmfc 是不同 session、不同任务、不同动物）⇒ **转分化在这里同样没有共同坐标系**
⇒ 形态判据①要成立，需要**同一记录、多个脑区、各自有行为标签**的**新数据**（见结案）。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 通过 ⇒ |
|---|---|---|
| **C0** | **管线自检**（exp90 的教训 ⇒ 显式前置）：每个 session 的解码准确率必须**高于**同 session 随机基线 | 标签/对齐/特征没问题 ⇒ 继续；**不过 ⇒ 全实验作废** |
| **C1** | **③-CENT**：跨 trial 泛化的伸手方向解码准确率 − 同 session 随机基线 **> 0**；证据量 = ①session 级符号检验（6 个 session ⇒ 最低 p=0.031）②trial 级 McNemar（更细，但注明 trial 不独立） | 运动皮层**携带**伸手方向信息 ⇒ ③-CENT 成立 |
| **C2** | **跨 session 泛化**（把 8 个方向按各 session 的目标角度对齐后，**零样本**跨 session 解码）| **只报**（探索性），不作判决 |
| **C3** | **③-FRONT**：`go_offset` 的预测相关 / 早-晚二分类准确率 − 随机基线 | **只报**（n=2–3 session ⇒ 证据极弱），**不作判决** |

## 用法（云端）

    python -u /domains/scripts/exp90b_nlb_function.py --out /domains/runs/illusionbrain-align/exp90b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

BIN_S = 0.020
N_DIR = 8
SEED = 0
DATA_DIR = Path("/domains/primate/nlb_trials")
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp90b")

CRITERIA = {
    "C0_pipeline_selfcheck": "每个 session 的解码准确率 > 同 session 随机基线（否则全实验作废）",
    "C1_CENT_reachdir": "跨 trial 泛化的伸手方向解码 − 随机基线 > 0；证据 = session 级符号检验 + trial 级 McNemar",
    "C2_cross_session": "零样本跨 session 解码（方向按各 session 目标角度对齐）⇒ 只报，不作判决",
    "C3_FRONT_decision": "go_offset 预测 / 早-晚二分类 − 随机基线 ⇒ 只报（n=2–3 session），不作判决",
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 统计小工具
# ---------------------------------------------------------------------------

def median_upper(xs):
    s = sorted(float(x) for x in xs if x == x)
    return s[len(s) // 2] if s else float("nan")


def sign_test_nonzero(diffs):
    d = [float(x) for x in diffs if x == x]
    n_zero = sum(1 for x in d if x == 0)
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 0, 0, 0, float("nan"), n_zero
    k = sum(1 for x in d if x > 0)
    m = n - k
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return n, k, m, float(min(1.0, p)), n_zero


def mcnemar(correct_a, correct_b):
    """配对二项（正确/错误）的 McNemar 检验（精确）⇒ (b, c, p)。

    b = A 对 B 错；c = A 错 B 对 ⇒ p 用二项精确检验。
    """
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, float("nan")
    lo = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return b, c, float(min(1.0, p))


# ---------------------------------------------------------------------------
# 特征
# ---------------------------------------------------------------------------

def trial_features(counts, t_start, t_stop):
    """每个 trial 的特征 = 窗内**每单元的平均放电率**（n_units 维）。"""
    n_tri = len(t_start)
    n_units = counts.shape[0]
    X = np.zeros((n_tri, n_units), dtype=np.float32)
    for i in range(n_tri):
        a = int(max(0, math.floor(t_start[i] / BIN_S)))
        b = int(min(counts.shape[1], math.ceil(t_stop[i] / BIN_S)))
        if b <= a:
            continue
        seg = counts[:, a:b].astype(np.float32)
        X[i] = seg.mean(axis=1)
    return X


def center_and_scale(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# ③-CENT：伸手方向解码
# ---------------------------------------------------------------------------

def decode_reachdir(X, y, n_folds=5, seed=SEED, shuffle=True):
    """session 内 trial 级 K 折 ⇒ 跨 trial 泛化的解码准确率。

    返回 (每 trial 的正确与否, 准确率, 随机基线的正确与否, 随机基线准确率)
    """
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    correct = np.zeros(n, dtype=bool)
    correct_shu = np.zeros(n, dtype=bool)
    for f in folds:
        te = f
        tr = np.setdiff1d(idx, te)
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[tr], y[tr])
        correct[te] = clf.predict(X[te]) == y[te]
        # 地板：打乱**训练**标签（保持测试标签不变）⇒ 期望准确率 = 1/n_classes
        y_shu = y[tr].copy()
        rng.shuffle(y_shu)
        clf2 = RidgeClassifier(alpha=1.0)
        clf2.fit(X[tr], y_shu)
        correct_shu[te] = clf2.predict(X[te]) == y[te]
    acc = float(correct.mean()) if n else float("nan")
    acc_s = float(correct_shu.mean()) if n else float("nan")
    return correct, acc, correct_shu, acc_s


def align_dirs_by_target(X, y, hit_pos, start_pts, session):
    """把 8 个方向标签**按该 session 的目标角度**重映射到全局角度网格
    ⇒ 跨 session 的标签才可比（每个 session 的迷宫方位不同）。"""
    angs = []
    for tgt, st in zip(hit_pos, start_pts):
        if not np.isfinite(tgt).all() or not np.isfinite(st).all():
            continue
        angs.append(math.atan2(tgt[1] - st[1], tgt[0] - st[0]))
    if len(angs) < N_DIR:
        return y, None
    # 8 个聚类中心（k-means 风格的贪心分配）
    cen = np.array([angs[0]])
    for a in angs[1:]:
        d = [abs((a - c + math.pi) % (2 * math.pi) - math.pi) for c in cen]
        j = int(np.argmin(d))
        if d[j] > math.pi / N_DIR:      # 超过 22.5° ⇒ 新方向
            cen = np.append(cen, a)
    if len(cen) != N_DIR:
        return y, None
    cen = np.sort(cen)
    # 全局网格：以「第一个 session 的中心角」为 0 号方向 ⇒ 用固定参考角
    ref = float(np.mean(cen))
    new_y = np.full(len(y), -1, dtype=np.int8)
    for i in range(len(y)):
        if y[i] < 0:
            continue
        a = cen[y[i]]
        g = int(round(((a - ref) + math.pi) % (2 * math.pi) / (2 * math.pi) * N_DIR)) % N_DIR
        new_y[i] = g
    return new_y, cen


# ---------------------------------------------------------------------------
# ③-FRONT：决策时程
# ---------------------------------------------------------------------------

def decode_gooffset(X, y, n_folds=5, seed=SEED):
    """预测 go_offset（连续）⇒ 相关系数；并做早/晚二分类准确率。"""
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    pred = np.full(n, np.nan)
    correct = np.zeros(n, dtype=bool)
    correct_shu = np.zeros(n, dtype=bool)
    med = float(np.median(y))
    yb = (y > med).astype(int)
    for f in folds:
        te = f
        tr = np.setdiff1d(idx, te)
        if len(te) == 0 or len(tr) < 5:
            continue
        r = Ridge(alpha=1.0)
        r.fit(X[tr], y[tr])
        pred[te] = r.predict(X[te])
        ys = y[tr].copy()
        rng.shuffle(ys)
        r2 = Ridge(alpha=1.0)
        r2.fit(X[tr], ys)
        # 地板：用打乱标签训练出的模型预测**原标签**的早/晚
        clf_s = RidgeClassifier(alpha=1.0)
        clf_s.fit(X[tr], (ys > med).astype(int))
        correct_shu[te] = clf_s.predict(X[te]) == yb[te]
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[tr], yb[tr])
        correct[te] = clf.predict(X[te]) == yb[te]
    ok = np.isfinite(pred)
    r = float(np.corrcoef(pred[ok], y[ok])[0, 1]) if ok.sum() > 3 else float("nan")
    acc = float(correct.mean()) if n else float("nan")
    acc_s = float(correct_shu.mean()) if n else float("nan")
    return pred, r, correct, acc, correct_shu, acc_s


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_churchland(files, args, rng):
    recs = []
    per_session = []
    all_correct, all_correct_shu = [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        sess = str(z["session"])
        counts = z["counts"]
        ts = z["trial_start"].astype(np.float64)
        te = z["trial_stop"].astype(np.float64)
        rd = z["reach_dir"].astype(np.int16)
        ok = (rd >= 0)
        if args.need_success:
            ok &= (z["task_success"].astype(int) == 1)
            ok &= (z["discard_trial"].astype(int) == 0)
        keep = np.flatnonzero(ok)
        if len(keep) < 50:
            log(f"  ✗ {sess}: 可用 trial 太少（{len(keep)}）⇒ 跳过")
            continue
        X = trial_features(counts, ts[keep], te[keep])
        X = center_and_scale(X)
        y = rd[keep]
        correct, acc, correct_shu, acc_s = decode_reachdir(X, y, args.n_folds)
        per_session.append({"session": sess, "n_trials": int(len(keep)),
                            "acc": acc, "acc_floor": acc_s, "diff": acc - acc_s})
        all_correct.append(correct)
        all_correct_shu.append(correct_shu)
        log(f"  ✓ {sess}: trial {len(keep)}  解码 {acc:.4f} vs 地板 {acc_s:.4f}"
            f"  差 {acc-acc_s:+.4f}")
    return per_session, all_correct, all_correct_shu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--need-success", type=int, default=1,
                    help="只留 task_success 且未 discard 的 trial")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("churchland__*.npz"))
    dm_files = sorted(data_dir.glob("dmfc-rsg__*.npz"))
    log(f"数据：churchland {len(files)} 个 session，dmfc {len(dm_files)} 个 session")
    if not files:
        log("✗ 没有 churchland 的抽取结果 ⇒ 先跑 nlb_extract_trial_labels.py")
        return 1

    # ---------- ③-CENT ----------
    log("\n=== ③-CENT：伸手方向解码（跨 trial 泛化，5 折）===")
    per_session, all_correct, all_correct_shu = run_churchland(files, args, np.random.default_rng(SEED))
    if not per_session:
        log("✗ 没有任何 session 可用 ⇒ 判决：INSUFFICIENT")
        verdict = {"verdict": "INSUFFICIENT"}
        (out_dir / "exp90b_nlb_function.json").write_text(
            json.dumps({"config": vars(args), "criteria": CRITERIA,
                        "per_session": per_session, "verdict": verdict},
                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return 1

    diffs = [s["diff"] for s in per_session]
    n_eff, n_pos, n_neg, p_sign, n_zero = sign_test_nonzero(diffs)
    ca = np.concatenate(all_correct) if all_correct else np.zeros(0, dtype=bool)
    cs = np.concatenate(all_correct_shu) if all_correct_shu else np.zeros(0, dtype=bool)
    b, c, p_mc = mcnemar(ca, cs)
    C0 = all(d > 0 for d in diffs)
    C1 = bool(n_eff >= 4 and p_sign <= 0.05 and median_upper(diffs) > 0)
    log(f"\n  逐 session 差值中位 {median_upper(diffs):+.4f}；符号检验 "
        f"{n_pos}/{n_eff} 正 p={p_sign:.4f}（零差 {n_zero}）")
    log(f"  trial 级 McNemar：解码独对 {b} / 地板独对 {c} ⇒ p={p_mc:.3g}")
    log(f"  C0 管线自检（每个 session 都高于地板）：{C0}")
    log(f"  C1 ③-CENT 成立：{C1}")

    # ---------- ③-FRONT（只报） ----------
    log("\n=== ③-FRONT：决策时程（go_offset）—— 只报，不作判决 ===")
    per_dmfc = []
    for fp in dm_files:
        z = np.load(fp, allow_pickle=True)
        sess = str(z["session"])
        counts = z["counts"]
        ts = z["trial_start"].astype(np.float64)
        te = z["trial_stop"].astype(np.float64)
        go = z["go_offset"].astype(np.float64)
        ok = np.isfinite(go)
        keep = np.flatnonzero(ok)
        if len(keep) < 30:
            log(f"  ✗ {sess}: trial 太少（{len(keep)}）⇒ 跳过")
            continue
        X = center_and_scale(trial_features(counts, ts[keep], te[keep]))
        pred, r, correct, acc, correct_shu, acc_s = decode_gooffset(X, go[keep], args.n_folds)
        per_dmfc.append({"session": sess, "n_trials": int(len(keep)),
                         "r_pred": r, "acc_earlylate": acc, "acc_floor": acc_s,
                         "diff": acc - acc_s})
        log(f"  ✓ {sess}: trial {len(keep)}  go_offset 预测 r={r:+.4f}  "
            f"早/晚准确率 {acc:.4f} vs 地板 {acc_s:.4f}")

    verdict = {
        "verdict": ("C1_CENT_OK" if (C0 and C1) else
                    ("C0_FAIL" if not C0 else "C1_FAIL")),
        "C0_pipeline_selfcheck": C0,
        "C1_CENT_reachdir": {
            "pass": C1, "median_diff": median_upper(diffs),
            "sign_test": {"n_eff": n_eff, "n_pos": n_pos, "n_neg": n_neg,
                          "p": p_sign, "n_zero": n_zero},
            "mcnemar_trial_level": {"b": b, "c": c, "p": p_mc,
                                    "n_trials": int(len(ca))},
        },
        "C3_FRONT_decision": {"report_only": True, "per_session": per_dmfc},
        "note": "C3 只报（n=2–3 session ⇒ 证据极弱 ⇒ 不是「执行控制」的证明）",
    }
    log(f"\n=== 判决：{verdict['verdict']} ===")
    (out_dir / "exp90b_nlb_function.json").write_text(
        json.dumps({"config": vars(args), "criteria": CRITERIA,
                    "per_session_cent": per_session, "per_session_frong": per_dmfc,
                    "verdict": verdict}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    log(f"结果写入 {out_dir}/exp90b_nlb_function.json")


if __name__ == "__main__":
    main()
