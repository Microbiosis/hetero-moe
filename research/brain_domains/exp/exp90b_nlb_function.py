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

    ⚠️ **n 很大（上千 trial）⇒ `2.0 ** n` 会 OverflowError** ⇒ 用对数域算 p。
    ⚠️ **trial 之间不独立**（同一 session、同一动物）⇒ 这个 p **偏乐观**
    ⇒ 因此它只作**描述性**证据；**主证据用 session 级符号检验**（见 CRITERIA C1）。
    """
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, float("nan")
    lo = min(b, c)
    s = sum(math.comb(n, i) for i in range(lo + 1))
    # p = 2 * s / 2^n ⇒ 对数域（避免 float 溢出）
    log2s = math.log2(s) if s > 0 else -math.inf
    expo = log2s - n + 1.0
    p = 2.0 ** expo if expo < 0 else 1.0
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


def decide(per_session, mcnemar_counts=None):
    """从**逐 session 原始记录**重算判决（阈值的**唯一出处** ⇒ 可离线重算）。

    ⚠️ **纯重构**：本函数是从 `main()` 原样搬出来的（**阈值一字未改**：
    `n_eff >= 4 and p_sign <= 0.05 and median_upper(diffs) > 0`），
    目的只是让 `results/parse_exp90b.py` 能用原始记录**离线重算判决**。

    `mcnemar_counts` = (解码正确, 地板正确) 两个 trial 级布尔数组；
    离线重算拿不到 ⇒ 该字段为 `None`（它是**描述性**的，**不进判决**）。
    """
    diffs = [s["diff"] for s in per_session]
    n_eff, n_pos, n_neg, p_sign, n_zero = sign_test_nonzero(diffs)
    C0 = all(d > 0 for d in diffs)
    C1 = bool(n_eff >= 4 and p_sign <= 0.05 and median_upper(diffs) > 0)
    if mcnemar_counts is not None:
        ca, cs = mcnemar_counts
        b, c, p_mc = mcnemar(ca, cs)
        mc = {"b": b, "c": c, "p": p_mc, "n_trials": int(len(ca)),
              "note": "**描述性**：trial 不独立 ⇒ p 偏乐观 ⇒ 主证据用 session 级符号检验"}
    else:
        mc = None
    return {
        "verdict": ("C1_CENT_OK" if (C0 and C1) else
                    ("C0_FAIL" if not C0 else "C1_FAIL")),
        "C0_pipeline_selfcheck": C0,
        "C1_CENT_reachdir": {
            "pass": C1, "median_diff": median_upper(diffs),
            "sign_test": {"n_eff": n_eff, "n_pos": n_pos, "n_neg": n_neg,
                          "p": p_sign, "n_zero": n_zero},
            "mcnemar_trial_level": mc,
        },
    }


def self_test():
    """判据实现的**自测**（exp90 的教训：**判据 bug 会伪装成结论** ⇒ 判据本身也要被测）。

    覆盖：① 上中位；② 符号检验的**零差剔除**（全 0 不得被数成「全负」）；
    ③ McNemar 的**大 n 不溢出**；④ `decide()` 的三条分支
    （`C1_CENT_OK` / `C1_FAIL` / `C0_FAIL`）。
    """
    ok = 0
    bad = []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  ✓ {name}")
        else:
            bad.append(name)
            print(f"  ✗ {name}")

    # ① 中位 = 上中位（偶数取偏大）
    chk("上中位(偶数 4 个取偏大)", median_upper([1.0, 2.0, 3.0, 4.0]) == 3.0)
    chk("上中位(奇数)", median_upper([1.0, 2.0, 3.0]) == 2.0)
    chk("上中位(空)=nan", math.isnan(median_upper([])))

    # ② 符号检验：全正 / 零差剔除 / 全零
    n, k, m, p, nz = sign_test_nonzero([0.1] * 6)
    chk("符号检验 6/6 正 ⇒ p=0.03125", (n, k, m) == (6, 6, 0) and abs(p - 0.03125) < 1e-12)
    n, k, m, p, nz = sign_test_nonzero([1.0, 0.0, -1.0])
    chk("符号检验 零差单独计数", (n, k, m, nz) == (2, 1, 1, 1))
    n, k, m, p, nz = sign_test_nonzero([0.0] * 6)
    chk("符号检验 全零 ⇒ n_eff=0 且 **n_neg=0**（不得被数成全负）",
        (n, k, m, nz) == (0, 0, 0, 6) and math.isnan(p))

    # ③ McNemar：大 n 不溢出（exp90b 的真实量级）
    a = [True] * 9558 + [False] * 328
    b = [False] * 9558 + [True] * 328
    bb, cc, pp = mcnemar(a, b)
    chk("McNemar 大 n 不溢出 ⇒ b/c 正确且 p∈[0,1]",
        (bb, cc) == (9558, 328) and 0.0 <= pp <= 1.0)
    chk("McNemar 无不一致对 ⇒ p=nan", math.isnan(mcnemar([True] * 5, [True] * 5)[2]))

    # ④ decide()：三条分支
    def sess(vals):
        return [{"session": f"s{i}", "n_trials": 100, "acc": 0.9, "acc_floor": 0.2,
                 "diff": v} for i, v in enumerate(vals)]

    v = decide(sess([0.7] * 6))
    chk("decide 6/6 正 ⇒ C1_CENT_OK", v["verdict"] == "C1_CENT_OK"
        and v["C0_pipeline_selfcheck"] and v["C1_CENT_reachdir"]["pass"]
        and abs(v["C1_CENT_reachdir"]["median_diff"] - 0.7) < 1e-12)
    v = decide(sess([0.7] * 4))
    chk("decide 4/4 正但 p=0.125 ⇒ C1_FAIL", v["verdict"] == "C1_FAIL"
        and v["C0_pipeline_selfcheck"] and not v["C1_CENT_reachdir"]["pass"])
    v = decide(sess([0.7, 0.7, 0.7, 0.7, 0.7, 0.0]))
    chk("decide 有一个持平 ⇒ C0_FAIL", v["verdict"] == "C0_FAIL"
        and not v["C0_pipeline_selfcheck"])
    v = decide(sess([0.0] * 6))
    chk("decide 全零 ⇒ C0_FAIL 且 sign_test 的 n_neg=0",
        v["verdict"] == "C0_FAIL" and v["C1_CENT_reachdir"]["sign_test"]["n_neg"] == 0)

    # ⑤ 特征抽取的形状（小合成数组）
    counts = np.zeros((3, 100), dtype=np.float32)
    counts[:, 10:20] = 1.0
    X = trial_features(counts, np.array([0.0]), np.array([0.4]))
    chk("trial_features 形状与窗内均值", X.shape == (1, 3) and abs(float(X[0, 0]) - 0.5) < 1e-5)
    Z = center_and_scale(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    chk("center_and_scale 均值≈0", abs(float(Z.mean())) < 1e-5)

    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--need-success", type=int, default=1,
                    help="只留 task_success 且未 discard 的 trial")
    ap.add_argument("--self-test", action="store_true",
                    help="只跑判据实现的自测（不需要数据）")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

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

    ca = np.concatenate(all_correct) if all_correct else np.zeros(0, dtype=bool)
    cs = np.concatenate(all_correct_shu) if all_correct_shu else np.zeros(0, dtype=bool)
    verdict = decide(per_session, mcnemar_counts=(ca, cs))
    _st = verdict["C1_CENT_reachdir"]["sign_test"]
    _mc = verdict["C1_CENT_reachdir"]["mcnemar_trial_level"]
    b, c, p_mc = _mc["b"], _mc["c"], _mc["p"]
    C0 = verdict["C0_pipeline_selfcheck"]
    C1 = verdict["C1_CENT_reachdir"]["pass"]
    log(f"\n  逐 session 差值中位 {verdict['C1_CENT_reachdir']['median_diff']:+.4f}；符号检验 "
        f"{_st['n_pos']}/{_st['n_eff']} 正 p={_st['p']:.4f}（零差 {_st['n_zero']}）")
    log(f"  trial 级 McNemar（**描述性**：trial 不独立 ⇒ p 偏乐观）⇒解码独对 {b} / "
        f"地板独对 {c} ⇒ p={p_mc:.3g}")
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

    verdict["C3_FRONT_decision"] = {"report_only": True, "per_session": per_dmfc}
    verdict["note"] = "C3 只报（n=2–3 session ⇒ 证据极弱 ⇒ 不是「执行控制」的证明）"
    log(f"\n=== 判决：{verdict['verdict']} ===")
    (out_dir / "exp90b_nlb_function.json").write_text(
        json.dumps({"config": vars(args), "criteria": CRITERIA,
                    "per_session_cent": per_session, "per_session_frong": per_dmfc,
                    "verdict": verdict}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    log(f"结果写入 {out_dir}/exp90b_nlb_function.json")


if __name__ == "__main__":
    main()
