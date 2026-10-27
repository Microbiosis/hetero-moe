#!/usr/bin/env python3
"""exp101_condition.py — ②③ 条件生成线的**第一个里程碑**：**脑侧条件的可辨识性**（THINGS-EEG）。

## 这一轮要回答什么（**不训任何生成模型**）

②③ 是「脑 → 图像」的**条件生成**（脑侧编码器 + 图像侧生成先验；见 `MODEL_FORM.md` §7.3.2 的路线 ②/③）。
⚠️ **在烧 GPU 训扩散之前，先判一件事**：**我们的脑侧条件到底能不能认出「是哪张图」？**
⇒ ⇒ 若连「在 50 张候选里挑出正确的那张」都做不到 ⇒ ⇒ **条件生成这条路现在不该开**
⇒ ⇒ **这一轮零训练**（用**已训好并已验证**的 exp93b 对比学习 encoder，冻结）。

## 口径（**与文献表同口径，以便对标**）

`MODEL_FORM.md` §7.3.3：THINGS-EEG = **1654 概念训练 / 200 测试概念 × 10 被试**；
文献的 200-way 零样本检索是「**依赖被试（intra-subject）+ 试次平均**」口径
⇒ ⇒ 本轮**两个口径都报**：**单试次**（现实设定）与**试次平均**（文献口径）。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 阈值 |
|---|---|---|
| **C0** | **条件自检**：200-way 单试次 top-1 中位 **≥ 3×随机**（1/200 ⇒ ≥0.015）且 **> 0**（对照 exp93b 的 **0.0589** ⇒ 这是**参照**，不是新发现）| 不过 ⇒ **全实验作废**（条件根本没学到东西 ⇒ 后面的检索数字无意义）|
| **C1**（主判据）| **50-way 条件可辨识**：逐被试 50-way top-1 ⇒ **显著高于随机**（1/50 = 2%）| **≥8 被试、p ≤ 0.05、中位 > 2%** |
| **C2** | **2-way 条件可辨识**：逐被试 2-way top-1 ⇒ **显著高于随机**（50%）| **≥8 被试、p ≤ 0.05、中位 > 50%** |
| **C3**（只报）| ① 200-way（单试次 / 试次平均）② 50-way / 2-way 的**试次平均**版 ③ **与 ATM（NeurIPS 2024）的并排参照**：他们报 2-way **83.1%**、50-way top-1 **28.4%**、FID 24.6 ⚠️ **他们的是「生成图」的指标、我们的是「条件」的指标 ⇒ 协议不同 ⇒ 只作量级参照，不是同协议对标** ④ 每概念召回（只报）| — |

### 决策规则（**预登记**）

- **C1 过** ⇒ **②③ 值得开**（下一步才是训生成模型：条件已能认出图像 ⇒ 条件是有信息的）。
- **C1 不过** ⇒ ⇒ **先别训生成模型** ⇒ ⇒ 要先改**脑侧**（更强的 encoder / 换码本锚 / 换数据），
  ⇒ ⇒ 否则就是拿一个认不出图的信号去条件化生成 ⇒ ⇒ **白烧 GPU**。

## 用法

```powershell
python -u research/brain_domains/exp/exp101_condition.py --self-test
# 云端
hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
  python -u /domains/scripts/exp101_condition.py --eval-subjects 1 2 3 4 5 6 7 8 9 10 \
  -v hf://buckets/illusion-brain/training-data:/training:ro \
  -v hf://buckets/illusion-brain/brain-domains:/domains \
  --flavor a10g-large --timeout 60m -d --name t101-cond
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
for _p in (str(EXP_DIR), str(EXP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp93b_contrastive as E93B  # noqa: E402

SEED = 0
CHANCE_200 = 1.0 / 200.0
CHANCE_50 = 1.0 / 50.0
CHANCE_2 = 0.5
MIN_SUBJECTS = 8
ALPHA = 0.05
# 外部参照（**C 类：别人发表的系统/数字 ⇒ 只作量级参照，不当我们的结果**）
# ATM (NeurIPS 2024, 64-ch EEG / THINGS)：2-way 83.1% / 50-way top-1 28.4% / FID 24.6
# ⚠️ 那是「**生成图**」的指标；我们这里量的是「**条件**」的指标 ⇒ 协议不同
ATM_REF = {"two_way_pct": 83.1, "fifty_way_top1_pct": 28.4, "fid": 24.6}


def log(*a):
    print(*a, flush=True)


def med_up(xs):
    s = sorted(float(x) for x in xs)
    return s[len(s) // 2] if s else float("nan")


def sign_test_nonzero(diffs):
    from math import comb
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    zero = n - pos - neg
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)) if n else 1.0
    return n, pos, neg, p, zero


def paired_test(diffs, name, out):
    n, pos, neg, p, zero = sign_test_nonzero(diffs)
    out[name] = {"n_eff": n, "n_pos": pos, "n_neg": neg, "p_sign": p,
                 "n_zero": zero, "median": med_up(diffs)}
    return out[name]


# ---------------------------------------------------------------------------
# 检索（条件 z ↔ 图像特征，余弦）
# ---------------------------------------------------------------------------
def make_candidate_structure(n_img, n_groups=4, seed=SEED):
    """50 组划分与 2-way 负样本：**按图像身份固定** ⇒ 跨被试一致、可比。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_img)
    group = np.zeros(n_img, dtype=np.int64)
    for gi in range(n_groups):
        group[perm[gi::n_groups]] = gi
    neg = np.empty(n_img, dtype=np.int64)
    for i in range(n_img):
        cand = rng.choice(np.delete(np.arange(n_img), i), size=1, replace=False)
        neg[i] = int(cand[0])
    return group, neg


def retrieval_metrics(Z, Vc, group, neg):
    """Z : (n_img, n_rep, D) 条件；Vc : (n_img, D) 图像特征。"""
    n_img, n_rep, D = Z.shape
    flat = Z.reshape(-1, D).astype(np.float64)
    fn = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    cn = Vc.astype(np.float64)
    cn = cn / np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
    sim = fn @ cn.T
    true = np.repeat(np.arange(n_img), n_rep)
    # 200-way（参照）
    acc200 = float((sim.argmax(axis=1) == true).mean())
    # 50-way：只在同组的 50 个候选里比
    mask = np.zeros_like(sim, dtype=bool)
    for gi in np.unique(group):
        cols = np.flatnonzero(group == gi)
        rows = np.flatnonzero(group[true] == gi)
        mask[np.ix_(rows, cols)] = True
    sim50 = np.where(mask, sim, -np.inf)
    acc50 = float((sim50.argmax(axis=1) == true).mean())
    # 2-way：每行只比 [正, 负]
    ng = neg[true]
    s_true = sim[np.arange(len(true)), true]
    s_neg = sim[np.arange(len(true)), ng]
    acc2 = float((s_true > s_neg).mean())
    # 试次平均版（文献口径）
    avg = Z.astype(np.float64).mean(axis=1)
    an = avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)
    sim_avg = an @ cn.T
    t = np.arange(n_img)
    acc200_avg = float((sim_avg.argmax(axis=1) == t).mean())
    mask_a = np.zeros_like(sim_avg, dtype=bool)
    for gi in np.unique(group):
        cols = np.flatnonzero(group == gi)
        rows = np.flatnonzero(group == gi)
        mask_a[np.ix_(rows, cols)] = True
    acc50_avg = float((np.where(mask_a, sim_avg, -np.inf).argmax(axis=1) == t).mean())
    s_true_a = sim_avg[t, t]
    s_neg_a = sim_avg[t, neg]
    acc2_avg = float((s_true_a > s_neg_a).mean())
    return {"acc200": acc200, "acc50": acc50, "acc2": acc2,
            "acc200_avg": acc200_avg, "acc50_avg": acc50_avg, "acc2_avg": acc2_avg}


def eval_subject(model, sid, IMG_test, group, neg, args, device):
    import torch
    eeg = E93B.load_test_subject(sid)
    n_img, n_rep, n_ch, T = eeg.shape
    flat = eeg.reshape(n_img * n_rep, n_ch, T)
    Z = E93B.encode_all(model, flat, device, args.batch).reshape(n_img, n_rep, -1)
    rec = {"subject": f"sub-{sid:02d}"}
    rec.update(retrieval_metrics(Z, IMG_test, group, neg))
    del eeg, Z
    return rec


# ---------------------------------------------------------------------------
# 判决（**预登记**）
# ---------------------------------------------------------------------------
def decide(records, min_subjects=MIN_SUBJECTS, alpha=ALPHA):
    out = {}
    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"verdict": "INSUFFICIENT", "why": "没有可用被试"}

    # ---- C0 条件自检（200-way 单试次）----
    a200 = [r["acc200"] for r in ok if "acc200" in r]
    c0 = {"acc200_median": med_up(a200),
          "acc200_above_chance3": bool(med_up(a200) >= 3 * CHANCE_200),
          "acc200_positive": bool(med_up(a200) > 0),
          "exp93b_reference": 0.0589}
    out["C0"] = c0
    out["C0_pass"] = bool(c0["acc200_above_chance3"] and c0["acc200_positive"])
    if not out["C0_pass"]:
        return {"verdict": "CONDITION_SELFCHECK_FAIL", "C0": c0,
                "why": "条件连 200-way 都不显著高于随机 ⇒ 条件没学到东西 ⇒ 后面的检索数字无意义 ⇒ 停"}

    # ---- C1（主判据）50-way ----
    c1 = {}
    diffs = [r["acc50"] - CHANCE_50 for r in ok if "acc50" in r]
    if diffs:
        v = paired_test(diffs, "acc50_minus_chance", c1)
        v["pass"] = bool(v["n_eff"] >= min_subjects and v["p_sign"] <= alpha
                         and (v.get("median") or 0) > 0)
    out["C1"] = c1
    out["C1_pass"] = bool(c1) and all(v.get("pass") for v in c1.values())

    # ---- C2 2-way ----
    c2 = {}
    diffs2 = [r["acc2"] - CHANCE_2 for r in ok if "acc2" in r]
    if diffs2:
        v = paired_test(diffs2, "acc2_minus_chance", c2)
        v["pass"] = bool(v["n_eff"] >= min_subjects and v["p_sign"] <= alpha
                         and (v.get("median") or 0) > 0)
    out["C2"] = c2
    out["C2_pass"] = bool(c2) and all(v.get("pass") for v in c2.values())

    # ---- C3 只报 ----
    out["C3_report"] = {
        "acc200_median": med_up([r["acc200"] for r in ok if "acc200" in r]),
        "acc50_median": med_up([r["acc50"] for r in ok if "acc50" in r]),
        "acc2_median": med_up([r["acc2"] for r in ok if "acc2" in r]),
        "acc200_avg_median": med_up([r["acc200_avg"] for r in ok if "acc200_avg" in r]),
        "acc50_avg_median": med_up([r["acc50_avg"] for r in ok if "acc50_avg" in r]),
        "acc2_avg_median": med_up([r["acc2_avg"] for r in ok if "acc2_avg" in r]),
        "atm_reference": ATM_REF,
        "atm_caveat": "ATM 报的是**生成图**的指标；我们这里量的是**条件**的指标 ⇒ 协议不同 ⇒ 只作量级参照",
    }

    if out["C1_pass"]:
        out["verdict"] = "CONDITION_USABLE"
        out["why"] = ("条件在 50 张候选里显著认出正确图像 ⇒ **②③ 值得开**"
                      "（下一步才是训生成模型）")
    else:
        out["verdict"] = "CONDITION_TOO_WEAK"
        out["why"] = ("条件在 50 张候选里认不出正确图像 ⇒ ⇒ **先别训生成模型**"
                      "（白烧 GPU）⇒ ⇒ 要先改脑侧：更强的 encoder / 换码本锚 / 换数据")
    return out


# ---------------------------------------------------------------------------
def _fake_records(n=10, acc50=0.30, acc2=0.80, acc200=0.059):
    recs = []
    for i in range(n):
        recs.append({"subject": f"sub-{i:02d}", "acc200": acc200 + 0.0001 * i,
                     "acc50": acc50 + 0.001 * i, "acc2": acc2 + 0.001 * i,
                     "acc200_avg": 0.23, "acc50_avg": 0.5, "acc2_avg": 0.9})
    return recs


def self_test():
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # ① 候选结构：按图像身份固定 ⇒ 跨被试一致
    g, n = make_candidate_structure(200)
    chk("候选结构：每组 50 张、共 4 组", sorted(np.bincount(g).tolist()) == [50, 50, 50, 50])
    chk("候选结构：负样本不是它自己", bool((n != np.arange(200)).all()))

    # ② 检索：完美的条件 ⇒ 各口径都满分；随机条件 ⇒ 机会水平
    rng = np.random.default_rng(0)
    Vc = rng.normal(size=(200, 32))
    Z_perfect = Vc[:, None, :] + 0.01 * rng.normal(size=(200, 4, 32))
    m = retrieval_metrics(Z_perfect, Vc, g, n)
    chk("检索：完美条件 ⇒ 200-way/50-way/2-way 都接近满分",
        m["acc200"] > 0.95 and m["acc50"] > 0.95 and m["acc2"] > 0.95)
    Z_rand = rng.normal(size=(200, 4, 32))
    m2 = retrieval_metrics(Z_rand, Vc, g, n)
    chk("检索：随机条件 ⇒ 50-way ≈ 2%、2-way ≈ 50%",
        abs(m2["acc50"] - 0.02) < 0.05 and abs(m2["acc2"] - 0.5) < 0.12)

    # ③ decide() 的判决分支
    v = decide(_fake_records(10, acc50=0.30, acc2=0.80))
    chk("decide：条件可用 ⇒ CONDITION_USABLE",
        v["verdict"] == "CONDITION_USABLE" and v["C1_pass"] and v["C2_pass"])
    # ⚠️ 用**明显低于机会**的档位（否则 +0.001*i 的抖动会让它变成显著的）
    v = decide(_fake_records(10, acc50=0.00, acc2=0.40))
    chk("decide：条件太弱 ⇒ CONDITION_TOO_WEAK（**先别训生成模型**）",
        v["verdict"] == "CONDITION_TOO_WEAK" and not v["C1_pass"])
    v = decide(_fake_records(10, acc200=0.001))
    chk("decide：200-way 都在随机水平 ⇒ CONDITION_SELFCHECK_FAIL（停）",
        v["verdict"] == "CONDITION_SELFCHECK_FAIL")
    v = decide(_fake_records(5, acc50=0.30, acc2=0.80))
    chk("decide：被试不够（5 < 8）⇒ C1 不过",
        not v.get("C1_pass", False))

    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-subjects", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ckpt", default="/domains/runs/illusionbrain-align/exp93b/encoder_best.pt")
    ap.add_argument("--out-dir", default="/domains/runs/illusionbrain-align/exp101")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    log(f"[条件] exp93b encoder_best.pt（epoch={ck.get('epoch')}，**冻结**）")

    fte = torch.load(E93B.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    group, neg = make_candidate_structure(len(IMG_test))
    log(f"[候选] {len(IMG_test)} 张未见图 ⇒ 4 组 × 50；2-way 负样本固定")

    per = []
    for sid in args.eval_subjects:
        try:
            rec = eval_subject(model, sid, IMG_test, group, neg, args, device)
            per.append(rec)
            log(f"  ✓ {rec['subject']}: 200-way {rec['acc200']:.4f}｜50-way {rec['acc50']:.4f}"
                f"｜2-way {rec['acc2']:.4f}（试次平均：{rec['acc200_avg']:.4f} / "
                f"{rec['acc50_avg']:.4f} / {rec['acc2_avg']:.4f}）")
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}", "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-600:]}
            log(f"  ✗ {rec['error']}")
            per.append(rec)

    verdict = decide(per)
    payload = {"config": vars(args), "atm_reference": ATM_REF, "per_subject": per,
               "verdict": verdict}
    (out_dir / "exp101_condition.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for r in per:
        print("===REC===" + json.dumps(r, ensure_ascii=False, default=str), flush=True)
    print("===VERDICT===" + json.dumps(verdict, ensure_ascii=False, default=str), flush=True)
    log(f"\n=== 判决：{verdict['verdict']} ===")
    log(f"  {verdict.get('why', '')}")
    log(f"结果写入 {out_dir}/exp101_condition.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
