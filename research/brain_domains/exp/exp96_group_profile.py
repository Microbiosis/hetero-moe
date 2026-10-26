#!/usr/bin/env python3
"""exp96_group_profile.py — 三组（POST / CENT / FRONT）在**模型层**的剖面 + 同尺寸随机组地板。

## 它回答什么（这是整条线的**开关**）

`BRAIN_STRUCTURE_SPEC.md` §4.1：三组切分（**POST 感觉/语义 34 / CENT 运动 14 / FRONT 执行 15**）
目前只是「**按脑科学解剖的候选**」（R3：原语个数只能由判据①「转分化代价」定）。

exp89 在**特征层**已经测过三组：

| 组 | 通道 | 语义对齐 RSA | 显著被试 |
|---|---:|---:|---:|
| POST | 34 | +0.0628 | 10/10 |
| CENT | 14 | +0.0082 | 3/10 |
| FRONT | 15 | +0.0064 | 2/10 |
| ANT（CENT+FRONT）| 29 | +0.0115 | 7/10 |

⇒ **CENT 与 FRONT 在「图像语义」这一维上不可分**（差 0.0018）⇒ 语义轴支撑不了这个拆分
⇒ 但**模型层从未测过** ⇒ 而我们真正的模型是模型层的（exp90 的底座就是 S1a/S1b）。

⇒⇒ **exp96 = exp89 的组剖面在模型层的版本** ⇒ 它决定：
- **FRONT 在模型层有没有独特身份**（脱离同尺寸随机组地板）
  - 有 ⇒ **三组成立** ⇒ exp90 走 **3×3** 转分化矩阵、**NLB 功能线（③-CENT / ③-FRONT）可开**；
  - 没有 ⇒ **三组不成立** ⇒ exp90 退回 **2×2**、**NLB 功能线不开**（省掉一整条线）。

## 口径纪律（R4：同口径才可比）

1. **只做同口径内的比较**：模型层内部比、特征层内部比。
   ⚠️ **模型层的绝对值不得与特征层的数字横向比**（掩码是把非选中通道**置零**再前向
   ⇒ 输入分布被破坏 ⇒ 测的是「组别对模型读出的影响」，不是「组别的语义含量」）。
   exp89 选特征层正是因为这条。
2. **地板分两种**（这是 exp92-A 教训的直接落实）：
   - **mixed**（从全部 63 通道抽同尺寸）⇒ exp89 口径（可横向比），但**天然含本组通道**
     （34 通道组平均含 ~18 个后部通道）⇒ 地板被本组信号抬高
     ⇒ 对 POST 是**保守**的（真值要更高才能赢 ⇒ 当**锚点**用）；
   - **pure**（从**不含本组通道**的通道里抽同尺寸）⇒ **内容匹配**
     ⇒ 「本组 vs 任意同样多但不含本组的通道」⇒ 判「本组有没有独特身份」**只能用它**
     （FRONT/CENT/ANT 都造得出来；**POST 造不出来**：非后部只有 29 < 34 ⇒ 记 None）。
   ⇒⇒ 判据 **B 用 mixed（保守锚点）**、判据 **C 用 pure（内容匹配）**。
3. **双骨干**（S1a `recon` / S1b `onehot_mean`）⇒ exp94/95 的教训：结论要在两个骨干上都成立。
4. **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）⇒ 与 exp89 汇总口径逐字一致。

## 预登记判据（**写死**，随结果一起落盘 ⇒ 可离线重算）

| # | 判据 | 通过 | 不过 ⇒ |
|---|---|---|---|
| **A** | **管线自检**：特征层复算的 r 与 exp89 的**逐被试**记录（ALL/POST/CENT/FRONT/ANT）逐被试绝对差 ≤ 1e-3 ⇒ **管线零漂移**（不是比中位 ⇒ 冒烟只跑 1 个被试也能判）| 数据/口径没漂 | ⇒ 停（`PIPELINE_DRIFT`），不往下看 |
| **B** | **POST 锚点**（**两个骨干都要**，地板 = **mixed**）：模型层 POST 的「真值 − 同尺寸随机组中位」配对差 **> 0**，且**剔除零差**后 n_eff ≥ 8、全部为正（符号检验 p ≤ 0.002）| 模型层读出口径可信 | ⇒ 停（`ANCHOR_FAIL`），**不据此判三组** |
| **C** | **开关判据**（**两个骨干都要**，**方向中立**，地板 = **pure**）：模型层 FRONT 的「真值 − **不含 FRONT 通道**的同尺寸随机组中位」配对差**显著非零**（剔零后 n_eff ≥ 8、同号，p ≤ 0.05）| ⇒ **三组成立**（`THREE_GROUPS_HOLD`）：exp90 走 3×3、NLB 线可开 | ⇒ **三组不成立**（`THREE_GROUPS_FAIL`）：exp90 退回 2×2、NLB 线不开 |

⚠️ **C 的方向不预判**（高 = 带语义、低 = 负空间、零 = 噪声）：形态学关心的是「**有没有独特身份**」
（可分离性），不是「带不带语义」⇒ 显著高于地板**也算**可分离，方向在结果里报告。

⚠️ **零差（与地板持平）必须先剔除再检验**：否则符号检验会把「10 个全 0」数成「全负」⇒ p=0.002
⇒ 会把「三组**不**成立」误判成「成立」⇒ **错误地开 NLB 线**（自测当场抓到的判据 bug）。

⚠️ **判据的实现本身也要被测**（exp94/95 两次「判据 bug 伪装成结论」的教训）⇒
`--self-test` 用**合成数据**分别造出「A 不过」「B 不过」「C 过」「C 不过」四种情形，
检查 `decide()` 分别给出**预期**判决。

## 用法

    # 自测（本地，合成数据；不碰数据）
    python -u research/brain_domains/exp/exp96_group_profile.py --self-test

    # 冒烟（云端，1 被试，用来量成本）
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp96_group_profile.py --subjects 1 --n-draw 1 --n-random 4 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 60m -d --name t96-smoke

    # 正式
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp96_group_profile.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 300m -d --name t96-groups
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((p for p in _PKG_CANDIDATES if (p / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp73_decode200 as E73      # noqa: E402
import exp81_tokenizer_eval as E81  # noqa: E402
import exp89_posaxis as E89        # noqa: E402
import exp93c_dose_response as E93C  # noqa: E402

CKPT_S1A = "/domains/runs/illusionbrain-align/exp80/k1024/tokenizer_sub-01.pt"
CKPT_S1B = "/domains/runs/illusionbrain-align/exp86v2/ffn-plain/tokenizer_assembled.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp96")
SEED = 0

# 真值组（ALL 是参照，ANT 是合并组）
GROUPS = ("ALL", "POST", "CENT", "FRONT", "ANT")
# 需要配地板的组（ALL 是全部通道 ⇒ 不需要地板）
FLOOR_GROUPS = ("POST", "CENT", "FRONT", "ANT")
# 模型层读出：骨干 → 读出名
MODEL_READOUTS = (("s1a", CKPT_S1A, "recon"), ("s1b", CKPT_S1B, "onehot_mean"))

# exp89（2026-10-24，零训练，10 被试）的**逐被试**特征层 r
# ⚠️ 判据 A 用**逐被试精确复现**（而不是「10 被试中位的容差」）：
#    后者要求被试集完整 ⇒ 只跑 1 个被试时会把「中位不同」误报成「管线漂了」；
#    前者每个被试各自可比 ⇒ 冒烟也能判，且强得多。
REF89 = {
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
                 "sub-06", "sub-07", "sub-08", "sub-09", "sub-10"],
    "r": {
        "ALL": {"sub-01": +0.0337, "sub-02": +0.0344, "sub-03": +0.0504, "sub-04": +0.0537, "sub-05": +0.0483, "sub-06": +0.0612, "sub-07": +0.0418, "sub-08": +0.0673, "sub-09": +0.0337, "sub-10": +0.0621},
        "POST": {"sub-01": +0.0402, "sub-02": +0.0527, "sub-03": +0.0594, "sub-04": +0.0748, "sub-05": +0.0628, "sub-06": +0.0846, "sub-07": +0.0472, "sub-08": +0.0790, "sub-09": +0.0442, "sub-10": +0.0774},
        "CENT": {"sub-01": +0.0077, "sub-02": +0.0001, "sub-03": +0.0152, "sub-04": +0.0007, "sub-05": +0.0090, "sub-06": +0.0082, "sub-07": +0.0148, "sub-08": +0.0261, "sub-09": +0.0048, "sub-10": +0.0081},
        "FRONT": {"sub-01": +0.0062, "sub-02": -0.0015, "sub-03": +0.0038, "sub-04": -0.0010, "sub-05": +0.0065, "sub-06": +0.0126, "sub-07": +0.0074, "sub-08": +0.0136, "sub-09": +0.0052, "sub-10": +0.0064},
        "ANT": {"sub-01": +0.0115, "sub-02": +0.0020, "sub-03": +0.0123, "sub-04": +0.0028, "sub-05": +0.0084, "sub-06": +0.0109, "sub-07": +0.0166, "sub-08": +0.0265, "sub-09": +0.0040, "sub-10": +0.0129},
    },
}
# exp89 记录的 10 被试中位（只作报告用）
REF_MEDIAN = {"ALL": 0.0504, "POST": 0.0628, "ANT": 0.0115}
# 逐被试复现的容差（exp89 的 JSON 里 r 只存 4 位小数 ⇒ 真值在 ±5e-5 内 ⇒ 1e-3 够宽也够严）
REF_TOL = 1e-3
# 符号检验的**有效样本**下限：10 个被试里至少 8 个非零同号才算「显著」
# ⚠️ 零差 = 「与地板持平」⇒ 单独计数，不参与符号检验（见 sign_test_nonzero）
MIN_NONZERO = 8

# 预登记判据（随结果落盘）
CRITERIA = {
    "A_pipeline_drift": {
        "what": "特征层复算 vs exp89 的**逐被试** r（ALL/POST/CENT/FRONT/ANT），逐被试 |差| ≤ 1e-3 ⇒ 管线零漂移",
        "ref_per_subject": REF89, "tol": REF_TOL, "ref_median": REF_MEDIAN,
    },
    "B_post_anchor": {
        "what": "模型层 POST 的（真值 − 同尺寸随机组中位）配对差 > 0 且剔零后 n_eff ≥ 8、全为正，p ≤ 0.002；两个骨干都要",
        "p_max": 0.002, "min_nonzero": MIN_NONZERO,
    },
    "C_front_separable": {
        "what": "模型层 FRONT 的（真值 − 同尺寸随机组中位）配对差显著非零（剔零后 n_eff ≥ 8、同号，p ≤ 0.05）；两个骨干都要；方向中立",
        "p_max": 0.05, "min_nonzero": MIN_NONZERO,
    },
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 统计小工具（与 exp89 口径逐字一致：上中位 + 精确符号检验）
# ---------------------------------------------------------------------------

def median_upper(xs):
    """**上中位**（`sorted[n//2]`，偶数取偏大）⇒ 与 exp89 汇总逐字一致。

    ⚠️ 右偏分布用下中位会**低报** 30–40%。
    """
    s = sorted(float(x) for x in xs)
    if not s:
        return float("nan")
    return s[len(s) // 2]


def sign_test(diffs):
    """双侧符号检验（精确二项）⇒ (n_pos, n_neg, p)。"""
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return k, m, float(min(1.0, p))


def sign_test_nonzero(diffs):
    """**剔除零差**后的双侧符号检验 ⇒ (n_eff, n_pos, n_neg, p, n_zero)。

    ⚠️ **为什么必须先剔零**（自测当场抓到的判据 bug）：
    若 10 个被试的差值**全是 0**（真值与地板持平），原式会数成「0 正 10 负」
    ⇒ lo=0 ⇒ p = 2/1024 = **0.002 ⇒ 判成「显著低于地板」**
    ⇒ ⇒ 会把「三组**不**成立」误判成「成立」⇒ **错误地开 NLB 功能线**。
    ⇒ 所以：零差（持平）必须**单独计数**，不参与符号检验；
    n_eff = 0 ⇒ 「与地板不可区分」⇒ 判据不过。
    """
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


# ---------------------------------------------------------------------------
# 地板抽样
# ---------------------------------------------------------------------------

def random_group(n_ch, size, self_idx, rng, exclude_self=False):
    """抽一个**同尺寸**的随机通道组。

    `exclude_self=False` ⇒ **mixed** 地板：从全部 63 通道抽 ⇒ exp89 口径（可横向比），
        但它**天然含本组通道**（34 通道组平均含 ~18 个后部通道）⇒
        exp92-A 的教训：这种地板**不是内容匹配的** ⇒ 只能当**保守锚点**用。
    `exclude_self=True` ⇒ **pure** 地板：从**不含本组通道**的通道里抽 ⇒
        「本组 vs 任意同样多但不含本组的通道」⇒ **内容匹配** ⇒ 判「本组有没有独特身份」
        只能用这种地板。
        ⚠️ 当「非本组通道数 < size」时造不出来（POST：非后部只有 29 < 34）⇒ 返回 None。

    返回 (通道下标列表, 该组里含本组通道的个数 k_self)；造不出时返回 (None, None)。
    """
    if exclude_self:
        pool = [c for c in range(n_ch) if c not in self_idx]
        if len(pool) < size:
            return None, None
        sub = rng.choice(pool, size=size, replace=False).tolist()
    else:
        sub = rng.choice(n_ch, size=size, replace=False).tolist()
    k_self = int(sum(1 for c in sub if c in self_idx))
    return [int(c) for c in sub], k_self


def _make_floors(n_ch, idx, rng, n_draw, r_true, r_of):
    """造两类地板，返回 {kind: {...} | None}（None = 造不出来）。

    - **mixed**：从全部 63 通道抽 ⇒ exp89 口径（可横向比），但**天然含本组通道**
      ⇒ 地板被本组信号抬高 ⇒ 对 POST 是**保守**的（真值要更高才能赢），
      对 FRONT 会**夸大**「低于地板」⇒ 只能当锚点用。
    - **pure**：从**不含本组通道**的通道里抽 ⇒ **内容匹配**
      ⇒ 「本组 vs 任意同样多但不含本组的通道」⇒ 判「本组有没有独特身份」只能用它。
      ⚠️ 非本组通道数 < 组大小 ⇒ 造不出来（POST：29 < 34）⇒ None。
    """
    out = {}
    for kind, excl in (("mixed", False), ("pure", True)):
        rs, ks = [], []
        for _ in range(n_draw):
            sub, k_self = random_group(n_ch, len(idx), set(idx), rng,
                                       exclude_self=excl)
            if sub is None:
                break
            rs.append(float(r_of(sub)))
            ks.append(k_self)
        if not rs:
            out[kind] = None
            continue
        out[kind] = {
            "rs": rs, "k_self": ks, "r_median": median_upper(rs),
            "frac_ge_true": float(np.mean([1.0 if x >= r_true else 0.0 for x in rs])),
            "n_k_self_zero": int(sum(1 for k in ks if k == 0)),
        }
    return out


# ---------------------------------------------------------------------------
# 判据（**离线可重算** ⇒ 只吃原始记录）
# ---------------------------------------------------------------------------

def decide(records):
    """从逐被试原始记录算出判决。

    ⚠️ 只吃记录里的数（r / rs / k_self）⇒ 判决可离线重算（D50）。
    ⚠️ 方向与显著性**同时**查（D50：exp95 只看 p 不看方向 ⇒ 把「显著更差」读成「显著更好」）。
    """
    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"verdict": "INSUFFICIENT", "why": "没有可用被试"}

    # ---- A：管线自检（特征层**逐被试**复现 exp89）----
    a_detail = {"per_subject": [], "n_checked": 0, "max_abs_diff": 0.0,
                 "ref_median": REF_MEDIAN}
    max_diff = 0.0
    for r in ok:
        subj = r.get("subject")
        for gname, ref_map in REF89["r"].items():
            if subj not in ref_map:
                continue
            got = r.get("feature", {}).get(gname, {}).get("r")
            if got is None:
                return {"verdict": "INSUFFICIENT",
                        "why": f"特征层缺 {subj}/{gname}"}
            d = float(got) - ref_map[subj]
            max_diff = max(max_diff, abs(d))
            a_detail["per_subject"].append(
                {"subject": subj, "group": gname, "got": got,
                 "ref": ref_map[subj], "diff": d})
            a_detail["n_checked"] += 1
    if a_detail["n_checked"] == 0:
        return {"verdict": "INSUFFICIENT", "why": "没有可与 exp89 逐被试比对的被试"}
    a_detail["max_abs_diff"] = max_diff
    if max_diff > REF_TOL:
        return {"verdict": "PIPELINE_DRIFT", "A": a_detail,
                "why": f"特征层复算与 exp89 逐被试参照最大差 {max_diff:.2e} > {REF_TOL:.0e} "
                       "⇒ 数据/口径漂了 ⇒ 停"}

    # ---- 逐骨干：B（POST 锚点）与 C（FRONT 可分离性）----
    b_detail, c_detail = {}, {}
    for tag, _ck, readout in MODEL_READOUTS:
        # B：POST（**mixed 地板** = 含 POST 通道的同尺寸随机组 ⇒ 对 POST 是**保守**的；
        #    剔零后计数）
        diffs_b = []
        for r in ok:
            m = r.get("model", {}).get(tag)
            if not m:
                continue
            rt = m["groups"].get("POST", {}).get("r")
            fl = (m.get("floors", {}).get("POST", {}) or {}).get("mixed")
            if rt is None or not fl:
                continue
            diffs_b.append(rt - fl["r_median"])
        if not diffs_b:
            return {"verdict": "INSUFFICIENT", "why": f"{tag} 缺 POST 地板"}
        n_eff, k, m_, p, n_zero = sign_test_nonzero(diffs_b)
        med = median_upper(diffs_b)
        b_pass = bool(n_eff >= MIN_NONZERO and k == n_eff and med > 0
                      and p <= CRITERIA["B_post_anchor"]["p_max"])
        b_detail[tag] = {"floor_kind": "mixed", "diffs": diffs_b, "n": len(diffs_b),
                         "n_eff": n_eff, "n_zero": n_zero, "n_pos": k, "n_neg": m_,
                         "p_sign": p, "median": med, "pass": b_pass}
        if not b_pass:
            return {"verdict": "ANCHOR_FAIL", "A": a_detail, "B": b_detail,
                    "why": f"{tag} 的 POST 锚点没过 ⇒ 模型层读出口径不可信 ⇒ 不据此判三组"}

        # C：FRONT（**pure 地板** = 不含 FRONT 通道的同尺寸随机组 ⇒ **内容匹配**
        #    ⇒ 判「FRONT 有没有独特身份」只能用这个；方向中立；剔零后计数）
        diffs_c = []
        for r in ok:
            m = r.get("model", {}).get(tag)
            if not m:
                continue
            rt = m["groups"].get("FRONT", {}).get("r")
            fl = (m.get("floors", {}).get("FRONT", {}) or {}).get("pure")
            if rt is None or not fl:
                continue
            diffs_c.append(rt - fl["r_median"])
        if not diffs_c:
            return {"verdict": "INSUFFICIENT",
                    "why": f"{tag} 缺 FRONT 的 pure 地板（不含 FRONT 通道的同尺寸随机组）"}
        n_eff2, k2, m2, p2, n_zero2 = sign_test_nonzero(diffs_c)
        med2 = median_upper(diffs_c)
        same_sign = bool(n_eff2 > 0 and (k2 == n_eff2 or m2 == n_eff2))
        c_pass = bool(n_eff2 >= MIN_NONZERO and same_sign
                      and p2 <= CRITERIA["C_front_separable"]["p_max"])
        c_detail[tag] = {"floor_kind": "pure", "diffs": diffs_c, "n": len(diffs_c),
                         "n_eff": n_eff2, "n_zero": n_zero2, "n_pos": k2, "n_neg": m2,
                         "p_sign": p2, "median": med2, "pass": c_pass,
                         "direction": ("above_floor" if med2 > 0 else
                                       "below_floor" if med2 < 0 else "same_as_floor")}
        if not c_pass:
            return {"verdict": "THREE_GROUPS_FAIL", "A": a_detail, "B": b_detail,
                    "C": c_detail,
                    "why": f"{tag} 上 FRONT 与同尺寸随机组不可区分 ⇒ 三组不成立 ⇒ "
                           "exp90 退回 2×2、NLB 功能线不开"}

    return {"verdict": "THREE_GROUPS_HOLD", "A": a_detail, "B": b_detail, "C": c_detail,
            "why": "两个骨干上 FRONT 都脱离同尺寸随机组地板 ⇒ 三组成立 ⇒ "
                   "exp90 走 3×3 转分化矩阵、NLB 功能线（③-CENT/③-FRONT）可开"}


# ---------------------------------------------------------------------------
# 自测（**判据实现也要被测**）
# ---------------------------------------------------------------------------

def _fake_records(n=10, post_diff=0.03, front_diff=-0.02, feature_shift=0.0,
                  anchor_ok=True):
    """造一份合成记录，用来驱动 decide()。

    特征层默认**逐被试等于 exp89 的记录**（⇒ 判据 A 天然过）；
    `feature_shift` 给全部值加一个偏移 ⇒ 造出「管线漂移」。
    """
    recs = []
    for i in range(n):
        subj = f"sub-{i+1:02d}"
        rec = {"subject": subj, "feature": {}, "model": {}}
        for gname in ("ALL", "POST", "CENT", "FRONT", "ANT"):
            rec["feature"][gname] = {
                "r": float(REF89["r"][gname][subj]) + feature_shift, "p": 0.001}
        for tag, _ck, _ro in MODEL_READOUTS:
            floor_post = [0.01] * 3
            floor_front = [0.01] * 3
            rec["model"][tag] = {
                "groups": {
                    "ALL": {"r": 0.05}, "POST": {"r": 0.01 + post_diff},
                    "CENT": {"r": 0.008}, "FRONT": {"r": 0.01 + front_diff},
                    "ANT": {"r": 0.011},
                },
                "floors": {
                    # POST 的 pure 地板**造不出来**（非后部只有 29 < 34）⇒ None
                    "POST": {"mixed": {"rs": floor_post, "k_self": [5, 6, 7],
                                       "r_median": median_upper(floor_post),
                                       "frac_ge_true": 0.0},
                             "pure": None},
                    "FRONT": {"mixed": {"rs": floor_front, "k_self": [1, 0, 1],
                                        "r_median": median_upper(floor_front),
                                        "frac_ge_true": 0.0},
                              "pure": {"rs": floor_front, "k_self": [0, 0, 0],
                                       "r_median": median_upper(floor_front),
                                       "frac_ge_true": 0.0}},
                },
            }
        if not anchor_ok:
            for tag, _ck, _ro in MODEL_READOUTS:
                rec["model"][tag]["groups"]["POST"]["r"] = 0.005   # 低于地板
        recs.append(rec)
    return recs


def self_test():
    log("=== 自测 1：组构造 ===")
    names = ([f"O{i}" for i in range(8)] + [f"P{i}" for i in range(16)]
             + [f"T{i}" for i in range(10)] + [f"C{i}" for i in range(14)]
             + [f"F{i}" for i in range(15)])
    groups, _lab = E89.build_groups(names)
    log("  " + "  ".join(f"{k}={len(groups[k])}" for k in
                         ("ALL", "POST", "CENT", "FRONT", "ANT")))
    assert len(groups["POST"]) == 34, groups["POST"]
    assert len(groups["CENT"]) == 14
    assert len(groups["FRONT"]) == 15
    assert len(groups["ANT"]) == 29
    assert not (set(groups["POST"]) & set(groups["ANT"])), "POST 与 ANT 必须不交"
    assert set(groups["POST"]) | set(groups["ANT"]) == set(groups["ALL"])
    log("  ✓ POST/CENT/FRONT = 34/14/15；POST ∩ ANT = ∅；POST ∪ ANT = ALL")

    log("=== 自测 2：地板抽样（mixed / pure 两种）===")
    rng = np.random.default_rng(0)
    post_set = set(groups["POST"])
    for _ in range(20):
        sub, k_self = random_group(63, 34, post_set, rng)
        assert len(sub) == 34 and len(set(sub)) == 34
        assert 0 <= k_self <= 34
        assert sum(1 for c in sub if c in post_set) == k_self
        # pure：不含本组通道
        sub_p, k_self_p = random_group(63, 15, set(groups["FRONT"]), rng,
                                        exclude_self=True)
        assert sub_p is not None and len(sub_p) == 15 and len(set(sub_p)) == 15
        assert k_self_p == 0, "pure 地板里不该有本组通道"
        assert not (set(sub_p) & set(groups["FRONT"]))
    # POST 的 pure 造不出来：非后部只有 29 < 34
    sub_none, k_none = random_group(63, 34, post_set, rng, exclude_self=True)
    assert sub_none is None and k_none is None
    log("  ✓ mixed 同尺寸无重复；pure 不含本组通道（k_self=0）；POST 的 pure 造不出来（29<34）")

    log("=== 自测 3：判据实现（**方向 + 显著性同时查**）===")
    v = decide(_fake_records(post_diff=0.03, front_diff=-0.02))["verdict"]
    assert v == "THREE_GROUPS_HOLD", v
    log("  ✓ FRONT 显著低于地板 ⇒ 判『三组成立』")

    v = decide(_fake_records(post_diff=0.03, front_diff=0.0))["verdict"]
    assert v == "THREE_GROUPS_FAIL", v
    log("  ✓ FRONT 与地板持平 ⇒ 判『三组不成立』")

    v = decide(_fake_records(post_diff=0.03, front_diff=+0.02))["verdict"]
    assert v == "THREE_GROUPS_HOLD", v
    log("  ✓ FRONT 显著**高于**地板（方向中立）⇒ 仍判『可分离』")

    # 零差（持平）必须被剔除：5 个持平 + 5 个为负 ⇒ n_eff=5 < 8 ⇒ 判不过
    recs = _fake_records(post_diff=0.03, front_diff=-0.02)
    for i, r in enumerate(recs):
        if i < 5:
            for tag, _ck, _ro in MODEL_READOUTS:
                m = r["model"][tag]
                m["groups"]["FRONT"]["r"] = m["floors"]["FRONT"]["pure"]["r_median"]
    v = decide(recs)["verdict"]
    assert v == "THREE_GROUPS_FAIL", v
    log("  ✓ 一半被试持平（零差剔除）⇒ n_eff=5 < 8 ⇒ 判『三组不成立』")

    v = decide(_fake_records(post_diff=-0.02, front_diff=-0.02))["verdict"]
    assert v == "ANCHOR_FAIL", v
    log("  ✓ POST 锚点没过 ⇒ 判『停』（不据此判三组）")

    v = decide(_fake_records(feature_shift=0.05))["verdict"]
    assert v == "PIPELINE_DRIFT", v
    log("  ✓ 特征层逐被试漂移 ⇒ 判『停』")

    v = decide([])["verdict"]
    assert v == "INSUFFICIENT", v
    log("  ✓ 无数据 ⇒ 判『数据不足』")

    log("=== 自测 4：中位是上中位 ===")
    assert median_upper([1.0, 2.0, 3.0, 4.0]) == 3.0
    assert median_upper([1.0, 2.0, 3.0]) == 2.0
    log("  ✓ 上中位（偶数取偏大）")

    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_subject(subject, IMG_test, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    t0 = time.time()

    eeg_te, npz_keys = E89.load_test_eeg(subject)
    rec["npz_keys"] = npz_keys
    n_img, n_rep, n_ch, T = eeg_te.shape
    ch_names, _ = E89.load_train_chunk(subject, 1)
    groups, labels = E89.build_groups(ch_names)
    rec["group_labels"] = labels
    rec["groups"] = {k: list(map(int, v)) for k, v in groups.items()}
    log("  通道分组：" + "  ".join(f"{k}={len(groups[k])}" for k in GROUPS))

    # ---------- 特征层（零成本；组间切片，口径与 exp89 逐字一致）----------
    feats_te = E73.per_rep_targets(eeg_te, chunk=args.chunk)
    feat = {}
    for gname in GROUPS:
        idx = groups.get(gname, [])
        if not idx:
            continue
        f = E89.slice_group(feats_te, idx, n_ch)
        r, p = E89.rsa(f, IMG_test, args.n_perm, rng)
        feat[gname] = {"r": r, "p": p, "n_ch": len(idx)}
        log(f"  [特征] {gname:<5} n_ch={len(idx):<3} r={r:+.4f} p={p:.4f}")
    rec["feature"] = feat

    feat_floor = {}
    for gname in FLOOR_GROUPS:
        idx = groups.get(gname, [])
        if not idx:
            continue
        r_true = feat[gname]["r"]

        def _r_of(sub, _feats=feats_te, _nch=n_ch, _rng=rng):
            f = E89.slice_group(_feats, sub, _nch)
            r, _p = E89.rsa(f, IMG_test, args.n_perm_null, _rng)
            return r

        fl = _make_floors(n_ch, idx, rng, args.n_random, r_true, _r_of)
        for kind, d in fl.items():
            if d is None:
                log(f"  [特征地板/{kind}] {gname:<5} **造不出来**"
                    "（非本组通道数 < 组大小）")
                continue
            log(f"  [特征地板/{kind}] {gname:<5} 中位 {d['r_median']:+.4f} "
                f"| 真值 {r_true:+.4f} | 随机组中≥真值 {d['frac_ge_true']:.0%} "
                f"| k_self=0 的抽样 {d['n_k_self_zero']}/{len(d['rs'])}")
        feat_floor[gname] = fl
    rec["feature_floor"] = feat_floor

    # ---------- 模型层 ----------
    model_rec = {}
    for tag, ckpt, readout in MODEL_READOUTS:
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt} ⇒ 跳过 {tag}")
            continue
        model, _ck, rnd_cb = E81.load_model(ckpt)
        g_rec, f_rec = {}, {}
        for gname in GROUPS:
            idx = groups.get(gname, [])
            if not idx:
                continue
            f = E93C.model_readouts(model, rnd_cb, E93C.mask_channels(eeg_te, idx),
                                    n_ch, (readout,), args.batch_size)
            r, _p = E89.rsa(f[readout], IMG_test, args.n_perm_null, rng)
            g_rec[gname] = {"r": r, "n_ch": len(idx)}
            log(f"  [{tag}/{readout}] {gname:<5} n_ch={len(idx):<3} r={r:+.4f}")
        for gname in FLOOR_GROUPS:
            idx = groups.get(gname, [])
            if not idx:
                continue
            r_true = g_rec[gname]["r"]

            def _r_of(sub, _model=model, _cb=rnd_cb, _ro=readout):
                f = E93C.model_readouts(_model, _cb, E93C.mask_channels(eeg_te, sub),
                                        n_ch, (_ro,), args.batch_size)
                r, _p = E89.rsa(f[_ro], IMG_test, args.n_perm_null, rng)
                return r

            fl = _make_floors(n_ch, idx, rng, args.n_draw, r_true, _r_of)
            for kind, d in fl.items():
                if d is None:
                    log(f"  [{tag}/{readout} 地板/{kind}] {gname:<5} **造不出来**"
                        "（非本组通道数 < 组大小）")
                    continue
                log(f"  [{tag}/{readout} 地板/{kind}] {gname:<5} 中位 {d['r_median']:+.4f} "
                    f"| 真值 {r_true:+.4f} | 随机组中≥真值 {d['frac_ge_true']:.0%} "
                    f"| k_self=0 的抽样 {d['n_k_self_zero']}/{len(d['rs'])}")
            f_rec[gname] = fl
        model_rec[tag] = {"readout": readout, "groups": g_rec, "floors": f_rec}
        del model
    rec["model"] = model_rec

    rec["elapsed_s"] = round(time.time() - t0, 1)
    log(f"  用时 {rec['elapsed_s']}s")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--n-perm", type=int, default=200, help="特征层真值组的置换次数")
    ap.add_argument("--n-perm-null", type=int, default=50, help="地板与模型层的置换次数")
    ap.add_argument("--n-random", type=int, default=10,
                    help="特征层每组的随机组抽样次数（**只作描述，不进任何判据** ⇒ 不必多抽）")
    ap.add_argument("--n-draw", type=int, default=3, help="模型层每组的随机组抽样次数（贵）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fte = torch.load(E89.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = run_subject(sid, IMG_test, args, np.random.default_rng(SEED + sid))
            per_subject.append(rec)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)
        log(f"  被试用时 {time.time()-t0:.0f}s")

    n_err = sum(1 for r in per_subject if "error" in r)
    log(f"\n完成：{len(per_subject)-n_err}/{len(per_subject)} 成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))

    verdict = decide(per_subject)
    log("\n=== 判决 ===")
    log(f"  {verdict['verdict']}：{verdict.get('why', '')}")
    log(json.dumps({k: v for k, v in verdict.items() if k in ("A", "B", "C")},
                   indent=2, ensure_ascii=False, default=str))

    (out_dir / "exp96_group_profile.json").write_text(
        json.dumps({"config": vars(args), "criteria": CRITERIA,
                    "per_subject": per_subject, "verdict": verdict},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp96_group_profile.json")


if __name__ == "__main__":
    main()
