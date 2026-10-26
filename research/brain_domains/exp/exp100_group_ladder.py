#!/usr/bin/env python3
"""exp100_group_ladder.py — 判据① 的**「组内 vs 跨组」验收**（Allen VC-NP，**零新抽取**）。

## 这一轮为什么存在（**从 exp98 的三个诊断直接推出来**）

exp98（Allen VC-NP 上判据① 的正/负对照）判决 `C1_FAIL`，而它的**事后观察**指出了三处具体病因：

1. **单一对比太脆**：预登记的正对照（VISp↔VISam）恰好落在 Δ 阶梯的**最底端**
   ⇒ ⇒ 「挑对一对」不可靠 ⇒ ⇒ **本轮不再挑任何一对**，改成**组内 vs 跨组**的整体比较。
2. **session 间方差压过了对比间的差**：即便把正对照换成 Δ 最大的 CA1，配对检验仍不显著
   （10/17 正、p=0.6291）⇒ ⇒ 症结是**统计单元选错**（session 级配对）
   ⇒ ⇒ **本轮把统计量改成 session 内部**（每 session 内先算「跨组 Δ 中位 − 同组 Δ 中位」）
   ⇒ ⇒ session 偏移在 session 内被消掉，跨 session 只检验**方向**。
3. **fallback 解释规则太弱**：exp98 的 fallback 只看「正对照 Δ > 0」，
   而**负对照的 Δ 也是正的**（+14.81）⇒ ⇒ 「正对照 Δ > 0」分辨不了
   「检验力不足」与「分不出对比强度」
   ⇒ ⇒ **本轮的 fallback 直接比较组内与跨组**（stat 本身就是这个差）⇒ 规则**变强**。

⇒ ⇒ **判据① 的定义一字未动**（转分化代价 = d(跨带) − d(同带劈半自对照)），
变的是**「拿哪些对比、怎么汇总、怎么判」**。

## 分组的外部依据（**注册源级事实，不是从数据里挑的**）

Allen Institute 官方 *Visual Coding* 页（`brain-map.org/our-research/circuits-behavior/visual-coding`）原文：

> "Data was collected from **primary visual cortex (VISp)** and **five higher visual areas
> (VISl, VISal, VISpm, VISam, VISrl)**."

⇒ ⇒ **primary 组 = {VISp}**，**higher 组 = {VISl, VISal, VISpm, VISam, VISrl}**，
**other 组 = {CA1}**（海马，Allen 另说「Spikes in the hippocampal formation (CA1, CA3, and
dentate gyrus) are also measured routinely」⇒ 属**不同脑区划分** ⇒ **不与皮层视觉区同类**）。

⚠️ **CA1 为什么不进主判据**（预登记决定）：exp98 实测 **CA1 几乎一律不可读**
（解码准确率多在 0.007–0.023、机会 0.0085；置换 p 多在 0.10–0.75）⇒
若把 CA1 的对放进「跨组」，**有的 session 有、有的没有** ⇒ 跨 session 的统计量口径不一致
⇒ ⇒ **CA1 的对只报不判**（C3）。

⇒ ⇒ **主判据只问一件事**：**primary（VISp）与 higher（其余 5 个皮层视觉区）能不能被转分化代价分开**。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 阈值 |
|---|---|---|
| **C0″** | **逐带可读性**（同 exp98）：每带把**刺激身份（K = 118 张自然图像）**解码到显著高于机会 ⇒ 不可读的带**不能**参与任何带对（**逐对门槛**：一对的两个端点都不可读 ⇒ 该对被剔除）。排除规则**与 Δ 无关** ⇒ 报出进入/剔除数与原因 | acc > **1/118** 且置换 p < **0.05**（n_perm=100）|
| **C1** | **组内 vs 跨组分离（主判据）**：逐 session 统计量 = `median(跨组对的 Δ) − median(同组对的 Δ)`（Δ 双向平均，与 exp98 同口径；**同组 = higher 组内部的对**，**跨组 = VISp ↔ higher**）⇒ 证据 = **session 级符号检验**（**≥ 8 个 session** ⇒ p ≤ 0.05）**且**中位 > 0 | n ≥ 8、p ≤ 0.05、中位 > 0 |
| **C2** | **退化守卫**（exp90 的教训）：主判据统计量不得是**非零常数**（恒为 0 不算退化 ⇒ 那是没信号 ⇒ 归 `C1_FAIL`）| 离散度 > 0 |
| **C3** | **只报不判**：① 每个带对的 Δ（全部 21 个无序对）；② 每带解码准确率；③ 每 session 的同组/跨组中位与对数；④ CA1 参与的 Δ；⑤ 进入/剔除明细 | — |

### 预登记的**解释规则**（**比 exp98 强**：直接比较组内 vs 跨组）

- `C1_OK` ⇒ **判据① 有判别力**（能把 primary 与 higher 视觉区分开 ⇒ 位置轴的「分组」用法成立）。
- `C1_FAIL` ⇒ **必须**再看两件事才能定性：
  ① **中位统计量 > 0**（方向对：跨组确实更贵）但配对差不显著 ⇒ 只能读作「**方向对、检验力不足**」；
  ② **中位统计量 ≤ 0**（跨组不比同组贵）⇒ 才读作「**判据① 在这个数据上分不出功能分组**」。
- ⇒ ⇒ **「工具无效」与「原理无效」必须分开记**（exp90/exp97a 的教训）；
  ⚠️ 且 `C1_FAIL` **不等于**「判据① 没判别力」（exp98 的不对称规则仍然适用）。

## 协议（**与 exp98 逐字相同**，因为数据与码本锚都没变）

- 数据：DANDI `000021`（Allen Visual Coding–Neuropixels, Brain Observatory 1.1，**CC-BY-4.0**）的
  **32 个 session 全部**；**复用 exp98 已抽好的 npz**（`/domains/allen_trials/*.npz`）⇒ **零新抽取**。
- 类别：`frame` 列（`stimulus_name` 全同、`stimulus_index` 恒为块号 ⇒ 不可用）⇒ **K = 118**（空白屏剔除）。
- 窗：呈现后 **[0, 200] ms**；bin = 10 ms；train/test = **按时间前 70% / 后 30%**；
  共同空间 = **两带联合 PCA**（前 16 维）；码本 = **类条件均值**（每类 1 原型）；
  每带最少 **20** 单元；**同带自对照** = 目标带单元随机劈两半；**不做质量过滤**。

## 统计口径（与 exp89/exp90/exp96/exp97c/exp98 逐字一致）

中位 = **上中位**（`sorted[n//2]`，偶数取偏大）；**零差剔除**；**统计单元 = session**。

## 实现说明

- 低层机器（`perm_p` / `joint_pca` / `build_codebook` / `proto_read` / 统计小工具）**从
  `exp98_posneg_control.py` 直接 import**（该文件自 `64a4052` 起**未再改动**，SHA 记在
  `results/check_findings_consistency.py` 的守卫里 ⇒ 机器可核）。
  ⇒ 本文件只新增：**全对计算 + 分组汇总 + 判决 + 解释规则 + 自测**。
- **判据脚本本体不依赖 exp98 的判决逻辑**（`decide()` 是本文件自己的）⇒ 两轮判决互不污染。

## 用法

```powershell
python -u research/brain_domains/exp/exp100_group_ladder.py --data /domains/allen_trials --out /domains/runs/illusionbrain-align/exp100
python -u research/brain_domains/results/parse_exp100.py     # 离线重算（与 Job 判决逐字比对）
```
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from itertools import combinations

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from exp98_posneg_control import (  # noqa: E402  （低层机器：SHA 固定的共享实现）
    BANDS, MIN_UNITS_PER_BAND, ALPHA_READ,
    median_upper, sign_test_nonzero, perm_p, perm_p_ref,
    joint_pca, build_codebook, proto_read,
)

# ---------------------------------------------------------------------------
# 分组（**Allen 官方措辞**，见 docstring）
# ---------------------------------------------------------------------------
GROUPS = {
    "VISp": "primary",     # Allen: primary visual cortex
    "VISl": "higher",      # Allen: five higher visual areas
    "VISrl": "higher",
    "VISam": "higher",
    "VISal": "higher",
    "VISpm": "higher",
    "CA1": "other",        # hippocampal formation ⇒ 只报，不进主判据
}
MIN_PAIRS_PER_GROUP = 2   # 每组至少 2 对（否则中位数就是单值）


def _pair_sets():
    same, cross, other = [], [], []
    for a, b in combinations(sorted(GROUPS), 2):
        ga, gb = GROUPS[a], GROUPS[b]
        if ga == gb:
            same.append((a, b))
        elif "other" in (ga, gb):
            other.append((a, b))
        else:
            cross.append((a, b))
    return same, cross, other


SAME_PAIRS, CROSS_PAIRS, OTHER_PAIRS = _pair_sets()
ALL_PAIRS = SAME_PAIRS + CROSS_PAIRS + OTHER_PAIRS


def pair_group(a, b):
    ga, gb = GROUPS.get(a), GROUPS.get(b)
    if ga is None or gb is None:
        return None
    if ga == gb:
        return "same"
    if "other" in (ga, gb):
        return "other"
    return "cross"


# ---------------------------------------------------------------------------
# 逐 session 计算
# ---------------------------------------------------------------------------
def prep_session(z):
    feat = np.asarray(z["feat"], dtype=np.float32)
    area = np.asarray(z["unit_area"], dtype=object)
    lab = np.asarray(z["trial_label"], dtype=np.int64)
    t = np.asarray(z["trial_time"], dtype=np.float64)
    n_tri = feat.shape[0]
    order = np.argsort(t)
    n_tr = int(round(0.7 * n_tri))
    tr = np.zeros(n_tri, dtype=bool)
    tr[order[:n_tr]] = True
    te = ~tr
    bands = {}
    for b in BANDS:
        idx = np.flatnonzero(np.array([str(x) == b for x in area]))
        if len(idx) >= MIN_UNITS_PER_BAND:
            bands[b] = idx
    n_classes = int(lab.max()) + 1 if len(lab) else 0
    return feat, lab, tr, te, bands, str(z["session"]), n_classes


def session_group_stat(z, seed=0):
    """⇒ 该 session 的全部带对 Δ、逐带可读性、组内/跨组中位与主判据统计量。"""
    feat, lab, tr, te, bands, sess, n_classes = prep_session(z)
    out = {"session": sess, "bands": {b: int(len(v)) for b, v in bands.items()},
           "n_trials": int(feat.shape[0]), "n_train": int(tr.sum()),
           "n_test": int(te.sum()), "n_classes": n_classes,
           "readable": {}, "pairs": {}}
    if tr.sum() < 40 or te.sum() < 20:
        out["skip"] = "trial 不够"
        return out
    chance = 1.0 / max(n_classes, 2)
    Z = {}
    for b, idx in bands.items():
        X = feat[:, idx].astype(np.float64)
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True) + 1e-9
        Z[b] = (X - mu) / sd
    # C0″：每带能不能把**刺激身份**解码到显著高于机会
    for b in bands:
        Xtr, ytr = Z[b][tr], lab[tr]
        Xte, yte = Z[b][te], lab[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            out["readable"][b] = {"acc": float("nan"), "p": float("nan"), "ok": False}
            continue
        acc, p = perm_p(Xtr, ytr, Xte, yte, seed=seed)
        out["readable"][b] = {"acc": acc, "p": p, "chance": chance,
                              "ok": bool(acc > chance and p < ALPHA_READ)}
    # **全部**有向带对（21 个无序对 ⇒ 42 个有向对）⇒ 与 exp98 不同（那里只算 VISp 配对）
    for A in BANDS:
        for B in BANDS:
            if A == B or A not in bands or B not in bands:
                continue
            mu, W = joint_pca(Z[A][tr], Z[B][tr])
            nA = Z[A].shape[1]
            zA = (Z[A] - mu[:, :nA]) @ W[:nA, :]
            zB = (Z[B] - mu[:, nA:]) @ W[nA:, :]
            PA, usedA = build_codebook(zA, lab, tr, n_classes, seed=seed)
            PB, usedB = build_codebook(zB, lab, tr, n_classes, seed=seed + 100)
            if PA is None or PB is None:
                continue
            used = sorted(set(usedA) & set(usedB))
            if len(used) < 2:
                continue
            PAu = np.vstack([PA[i] for i in range(len(used))])
            PBu = np.vstack([PB[i] for i in range(len(used))])
            r_own = proto_read(zB[te], PBu)
            r_other = proto_read(zB[te], PAu)
            d_cross = r_other - r_own
            rng = np.random.default_rng(seed + 7)
            idxB = bands[B]
            perm = rng.permutation(len(idxB))
            h = len(idxB) // 2
            loc1, loc2 = perm[:h], perm[h:]
            mu2, W2 = joint_pca(Z[B][tr], Z[B][tr])
            nB = Z[B].shape[1]
            muB = mu2[:, :nB]
            WB = W2[:nB, :]
            z1 = (Z[B][:, loc1] - muB[:, loc1]) @ WB[loc1, :]
            z2 = (Z[B][:, loc2] - muB[:, loc2]) @ WB[loc2, :]
            P1, u1 = build_codebook(z1, lab, tr, n_classes, seed=seed + 11)
            P2, u2 = build_codebook(z2, lab, tr, n_classes, seed=seed + 13)
            if P1 is None or P2 is None:
                continue
            used2 = sorted(set(u1) & set(u2))
            if len(used2) < 2:
                continue
            c1 = np.vstack([P1[i] for i in range(len(used2))])
            c2 = np.vstack([P2[i] for i in range(len(used2))])
            d_ctrl = proto_read(z2[te], c1) - proto_read(z2[te], c2)
            out["pairs"][f"{A}->{B}"] = {
                "d_cross": d_cross, "d_ctrl": d_ctrl, "delta": d_cross - d_ctrl,
                "n_classes": len(used), "n_units_B": int(len(idxB))}
    # ---- 分组汇总（**逐对门槛**：两个端点都不可读 ⇒ 该对剔除）
    def pair_mean(a, b):
        vals = []
        for key in (f"{a}->{b}", f"{b}->{a}"):
            d = (out["pairs"].get(key) or {}).get("delta")
            if d is not None and d == d:
                vals.append(d)
        return float(np.mean(vals)) if vals else None

    def readable(b):
        return bool((out["readable"].get(b) or {}).get("ok"))

    buckets = {"same": [], "cross": [], "other": []}
    used_pairs = {"same": [], "cross": [], "other": []}
    for a, b in ALL_PAIRS:
        if not (readable(a) and readable(b)):
            continue
        v = pair_mean(a, b)
        if v is None:
            continue
        g = pair_group(a, b)
        buckets[g].append(v)
        used_pairs[g].append(f"{a}<->{b}")
    for g in ("same", "cross", "other"):
        out[f"{g}_values"] = buckets[g]
        out[f"{g}_pairs"] = used_pairs[g]
        out[f"n_{g}_pairs"] = len(buckets[g])
        out[f"{g}_median"] = median_upper(buckets[g]) if buckets[g] else None
    out["group_stat"] = (out["cross_median"] - out["same_median"]
                          if out["cross_median"] is not None and out["same_median"] is not None
                          else None)
    out["enough_pairs"] = (out["n_same_pairs"] >= MIN_PAIRS_PER_GROUP
                           and out["n_cross_pairs"] >= MIN_PAIRS_PER_GROUP)
    return out


# ---------------------------------------------------------------------------
# 判决
# ---------------------------------------------------------------------------
def decide(per_session, min_sessions=8, alpha=0.05):
    """从逐 session 原始记录重算判决（阈值的唯一出处 ⇒ 可离线重算）。"""
    problems = []
    used, excluded = [], []
    for r in per_session:
        if r.get("skip"):
            excluded.append((r.get("session"), r.get("skip")))
            continue
        if not r.get("enough_pairs"):
            excluded.append((r.get("session"),
                             f"同组/跨组可用的带对不够（各需 ≥{MIN_PAIRS_PER_GROUP} 对；"
                             f"实际 同组 {r.get('n_same_pairs')}、跨组 {r.get('n_cross_pairs')}）"))
            continue
        if r.get("group_stat") is None:
            excluded.append((r.get("session"), "缺主判据统计量"))
            continue
        used.append(r)
    prim = [r["group_stat"] for r in used]
    n_eff, n_pos, n_neg, p_sign, n_zero = sign_test_nonzero(prim)
    med = median_upper(prim)
    c1 = bool(n_eff >= min_sessions and p_sign <= alpha and med > 0)
    if len(prim) > 1:
        spread = float(max(prim) - min(prim))
        max_abs = float(max(abs(x) for x in prim))
    else:
        spread, max_abs = 0.0, 0.0
    degenerate = (spread <= 1e-12 and max_abs > 1e-9)
    c2 = not degenerate
    if len(prim) < min_sessions:
        verdict = "INSUFFICIENT_SESSIONS"
    elif not c2:
        verdict = "DEGENERATE"
    elif c1:
        verdict = "C1_OK"
    else:
        verdict = "C1_FAIL"
    # 解释规则（**比较组内 vs 跨组** ⇒ 比 exp98 强）
    same_pooled = [v for r in used for v in (r.get("same_values") or [])]
    cross_pooled = [v for r in used for v in (r.get("cross_values") or [])]
    same_med = median_upper(same_pooled)
    cross_med = median_upper(cross_pooled)
    if verdict == "C1_OK":
        interp = "判据① **有判别力**（能把 primary 与 higher 视觉区分开）"
    elif verdict == "C1_FAIL":
        if med == med and med > 0:
            interp = ("方向对、检验力不足（跨组中位 > 同组中位，但配对差不显著）"
                      "⇒ **不得**读作「判据① 没判别力」")
        else:
            interp = ("跨组中位 **≤** 同组中位 ⇒ 读作「**判据① 在这个数据上分不出功能分组**」")
    else:
        interp = "判决未成立（session 数不足 / 退化）"
    # 只报项：各带对
    per_pair = {}
    for key in sorted({k for r in per_session for k in (r.get("pairs") or {})}):
        vals = [r["pairs"][key]["delta"] for r in per_session if key in (r.get("pairs") or {})]
        if vals:
            ne, np_, nn, ps, nz = sign_test_nonzero(vals)
            per_pair[key] = {"n": len(vals), "median": median_upper(vals),
                             "n_pos": np_, "p_sign": ps, "n_zero": nz}
    if len(prim) < min_sessions:
        problems.append(f"进入主判据的 session 只有 {len(prim)} 个（要求 ≥ {min_sessions}）")
    if not c2:
        problems.append("退化守卫不过（主判据统计量是非零常数）⇒ 判决作废")
    return {
        "verdict": verdict,
        "C0_pipeline_selfcheck": len(prim) >= min_sessions,
        "n_sessions_total": len(per_session),
        "n_sessions_primary_used": len(prim),
        "sessions_excluded": [{"session": s, "reason": w} for s, w in excluded],
        "C1_has_discriminative_power": c1,
        "C1": {"n_eff": n_eff, "n_pos": n_pos, "n_neg": n_neg, "p": p_sign,
               "n_zero": n_zero, "median_delta": med, "min_sessions": min_sessions,
               "alpha": alpha, "statistic": "median(跨组Δ) − median(同组Δ)",
               "same_median_pooled": same_med, "cross_median_pooled": cross_med,
               "primary_stats": prim},
        "C1_interpretation": interp,
        "C2_degenerate_guard": c2,
        "C2_spread": {"primary": {"spread": spread, "max_abs": max_abs}},
        "per_pair": per_pair,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/domains/allen_trials")
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp100")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    data_dir = pathlib.Path(args.data)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("*.npz"))
    print(f"数据：{len(files)} 个 session（{data_dir}）", flush=True)
    if not files:
        print("✗ 没有抽取结果 ⇒ 先跑 exp98_extract.py")
        return 1
    per_session = []
    for i, fp in enumerate(files):
        try:
            print(f"  → [{i+1}/{len(files)}] {fp.stem}", flush=True)
            z = np.load(fp, allow_pickle=True)
            rec = session_group_stat(z)
            per_session.append(rec)
            print(f"  ✓ {rec['session']}：同组 {rec.get('n_same_pairs')} 对 "
                  f"(中位 {rec.get('same_median')})；跨组 {rec.get('n_cross_pairs')} 对 "
                  f"(中位 {rec.get('cross_median')})；统计量 {rec.get('group_stat')}", flush=True)
        except Exception as e:
            print(f"  ✗ {fp.name}: {type(e).__name__}: {e}", flush=True)
    verdict = decide(per_session)
    payload = {"config": vars(args), "criteria": {
        "C0double_prime": "逐带可读性（acc>1/118 且置换 p<0.05）；逐对门槛：两个端点都不可读 ⇒ 该对剔除"
                          "（排除规则与 Δ 无关）",
        "C1": "组内 vs 跨组分离：逐 session 统计量 = median(跨组Δ) − median(同组Δ)"
              "（同组 = higher 组内部，跨组 = VISp↔higher）；session 级符号检验 ≥8 session 且 p≤0.05 且中位>0",
        "C2": "退化守卫：主判据统计量不得是非零常数",
        "C3": "只报：全部 21 个无序带对的 Δ / 每带解码准确率 / 每 session 组内-跨组中位 / CA1 的 Δ",
    }, "per_session": per_session, "verdict": verdict}
    (out_dir / "exp100_group_ladder.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for _r in per_session:
        print("===REC===" + json.dumps(_r, ensure_ascii=False, default=str), flush=True)
    print("===VERDICT===" + json.dumps(verdict, ensure_ascii=False, default=str), flush=True)
    print(f"\n=== 判决：{verdict['verdict']} ===")
    print(f"  进入主判据：{verdict['n_sessions_primary_used']} / {verdict['n_sessions_total']}")
    print(f"  符号检验：{verdict['C1']['n_pos']}/{verdict['C1']['n_eff']} 正，"
          f"p={verdict['C1']['p']:.4f}，中位 Δ={verdict['C1']['median_delta']:+.4f}")
    print(f"  同组中位 Δ={verdict['C1']['same_median_pooled']:+.4f}；"
          f"跨组中位 Δ={verdict['C1']['cross_median_pooled']:+.4f}")
    print(f"  解释：{verdict['C1_interpretation']}")
    print(f"  C2 退化守卫：{verdict['C2_degenerate_guard']}")
    print(f"结果写入 {out_dir}/exp100_group_ladder.json")
    return 0


# ---------------------------------------------------------------------------
def self_test():
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # ① 分组集合（**预登记的分组常量**：主判据只用 primary↔higher 与 higher 内部）
    chk("分组：同组 10 对（higher 内部）、跨组 5 对（VISp↔higher）、other 6 对（CA1）",
        len(SAME_PAIRS) == 10 and len(CROSS_PAIRS) == 5 and len(OTHER_PAIRS) == 6)
    chk("分组：VISp↔VISl 属跨组、VISl↔VISam 属同组、VISp↔CA1 属 other",
        pair_group("VISp", "VISl") == "cross" and pair_group("VISl", "VISam") == "same"
        and pair_group("VISp", "CA1") == "other")

    # ② perm_p 优化版 vs 参考实现（对拍 ⇒ 等价性机器可判）
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(240, 14))
    yr = rng.integers(0, 5, 240)
    Xs = rng.normal(size=(80, 14))
    ys = rng.integers(0, 5, 80)
    o1, p1 = perm_p(Xr, yr, Xs, ys, n_perm=30)
    o2, p2 = perm_p_ref(Xr, yr, Xs, ys, n_perm=30)
    chk("perm_p 优化版与参考实现逐位一致", o1 == o2 and p1 == p2)

    # ③ decide() 的判决分支
    def fake_rec(stat=1.0, same=1.0, cross=2.0, ok=True, jitter=0.3, seed=0):
        r = np.random.default_rng(seed)
        rd = {b: {"acc": 0.5 if ok else 0.02, "p": 0.001 if ok else 0.9, "ok": ok}
              for b in ("VISp", "VISl", "VISam", "VISal")}
        return {"session": f"f{seed}", "readable": rd, "pairs": {},
                "n_same_pairs": 3, "n_cross_pairs": 3, "enough_pairs": ok,
                "same_values": [same], "cross_values": [cross],
                "same_median": same, "cross_median": cross,
                "group_stat": stat + jitter * r.normal()}

    v = decide([fake_rec(1.0, seed=i) for i in range(10)])
    chk("decide 10 个 session 全正 ⇒ C1_OK",
        v["verdict"] == "C1_OK" and v["C1_has_discriminative_power"])
    v = decide([fake_rec(-1.0, seed=i) for i in range(10)])
    chk("decide 全负 ⇒ C1_FAIL", v["verdict"] == "C1_FAIL")
    v = decide([fake_rec(1.0, jitter=0.0, seed=i) for i in range(10)])
    chk("decide 主统计量是非零常数 ⇒ DEGENERATE", v["verdict"] == "DEGENERATE")
    v = decide([fake_rec(1.0, seed=i) for i in range(5)])
    chk("decide 只有 5 个 session ⇒ INSUFFICIENT_SESSIONS",
        v["verdict"] == "INSUFFICIENT_SESSIONS")
    recs = [fake_rec(1.0, seed=i) for i in range(10)]
    recs[0]["enough_pairs"] = False
    v = decide(recs)
    chk("decide 对数不够 ⇒ 只剔这个 session（9 个够 ⇒ C1_OK）",
        v["verdict"] == "C1_OK" and v["n_sessions_primary_used"] == 9
        and len(v["sessions_excluded"]) == 1)
    v = decide([fake_rec(-1.0, seed=i) for i in range(10)])
    chk("decide C1_FAIL 且中位 ≤ 0 ⇒ 读作「分不出功能分组」",
        "分不出功能分组" in v["C1_interpretation"])

    # ④ **端到端**：合成 session —— VISp 单独一套表征，3 个 higher 共享另一套
    #    ⇒ 跨组 Δ 大、同组 Δ 小 ⇒ 统计量 > 0 ⇒ 10 个 session ⇒ C1_OK
    def fake_session(seed=0, n_tri=480, n_per=40, k=8, structured=True):
        rng = np.random.default_rng(seed)
        cls = rng.permutation(np.repeat(np.arange(k), n_tri // k))
        prim = rng.normal(size=(k, n_per))
        high = rng.normal(size=(k, n_per)) if structured else prim
        names = ["VISp", "VISl", "VISam", "VISal"]
        pats = [prim, high, high, high]
        feat = np.zeros((n_tri, len(names) * n_per), dtype=np.float32)
        for bi, pat in enumerate(pats):
            feat[:, bi * n_per:(bi + 1) * n_per] = pat[cls] + 0.6 * rng.normal(size=(n_tri, n_per))
        return {"feat": feat,
                "unit_area": np.array([n for n in names for _ in range(n_per)], dtype=object),
                "trial_label": cls,
                "trial_time": np.arange(n_tri, dtype=np.float64),
                "session": f"syn{seed}"}

    rec = session_group_stat(fake_session(0))
    chk("端到端：合成 session 产出同组/跨组中位与统计量",
        rec.get("same_median") is not None and rec.get("cross_median") is not None
        and rec.get("group_stat") is not None)
    chk("端到端：有分组结构 ⇒ **跨组 Δ > 同组 Δ**（统计量 > 0）",
        rec["group_stat"] > 0)
    rec_flat = session_group_stat(fake_session(0, structured=False))
    chk("端到端：无分组结构 ⇒ 统计量 ≈ 0（不会凭空分出组）",
        abs(rec_flat["group_stat"]) < abs(rec["group_stat"]))
    v = decide([session_group_stat(fake_session(i)) for i in range(10)])
    chk("端到端：10 个有结构的合成 session ⇒ decide ⇒ C1_OK", v["verdict"] == "C1_OK")
    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
