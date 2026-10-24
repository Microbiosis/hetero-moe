"""cross_validate_branches.py — **三条支线的交叉验证**（exp92 的 A/B/C + 先前的 exp89/90/91a）。

## 交叉验证要回答的四件事

| 交叉 | 问题 | 数据来源 |
|---|---|---|
| **A × exp91a** | 位置轴的「后>前」在**特征层**与**模型层**是否一致？ | exp92-A vs `readout-results/exp91a_readout.json` |
| **A × C** | 模型层的分工成立**且**档位仍低 ⇒ 瓶颈是训练目标/容量，不是位置轴 | exp92-A vs exp92-C |
| **B × exp90** | 扣掉全局左右增益差后，后部特有的右侧优势**是否仍在** | exp92-B vs `lateral-results/exp90_lateral.json` |
| **C × A13** | 我们模型层的 200-way top-1 离文献（NICE-EEG 15.6%）有多远 | exp92-C vs 台账 A13 |

## 数据来源

- exp92 三条支线的 JSON 在桶里 `/domains/runs/illusionbrain-align/exp92/<branch>/<branch>.json`
  （本机写通道断 ⇒ 读要重试；读不到就退化为**从 Job 日志重建**）
- 先前结果都在本地：`readout-results/`、`lateral-results/`、`posaxis-results/`

## 输出

`results/cross-validation/cross_validation.{json,md}`
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cross-validation"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align/exp92"
BRANCHES = ("posaxis_model", "lateral_baseline", "level")
LIT_TOP1 = 15.6          # NICE-EEG（arXiv:2308.13234）同一份 THINGS-EEG 的 200-way 零样本 top-1


def sign_test_p(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    p = 2.0 * sum(math.comb(n, i) for i in range(min(k, m) + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


def med(xs):
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return float(s[len(s) // 2])


def fetch(branch, retries=5):
    """从桶里取支线 JSON（读通道不稳 ⇒ 重试）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{branch}.json"
    if dst.exists() and dst.stat().st_size > 0:
        return json.loads(dst.read_text(encoding="utf-8"))
    src = f"{BUCKET}/{branch}/{branch}.json"
    for i in range(retries):
        try:
            r = subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return json.loads(dst.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            pass
        time.sleep(15)
    return None


def summarize_A(d):
    """支线 A：模型层的通道组消融。"""
    out = {}
    for rec in d.get("per_subject", []):
        for tag, groups in (rec.get("groups") or {}).items():
            for g, e in groups.items():
                out.setdefault(tag, {}).setdefault(g, []).append(e["r"])
        for tag, groups in (rec.get("random_null") or {}).items():
            for g, e in groups.items():
                out.setdefault("null", {}).setdefault(f"{tag}.{g}", []).append(e["r_median"])
    res = {}
    for tag, groups in out.items():
        if tag == "null":
            continue
        for g, rs in groups.items():
            res.setdefault(tag, {})[g] = {"median": med(rs), "n": len(rs)}
        # 后 − 前 的配对差（逐被试）
        diffs = []
        for rec in d.get("per_subject", []):
            gg = (rec.get("groups") or {}).get(tag) or {}
            if "POST" in gg and "ANT" in gg:
                diffs.append(gg["POST"]["r"] - gg["ANT"]["r"])
        if diffs:
            k, m, p = sign_test_p(diffs)
            res[tag]["_POST_minus_ANT"] = {"median": med(diffs), "n_pos": k,
                                          "n_neg": m, "p_sign": p, "n": len(diffs)}
    for key, rs in out.get("null", {}).items():
        res.setdefault("null", {})[key] = {"median": med(rs), "n": len(rs)}
    return res


def summarize_B(d):
    out = {"post_RL": [], "ant_RL": [], "resid": [], "mid": []}
    for rec in d.get("per_subject", []):
        v = rec.get("vals") or {}
        for k, key in (("post_RL", "post_RL"), ("ant_RL", "ant_RL"),
                       ("post_resid_ant_base", "resid"), ("mid_mean", "mid")):
            if k in v:
                out[key].append(v[k])
    res = {}
    for k, xs in out.items():
        if not xs:
            continue
        kk, mm, p = sign_test_p(xs) if k != "mid" else (0, 0, float("nan"))
        res[k] = {"median": med(xs), "n": len(xs), "n_pos": kk, "n_neg": mm,
                  "p_sign": p}
    return res


def summarize_C(d):
    out = {}
    for rec in d.get("per_subject", []):
        for tag, e in (rec.get("level") or {}).items():
            out.setdefault(tag, []).append(e["acc1"])
    res = {}
    for tag, xs in out.items():
        res[tag] = {"median": med(xs), "n": len(xs), "all": xs}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    branches = {}
    for b in BRANCHES:
        d = fetch(b)
        if d is None:
            print(f"✗ 取不到支线 {b} 的结果（读通道失败）")
        else:
            ps = d.get("per_subject", [])
            # ⚠️ **必须查「全部被试都报错」**：否则汇总会静默出空表
            #    （踩过：--n-random 0 让随机对照块抛错 ⇒ 10 个被试全是 error 记录）
            errs = [p for p in ps if p.get("error")]
            if ps and len(errs) == len(ps):
                print(f"✗ 支线 {b}: {len(ps)} 个被试**全部报错**（{errs[0].get('error','')[:120]}）"
                      f" ⇒ 该支线无有效结果")
                d = None
            elif errs:
                print(f"⚠ 支线 {b}: {len(errs)}/{len(ps)} 个被试报错")
            if d is not None:
                branches[b] = d
                print(f"✓ 支线 {b}: {len(ps)} 被试")
    if not branches:
        return 1

    A = summarize_A(branches["posaxis_model"]) if "posaxis_model" in branches else {}
    B = summarize_B(branches["lateral_baseline"]) if "lateral_baseline" in branches else {}
    C = summarize_C(branches["level"]) if "level" in branches else {}

    # 先前结果（本地）
    def load(p):
        fp = HERE / p
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None

    e91a = load("readout-results/exp91a_readout.json")
    e90 = load("lateral-results/exp90_lateral.json")
    e89 = load("posaxis-results/exp89_posaxis.json")

    cross = {}

    # ---- A × exp91a：特征层 vs 模型层 ----
    if A and e91a:
        s91 = e91a.get("summary", {})
        rows = []
        for tag in ("s1a", "s1b"):
            if tag not in A:
                continue
            d_model = A[tag].get("_POST_minus_ANT")
            # 特征层：amp630 / bandpow 的后−前
            for ro in ("amp630", "bandpow"):
                if ro in s91 and "_POST_minus_ANT" in s91[ro]:
                    e = s91[ro]["_POST_minus_ANT"]
                    rows.append({"layer": "特征层", "readout": ro,
                                 "median": e["median"], "n_pos": e["n_pos"],
                                 "n": e["n"], "p": e["p_sign"]})
            if d_model:
                rows.append({"layer": "模型层", "readout": tag,
                             "median": d_model["median"],
                             "n_pos": d_model["n_pos"], "n": d_model["n"],
                             "p": d_model["p_sign"]})
        cross["A_x_exp91a"] = rows
        cross["A_x_exp91a_verdict"] = (
            "一致" if rows and all(r["median"] > 0 and r["n_pos"] == r["n"]
                                    for r in rows) else "不一致/部分")

    # ---- A × C ----
    if A and C:
        rows = []
        for tag in ("s1a", "s1b"):
            if tag in A and "_POST_minus_ANT" in A[tag]:
                rows.append({"model": tag,
                             "POST_minus_ANT": A[tag]["_POST_minus_ANT"]["median"],
                             "p": A[tag]["_POST_minus_ANT"]["p_sign"]})
        levels = {k: v["median"] for k, v in C.items()}
        cross["A_x_C"] = {"posaxis": rows, "level": levels,
                          "lit_top1": LIT_TOP1}
        best = max(levels.values()) if levels else 0.0
        cross["A_x_C_verdict"] = (
            "模型层分工成立、但档位远低于文献 ⇒ 瓶颈在训练目标/容量，不在位置轴"
            if rows and all(r["POST_minus_ANT"] > 0 for r in rows)
            and best < LIT_TOP1 / 2 else "见数")

    # ---- B × exp90 ----
    if B and e90:
        s90 = (e90.get("v2") or {}).get("summary", {})
        cross["B_x_exp90"] = {
            "raw_sitepair_median": (
                (s90.get("LEFT_minus_RIGHT") or {}).get("median")),
            "raw_sitepair_p": (
                (s90.get("LEFT_minus_RIGHT") or {}).get("p_sign")),
            "post_RL_median": B.get("post_RL", {}).get("median"),
            "ant_RL_median": B.get("ant_RL", {}).get("median"),
            "residual_median": B.get("resid", {}).get("median"),
            "residual_n_pos": B.get("resid", {}).get("n_pos"),
            "residual_n": B.get("resid", {}).get("n"),
            "residual_p": B.get("resid", {}).get("p_sign"),
        }

    # ---- C × A13 ----
    if C:
        cross["C_x_A13"] = {
            "ours": {k: v["median"] for k, v in C.items()},
            "lit_top1": LIT_TOP1,
            "lit_source": "NICE-EEG arXiv:2308.13234（同一份 THINGS-EEG）",
            "A13_baseline": 0.0111,
        }

    (OUT / "cross_validation.json").write_text(
        json.dumps({"branches": {"A": A, "B": B, "C": C}, "cross": cross},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# exp92 三条支线的交叉验证", ""]
    if A:
        L += ["## 支线 A：模型层的通道组消融（通道掩码）", "",
              "| 模型 | ALL | 后部 POST | 前部 ANT | 枕叶 OCC | 后−前（配对） |", "|---|---|---|---|---|---|"]
        for tag in ("s1a", "s1b"):
            if tag not in A:
                continue
            e = A[tag]
            d = e.get("_POST_minus_ANT", {})
            L.append(f"| {tag} | {e.get('ALL',{}).get('median',float('nan')):+.4f} "
                     f"| {e.get('POST',{}).get('median',float('nan')):+.4f} "
                     f"| {e.get('ANT',{}).get('median',float('nan')):+.4f} "
                     f"| {e.get('OCC',{}).get('median',float('nan')):+.4f} "
                     f"| {d.get('median',float('nan')):+.4f} "
                     f"({d.get('n_pos','?')}/{d.get('n','?')}, p={d.get('p_sign',float('nan')):.4f}) |")
        for k, v in (A.get("null") or {}).items():
            L.append(f"- 同尺寸随机组对照 `{k}`：中位 {v['median']:+.4f}")
        L.append("")
    if B:
        L += ["## 支线 B：左右轴的基线扣除", "",
              "| 量 | 中位 | 正向数 | p |", "|---|---|---|---|"]
        for k, label in (("post_RL", "后部 R−L（原始）"),
                         ("ant_RL", "前部 R−L（全局增益差的代理）"),
                         ("resid", "后部残余（后部 R−L − 前部 R−L）")):
            e = B.get(k)
            if not e:
                continue
            L.append(f"| {label} | {e['median']:+.4f} | "
                     f"{e.get('n_pos','-')}/{e.get('n','-')} | "
                     f"{e.get('p_sign', float('nan')):.4f} |")
        if B.get("mid"):
            # ⚠️ 中线 7 通道**没有左右**⇒ 只能当幅度参照，**不能**当偏倚检验
            L.append(f"| （参照）中线 7 通道平均幅度 | "
                     f"{B['mid']['median']:.4f} | — | — |")
        L.append("")
    if C:
        L += ["## 支线 C：绝对档位（200-way top-1，ridge 200 选 1）", "",
              "| 读出 | acc1 中位 | n |", "|---|---|---|"]
        for k, v in C.items():
            L.append(f"| {k} | {v['median']:.4f} | {v['n']} |")
        L.append(f"| **文献（NICE-EEG，同一份 THINGS-EEG）** | **{LIT_TOP1:.4f}** | — |")
        L.append("")
    L += ["## 交叉验证判读", ""]
    for k, v in cross.items():
        if k.endswith("_verdict"):
            L.append(f"- **{k[:-8]}**：{v}")
    L.append("")
    (OUT / "cross_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"写入 {OUT}/cross_validation.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
