"""merge_exp93a.py — 把 exp93a 的多段结果（按被试去重）合并成一张因子表。

## 为什么必须按被试去重

exp93a 跑过**两段**（s1-5 与 s6-10），每段有两份来源：

- 桶内 JSON（`exp93a_level_audit_sub-XX-sub-YY.json`，按被试范围自动命名）
- Job 日志解析（`exp93a_s1-5.from_log.json`，取自第一段）

⇒ **同一批被试在两份来源里都会出现**。若按「把每段的数值累进列表」来算，
s1-5 的 5 个被试会被**计两次** ⇒ 中位被拉偏（本轮实测：0.0321 → 0.0297），
而 n 会显示成 15/14/10 这种**不自洽**的数字。

⇒ ⇒ 正确做法：**先按 subject 建成一个字典**（JSON 优先，日志只补 JSON 里没有的），
再从字典聚合每一格。

## 统计口径

**中位 = 上中位**（`sorted[n//2]`，偶数取偏大）⇒ 右偏分布会报高 30–40%。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp93-results"
LOG = OUT / "exp93a_s1-5.from_log.json"
CHANCE = 1.0 / 200.0


def med(xs):
    """**上中位**（本项目约定）。"""
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else None


def from_json(rec):
    """桶内 JSON 的一条被试记录 → {cell: value}。"""
    out = {}
    b = rec.get("baseline_ridge") or {}
    if b.get("acc1_cos") is not None:
        out["P1/ridge/16540"] = b["acc1_cos"]
    if b.get("acc1_featavg_cos") is not None:
        out["P1/ridge/16540/试次平均"] = b["acc1_featavg_cos"]
    for n, v in (rec.get("F1_train_size") or {}).items():
        if v.get("acc1_cos") is not None:
            out[f"P1/ridge/{n}"] = v["acc1_cos"]
    for tag, v in (rec.get("F2_decoder") or {}).items():
        if tag == "ridge":
            continue                      # 与基准格同一次计算（重复计数会虚增 n）
        if v.get("acc1_cos") is not None:
            out[f"F2/{tag}/16540"] = v["acc1_cos"]
    f4 = rec.get("F4_target_pca128") or {}
    if f4.get("acc1_cos") is not None:
        out["P1/ridge+PCA128目标/16540"] = f4["acc1_cos"]
    f5 = rec.get("F5_protocol") or {}
    if f5.get("P3_same_set_identity") is not None:
        out["P3/同集身份解码"] = f5["P3_same_set_identity"]
    s2 = rec.get("self_check_S2_same_set_retrieval") or {}
    if s2.get("acc1_cos") is not None:
        out["P2/同集检索"] = s2["acc1_cos"]
    return out


def from_log(rec):
    """Job 日志解析出的一条被试记录 → {cell: value}。"""
    out = {}
    if "base" in rec:
        out["P1/ridge/16540"] = rec["base"]
    for n, acc in (rec.get("f1") or {}).items():
        out[f"P1/ridge/{n}"] = acc
    for tag, acc in (rec.get("f2") or {}).items():
        out[f"F2/{tag}/16540"] = acc
    if "p3" in rec:
        out["P3/同集身份解码"] = rec["p3"]
    if "p2" in rec:
        out["P2/同集检索"] = rec["p2"]
    return out


def main():
    per = {}          # subject → {cell: value}
    src = {}

    # ---- 1) 桶内 JSON（优先）----
    for fp in sorted(OUT.glob("exp93a_level_audit_*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("per_subject", []):
            if "error" in r or "subject" not in r:
                continue
            per[r["subject"]] = from_json(r)
            src[r["subject"]] = "json"

    # ---- 2) Job 日志（**只补 JSON 里没有的**）----
    if LOG.exists():
        d = json.loads(LOG.read_text(encoding="utf-8"))
        added = 0
        for s, rec in (d.get("per_subject") or {}).items():
            if s in per:
                continue
            per[s] = from_log(rec)
            src[s] = "log"
            added += 1
        if added:
            print(f"日志补了 {added} 个被试")

    cells = {}
    for s, cv in per.items():
        for c, v in cv.items():
            cells.setdefault(c, []).append(v)

    out = {}
    for cell, xs in cells.items():
        out[cell] = {"n": len(xs), "median_up": med(xs), "mean": mean(xs),
                     "min": min(xs), "max": max(xs)}
    (OUT / "exp93a_merged.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    ORDER = ["P1/ridge/16540", "P1/ridge/16540/试次平均", "P1/ridge/4000",
             "P1/ridge/1000", "F2/ridge_unit_target/16540",
             "F2/rrr_k64/16540", "F2/rrr_k256/16540", "F2/cca_k32/16540",
             "F2/cca_k128/16540", "P1/ridge+PCA128目标/16540",
             "P2/同集检索", "P3/同集身份解码"]
    CH = {"P2/同集检索": 1 / 2000.0}
    print(f"被试 {len(per)} 个（来源：{sum(1 for v in src.values() if v=='json')} json + "
          f"{sum(1 for v in src.values() if v=='log')} log）\n")
    print(f"{'cell':<32} {'n':>3} {'中位(上)':>9} {'均值':>9}  {'×随机':>7}")
    for cell in ORDER:
        if cell not in out:
            continue
        v = out[cell]
        ch = CH.get(cell, CHANCE)
        print(f"{cell:<32} {v['n']:>3} {v['median_up']:>9.4f} {v['mean']:>9.4f}"
              f"  {v['median_up']/ch:>6.1f}×")

    L = ["# exp93a 档位因子表（**按被试去重的合并版**）", "",
         f"被试 {len(per)} 个（JSON 优先，日志只补缺失的）",
         "**中位 = 上中位**（右偏分布会报高 30–40%）。", "",
         "| 协议 / 解码器 / 训练图数 | 候选数 | n | 中位(上) | 均值 | ×随机 |",
         "|---|---:|---:|---:|---:|---:|"]
    for cell in ORDER:
        if cell not in out:
            continue
        v = out[cell]
        ch = CH.get(cell, CHANCE)
        L.append(f"| `{cell}` | {int(round(1/ch))} | {v['n']} | {v['median_up']:.4f} "
                 f"| {v['mean']:.4f} | {v['median_up']/ch:.1f}× |")
    (OUT / "exp93a_merged.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n写入 {OUT/'exp93a_merged.md'}")


if __name__ == "__main__":
    main()
