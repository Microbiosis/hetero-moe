"""merge_exp93a.py — 把 exp93a 的两段结果（s1-5 来自 Job 日志、s6-10 来自桶内 JSON）合并。

**为什么**：两个 Job 写**同一个**桶内路径 ⇒ 后完成的会覆盖先完成的
⇒ 必须分别保存、分别解析，再按被试合并。

口径与 `parse_exp93.py` 一致：**中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp93-results"
LOG = OUT / "exp93a_s1-5.from_log.json"


def med(xs):
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else None


def main():
    cells = {}

    def add(cell, vals):
        cells.setdefault(cell, []).extend([float(v) for v in vals])

    # --- s1-5（Job 日志）---
    if LOG.exists():
        a = json.loads(LOG.read_text(encoding="utf-8"))
        for _s, v in a["per_subject"].items():
            if "base" in v:
                add("P1/ridge/16540", [v["base"]])
            for n, acc in (v.get("f1") or {}).items():
                add(f"P1/ridge/{n}", [acc])
            if "p3" in v:
                add("P3/同集身份解码", [v["p3"]])
            if "p2" in v:
                add("P2/同集检索", [v["p2"]])

    # --- 桶内 JSON 段（**按被试去重**，不依赖固定文件名）---
    for jf in sorted(OUT.glob("exp93a_level_audit*.json")):
        if jf.name.endswith(".raw.json") or jf.name.endswith(".summary.json"):
            continue
        d = json.loads(jf.read_text(encoding="utf-8"))
        for r in d.get("per_subject", []):
            if "error" in r:
                continue
            b = r.get("baseline_ridge") or {}
            if b.get("acc1_cos") is not None:
                add("P1/ridge/16540", [b["acc1_cos"]])
            if b.get("acc1_featavg_cos") is not None:
                add("P1/ridge/16540/试次平均", [b["acc1_featavg_cos"]])
            for n, v in (r.get("F1_train_size") or {}).items():
                if v.get("acc1_cos") is not None:
                    add(f"P1/ridge/{n}", [v["acc1_cos"]])
            for tag, v in (r.get("F2_decoder") or {}).items():
                if tag == "ridge":
                    continue          # ← 与基准格**同一次计算**（重复计数会虚增 n）
                if v.get("acc1_cos") is not None:
                    add(f"F2/{tag}/16540", [v["acc1_cos"]])
            f4 = r.get("F4_target_pca128") or {}
            if f4.get("acc1_cos") is not None:
                add("P1/ridge+PCA128目标/16540", [f4["acc1_cos"]])
            f5 = r.get("F5_protocol") or {}
            if f5.get("P3_same_set_identity") is not None:
                add("P3/同集身份解码", [f5["P3_same_set_identity"]])
            s2 = r.get("self_check_S2_same_set_retrieval") or {}
            if s2.get("acc1_cos") is not None:
                add("P2/同集检索", [s2["acc1_cos"]])

    out = {}
    for cell, xs in cells.items():
        out[cell] = {"n": len(xs), "median_up": med(xs), "mean": mean(xs),
                     "min": min(xs), "max": max(xs)}
    (OUT / "exp93a_merged.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # 人读的表（**完整 10 被试版**；这里只带**有日志/JSON 两段合并**的格子）
    ORDER = ["P1/ridge/16540", "P1/ridge/16540/试次平均", "P1/ridge/4000",
             "P1/ridge/1000", "P2/同集检索", "P3/同集身份解码"]
    CH = {"P2/同集检索": 1 / 2000.0}
    L = ["# exp93a 档位因子表（**合并后的完整版**）", "",
         "来源：s1-5 段来自 Job 日志解析、s6-10 段来自桶内 JSON ⇒ 按被试合并。",
         "**中位 = 上中位**（右偏分布会报高 30–40%）。", "",
         "| 协议 / 解码器 / 训练图数 | 候选数 | n | 中位(上) | 均值 | ×随机 |",
         "|---|---:|---:|---:|---:|---:|"]
    for cell in ORDER:
        if cell not in out:
            continue
        v = out[cell]
        ch = CH.get(cell, 1 / 200.0)
        L.append(f"| `{cell}` | {int(round(1/ch))} | {v['n']} | {v['median_up']:.4f} "
                 f"| {v['mean']:.4f} | {v['median_up']/ch:.1f}× |")
    (OUT / "exp93a_merged.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{'cell':<32} {'n':>3} {'中位(上)':>9} {'均值':>9}  {'×随机':>7}")
    chance = {"P2/同集检索": 1 / 2000.0}
    for cell in sorted(out):
        v = out[cell]
        ch = chance.get(cell, 1 / 200.0)
        print(f"{cell:<32} {v['n']:>3} {v['median_up']:>9.4f} {v['mean']:>9.4f}"
              f"  {v['median_up']/ch:>6.1f}×")


if __name__ == "__main__":
    main()
