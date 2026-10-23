"""table_from_logs.py — **不下载文件**，直接从 Job 日志重建最终表。

为什么走日志：`hf buckets cp`（下载）会被间歇拦截，而 `hf jobs logs`（读）
一直可用。日志里每个被试每个模型都打印了一行完整数字：

    [cbramod] A r=+0.0105 p=0.037 | B acc1=0.0054 α=100 (零中位 0.0050)

⇒ 按模型收集这些行，算中位/显著被试数，即得最终表。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

LINE = re.compile(
    r"\[(?P<name>[^\]]+)\]\s*A r=(?P<r>[+-][\d.]+)\s+p=(?P<p>[\d.]+)\s*\|\s*"
    r"B acc1=(?P<acc>[\d.]+)(?:.*?零中位 (?P<null>[\d.]+))?")

MODELS = ["cbramod", "eegdino-small", "biot", "signal-jepa",
          "mirepnet", "braintokenizer", "基线 630-d"]
ALIAS = {"基线 630-d": "630-d 原始特征"}


def all_jobs():
    r = subprocess.run(["hf", "jobs", "list", "-a", "--format", "json"],
                       capture_output=True, text=True, timeout=300)
    out = r.stdout or ""
    i, j = out.find("["), out.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(out[i:j + 1])
    except Exception:
        return []


def logs_of(job_id):
    r = subprocess.run(["hf", "jobs", "logs", f"redmingwei/{job_id}"],
                       capture_output=True, text=True, timeout=300)
    return r.stdout or ""


def main():
    jobs = all_jobs()
    # name 形如 t2-s01-stable / t3-s01-dev
    by_key = {}
    for j in jobs:
        nm = (j.get("name") or "")
        m = re.match(r"t[23]-s(\d\d)-(stable|dev)$", nm)
        st = j.get("status")
        stage = st.get("stage") if isinstance(st, dict) else st
        if m and stage == "COMPLETED":
            by_key[f"s{m.group(1)}:{m.group(2)}"] = j["id"]
    print(f"找到 {len(by_key)} 个可用 Job", flush=True)
    miss_key = [f"s{s:02d}:{p}" for s in range(1, 11) for p in ("stable", "dev")
                if f"s{s:02d}:{p}" not in by_key]
    if miss_key:
        print(f"⚠ 缺 Job：{miss_key}", flush=True)

    # 每个被试：stable 趟 + dev 趟
    per_subject = {}
    for s in range(1, 11):
        rows = {}
        for pas in ("stable", "dev"):
            jid = by_key.get(f"s{s:02d}:{pas}")
            if not jid:
                continue
            for line in logs_of(jid).splitlines():
                mm = LINE.search(line)
                if not mm:
                    continue
                name = mm.group("name").strip()
                if name not in MODELS:
                    continue
                rows[name] = {
                    "r": float(mm.group("r")), "p": float(mm.group("p")),
                    "acc": float(mm.group("acc")),
                    "null": float(mm.group("null")) if mm.group("null") else None,
                }
        if rows:
            per_subject[f"sub-{s:02d}"] = rows
    print(f"解析到 {len(per_subject)} 个被试", flush=True)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return sorted(vals)[len(vals) // 2] if vals else None

    out = []
    for name in MODELS:
        rs = [per_subject[k][name]["r"] for k in per_subject if name in per_subject[k]]
        ps = [per_subject[k][name]["p"] for k in per_subject if name in per_subject[k]]
        ac = [per_subject[k][name]["acc"] for k in per_subject if name in per_subject[k]]
        nu = [per_subject[k][name]["null"] for k in per_subject if name in per_subject[k]]
        if not rs:
            continue
        out.append({
            "model": name, "n": len(rs),
            "r": med(rs), "p": med(ps), "n_sig": sum(1 for p in ps if p < 0.05),
            "acc": med(ac), "null": med(nu),
            "ratio": (med(ac) / med(nu)) if (med(ac) and med(nu)) else None,
        })

    print("\n" + "=" * 104)
    print(f"7 基模脑对齐表（{len(per_subject)} 个被试，**由 Job 日志重建**）")
    print("=" * 104)
    print("| 模型 | A: r_img (中位) | A: p (中位) | A 显著被试 | B: acc1 (中位) | "
          "B: 零中位 | B: ×随机 |")
    print("|---|---|---|---|---|---|---|")
    for o in out:
        rr = f"{o['r']:+.4f}" if o["r"] is not None else "—"
        pp = f"{o['p']:.3f}" if o["p"] is not None else "—"
        aa = f"{o['acc']:.4f}" if o["acc"] is not None else "—"
        nn = f"{o['null']:.4f}" if o["null"] is not None else "—"
        ratio = f"{o['ratio']:.1f}×" if o["ratio"] else "—"
        print(f"| {o['model']} | {rr} | {pp} | {o['n_sig']}/{o['n']} | {aa} | {nn} | {ratio} |")

    Path("alignment-results").mkdir(exist_ok=True)
    with open("alignment-results/table_from_logs.md", "w", encoding="utf-8") as f:
        f.write(f"# 7 基模脑对齐表（{len(per_subject)} 被试，由 Job 日志重建）\n\n")
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open("alignment-results/parsed_from_logs.json", "w", encoding="utf-8") as f:
        json.dump(per_subject, f, ensure_ascii=False, indent=2)
    print(f"\n每被试原始行已存 alignment-results/parsed_from_logs.json")


if __name__ == "__main__":
    main()
