"""parse_readout_from_logs.py — 从 Job 日志重建 **exp91a（读出强度检验）**的结果。

## 关键统计量

**每种读出下的「后部 − 前部」配对差** —— 它回答：
「换更强的读出后，前部还是读不到吗？」

| 读出 | 补上了什么 | 对哪种码型敏感 |
|---|---|---|
| `amp630` | （基线）逐通道幅度 10×100 ms | 带符号的**时域幅度**码 |
| `bandpow` | 5 个频带的对数带功率 | **谱形**码（功率是平方） |
| `covpat` | 组内通道协方差（空间模式） | 同上 + 空间共变结构 |
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "readout-results"
JOBS = ("t91a-readout-s1-5", "t91a-readout-s6-10")

SUBJ = re.compile(r"===\s*sub-(\d+)\s*===")
RSA = re.compile(
    r"^\s*RSA (?P<readout>[a-z0-9]+)\s+(?P<group>[A-Z]+)\s+n_ch=(?P<n>\d+)\s+"
    r"dim=(?P<dim>\d+)\s+r=(?P<r>[+-][\d.]+)\s+p=(?P<p>[\d.]+)")
MB = re.compile(r"^\s*MB\s+[a-z0-9]+\s+(?P<group>[A-Z]+)\s+acc1=(?P<acc>[\d.]+)\s+"
                r"\(零中位 (?P<null>[\d.]+)\)")
AMPALL = re.compile(r"^\s*AMPALL (?P<vals>[-+0-9.,eE]+)$")


def all_jobs():
    r = subprocess.run(["hf", "jobs", "list", "-a", "--format", "json", "--limit", "60"],
                       capture_output=True, text=True, timeout=900)
    s = r.stdout or ""
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return []


def job_ids():
    best = {}
    for j in all_jobs():
        nm = j.get("name") or (j.get("labels") or {}).get("name") or ""
        if nm not in best or (j.get("created_at") or "") > (best[nm].get("created_at") or ""):
            best[nm] = j
    return {nm: j["id"] for nm, j in best.items()}


def logs_of(job_id):
    r = subprocess.run(["hf", "jobs", "logs", f"redmingwei/{job_id}"],
                       capture_output=True, text=True, timeout=1800)
    return r.stdout or ""


def sign_test_p(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    p = 2.0 * sum(math.comb(n, i) for i in range(min(k, m) + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


def parse(text):
    per_subject, cur = [], None
    for line in text.splitlines():
        m = SUBJ.search(line)
        if m:
            cur = {"subject": f"sub-{m.group(1)}", "rsa": {}, "mb": {},
                   "amp": None}
            per_subject.append(cur)
            continue
        if cur is None:
            continue
        m = RSA.match(line)
        if m:
            cur["rsa"].setdefault(m.group("readout"), {})[m.group("group")] = {
                "r": float(m.group("r")), "p": float(m.group("p")),
                "n_ch": int(m.group("n")), "dim": int(m.group("dim"))}
            continue
        m = MB.match(line)
        if m:
            cur["mb"][m.group("group")] = {"acc1": float(m.group("acc")),
                                           "null_acc1": float(m.group("null"))}
            continue
        m = AMPALL.match(line)
        if m:
            vals = [float(x) for x in m.group("vals").split(",") if x]
            if vals:
                cur["amp"] = vals
            continue
    return per_subject


def summarize(per_subject):
    out = {}
    readouts = sorted({ro for r in per_subject for ro in r.get("rsa", {})})
    for ro in readouts:
        groups = sorted({g for r in per_subject for g in r.get("rsa", {}).get(ro, {})})
        e = {}
        for g in groups:
            rs = [r["rsa"][ro][g]["r"] for r in per_subject
                  if g in r.get("rsa", {}).get(ro, {})]
            if not rs:
                continue
            srt = sorted(rs)
            e[g] = {"n": len(rs), "r_median": srt[len(srt) // 2],
                    "n_sig": sum(1 for r in per_subject
                                 if g in r.get("rsa", {}).get(ro, {})
                                 and r["rsa"][ro][g]["p"] < 0.05),
                    "r_all": rs,
                    "dim": next(iter([r["rsa"][ro][g]["dim"]
                                      for r in per_subject
                                      if g in r.get("rsa", {}).get(ro, {})]), None)}
        # 关键：后 − 前 的配对差
        diffs = [r["rsa"][ro]["POST"]["r"] - r["rsa"][ro]["ANT"]["r"]
                 for r in per_subject
                 if "POST" in r.get("rsa", {}).get(ro, {})
                 and "ANT" in r.get("rsa", {}).get(ro, {})]
        if diffs:
            k, m, p = sign_test_p(diffs)
            srt = sorted(diffs)
            e["_POST_minus_ANT"] = {"n": len(diffs), "n_pos": k, "n_neg": m,
                                    "p_sign": p, "median": srt[len(srt) // 2],
                                    "diffs": diffs}
        out[ro] = e
    mb = {}
    for g in sorted({g for r in per_subject for g in r.get("mb", {})}):
        acc = [r["mb"][g]["acc1"] for r in per_subject if g in r.get("mb", {})]
        nul = [r["mb"][g]["null_acc1"] for r in per_subject if g in r.get("mb", {})]
        if acc:
            srt = sorted(acc)
            mb[g] = {"n": len(acc), "acc1_median": srt[len(srt) // 2],
                     "null_median": sorted(nul)[len(nul) // 2], "acc1_all": acc}
    if mb:
        out["metric_b_covpat"] = mb
    amps = [r["amp"] for r in per_subject if r.get("amp")]
    if amps:
        out["amp_te"] = {"n": len(amps), "n_ch": len(amps[0])}
    return out


def markdown(summary):
    L = ["# exp91a 读出强度检验：换更强的读出后，前部还是读不到吗？", ""]
    L.append("协议同 exp89：630-d 逐通道幅度为基线，另加 **频带功率**与**空间模式协方差**两种读出；")
    L.append("同一份 test split、同一套 RSA 协议、零训练。")
    L.append("")
    for ro, e in summary.items():
        if ro in ("metric_b_covpat", "amp_te"):
            continue
        L.append(f"## 读出 `{ro}`")
        L.append("")
        L.append("| 组 | 维度 | 中位 r | 显著被试 | 全部被试 r |")
        L.append("|---|---|---|---|---|")
        for g in sorted(k for k in e if not k.startswith("_")):
            v = e[g]
            vals = " ".join(f"{x:+.4f}" for x in v["r_all"])
            L.append(f"| {g} | {v.get('dim')} | {v['r_median']:+.4f} | "
                     f"{v['n_sig']}/{v['n']} | {vals} |")
        d = e.get("_POST_minus_ANT")
        if d:
            L.append("")
            L.append(f"**后部 − 前部（配对）= 中位 {d['median']:+.4f}，"
                     f"{d['n_pos']}/{d['n']} 为正（符号检验 p = {d['p_sign']:.4f}）**")
        L.append("")
    if "metric_b_covpat" in summary:
        L.append("## 指标 B（ridge 200 选 1，covpat 读出）")
        L.append("")
        L.append("| 组 | acc1 中位 | 零中位 | n |")
        L.append("|---|---|---|---|")
        for g, v in summary["metric_b_covpat"].items():
            L.append(f"| {g} | {v['acc1_median']:.4f} | {v['null_median']:.4f} | {v['n']} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = job_ids()
    print(f"找到 {len(ids)} 个 Job", flush=True)
    per_subject, used = [], []
    for name in JOBS:
        jid = ids.get(name)
        if not jid:
            print(f"  缺 Job {name}")
            continue
        ps = parse(logs_of(jid))
        if not ps:
            print(f"  {name}: 日志里没解析出被试（可能还在跑）")
            continue
        per_subject += ps
        used.append({"job": name, "job_id": jid, "n": len(ps)})
        print(f"  ✓ {name}: {len(ps)} 被试")
    if not per_subject:
        print("还没有可用结果")
        return 1
    summary = summarize(per_subject)
    (OUT / "exp91a_readout.json").write_text(
        json.dumps({"source": "job-logs", "jobs": used, "per_subject": per_subject,
                    "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "exp91a_readout.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    print(f"写入 {OUT}/exp91a_readout.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
