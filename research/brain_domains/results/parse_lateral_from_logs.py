"""parse_lateral_from_logs.py — **从 Job 日志**重建 exp90（左右轴）的结果。

## 为什么不用下载

同 `parse_posaxis_from_logs.py`：本机 `hf buckets cp` 的读通道时断时续，
而 `hf jobs logs` 一直可用 ⇒ 结果一律从日志重建。

## 两个 Job 的分工

| Job | 脚本版本 | 用途 |
|---|---|---|
| `t90-lateral` | **v1** | 第一版：L3（跨半球 transfer）**维度未按位点对齐** ⇒ 其 transfer 数字**无意义**（假对照）；但 L1/L2 可用 |
| `t90-lateral-v2` | **v2** | 修正版：左右两组按**同一位点镜像**对齐 + 加**错位对照** + L2 加**同侧分半上限** ⇒ 全部测量有效 |

⇒ 以 **v2 为准**；v1 只作为 L1/L2 的**复现对照**（两次独立跑，结果应逐位相同）。

## 解析什么

- 逐被试：`RSA <组> n_ch=.. r=.. p=..`、`RDMRED L-R=.. L-M=.. R-M=.. | 同侧分半上限=..`、
  `MB <tag> acc1=.. (零中位 ..)`、`TRANSFER k=v ...`
- 汇总：脚本末尾的 `KEY <name> <value>` 行
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "lateral-results"

JOBS = {"t90-lateral-v2": "v2", "t90-lateral": "v1"}

SUBJ = re.compile(r"===\s*sub-(\d+)\s*===")
RSA = re.compile(r"^\s*RSA (?P<group>[A-Z0-9]+)\s+n_ch=(?P<n>\d+)\s+"
                 r"r=(?P<r>[+-][\d.]+)\s+p=(?P<p>[\d.]+)")
RDMRED = re.compile(r"RDMRED L-R=(?P<lr>[+-][\d.]+)\s+L-M=(?P<lm>[+-][\d.]+)\s+"
                    r"R-M=(?P<rm>[+-][\d.]+)(?:\s*\|\s*同侧分半上限=(?P<ce>[+-][\d.]+))?")
MB = re.compile(r"^\s*MB (?P<tag>[A-Z0-9]+)\s+acc1=(?P<acc>[\d.]+)\s+"
                r"\(零中位 (?P<null>[\d.]+)\)")
TRANSFER = re.compile(r"^\s*TRANSFER (?P<body>.+)$")
KEY = re.compile(r"^\s*KEY (?P<name>[A-Z0-9_]+) (?P<val>[-+][\d.eE]+)$")


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
        nm = j.get("name") or j.get("labels", {}).get("name") or ""
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
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


def parse(text):
    per_subject, cur = [], None
    keys = {}
    for line in text.splitlines():
        m = SUBJ.search(line)
        if m:
            cur = {"subject": f"sub-{m.group(1)}", "group_rsa": {},
                   "rdm": {}, "metric_b": {}, "transfer": {}}
            per_subject.append(cur)
            continue
        mk = KEY.match(line)
        if mk:
            keys[mk.group("name")] = float(mk.group("val"))
            continue
        if cur is None:
            continue
        mr = RSA.match(line)
        if mr:
            cur["group_rsa"][mr.group("group")] = {
                "r": float(mr.group("r")), "p": float(mr.group("p")),
                "n_ch": int(mr.group("n"))}
            continue
        md = RDMRED.search(line)
        if md:
            cur["rdm"] = {"L_R": float(md.group("lr")), "L_M": float(md.group("lm")),
                          "R_M": float(md.group("rm"))}
            if md.group("ce"):
                cur["rdm"]["CEILING"] = float(md.group("ce"))
            continue
        mb = MB.match(line)
        if mb:
            cur["metric_b"][mb.group("tag")] = {
                "acc1": float(mb.group("acc")), "null_acc1": float(mb.group("null"))}
            continue
        mt = TRANSFER.match(line)
        if mt:
            for tok in mt.group("body").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        cur["transfer"][k] = float(v)
                    except ValueError:
                        pass
            continue
    return per_subject, keys


def summarize(per_subject, keys):
    out = {"KEY": keys}

    def med(vals):
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return None
        s = sorted(vals)
        return float(s[len(s) // 2])

    d = [r["group_rsa"]["LEFT"]["r"] - r["group_rsa"]["RIGHT"]["r"]
         for r in per_subject
         if "LEFT" in r.get("group_rsa", {}) and "RIGHT" in r.get("group_rsa", {})]
    if d:
        k, m, p = sign_test_p(d)
        out["LEFT_minus_RIGHT"] = {"n": len(d), "n_pos": k, "n_neg": m, "p_sign": p,
                                   "median": med(d), "mean": sum(d) / len(d),
                                   "diffs": d}
    for g in ("LEFT", "RIGHT", "MID", "ALL"):
        rs = [r["group_rsa"][g]["r"] for r in per_subject if g in r.get("group_rsa", {})]
        ps = [r["group_rsa"][g]["p"] for r in per_subject if g in r.get("group_rsa", {})]
        if rs:
            out[f"{g}_RSA"] = {"median": med(rs), "n": len(rs),
                               "n_sig": sum(1 for p in ps if p < 0.05),
                               "all": rs}
    for k in ("L_R", "L_M", "R_M", "CEILING"):
        v = [r["rdm"][k] for r in per_subject if k in r.get("rdm", {})]
        if v:
            out[f"RDMRED_{k}"] = {"median": med(v), "n": len(v)}
    for tag in ("SL2L", "SR2R", "XL2R", "XM2R", "XR2L", "ALL2ALL"):
        v = [r["metric_b"][tag]["acc1"] for r in per_subject
             if tag in r.get("metric_b", {})]
        nv = [r["metric_b"][tag]["null_acc1"] for r in per_subject
              if tag in r.get("metric_b", {})]
        if v:
            out[f"MB_{tag}"] = {"median": med(v), "null_median": med(nv), "n": len(v)}
    for k in ("XL2R_over_SL2L", "XR2L_over_SR2R", "XL2R_over_XM2R"):
        v = [r["transfer"][k] for r in per_subject
             if k in r.get("transfer", {}) and r["transfer"][k] is not None]
        if v:
            out[f"TRANSFER_{k}"] = {"median": med(v), "n": len(v)}
    return out


def markdown(summary, per_subject):
    L = ["# exp90 左右轴（外侧轴）零训练测量 — 结果表", ""]
    L.append("> 协议同 exp89：630-d 原始特征 + `metric_a_rsa`；ridge 200 选 1 的拟合集 = "
             "train split 前 1000 张，test 200 图。")
    L.append("> 蒙太奇：28 左 / 28 右 / 7 中线（无乳突电极）。")
    L.append("")
    L.append("## L1 偏侧（RSA）")
    L.append("")
    L.append("| 组 | 通道数 | 中位 r | 显著被试 | 全部被试 r |")
    L.append("|---|---|---|---|---|")
    nch = {g: per_subject[0]["group_rsa"][g]["n_ch"]
           for g in per_subject[0].get("group_rsa", {})} if per_subject else {}
    for g in ("LEFT", "RIGHT", "MID", "ALL"):
        e = summary.get(f"{g}_RSA")
        if not e:
            continue
        vals = " ".join(f"{x:+.4f}" for x in e["all"])
        L.append(f"| {g} | {nch.get(g, '')} | {e['median']:+.4f} | "
                 f"{e['n_sig']}/{e['n']} | {vals} |")
    L.append("")
    lm = summary.get("LEFT_minus_RIGHT")
    if lm:
        L.append(f"**左 − 右（配对）= 中位 {lm['median']:+.4f}，"
                 f"{lm['n_pos']}/{lm['n']} 为正（符号检验 p={lm['p_sign']:.4f}）**")
        L.append("")
    L.append("## L2 左右结构冗余度（+ 同侧分半上限）")
    L.append("")
    L.append("| 量 | 中位 | n |")
    L.append("|---|---|---|")
    for k in ("L_R", "L_M", "R_M", "CEILING"):
        e = summary.get(f"RDMRED_{k}")
        if e:
            L.append(f"| {k} | {e['median']:+.4f} | {e['n']} |")
    L.append("")
    L.append("## L3 跨半球 transfer（位点对齐 + 错位对照）")
    L.append("")
    L.append("| 指标 | 中位 | n |")
    L.append("|---|---|---|")
    for tag in ("SL2L", "SR2R", "XL2R", "XM2R", "XR2L", "ALL2ALL"):
        e = summary.get(f"MB_{tag}")
        if e:
            L.append(f"| MB {tag} | acc1 {e['median']:.4f}（零中位 {e['null_median']:.4f}） | {e['n']} |")
    for k in ("XL2R_over_SL2L", "XR2L_over_SR2R", "XL2R_over_XM2R"):
        e = summary.get(f"TRANSFER_{k}")
        if e:
            L.append(f"| TRANSFER {k} | {e['median']:.3f} | {e['n']} |")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = job_ids()
    print(f"找到 {len(ids)} 个 Job", flush=True)
    result = {}
    for name, ver in JOBS.items():
        jid = ids.get(name)
        if not jid:
            print(f"  缺 Job {name}")
            continue
        text = logs_of(jid)
        ps, keys = parse(text)
        if not ps:
            print(f"  {name}: 日志里没解析出被试（可能还在跑）")
            continue
        summ = summarize(ps, keys)
        result[ver] = {"job": name, "job_id": jid, "per_subject": ps, "summary": summ}
        print(f"  ✓ {name} ({ver}): {len(ps)} 被试")
    if "v2" not in result:
        print("还没有可用的 v2 结果")
        return 1
    v2 = result["v2"]
    (OUT / "exp90_lateral.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "exp90_lateral.md").write_text(markdown(v2["summary"], v2["per_subject"]),
                                           encoding="utf-8")
    # v1 的 L1/L2 复现对照
    if "v1" in result:
        a = result["v1"]["summary"]
        b = v2["summary"]
        for g in ("LEFT", "RIGHT", "MID", "ALL"):
            if g in a and g in b:
                d = abs(a[g]["median"] - b[g]["median"])
                print(f"  复现对照 {g}: v1={a[g]['median']:+.4f} v2={b[g]['median']:+.4f} "
                      f"Δ={d:.2e}")
    print(markdown(v2["summary"], v2["per_subject"]))
    print(f"写入 {OUT}/exp90_lateral.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
