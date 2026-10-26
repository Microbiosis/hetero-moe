"""parse_posaxis_from_logs.py — **从 Job 日志**重建 exp89（位置轴三测量）的结果。

## 为什么不用下载

同 `parse_assembled_from_logs.py`：本机 `hf buckets cp` 的读通道时断时续，
而 `hf jobs logs` 的读通道一直可用 ⇒ 结果一律从日志重建。

## 解析哪些行（脚本 exp89_posaxis.py 的输出格式）

```
=== sub-01 ===
  ORDER 逐通道 ERP 相关：中位 +0.9979 | >0.8 的通道 61/63 | 幅度图 r=+0.9710 sp=+0.9512 | 幅度~解剖位 sp=+0.8120
  AMP  OCC=1.234 PARI=1.100 ...
  RSA ALL     n_ch=63   r=+0.0504 p=0.0050
  CHMAP O1=+0.0810(occi) Oz=+0.0780(occi) ...
  NULL OCC    同尺寸随机组中位 +0.0123 最大 +0.0456 | 真值 +0.0504 超过 90% 随机组
  SHUF #0 ALL r=+0.0504（真值 +0.0504） POST +0.0481 ANT +0.0120 差 +0.0361
  MB ALL     acc1=0.0098 (零中位 0.0044)
```

## 输出

1. `results/posaxis-results/exp89_posaxis.json`（结构与云端 json 同构）
2. `results/posaxis-results/exp89_posaxis.md`（人读的表）
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "posaxis-results"

JOB_NAMES = ("t89-posaxis-s1-5", "t89-posaxis-s6-10")
CURVE_JOB = "t89b-curve"

SUBJ = re.compile(r"===\s*sub-(\d+)\s*===")
CHLIST = re.compile(r"^\s*CHLIST (?P<names>\S+)$")
CURVE2 = re.compile(
    r"^\s*CURVE2 sub-(\d+) (?P<vals>[+-][\d.,eE+-]+)$")
CURVE2_MED = re.compile(r"^\s*CURVE2_MEDIAN (?P<vals>[+-][\d.,eE+-]+)$")
ORDER = re.compile(
    r"ORDER 逐通道 ERP 相关：中位 (?P<wave_median>[+-][\d.]+)\s*\|\s*"
    r">0\.8 的通道 (?P<wave_n>\d+)/(?P<wave_tot>\d+)\s*\|\s*"
    r"幅度图 r=(?P<amp_r>[+-][\d.]+) sp=(?P<amp_sp>[+-][\d.]+)\s*\|\s*"
    r"幅度~解剖位 sp=(?P<amp_anat>[+-][\d.]+)")
AMP = re.compile(r"^\s*AMP\s+(?P<body>.+)$")
RSA = re.compile(
    r"^\s*RSA (?P<group>[A-Z]+)\s+n_ch=(?P<n_ch>\d+)\s+"
    r"r=(?P<r>[+-][\d.]+)\s+p=(?P<p>[\d.]+)")
CHMAP = re.compile(r"^\s*CHMAP\s+(?P<body>.+)$")
NULL = re.compile(
    r"^\s*NULL (?P<group>[A-Z]+)\s+同尺寸随机组中位 (?P<med>[+-][\d.]+)\s+"
    r"最大 (?P<mx>[+-][\d.]+)\s*\|\s*真值 (?P<true>[+-][\d.]+)"
    # 旧措辞：「真值 X 超过 N% 随机组」；新措辞：「随机组中≥真值的占 N%」
    # ⇒ **两种措辞存的数都是「随机组里 ≥ 真值的比例」**（frac_null_ge_true）
    r"(?:\s+超过 (?P<frac_old>[\d.]+)% 随机组)?"
    r"(?:\s*\|\s*随机组中≥真值的占 (?P<frac_new>[\d.]+)%)?")
SHUF = re.compile(
    r"^\s*SHUF #(?P<k>\d+) ALL r=(?P<r_all>[+-][\d.]+)（真值 (?P<r_all_true>[+-][\d.]+)）"
    r"\s+POST (?P<r_post>[+-][\d.]+) ANT (?P<r_ant>[+-][\d.]+) 差 (?P<diff>[+-][\d.]+)")
MB = re.compile(
    r"^\s*MB (?P<group>[A-Z]+)\s+acc1=(?P<acc>[\d.]+)\s+\(零中位 (?P<null>[\d.]+)\)")


def all_jobs():
    r = subprocess.run(["hf", "jobs", "list", "-a", "--format", "json"],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout or ""
    i, j = out.find("["), out.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(out[i:j + 1])
    except Exception:
        return []


def job_ids():
    """{Job 名: 最新一个的 id}。"""
    best = {}
    for j in all_jobs():
        nm = j.get("name") or ""
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
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return k, m, float(min(1.0, p))


def parse(text):
    per_subject = []
    cur = None
    for line in text.splitlines():
        m = SUBJ.search(line)
        if m:
            cur = {"subject": f"sub-{m.group(1)}", "group_rsa": {},
                   "random_group_null": {}, "shuffled": [], "metric_b": {}}
            per_subject.append(cur)
            continue
        if cur is None:
            continue
        mo = ORDER.search(line)
        if mo:
            cur["position"] = {
                "wave_median": float(mo.group("wave_median")),
                "wave_n_gt_0.8": int(mo.group("wave_n")),
                "wave_total": int(mo.group("wave_tot")),
                "amp_r": float(mo.group("amp_r")),
                "amp_spearman": float(mo.group("amp_sp")),
                "amp_vs_anat_spearman": float(mo.group("amp_anat")),
            }
            continue
        ma = AMP.match(line)
        if ma:
            cur["amp_group_mean"] = {}
            for kv in ma.group("body").split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        cur["amp_group_mean"][k] = float(v)
                    except ValueError:
                        pass
            continue
        mr = RSA.match(line)
        if mr:
            cur["group_rsa"][mr.group("group")] = {
                "r": float(mr.group("r")), "p": float(mr.group("p")),
                "n_ch": int(mr.group("n_ch"))}
            continue
        mc = CHMAP.match(line)
        if mc:
            cur["chmap"] = {}
            for tok in mc.group("body").split():
                mm = re.match(r"^([A-Za-z0-9]+)=([+-][\d.]+)\((\w+)\)$", tok)
                if mm:
                    cur["chmap"][mm.group(1)] = {"r": float(mm.group(2)),
                                                 "group": mm.group(3)}
            continue
        mn = NULL.match(line)
        if mn:
            frac = mn.group("frac_old")
            if frac is None:
                frac = mn.group("frac_new")
            cur["random_group_null"][mn.group("group")] = {
                "r_null_median": float(mn.group("med")),
                "r_null_max": float(mn.group("mx")),
                "r_true": float(mn.group("true")),
                # ⚠️ 语义 = **随机组里 ≥ 真值** 的比例
                #    0% ⇒ 真值高于全部随机组；100% ⇒ 真值低于全部随机组
                "frac_null_ge_true": float(frac) / 100.0}
            continue
        ms = SHUF.match(line)
        if ms:
            cur["shuffled"].append({
                "r_all": float(ms.group("r_all")),
                "r_all_true": float(ms.group("r_all_true")),
                "r_post": float(ms.group("r_post")),
                "r_ant": float(ms.group("r_ant")),
                "diff": float(ms.group("diff"))})
            continue
        mb = MB.match(line)
        if mb:
            cur["metric_b"][mb.group("group")] = {
                "acc1": float(mb.group("acc")), "null_acc1": float(mb.group("null"))}
            continue
    return per_subject


def summarize(per_subject):
    out = {}
    names = sorted({g for r in per_subject for g in r.get("group_rsa", {})})
    for g in names:
        rs = [r["group_rsa"][g]["r"] for r in per_subject if g in r.get("group_rsa", {})]
        ps = [r["group_rsa"][g]["p"] for r in per_subject if g in r.get("group_rsa", {})]
        if not rs:
            continue
        srt = sorted(rs)
        out[g] = {"n": len(rs), "r_median": srt[len(srt) // 2],
                  "r_mean": sum(rs) / len(rs),
                  "n_sig": sum(1 for p in ps if p < 0.05),
                  "r_all": rs}
    diffs = [r["group_rsa"]["POST"]["r"] - r["group_rsa"]["ANT"]["r"]
             for r in per_subject
             if "POST" in r.get("group_rsa", {}) and "ANT" in r.get("group_rsa", {})]
    if diffs:
        k, m, p = sign_test_p(diffs)
        out["POST_minus_ANT"] = {"n": len(diffs), "n_pos": k, "n_neg": m,
                                 "p_sign": p,
                                 "median": sorted(diffs)[len(diffs) // 2],
                                 "mean": sum(diffs) / len(diffs), "diffs": diffs}
    null_diffs = [s["diff"] for r in per_subject for s in r.get("shuffled", [])]
    if null_diffs and diffs:
        nd = sorted(null_diffs)
        out["POST_minus_ANT_null"] = {
            "n_null": len(null_diffs),
            "null_median": nd[len(nd) // 2],
            "null_max": nd[-1],
            "null_p95": nd[int(0.95 * (len(nd) - 1))],
            "n_true_above_null_max": sum(1 for x in diffs if x > nd[-1]),
            "n_true": len(diffs)}
    sh_all = [s["r_all"] for r in per_subject for s in r.get("shuffled", [])]
    sh_true = [s["r_all_true"] for r in per_subject for s in r.get("shuffled", [])]
    if sh_all:
        d = [abs(a - b) for a, b in zip(sh_all, sh_true)]
        out["permutation_invariance_ALL"] = {
            "max_abs_delta": max(d), "n": len(d),
            "r_true_median": out.get("ALL", {}).get("r_median"),
            "r_shuffled_median": sorted(sh_all)[len(sh_all) // 2]}
    for key, path in (("position", "position"),):
        vals = [r[path] for r in per_subject if path in r]
        if vals:
            agg = {}
            for k in vals[0]:
                xs = [v[k] for v in vals]
                agg[k] = {"median": sorted(xs)[len(xs) // 2], "min": min(xs),
                          "max": max(xs)}
            out["position_summary"] = agg
    # (c1) 随机组零分布
    rnd = {}
    for g in sorted({g for r in per_subject for g in r.get("random_group_null", {})}):
        fr = [r["random_group_null"][g]["frac_null_ge_true"]
              for r in per_subject if g in r.get("random_group_null", {})]
        meds = [r["random_group_null"][g]["r_null_median"]
                for r in per_subject if g in r.get("random_group_null", {})]
        if fr:
            rnd[g] = {"mean_frac_null_ge_true": sum(fr) / len(fr),
                      "median_of_null_medians": sorted(meds)[len(meds) // 2]}
    if rnd:
        out["random_group_null_summary"] = rnd
    # 指标 B
    mbs = {}
    for g in sorted({g for r in per_subject for g in r.get("metric_b", {})}):
        acc = [r["metric_b"][g]["acc1"] for r in per_subject if g in r.get("metric_b", {})]
        nul = [r["metric_b"][g]["null_acc1"] for r in per_subject if g in r.get("metric_b", {})]
        if acc:
            mbs[g] = {"acc1_median": sorted(acc)[len(acc) // 2],
                      "null_median": sorted(nul)[len(nul) // 2],
                      "n": len(acc), "acc1_all": acc}
    if mbs:
        out["metric_b_summary"] = mbs
    return out


def markdown(per_subject, summary):
    L = ["# exp89 位置轴三测量（零训练）— 结果表", ""]
    L.append("## (a) 通道组消融（指标 A：RSA，630-d 原始特征，同基线表协议）")
    L.append("")
    L.append("| 组 | 通道数 | 中位 r | 显著被试 | 全部被试 r |")
    L.append("|---|---|---|---|---|")
    nch = {g: per_subject[0]["group_rsa"][g]["n_ch"]
           for g in per_subject[0].get("group_rsa", {})} if per_subject else {}
    for g, e in summary.items():
        if not isinstance(e, dict) or "r_median" not in e:
            continue
        vals = " ".join(f"{x:+.4f}" for x in e["r_all"])
        L.append(f"| {g} | {nch.get(g, '')} | {e['r_median']:+.4f} | "
                 f"{e['n_sig']}/{e['n']} | {vals} |")
    L.append("")
    if "POST_minus_ANT" in summary:
        e = summary["POST_minus_ANT"]
        L.append(f"**后部 − 前部（配对）**：中位 {e['median']:+.4f}，"
                 f"{e['n_pos']}/{e['n']} 为正（符号检验 p={e['p_sign']:.4f}）")
        L.append("")
    if "POST_minus_ANT_null" in summary:
        e = summary["POST_minus_ANT_null"]
        L.append(f"**打乱通道顺序后的零分布**：中位 {e['null_median']:+.4f}，"
                 f"最大 {e['null_max']:+.4f}（n={e['n_null']}）；"
                 f"真实差超过零分布最大值的被试：{e['n_true_above_null_max']}/{e['n_true']}")
        L.append("")
    if "random_group_null_summary" in summary:
        L.append("## (c1) 同尺寸随机通道组零分布")
        L.append("")
        L.append("（**「随机组中 ≥ 真值」的比例**：0% ⇒ 真值高于**全部**同尺寸随机组；"
                 "100% ⇒ 真值低于**全部**同尺寸随机组）")
        L.append("")
        L.append("| 组 | 随机组中≥真值的比例（均值） | 随机组 r 中位的中位 |")
        L.append("|---|---|---|")
        for g, e in summary["random_group_null_summary"].items():
            L.append(f"| {g} | {e['mean_frac_null_ge_true']:.0%} | "
                     f"{e['median_of_null_medians']:+.4f} |")
        L.append("")
    if "permutation_invariance_ALL" in summary:
        e = summary["permutation_invariance_ALL"]
        L.append(f"## 置换不变性（自检）")
        L.append("")
        L.append(f"打乱通道顺序后 ALL 组 RSA 与真值的最大偏差 = "
                 f"{e['max_abs_delta']:.2e}（n={e['n']}）")
        L.append("")
    if "position_summary" in summary:
        e = summary["position_summary"]
        L.append("## (b) 位置反推：跨 split 通道顺序一致性 + 诱发幅度")
        L.append("")
        L.append("| 量 | 中位 | 最小 | 最大 |")
        L.append("|---|---|---|---|")
        for k, v in e.items():
            L.append(f"| {k} | {v['median']:+.4f} | {v['min']:+.4f} | {v['max']:+.4f} |")
        L.append("")
    # (b) 各通道组的视觉诱发幅度（峰峰值，48–500 ms）
    amp_by_group = {}
    for r in per_subject:
        for g, v in (r.get("amp_group_mean") or {}).items():
            amp_by_group.setdefault(g, []).append(v)
    if amp_by_group:
        L.append("### (b) 各通道组的视觉诱发幅度（峰峰值；跨被试中位）")
        L.append("")
        L.append("| 组 | 幅度中位 | 最小 | 最大 |")
        L.append("|---|---|---|---|")
        order = ["OCC", "PARI", "TEMP", "POST", "CENT", "FRONT", "ANT", "ALL"]
        for g in order:
            if g not in amp_by_group:
                continue
            xs = sorted(amp_by_group[g])
            L.append(f"| {g} | {xs[len(xs)//2]:.4f} | {xs[0]:.4f} | {xs[-1]:.4f} |")
        L.append("")
    if "metric_b_summary" in summary:
        L.append("## 指标 B（ridge 200 选 1，次要确认）")
        L.append("")
        L.append("| 组 | acc1 中位 | 零中位 | n |")
        L.append("|---|---|---|---|")
        for g, e in summary["metric_b_summary"].items():
            L.append(f"| {g} | {e['acc1_median']:.4f} | {e['null_median']:.4f} | {e['n']} |")
        L.append("")
    return "\n".join(L) + "\n"


def parse_curve(text):
    """exp89b 的逐通道曲线：CHLIST + CURVE2 per subject + CURVE2_MEDIAN。"""
    names, curves, med = None, {}, None
    for line in text.splitlines():
        m = CHLIST.match(line)
        if m:
            names = m.group("names").split(",")
            continue
        m = CURVE2.match(line)
        if m:
            vals = [float(x) for x in m.group("vals").split(",") if x]
            if vals:
                curves[f"sub-{m.group(1)}"] = vals
            continue
        m = CURVE2_MED.match(line)
        if m:
            med = [float(x) for x in m.group("vals").split(",") if x]
    return names, curves, med


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = job_ids()
    print(f"找到 {len(ids)} 个 Job", flush=True)
    per_subject = []
    used = []
    for name in JOB_NAMES:
        jid = ids.get(name)
        if not jid:
            print(f"  缺 Job {name}")
            continue
        text = logs_of(jid)
        ps = parse(text)
        if not ps:
            print(f"  {name}: 日志里没解析出被试（可能还在跑）")
            continue
        per_subject += ps
        used.append({"job": name, "job_id": jid, "n": len(ps)})
        print(f"  ✓ {name}: {len(ps)} 被试")
    if not per_subject:
        print("没有可解析的结果")
        return 1
    summary = summarize(per_subject)
    (OUT / "exp89_posaxis.json").write_text(
        json.dumps({"source": "job-logs", "jobs": used,
                    "per_subject": per_subject, "summary": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "exp89_posaxis.md").write_text(markdown(per_subject, summary),
                                           encoding="utf-8")

    # ---- exp89b 的逐通道曲线（头皮图用）----
    jid = ids.get(CURVE_JOB)
    if jid:
        text = logs_of(jid)
        names, curves, med = parse_curve(text)
        if curves:
            (OUT / "per_channel_curve.json").write_text(
                json.dumps({"source": "job-logs", "job": CURVE_JOB,
                            "job_id": jid, "ch_names": names,
                            "per_subject_curves": curves,
                            "median_curve": med}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"  ✓ {CURVE_JOB}: {len(curves)} 条曲线，"
                  f"通道名 {len(names or [])} 个 → per_channel_curve.json")
        else:
            print(f"  ✗ {CURVE_JOB}: 没解析出曲线")
    else:
        print(f"  缺 Job {CURVE_JOB}")

    print(markdown(per_subject, summary))
    print(f"写入 {OUT}/exp89_posaxis.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
