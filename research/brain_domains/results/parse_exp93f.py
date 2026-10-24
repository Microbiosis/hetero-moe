"""parse_exp93f.py — exp93f 的汇总：**每种读出下** S1a（自监督）与 S1b（语义监督）的剂量斜率。

## 回答的问题

§二十九·五 由 exp93c 得到「**语义监督把位置特异性洗掉**」这条断言，
但 exp93c 的比较里 **读出类型与训练目标同时变了**（S1a 用 `recon`、S1b 用 `onehot_mean`）
⇒ exp93f 把**每个模型 × 每种读出**的剂量曲线都算出来，
在**同一读出**下比较斜率 ⇒ 区分「目标造成的」还是「读出造成的」。

## 判读

| 结果 | 读法 |
|---|---|
| 同一读出下 S1b 斜率**仍远小于** S1a | 混淆排除 ⇒ **目标**是原因 ⇒ 猕猴功能目标**有实质风险**，需在真数据上预检 |
| 同一读出下两者斜率**相近** | 差异来自**读出** ⇒ §二十九·五 那条断言**必须限定/改写** ⇒ 猕猴目标**不继承**这个风险 |

## 统计口径

**中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
**斜率** = r(k=34) − r(k=5)（越大 ⇒ 越依赖后部）；
**归一化斜率** = (r(34) − r(5)) / (r(ALL) − r(5))。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp93-results"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
LOGF = HERE / "exp93f_readout_control_job_log.txt"

RE_SUBJ = re.compile(r"^===\s*(sub-\d+)\s*===")
RE_K = re.compile(r"^\s*\[(?P<tag>\w+)\]\s*k=(?P<k>\d+)\s+(?P<rest>.*)$")
RE_REF = re.compile(r"^\s*\[参考 ALL\]\s+(?P<tag>\w+):\s*(?P<rest>.*)$")
RE_PAIR = re.compile(r"(?P<ro>[a-z0-9_]+)=(?P<val>[+-][\d.]+)")


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


def parse_log(path):
    if not path.exists():
        return None
    cur = None
    per = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_SUBJ.match(line.strip())
        if m:
            cur = m.group(1)
            per.setdefault(cur, {"ref": {}, "dose": {}})
            continue
        if cur is None:
            continue
        m = RE_REF.match(line)
        if m:
            tag = m.group("tag")
            for pm in RE_PAIR.finditer(m.group("rest")):
                per[cur]["ref"].setdefault(tag, {})[pm.group("ro")] = float(pm.group("val"))
            continue
        m = RE_K.match(line)
        if m:
            tag = m.group("tag")
            k = int(m.group("k"))
            for pm in RE_PAIR.finditer(m.group("rest")):
                per[cur]["dose"].setdefault(tag, {}).setdefault(pm.group("ro"), {})[str(k)] = \
                    float(pm.group("val"))
    return per or None


def fetch_json(retries=5):
    dst = OUT / "exp93f_readout_control.json"
    if dst.exists() and dst.stat().st_size > 0:
        return json.loads(dst.read_text(encoding="utf-8"))
    src = f"{BUCKET}/exp93f/exp93f_readout_control.json"
    for i in range(retries):
        try:
            r = subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                               capture_output=True, text=True, timeout=420)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return json.loads(dst.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            pass
        print(f"  重试 {i+1}/{retries}")
        time.sleep(6)
    return None


def slopes_from_dose(dose, ref):
    """每条曲线 → (斜率, 归一化斜率, 单调性)。"""
    out = {}
    ks = sorted({int(k) for ro in dose for k in dose[ro]})
    if not ks or "5" not in dose[next(iter(dose))] or str(ks[-1]) not in dose[next(iter(dose))]:
        return out
    k0, k1 = "5", str(ks[-1])
    for ro, curve in dose.items():
        if k0 not in curve or k1 not in curve:
            continue
        r0, r1 = curve[k0], curve[k1]
        slope = r1 - r0
        ref_all = (ref or {}).get(ro)
        norm = None
        if ref_all is not None and abs(ref_all - r0) > 1e-9:
            norm = slope / (ref_all - r0)
        seq = [curve.get(str(k)) for k in ks if str(k) in curve]
        mono = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
        out[ro] = {"k_min": int(k0), "k_max": int(ks[-1]),
                   "r_min": r0, "r_max": r1, "slope": slope, "norm_slope": norm,
                   "monotonic_steps": mono, "n_steps": len(seq) - 1,
                   "curve": {str(k): curve[str(k)] for k in ks if str(k) in curve},
                   "ref_all": ref_all}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = fetch_json()
    per = None
    if data:
        for r in data.get("per_subject", []):
            if "error" in r:
                continue
            per = per or {}
            per[r["subject"]] = {"ref": r.get("ref_all", {}), "dose": r.get("dose", {})}
    if not per:
        per = parse_log(LOGF)
        print("（桶内 JSON 未就位 ⇒ 用 Job 日志解析）")
    if not per:
        print("✗ 没有可用数据")
        return

    # ---- 每条曲线的中位（跨被试）----
    tags = sorted({t for v in per.values() for t in v["dose"]})
    rows = []
    for tag in tags:
        ros = sorted({ro for v in per.values() for ro in v["dose"].get(tag, {})})
        for ro in ros:
            curve = {}
            for k in sorted({int(kk) for v in per.values()
                             for kk in v["dose"].get(tag, {}).get(ro, {})}):
                vals = [v["dose"][tag][ro][str(k)] for v in per.values()
                        if str(k) in v["dose"].get(tag, {}).get(ro, {})]
                curve[str(k)] = {"median_up": med(vals), "mean": mean(vals), "n": len(vals)}
            ref_all = med([v["ref"][tag][ro] for v in per.values()
                           if tag in v["ref"] and ro in v["ref"][tag]])
            rows.append({"tag": tag, "readout": ro, "curve": curve, "ref_all": ref_all})

    print(f"被试 {len(per)} 个\n")
    hdr = (f"{'模型':<6} {'读出':<16} " +
           " ".join(f"k={k:<7}" for k in sorted({int(k) for r in rows for k in r['curve']})) +
           f" {'斜率':>9} {'归一斜率':>9} {'单调':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ks = sorted(r["curve"], key=int)
        vals = " ".join(f"{r['curve'][k]['median_up']:+.4f}  " for k in ks)
        ks_ = [float(k) for k in ks]
        r0 = r["curve"][ks[0]]["median_up"]
        r1 = r["curve"][ks[-1]]["median_up"]
        slope = r1 - r0
        norm = None
        if r["ref_all"] is not None and abs(r["ref_all"] - r0) > 1e-9:
            norm = slope / (r["ref_all"] - r0)
        seq = [r["curve"][k]["median_up"] for k in ks]
        mono = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
        print(f"{r['tag']:<6} {r['readout']:<16} {vals} {slope:>+9.4f} "
              f"{(f'{norm:>+9.3f}' if norm is not None else '      —  ')} "
              f"{mono}/{len(seq)-1:>4}")

    (OUT / "exp93f_summary.json").write_text(
        json.dumps({"n_subjects": len(per), "rows": rows}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    L = ["# exp93f 读出混淆的排除（每种读出下的剂量斜率）", "",
         f"被试 {len(per)} 个；**中位 = 上中位**；斜率 = r(k=34) − r(k=5)", "",
         "| 模型 | 读出 | " + " | ".join(f"k={k}" for k in
                                        sorted({int(k) for r in rows for k in r['curve']})) +
         " | 斜率 | 归一斜率 | 单调 |",
         "|---|---|" + "---:|" * (len({int(k) for r in rows for k in r['curve']}) + 3)]
    for r in rows:
        ks = sorted(r["curve"], key=int)
        r0 = r["curve"][ks[0]]["median_up"]
        r1 = r["curve"][ks[-1]]["median_up"]
        slope = r1 - r0
        norm = None
        if r["ref_all"] is not None and abs(r["ref_all"] - r0) > 1e-9:
            norm = slope / (r["ref_all"] - r0)
        seq = [r["curve"][k]["median_up"] for k in ks]
        mono = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
        L.append(f"| {r['tag']} | {r['readout']} | " +
                 " | ".join(f"{r['curve'][k]['median_up']:+.4f}" for k in ks) +
                 f" | {slope:+.4f} | " +
                 (f"{norm:+.3f} |" if norm is not None else "— |") +
                 f" {mono}/{len(seq)-1} |")
    (OUT / "exp93f_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n写入 {OUT/'exp93f_summary.md'}")


if __name__ == "__main__":
    main()
