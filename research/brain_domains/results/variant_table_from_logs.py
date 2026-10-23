"""variant_table_from_logs.py — 从 Job 日志重建**阶段 D 变体表**。

沿用基线表的路子：不下载文件，直接从 `hf jobs logs` 解析
（下载通道会被间歇拦截，读通道一直可用）。

比基线表多算两件事：

1. **供体贡献** = 变体行 − 同名 `_shuf` 对照行（**逐被试配对**）
   ⇒ 这是唯一能读出「供体有没有带来宿主之外的信息」的量。
   变体行自己的高低没有意义（维度变大本身就会抬高分数）。
2. **符号检验**（精确二项）：配对差值里正负各半的零假设下，
   单侧 p = P(X ≥ n+) ，X~Bin(n, 0.5)。
"""

import json
import re
import subprocess
import sys
from pathlib import Path

# [名字] A r=+0.0105 p=0.037 | B acc1=0.0054 (零中位 0.0050) d=200
LINE = re.compile(
    r"\[(?P<name>[^\]]+)\]\s*A r=(?P<r>[+-][\d.]+)\s+p=(?P<p>[\d.]+)\s*\|"
    r"\s*B acc1=(?P<acc>[\d.]+)\s*\(零中位 (?P<null>[\d.]+)\)")

JOB_RE = re.compile(r"^t77-s(\d\d)-(stable|dev)$")


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
                       capture_output=True, text=True, timeout=600)
    return r.stdout or ""


def sign_test_p(n_pos, n):
    """单侧精确二项检验：H0: 正负各半。返回 P(X >= n_pos)。"""
    if n == 0:
        return None
    from math import comb
    tail = sum(comb(n, k) for k in range(n_pos, n + 1))
    return tail / (2 ** n)


def med(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def main():
    jobs = all_jobs()
    by_key = {}
    for j in jobs:
        nm = (j.get("name") or "")
        m = JOB_RE.match(nm)
        st = j.get("status")
        stage = st.get("stage") if isinstance(st, dict) else st
        if m:
            by_key[f"s{m.group(1)}:{m.group(2)}"] = (j["id"], stage)
    done = {k: v[0] for k, v in by_key.items() if v[1] == "COMPLETED"}
    print(f"找到 {len(by_key)} 个 t77 Job，其中完成 {len(done)} 个", flush=True)
    miss = [f"s{s:02d}:{p}" for s in range(1, 11) for p in ("stable", "dev")
            if f"s{s:02d}:{p}" not in done]
    if miss:
        print(f"⚠ 未完成/缺失：{miss}", flush=True)

    per_subject = {}
    for key, jid in done.items():
        rows = {}
        for line in logs_of(jid).splitlines():
            mm = LINE.search(line)
            if not mm:
                continue
            name = mm.group("name").strip()
            if name.startswith("cbramod（宿主"):
                name = "cbramod"
            rows[name] = {"r": float(mm.group("r")), "p": float(mm.group("p")),
                          "acc": float(mm.group("acc")),
                          "null": float(mm.group("null"))}
        if rows:
            per_subject[key] = rows
    print(f"解析到 {len(per_subject)} 个被试-趟", flush=True)

    # 行名顺序：宿主 → 变体 → 对照
    vids = []
    for key in per_subject:
        for n in per_subject[key]:
            if n.endswith("_shuf"):
                continue
            if n != "cbramod" and n not in vids:
                vids.append(n)
    order = ["cbramod"] + vids + [v + "_shuf" for v in vids]

    out = []
    for name in order:
        rs = [per_subject[k][name]["r"] for k in per_subject if name in per_subject[k]]
        ps = [per_subject[k][name]["p"] for k in per_subject if name in per_subject[k]]
        ac = [per_subject[k][name]["acc"] for k in per_subject if name in per_subject[k]]
        nu = [per_subject[k][name]["null"] for k in per_subject if name in per_subject[k]]
        if not rs:
            continue
        out.append({"row": name, "n": len(rs), "r": med(rs), "p": med(ps),
                    "n_sig": sum(1 for p in ps if p < 0.05), "acc": med(ac),
                    "null": med(nu),
                    "ratio": (med(ac) / med(nu)) if (med(ac) and med(nu)) else None})

    # --- 供体贡献（配对差）---
    deltas = []
    for v in vids:
        c = v + "_shuf"
        dr, da = [], []
        for k in per_subject:
            if v in per_subject[k] and c in per_subject[k]:
                dr.append(per_subject[k][v]["r"] - per_subject[k][c]["r"])
                da.append(per_subject[k][v]["acc"] - per_subject[k][c]["acc"])
        if not dr:
            continue
        npos = sum(1 for x in dr if x > 0)
        npos_a = sum(1 for x in da if x > 0)
        deltas.append({
            "variant": v, "control": c, "n": len(dr),
            "d_r_median": med(dr), "d_r_n_pos": npos, "d_r_p": sign_test_p(npos, len(dr)),
            "d_acc_median": med(da), "d_acc_n_pos": npos_a,
            "d_acc_p": sign_test_p(npos_a, len(da)),
        })

    print("\n" + "=" * 104)
    print(f"阶段 D 变体表（{len(per_subject)} 个被试-趟，由 Job 日志重建）")
    print("=" * 104)
    print("| 行 | A: r_img (中位) | A: p (中位) | A 显著 | B: acc1 (中位) | "
          "B: 零中位 | B: ×随机 |")
    print("|---|---|---|---|---|---|---|")
    for o in out:
        rr = f"{o['r']:+.4f}" if o["r"] is not None else "—"
        pp = f"{o['p']:.3f}" if o["p"] is not None else "—"
        aa = f"{o['acc']:.4f}" if o["acc"] is not None else "—"
        nn = f"{o['null']:.4f}" if o["null"] is not None else "—"
        ratio = f"{o['ratio']:.1f}×" if o["ratio"] else "—"
        print(f"| {o['row']} | {rr} | {pp} | {o['n_sig']}/{o['n']} | {aa} | {nn} | {ratio} |")

    print("\n**供体贡献（变体 − 同名 shuffle 对照，逐被试配对）**")
    print("| 变体 | n | Δr (中位) | Δr 为正 | 符号检验 p | Δacc (中位) | Δacc 为正 | p |")
    print("|---|---|---|---|---|---|---|---|")
    for d in deltas:
        p1 = f"{d['d_r_p']:.4f}" if d["d_r_p"] is not None else "—"
        p2 = f"{d['d_acc_p']:.4f}" if d["d_acc_p"] is not None else "—"
        dr = f"{d['d_r_median']:+.4f}" if d["d_r_median"] is not None else "—"
        da = f"{d['d_acc_median']:+.4f}" if d["d_acc_median"] is not None else "—"
        print(f"| {d['variant']} | {d['n']} | {dr} | {d['d_r_n_pos']}/{d['n']} | {p1} | "
              f"{da} | {d['d_acc_n_pos']}/{d['n']} | {p2} |")

    outdir = Path("variant-results")
    outdir.mkdir(exist_ok=True)
    (outdir / "variant_table_from_logs.md").write_text(
        "# 阶段 D 变体表（由 Job 日志重建）\n\n"
        "## 行\n\n" + json.dumps(out, ensure_ascii=False, indent=2)
        + "\n\n## 供体贡献（配对差）\n\n" + json.dumps(deltas, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")
    (outdir / "variant_parsed_from_logs.json").write_text(
        json.dumps({"per_subject": per_subject, "rows": out, "deltas": deltas},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {outdir}/")


if __name__ == "__main__":
    main()
