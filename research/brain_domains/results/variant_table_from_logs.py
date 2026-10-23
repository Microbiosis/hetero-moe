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

# ⚠️ 名字里允许 "t77b"：修 bug 那一轮发过一个 t77b-s01-stable（Job 侧 sed 补丁版），
#    它是 sub-01 stable 的**唯一有效**结果（重发时用了同名但带 bug 的版本）。
JOB_RE = re.compile(r"^t77b?-s(\d\d)-(stable|dev)$")


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


def from_json_file(path):
    """从桶里的逐被试 JSON 取行（日志被平台清掉时的兜底）。

    结构：{"per_subject": [{"rows": [{name, ok, metric_a, metric_b}, ...]}]}
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for rec in data.get("per_subject", []):
        for row in rec.get("rows", []):
            if not row.get("ok", True):
                continue
            a = row.get("metric_a") or {}
            b = row.get("metric_b") or {}
            if "r_img" not in a or "acc1_cos" not in b:
                continue
            out[row["name"].strip()] = {
                "r": float(a["r_img"]), "p": float(a["p_img"]),
                "acc": float(b["acc1_cos"]),
                "null": float(__import__("numpy").median(b.get("null_acc1_cos", []))),
            }
    return out


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="",
                    help="桶里逐被试 JSON 的本地目录（按 stable/dev 子目录），"
                         "日志被平台清掉时用它兜底")
    args = ap.parse_args()

    jobs = all_jobs()
    # ⚠️ **同名 Job 会撞车**：修 bug 那一轮用过同样的名字（t77-s03-stable 等），
    #    列表里会同时存在旧（带 bug）与新（已修）两个 ⇒ 必须按
    #    「已完成优先、其次最新」挑，否则会解析到旧的那批（供体行是失败记录）。
    cands = {}
    for j in jobs:
        nm = (j.get("name") or "")
        m = JOB_RE.match(nm)
        if not m:
            continue
        st = j.get("status")
        stage = st.get("stage") if isinstance(st, dict) else st
        created = j.get("created_at") or ""
        rank = 2 if stage == "COMPLETED" else (1 if stage == "RUNNING" else 0)
        key = f"s{m.group(1)}:{m.group(2)}"
        prev = cands.get(key)
        if prev is None or (rank, created) > (prev["rank"], prev["created"]):
            cands[key] = {"id": j["id"], "stage": stage, "rank": rank,
                          "created": created}
    done = {k: v["id"] for k, v in cands.items() if v["rank"] == 2}
    print(f"找到 {len(cands)} 个被试-趟，其中已完成 {len(done)} 个", flush=True)
    miss = [f"s{s:02d}:{p}" for s in range(1, 11) for p in ("stable", "dev")
            if f"s{s:02d}:{p}" not in done]
    if miss:
        print(f"⚠ 未完成/缺失：{miss}", flush=True)

    per_subject = {}
    host_by_subject = {}      # ⚠️ 宿主行在两趟里是**同一个测量**（同一被试、同一协议）
                              #    ⇒ 聚合时按被试去重（取先到的那趟），否则 n 虚增一倍
    for key, jid in done.items():
        rows = {}
        for line in logs_of(jid).splitlines():
            mm = LINE.search(line)
            if not mm:
                continue
            name = mm.group("name").strip()
            if name.startswith("cbramod（宿主"):
                name = "cbramod"
            rec = {"r": float(mm.group("r")), "p": float(mm.group("p")),
                   "acc": float(mm.group("acc")),
                   "null": float(mm.group("null"))}
            rows[name] = rec
            if name == "cbramod":
                sid = key.split(":")[0]
                host_by_subject.setdefault(sid, rec)
        if rows:
            per_subject[key] = rows
    print(f"解析到 {len(per_subject)} 个被试-趟（日志）", flush=True)

    # --- 兜底：日志被平台清掉的被试-趟 ⇒ 从桶里的逐被试 JSON 补 ---
    if args.json_dir:
        jd = Path(args.json_dir)
        for s in range(1, 11):
            for trip in ("stable", "dev"):
                key = f"s{s:02d}:{trip}"
                if key in per_subject:
                    continue
                fp = jd / trip / f"variant_sub-{s:02d}.json"
                if not fp.exists():
                    continue
                rows = from_json_file(fp)
                if rows:
                    per_subject[key] = rows
                    for name, rec in rows.items():
                        if name == "cbramod":
                            host_by_subject.setdefault(key.split(":")[0], rec)
                    print(f"  ✓ {key} 由 JSON 兜底补上（{len(rows)} 行）", flush=True)

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
        if name == "cbramod":
            # 宿主行：按被试去重（两趟是同一次测量）
            src = [host_by_subject[k] for k in sorted(host_by_subject)]
        else:
            src = [per_subject[k][name] for k in per_subject
                   if name in per_subject[k]]
        rs = [s["r"] for s in src]
        ps = [s["p"] for s in src]
        ac = [s["acc"] for s in src]
        nu = [s["null"] for s in src]
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
        dr, da, dh, dha = [], [], [], []
        for k in per_subject:
            if v in per_subject[k] and c in per_subject[k]:
                dr.append(per_subject[k][v]["r"] - per_subject[k][c]["r"])
                da.append(per_subject[k][v]["acc"] - per_subject[k][c]["acc"])
            if v in per_subject[k] and "cbramod" in per_subject[k]:
                dh.append(per_subject[k][v]["r"] - per_subject[k]["cbramod"]["r"])
                dha.append(per_subject[k][v]["acc"] - per_subject[k]["cbramod"]["acc"])
        if not dr:
            continue
        npos = sum(1 for x in dr if x > 0)
        npos_a = sum(1 for x in da if x > 0)
        npos_h = sum(1 for x in dh if x > 0)
        npos_ha = sum(1 for x in dha if x > 0)
        deltas.append({
            "variant": v, "control": c, "n": len(dr),
            # 对照：供体的「真信息」减去「噪声稀释的代价」
            "d_r_median": med(dr), "d_r_n_pos": npos, "d_r_p": sign_test_p(npos, len(dr)),
            "d_acc_median": med(da), "d_acc_n_pos": npos_a,
            "d_acc_p": sign_test_p(npos_a, len(da)),
            # 宿主：接进来之后整体是升还是降
            "d_r_vs_host_median": med(dh), "d_r_vs_host_n_pos": npos_h,
            "d_r_vs_host_p": sign_test_p(npos_h, len(dh)) if dh else None,
            "d_acc_vs_host_median": med(dha), "d_acc_vs_host_n_pos": npos_ha,
            "d_acc_vs_host_p": sign_test_p(npos_ha, len(dha)) if dha else None,
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

    print("\n**供体贡献（逐被试配对）**")
    print("| 变体 | n | Δr vs 对照 | 正向 | p | Δacc vs 对照 | 正向 | p | "
          "Δr vs 宿主 | 正向 | p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in deltas:
        p1 = f"{d['d_r_p']:.4f}" if d["d_r_p"] is not None else "—"
        p2 = f"{d['d_acc_p']:.4f}" if d["d_acc_p"] is not None else "—"
        p3 = f"{d['d_r_vs_host_p']:.4f}" if d.get("d_r_vs_host_p") is not None else "—"
        dr = f"{d['d_r_median']:+.4f}" if d["d_r_median"] is not None else "—"
        da = f"{d['d_acc_median']:+.4f}" if d["d_acc_median"] is not None else "—"
        dh = (f"{d['d_r_vs_host_median']:+.4f}"
              if d.get("d_r_vs_host_median") is not None else "—")
        print(f"| {d['variant']} | {d['n']} | {dr} | {d['d_r_n_pos']}/{d['n']} | {p1} | "
              f"{da} | {d['d_acc_n_pos']}/{d['n']} | {p2} | {dh} | "
              f"{d.get('d_r_vs_host_n_pos', '—')}/{d['n']} | {p3} |")

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
