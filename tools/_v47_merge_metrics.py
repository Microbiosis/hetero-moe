#!/usr/bin/env python3
"""把 v47 两个 runner 的结果合并成一份 results/metrics.json。

两个 runner 各自写 metrics.json (各自只覆盖自己那半), 本脚本负责把
central_heldout.json 与 scale_hetero_heldout.json 的记录并进同一份快照,
并按包给出**结论方向是否复现**的判定。

两个判据的区别 (别混):
  * `verdicts` 里的是算子口径 —— 「treatment 的优势有没有活到样本外」;
  * `status`   是包级结论 —— 「本包原来那条结论, 样本外是否仍然成立」。
    一个本来就为负的结论 (C-1 有害 / 无叠加增益) 在样本外仍为负, 说明它站得住。
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "research/central_stability/v47_heldout_audit/results"
SOURCES = ["central_heldout.json", "scale_hetero_heldout.json"]
SCOPES = ["in_sample", "heldout_noise", "heldout_stimulus"]


def gain(rec, scope):
    t, b = rec[scope]["treat"], rec[scope]["base"]
    per = [(y - x) / y * 100 for x, y in zip(t, b)]
    return {"median_pct": round(statistics.median(per), 2),
            "wins": sum(1 for x in per if x > 0), "n": len(per)}


def sgn(x):
    return (x > 0) - (x < 0)


def main():
    merged_records, verdicts, claims, blocked = {}, {}, {}, {}
    seeds = 5
    for src in SOURCES:
        p = RES / src
        if not p.exists():
            print(f"缺 {src}, 跳过")
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        seeds = blob["seeds"]
        verdicts.update(blob.get("verdicts", {}))
        claims.update(blob.get("claims", {}))
        blocked.update(blob.get("blocked", {}))
        for rec in blob["records"]:
            merged_records[rec["name"]] = rec

    per_package = {}
    for name, rec in merged_records.items():
        row = {"in_sample_median_pct": gain(rec, "in_sample")["median_pct"]}
        si = sgn(row["in_sample_median_pct"])
        hos = []
        for scope in SCOPES[1:]:
            g = gain(rec, scope)
            row[scope] = {"median_pct": g["median_pct"], "wins": f"{g['wins']}/{g['n']}",
                          "operator_verdict": verdicts.get(name, {}).get(scope)}
            keep = (g["median_pct"] / row["in_sample_median_pct"]
                    if row["in_sample_median_pct"] > 0 else None)
            row[scope]["retention"] = None if keep is None else round(keep, 3)
            hos.append(g["median_pct"])
        sh = [sgn(x) for x in hos]
        if si > 0 and any(x < 0 for x in sh):
            row["status"] = "invalidated"
        elif si != 0 and all(x == si for x in sh):
            row["status"] = "clean"
        else:
            row["status"] = "pending"
        per_package[name] = row

    for name in blocked:
        per_package.setdefault(name, {"status": "pending", "blocked": blocked[name]})

    metrics = {
        "schema_version": "1.0", "version": "47.0",
        "package": "v47_heldout_audit", "line": "central_stability",
        "metric": "fuse_mse", "direction": "lower_is_better",
        "recorded_by": ("examples/run_v47_central_heldout.py + "
                        "examples/run_v47_scale_hetero_heldout.py"),
        "statistic": "配对逐种子中位数 + 逐种子胜出数 (不用均值 — v40 起的纪律)",
        "scopes": SCOPES, "seeds": seeds,
        "scope_definitions": {
            "in_sample": "与被审计 runner 同构 (v47 额外固定了 attn pool 与 numpy RNG 两个种子源)",
            "heldout_noise": "seed+1000, 同刺激换噪声抽样",
            "heldout_stimulus": "换文本/换代码的未见刺激",
        },
        "audited": sorted(merged_records),
        "claims": claims,
        "operator_verdicts": verdicts,
        "per_package": per_package,
        "status_counts": {
            s: sum(1 for r in per_package.values() if r.get("status") == s)
            for s in ("clean", "invalidated", "pending")
        },
        "caveat": ("status 判的是『本包原结论在样本外是否仍成立』, 与算子的 "
                   "operator_verdict (『treatment 的优势有没有活到样本外』) 是两个问题。"
                   "负向结论 (C-1 有害 / 无叠加增益) 三个口径同为负, 记 clean。"),
    }
    out = RES / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"合并 {len(per_package)} 个包 -> {out}")
    for name, row in per_package.items():
        print(f"  {name:<5} {row.get('status'):<12} "
              f"in={row.get('in_sample_median_pct')} "
              f"noise={row.get('heldout_noise', {}).get('median_pct')} "
              f"stim={row.get('heldout_stimulus', {}).get('median_pct')}")


if __name__ == "__main__":
    main()
