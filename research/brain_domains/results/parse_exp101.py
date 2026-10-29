"""parse_exp101.py — exp101（②③ 条件生成线的第一个里程碑：脑侧条件的可辨识性）的结果重建与**判决离线重算**。

1. 取回结果：桶 `runs/illusionbrain-align/exp101/exp101_condition.json`
   （退化通道：Job 日志的 `===REC===` / `===VERDICT===` 回显）。
2. 用判据脚本里的 `decide()` 从逐被试原始记录重算判决 ⇒ 与云端逐字比对。
3. 出**人读表**（逐被试的 200-way / 50-way / 2-way，单试次 + 试次平均）与**与 ATM 的并排参照**。

⚠️ **口径提醒**：ATM（NeurIPS 2024）报的是「**生成图**」的 2-way / 50-way / FID；
我们这里量的是「**条件**」的可辨识性 ⇒ **协议不同 ⇒ 只作量级参照，不是同协议对标**。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp101-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp101/exp101_condition.json"
RAW = OUT / "exp101_condition.json"
LOG = OUT / "judge_log.txt"


def _lines(p: Path):
    return p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []


def _extract_compact(log_path: Path):
    recs, verdict = [], None
    for ln in _lines(log_path):
        if ln.startswith("===REC==="):
            try:
                recs.append(json.loads(ln[len("===REC==="):]))
            except Exception:
                pass
        elif ln.startswith("===VERDICT==="):
            try:
                verdict = json.loads(ln[len("===VERDICT==="):])
            except Exception:
                verdict = None
    if not recs:
        return None
    return {"config": {"source": "compact-records"}, "per_subject": recs,
            "verdict": verdict or {}}


def fetch(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not force:
        return RAW
    payload = _extract_compact(LOG)
    if payload:
        RAW.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                       encoding="utf-8")
        print(f"✓ 退化通道：从日志抽回（{RAW.stat().st_size} 字节）")
        return RAW
    for i in range(5):
        subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{REL}", str(RAW)],
                       capture_output=True, text=True)
        if RAW.exists() and RAW.stat().st_size > 0:
            print(f"✓ 取回 {RAW}（第 {i+1} 次）")
            return RAW
        print(f"  ✗ 取回失败（第 {i+1} 次）")
        time.sleep(6)
    raise RuntimeError("取不回 exp101 的结果")


def fmt_table(rows, header):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    raw = fetch(force="--force" in sys.argv)
    data = json.loads(raw.read_text(encoding="utf-8"))
    per = data.get("per_subject") or []
    job_verdict = data.get("verdict") or {}

    import importlib
    E = importlib.import_module("exp101_condition")
    recomputed = E.decide(per)

    checks = {
        "verdict": (job_verdict.get("verdict"), recomputed.get("verdict")),
        "C0_pass": (job_verdict.get("C0_pass"), recomputed.get("C0_pass")),
        "C1_pass": (job_verdict.get("C1_pass"), recomputed.get("C1_pass")),
        "C2_pass": (job_verdict.get("C2_pass"), recomputed.get("C2_pass")),
        "C0.acc200_median": ((job_verdict.get("C0") or {}).get("acc200_median"),
                              (recomputed.get("C0") or {}).get("acc200_median")),
        "C1.median": (((job_verdict.get("C1") or {}).get("acc50_minus_chance") or {}).get("median"),
                      ((recomputed.get("C1") or {}).get("acc50_minus_chance") or {}).get("median")),
        "C2.median": (((job_verdict.get("C2") or {}).get("acc2_minus_chance") or {}).get("median"),
                      ((recomputed.get("C2") or {}).get("acc2_minus_chance") or {}).get("median")),
    }
    mismatches = []
    for k, (a, b) in checks.items():
        if isinstance(a, float) or isinstance(b, float):
            ok = (a is not None and b is not None
                  and (abs(float(a) - float(b)) < 1e-9 or (a != a and b != b)))
        else:
            ok = (a == b)
        if not ok:
            mismatches.append(f"{k}: Job={a!r} 离线={b!r}")
    same = not mismatches

    rep = recomputed.get("C3_report") or {}
    print(f"\n预登记判决（云端写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐被试）      : {recomputed.get('verdict')}  ⇒ 一致：{same}")
    for m in mismatches:
        print("   ✗ " + m)
    print(f"C0 条件自检：200-way 中位 {rep.get('acc200_median')}（参照 exp93b {E.ATM_REF and 0.0589}）")
    print(f"C1 50-way：中位 {rep.get('acc50_median')}（随机 0.02）｜试次平均 {rep.get('acc50_avg_median')}")
    print(f"C2 2-way：中位 {rep.get('acc2_median')}（随机 0.5）｜试次平均 {rep.get('acc2_avg_median')}")

    rows = [(r.get("subject"),
             f"{r.get('acc200'):.4f}", f"{r.get('acc50'):.4f}", f"{r.get('acc2'):.4f}",
             f"{r.get('acc200_avg'):.4f}", f"{r.get('acc50_avg'):.4f}",
             f"{r.get('acc2_avg'):.4f}") for r in per if "acc200" in r]
    atm = E.ATM_REF
    md = []
    md.append("# exp101 结果重建（②③ 条件生成线 · 第一个里程碑：脑侧条件的可辨识性）\n")
    md.append(f"**预登记判决（云端写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐被试原始记录）**：`{recomputed.get('verdict')}` ⇒ 与云端一致：**{same}**\n")
    md.append(f"**结论**：{recomputed.get('why', '')}\n")
    md.append("## 逐被试（条件 ↔ 图像特征 的检索）\n")
    md.append(fmt_table(rows, ["被试", "200-way", "50-way", "2-way",
                               "200-way(试次平均)", "50-way(试次平均)", "2-way(试次平均)"]))
    md.append("\n## 与 ATM（NeurIPS 2024）的并排参照\n")
    md.append("| 指标 | 随机水平 | **我们（条件）单试次** | **我们（条件）试次平均** | ATM（**生成图**）|")
    md.append("|---|---|---|---|---|")
    md.append(f"| 200-way top-1 | 0.50% | {rep.get('acc200_median'):.2%} | {rep.get('acc200_avg_median'):.2%} | — |")
    md.append(f"| 50-way top-1 | 2.00% | {rep.get('acc50_median'):.2%} | {rep.get('acc50_avg_median'):.2%} | {atm['fifty_way_top1_pct']:.1f}% |")
    md.append(f"| 2-way top-1 | 50.00% | {rep.get('acc2_median'):.2%} | {rep.get('acc2_avg_median'):.2%} | {atm['two_way_pct']:.1f}% |")
    md.append("\n⚠️ **协议不同**：ATM 量的是「**生成图**」的可辨识性（还含 FID），我们量的是「**条件**」的可辨识性 ⇒ "
              "**只作量级参照，不是同协议对标**。\n")
    (OUT / "exp101_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "exp101_summary.json").write_text(
        json.dumps({"job_verdict": job_verdict, "recomputed_verdict": recomputed,
                    "recompute_matches_job": same, "mismatches": mismatches,
                    "per_subject": per}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    compact = {"verdict": recomputed.get("verdict"),
               "recompute_matches_job": same,
               "C0": recomputed.get("C0"), "C0_pass": recomputed.get("C0_pass"),
               "C1": recomputed.get("C1"), "C1_pass": recomputed.get("C1_pass"),
               "C2": recomputed.get("C2"), "C2_pass": recomputed.get("C2_pass"),
               "C3_report": rep, "why": recomputed.get("why")}
    (OUT / "exp101_verdict_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"输出：{OUT / 'exp101_summary.json'} / {OUT / 'exp101_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
