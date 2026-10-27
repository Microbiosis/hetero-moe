"""parse_exp90c.py — exp90c（判据②：接口的因果价值，THINGS-EEG）的结果重建与**判决离线重算**。

## 做什么

1. **取回结果**：桶里 `runs/illusionbrain-align/exp90c/exp90c_interface.json`
   （`hf buckets cp` 读通道不稳 ⇒ 带重试；退化通道：Job 日志里的 `===REC===` / `===VERDICT===` 回显）。
2. **用判据脚本里的 `decide()` 从逐被试原始记录重算判决**（可审计、可复现）⇒ 与云端逐字比对。
3. 报出**接口的去留**（判据② 的主判决）与只报项（3×3 转分化、各带 RSA、潜变量离散度）。

⚠️ **读法提醒（§四十一·三）**：C3 的量是 `r_cut − r_full`（切接口后的读出残差 − 满接口的读出残差）
⇒ ⇒ **> 0 ⇒ 满接口时目标分支更一致 ⇒ 该接口有正价值（保留）**；**≤ 0 ⇒ 接口把目标分支的表征搞乱（负价值 ⇒ 砍掉）**。

## 输出

- `results/exp90c-results/exp90c_interface.json`（云端原始）
- `results/exp90c-results/exp90c_summary.{json,md}`
- `results/exp90c-results/exp90c_verdict_compact.json`（**紧凑判决**，守卫绑它）
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp90c-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp90c/exp90c_interface.json"
RAW = OUT / "exp90c_interface.json"
LOG = OUT / "judge_log.txt"
JUDGE = "exp90c_interface"


def _lines(p: Path):
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def _extract_compact(log_path: Path):
    """退化通道：日志里只有 `===REC===`（逐被试记录）+ `===VERDICT===`（云端判决）。"""
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
    raise RuntimeError("取不回 exp90c 的结果（日志里也没有、桶也读不到）")


def fmt_table(rows, header):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    force = "--force" in sys.argv
    raw = fetch(force=force)
    data = json.loads(raw.read_text(encoding="utf-8"))
    per = data.get("per_subject") or []
    job_verdict = data.get("verdict") or {}

    import importlib
    E = importlib.import_module(JUDGE)
    recomputed = E.decide(per)

    checks = {
        "verdict": (job_verdict.get("verdict"), recomputed.get("verdict")),
        "C3_pass": (job_verdict.get("C3_pass"), recomputed.get("C3_pass")),
        "C2_pass": (job_verdict.get("C2_pass"), recomputed.get("C2_pass")),
        "C1_pass": (job_verdict.get("C1_pass"), recomputed.get("C1_pass")),
        "C0_pass": (job_verdict.get("C0_pass"), recomputed.get("C0_pass")),
        "interfaces_keep": (job_verdict.get("interfaces_keep"), recomputed.get("interfaces_keep")),
        "interfaces_drop": (job_verdict.get("interfaces_drop"), recomputed.get("interfaces_drop")),
        "C0.acc1_median": ((job_verdict.get("C0") or {}).get("acc1_median"),
                            (recomputed.get("C0") or {}).get("acc1_median")),
        "C0.rsa_median": ((job_verdict.get("C0") or {}).get("rsa_median"),
                           (recomputed.get("C0") or {}).get("rsa_median")),
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

    c0 = recomputed.get("C0") or {}
    c3 = recomputed.get("C3") or {}
    print(f"\n预登记判决（云端写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐被试）      : {recomputed.get('verdict')}  ⇒ 一致：{same}")
    for m in mismatches:
        print("   ✗ " + m)
    print(f"C0 管线自检：200-way 中位 {c0.get('acc1_median')}（≥3×随机={c0.get('acc1_above_chance3')}）"
          f"｜RSA 中位 {c0.get('rsa_median')}")
    print(f"C2 退化守卫：{recomputed.get('C2_pass')}")
    print(f"C3 接口去留：保留 {recomputed.get('interfaces_keep')}｜砍掉 {recomputed.get('interfaces_drop')}")
    for k in sorted(c3):
        v = c3[k]
        reading = ("有正价值（保留）" if v.get("has_value") else v.get("reading"))
        print(f"   {k}: n={v.get('n_eff')} n_pos={v.get('n_pos')} p={v.get('p_sign'):.4f} "
              f"中位={v.get('median'):+.4f} ⇒ {reading}")

    iface_rows = []
    for k, v in sorted(c3.items()):
        # ⚠️ 判据脚本里的 `reading` **只在「没过」时才有意义**（它是 fallback 解释）
        #    ⇒ 通过的接口在汇总层给出正确读法（**不改已跑完的判决脚本**）
        reading = ("有正价值（保留）" if v.get("has_value")
                   else v.get("reading"))
        iface_rows.append((k, v.get("n_eff"), v.get("n_pos"),
                           f"{v.get('p_sign'):.4f}", f"{v.get('median'):+.4f}",
                           "保留" if v.get("has_value") else "砍掉", reading))
    disp_rows = []
    for k, v in sorted((recomputed.get("C2") or {}).items()):
        disp_rows.append((k, f"{v.get('median_disp'):.4f}", f"{v.get('min_disp'):.4f}",
                          "✓" if v.get("pass") else "✗"))
    rsa_rows = []
    for r in per:
        if "img_rsa" not in r:
            continue
        rsa_rows.append((r.get("subject"),
                         f"{r['img_rsa']['POST']['r']:+.4f}",
                         f"{r['img_rsa']['CENT']['r']:+.4f}",
                         f"{r['img_rsa']['FRONT']['r']:+.4f}",
                         f"{r.get('post_acc1'):.4f}" if r.get("post_acc1") is not None else "—"))
    td_rows = []
    for k, v in sorted((recomputed.get("C4_transdiff_report") or {}).items()):
        td_rows.append((k, v.get("n"), f"{v.get('median'):+.4f}"))

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_subject": per,
        "note": "C3 的量 = r_cut − r_full ⇒ >0 ⇒ 满接口时目标分支更一致（接口有正价值）；"
                "≤0 ⇒ 接口把目标分支的表征搞乱（负价值 ⇒ 砍掉）",
    }
    (OUT / "exp90c_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    compact = {
        "verdict": recomputed.get("verdict"),
        "recompute_matches_job": same,
        "C0": recomputed.get("C0"),
        "C0_pass": recomputed.get("C0_pass"),
        "C1_pass": recomputed.get("C1_pass"),
        "C2": recomputed.get("C2"),
        "C2_pass": recomputed.get("C2_pass"),
        "C3": recomputed.get("C3"),
        "C3_pass": recomputed.get("C3_pass"),
        "interfaces_keep": recomputed.get("interfaces_keep"),
        "interfaces_drop": recomputed.get("interfaces_drop"),
        "C4_transdiff_report": recomputed.get("C4_transdiff_report"),
        "why": recomputed.get("why"),
    }
    (OUT / "exp90c_verdict_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp90c 结果重建（判据②：接口的因果价值，THINGS-EEG）\n")
    md.append(f"**预登记判决（云端写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐被试原始记录）**：`{recomputed.get('verdict')}` ⇒ 与云端一致：**{same}**\n")
    md.append(f"**C0 管线自检**：200-way 中位 **{c0.get('acc1_median')}**（≥3×随机：{c0.get('acc1_above_chance3')}）"
              f"｜图像 RSA 中位 **{c0.get('rsa_median')}**\n")
    md.append(f"**C2 退化守卫**：{recomputed.get('C2_pass')}\n")
    md.append(f"**C3 接口去留**：保留 `{recomputed.get('interfaces_keep')}`｜砍掉 `{recomputed.get('interfaces_drop')}`\n")
    md.append("## 判据②：每条接口（量 = r_cut − r_full）\n")
    md.append(fmt_table(iface_rows, ["接口|受影响分支", "n", "n_pos", "p", "中位", "去留", "读法"]))
    md.append("\n## 退化守卫：各原语潜变量的相对离散度\n")
    md.append(fmt_table(disp_rows, ["原语", "中位离散度", "最小离散度", "过"]))
    md.append("\n## 逐被试：图像 RSA 与 200-way\n")
    md.append(fmt_table(rsa_rows, ["被试", "POST RSA", "CENT RSA", "FRONT RSA", "200-way"]))
    md.append("\n## 只报：3×3 转分化（各带对 Δ 的中位；⚠️ 原语个数只报不判）\n")
    md.append(fmt_table(td_rows, ["带对", "n", "中位Δ"]))
    md.append("\n## 读法提醒\n")
    md.append("- C3 的量 = **切接口后的读出残差 − 满接口的读出残差** ⇒ **> 0 ⇒ 满接口时目标分支更一致"
              "⇒ 该接口有正价值（保留）**；**≤ 0 ⇒ 接口把目标分支的表征搞乱（负价值 ⇒ 砍掉）**。\n")
    (OUT / "exp90c_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp90c_summary.json'} / {OUT / 'exp90c_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
