"""parse_exp98.py — exp98（Allen Visual Coding–Neuropixels 上判据① 的**正/负对照**）的
结果重建与**判决离线重算**。

## 做什么

1. **取回结果**：桶里 `runs/illusionbrain-align/exp98/exp98_posneg_control.json`
   （`hf buckets cp` 读通道不稳 ⇒ 带重试；退化通道：Job 日志里的 `===VERDICT===` 回显）。
2. **用判据脚本里的 `decide()` 从逐 session 原始记录重算判决**（可审计、可复现）⇒ 与云端逐字比对。
3. 报出**进入/剔除**的 session 数与原因（C0′ 可读性 ⇒ 排除规则与 Δ 无关）。
4. 报出**只报项**：其余带对（VISp↔VISrl / VISal / VISpm / CA1）的 Δ 阶梯、每带解码准确率。

⚠️ **口径提醒（§三十九·三/·五）**：负对照是「**层级距离最小**」的一对（VISp↔VISl），
**不是**「同层级」对照 ⇒ ⇒ **`C1_FAIL` 不等于「判据① 没判别力」**，必须同时看正对照自身的 Δ 中位。

## 输出

- `results/exp98-results/exp98_posneg_control.json`（云端原始）
- `results/exp98-results/exp98_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp98-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp98/exp98_posneg_control.json"
RAW = OUT / "exp98_posneg_control.json"
LOG = OUT / "judge_log.txt"


def _lines(p: Path):
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def _extract_compact(log_path: Path):
    """退化通道：日志里只有 `===REC===`（逐 session 记录）+ `===VERDICT===`（云端判决）。"""
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
    return {"config": {"source": "compact-records"}, "per_session": recs,
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
    raise RuntimeError("取不回 exp98 的结果（日志里也没有、桶也读不到）")


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
    per_session = data.get("per_session") or []
    job_verdict = data.get("verdict") or {}

    import exp98_posneg_control as E98
    recomputed = E98.decide(per_session)

    checks = {
        "verdict": (job_verdict.get("verdict"), recomputed.get("verdict")),
        "n_sessions_primary_used": (job_verdict.get("n_sessions_primary_used"),
                                    recomputed.get("n_sessions_primary_used")),
        "C1.n_eff": ((job_verdict.get("C1") or {}).get("n_eff"),
                     (recomputed.get("C1") or {}).get("n_eff")),
        "C1.n_pos": ((job_verdict.get("C1") or {}).get("n_pos"),
                     (recomputed.get("C1") or {}).get("n_pos")),
        "C1.p": ((job_verdict.get("C1") or {}).get("p"),
                 (recomputed.get("C1") or {}).get("p")),
        "C1.median_delta": ((job_verdict.get("C1") or {}).get("median_delta"),
                            (recomputed.get("C1") or {}).get("median_delta")),
        "C1_pos_median": (job_verdict.get("C1_pos_median"), recomputed.get("C1_pos_median")),
        "C1_neg_median": (job_verdict.get("C1_neg_median"), recomputed.get("C1_neg_median")),
        "C2_degenerate_guard": (job_verdict.get("C2_degenerate_guard"),
                                recomputed.get("C2_degenerate_guard")),
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

    c1 = recomputed.get("C1") or {}
    print(f"\n预登记判决（云端写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐 session）   : {recomputed.get('verdict')}  ⇒ 一致：{same}")
    for m in mismatches:
        print("   ✗ " + m)
    print(f"进入主判据：{recomputed.get('n_sessions_primary_used')} / "
          f"{recomputed.get('n_sessions_total')}"
          f"（剔除 {len(recomputed.get('sessions_excluded') or [])}）")
    print(f"符号检验：{c1.get('n_pos')}/{c1.get('n_eff')} 正，p={c1.get('p')}，"
          f"中位 Δ={c1.get('median_delta')}")
    print(f"正对照中位 Δ={recomputed.get('C1_pos_median')}；"
          f"负对照中位 Δ={recomputed.get('C1_neg_median')}")
    print(f"解释：{recomputed.get('C1_interpretation')}")

    sess_rows = []
    for r in per_session:
        rd = r.get("readable") or {}
        ps = r.get("pairs") or {}
        sess_rows.append((
            r.get("session", "").split("/")[-1],
            f"{r.get('primary_stat'):+.4f}" if r.get("primary_stat") is not None else "—",
            f"{r.get('neg_stat'):+.4f}" if r.get("neg_stat") is not None else "—",
            f"{r.get('pos_stat'):+.4f}" if r.get("pos_stat") is not None else "—",
            "✓" if all(rd.get(b, {}).get("ok") for b in ("VISp", "VISl", "VISam")) else "✗",
            r.get("skip") or "",
            ", ".join(f"{b}:{v.get('acc', float('nan')):.3f}/p{v.get('p', float('nan')):.3f}"
                      for b, v in sorted(rd.items())),
        ))
    sess_rows.sort(key=lambda x: x[1])

    per_pair = recomputed.get("per_pair") or {}
    pair_rows = []
    for key in sorted(per_pair):
        v = per_pair[key]
        pair_rows.append((key, v.get("n"), v.get("n_pos"),
                          f"{v.get('median'):+.4f}" if v.get("median") is not None else "—",
                          f"{v.get('p_sign'):.4f}" if v.get("p_sign") is not None else "—"))

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_session": per_session,
        "per_pair": per_pair,
        "note": "**鼠 ≠ 人**：本实验只回答「判据① 有没有判别力」，"
                "**不回填**「三个原语还是两个」（人 EEG 的问题）；"
                "负对照是「层级距离最小」的一对 ⇒ `C1_FAIL` 不等于「没判别力」（§三十九·五）",
    }
    (OUT / "exp98_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp98 结果重建（Allen VC-NP 上判据① 的正/负对照）\n")
    md.append(f"**预登记判决（云端写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐 session 原始记录）**：`{recomputed.get('verdict')}` ⇒ "
              f"与云端一致：**{same}**\n")
    md.append(f"**进入主判据**：{recomputed.get('n_sessions_primary_used')} / "
              f"{recomputed.get('n_sessions_total')}"
              f"（因 C0′ 可读性剔除 {len(recomputed.get('sessions_excluded') or [])}）\n")
    md.append(f"**符号检验**：{c1.get('n_pos')}/{c1.get('n_eff')} 正，p={c1.get('p')}，"
              f"中位 Δ={c1.get('median_delta')}（零差 {c1.get('n_zero')}）\n")
    md.append(f"**正对照中位 Δ** = {recomputed.get('C1_pos_median')}；"
              f"**负对照中位 Δ** = {recomputed.get('C1_neg_median')}\n")
    md.append(f"**解释**：{recomputed.get('C1_interpretation')}\n")
    md.append("## 逐 session\n")
    md.append(fmt_table(sess_rows,
                        ["session", "主统计量", "负对照Δ", "正对照Δ", "C0′", "剔除原因", "各带解码 acc/p"]))
    md.append("\n## 只报：各带对的 Δ（阶梯）\n")
    md.append(fmt_table(pair_rows, ["带对", "n", "n_pos", "中位Δ", "符号检验 p"]))
    md.append("\n## 诚实边界\n")
    md.append("- 排除规则**与 Δ 无关**（可读性是数据属性）⇒ 报出进入/剔除数与原因。\n")
    md.append("- 负对照是「**层级距离最小**」的一对（**不是**「同层级」）⇒ "
              "**`C1_FAIL` 不等于「判据① 没判别力」**（§三十九·五）。\n")
    md.append("- **鼠 ≠ 人**：只回答「判据① 有没有判别力」，**不回填**「三个原语还是两个」。\n")
    (OUT / "exp98_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp98_summary.json'} / {OUT / 'exp98_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
