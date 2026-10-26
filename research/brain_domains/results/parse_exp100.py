"""parse_exp100.py — exp100（判据① 的「组内 vs 跨组」验收，Allen VC-NP）的
结果重建与**判决离线重算**。

## 做什么

1. **取回结果**：桶里 `runs/illusionbrain-align/exp100/exp100_group_ladder.json`
   （`hf buckets cp` 读通道不稳 ⇒ 带重试；退化通道：Job 日志里的 `===REC===` / `===VERDICT===` 回显）。
2. **用判据脚本里的 `decide()` 从逐 session 原始记录重算判决**（可审计、可复现）⇒ 与云端逐字比对。
3. 报出**进入/剔除**的 session 数与原因（C0″ 可读性 + 逐对门槛 ⇒ 排除规则与 Δ 无关）。
4. 报出**只报项**：全部 21 个无序带对的 Δ（同组/跨组/CA1 分层）、每带解码准确率、每 session 的组内-跨组中位。

⚠️ **口径提醒（§四十·二/·四）**：分组来自 **Allen 官方分类**（primary={VISp} vs
higher={VISl/VISal/VISpm/VISam/VISrl}；CA1 只报不判）⇒ **绝不用实测 Δ 给脑区排序**；
`C1_FAIL` 时**必须**同时看中位统计量的方向（>0 = 方向对但检验力不足；≤0 = 分不出功能分组）。

## 输出

- `results/exp100-results/exp100_group_ladder.json`（云端原始）
- `results/exp100-results/exp100_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp100-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp100/exp100_group_ladder.json"
RAW = OUT / "exp100_group_ladder.json"
LOG = OUT / "judge_log.txt"
JUDGE = "exp100_group_ladder"


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
    raise RuntimeError("取不回 exp100 的结果（日志里也没有、桶也读不到）")


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

    import importlib
    E = importlib.import_module(JUDGE)
    recomputed = E.decide(per_session)

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
        "C1.same_median_pooled": ((job_verdict.get("C1") or {}).get("same_median_pooled"),
                                   (recomputed.get("C1") or {}).get("same_median_pooled")),
        "C1.cross_median_pooled": ((job_verdict.get("C1") or {}).get("cross_median_pooled"),
                                    (recomputed.get("C1") or {}).get("cross_median_pooled")),
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
    print(f"同组中位 Δ={c1.get('same_median_pooled')}；跨组中位 Δ={c1.get('cross_median_pooled')}")
    print(f"解释：{recomputed.get('C1_interpretation')}")

    sess_rows = []
    for r in per_session:
        rd = r.get("readable") or {}
        sess_rows.append((
            r.get("session", "").split("/")[-1],
            f"{r.get('group_stat'):+.4f}" if r.get("group_stat") is not None else "—",
            f"{r.get('same_median'):+.4f}" if r.get("same_median") is not None else "—",
            f"{r.get('cross_median'):+.4f}" if r.get("cross_median") is not None else "—",
            r.get("n_same_pairs"), r.get("n_cross_pairs"),
            "✓" if r.get("enough_pairs") else "✗",
            r.get("skip") or "",
            ", ".join(f"{b}:{v.get('acc', float('nan')):.3f}/p{v.get('p', float('nan')):.3f}"
                      for b, v in sorted(rd.items())),
        ))
    sess_rows.sort(key=lambda x: x[1])

    per_pair = recomputed.get("per_pair") or {}
    pair_rows = []
    for key in sorted(per_pair):
        v = per_pair[key]
        a, b = key.split("->")
        g = E.pair_group(a, b)
        pair_rows.append((key, g or "—", v.get("n"), v.get("n_pos"),
                          f"{v.get('median'):+.4f}" if v.get("median") is not None else "—",
                          f"{v.get('p_sign'):.4f}" if v.get("p_sign") is not None else "—"))

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_session": per_session,
        "per_pair": per_pair,
        "note": "**鼠 ≠ 人**：本实验只回答「判据① 有没有判别力」，**不回填**「三个原语还是两个」；"
                "分组来自 **Allen 官方分类**（绝不用实测 Δ 排序 ⇒ 不循环论证）；"
                "CA1 因几乎一律不可读**只报不判**",
    }
    (OUT / "exp100_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # **紧凑判决文件**（给守卫读）⇒ 只含守卫要绑的字段 ⇒ **几 KB**
    # （完整的 per_session 原始记录太大，留在桶里 + results 里另存，git 不必背）
    compact = {
        "verdict": recomputed.get("verdict"),
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "n_sessions_total": recomputed.get("n_sessions_total"),
        "n_sessions_primary_used": recomputed.get("n_sessions_primary_used"),
        "C1": {k: (recomputed.get("C1") or {}).get(k)
               for k in ("n_eff", "n_pos", "n_neg", "p", "n_zero", "median_delta",
                         "same_median_pooled", "cross_median_pooled")},
        "C2_degenerate_guard": recomputed.get("C2_degenerate_guard"),
        "C1_interpretation": recomputed.get("C1_interpretation"),
        "sessions_excluded": recomputed.get("sessions_excluded"),
    }
    (OUT / "exp100_verdict_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp100 结果重建（判据① 的「组内 vs 跨组」验收）\n")
    md.append(f"**预登记判决（云端写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐 session 原始记录）**：`{recomputed.get('verdict')}` ⇒ "
              f"与云端一致：**{same}**\n")
    md.append(f"**进入主判据**：{recomputed.get('n_sessions_primary_used')} / "
              f"{recomputed.get('n_sessions_total')}"
              f"（剔除 {len(recomputed.get('sessions_excluded') or [])}）\n")
    md.append(f"**符号检验**：{c1.get('n_pos')}/{c1.get('n_eff')} 正，p={c1.get('p')}，"
              f"中位 Δ={c1.get('median_delta')}（零差 {c1.get('n_zero')}）\n")
    md.append(f"**同组中位 Δ** = {c1.get('same_median_pooled')}；"
              f"**跨组中位 Δ** = {c1.get('cross_median_pooled')}\n")
    md.append(f"**解释**：{recomputed.get('C1_interpretation')}\n")
    md.append("## 逐 session\n")
    md.append(fmt_table(sess_rows,
                        ["session", "统计量(跨−同)", "同组中位", "跨组中位", "同组对数",
                         "跨组对数", "够对数", "剔除原因", "各带解码 acc/p"]))
    md.append("\n## 只报：全部有向带对的 Δ\n")
    md.append(fmt_table(pair_rows, ["带对", "分组", "n", "n_pos", "中位Δ", "符号检验 p"]))
    md.append("\n## 诚实边界\n")
    md.append("- 排除规则**与 Δ 无关**（可读性 + 逐对门槛都是数据属性）⇒ 报出进入/剔除数与原因。\n")
    md.append("- 分组来自 **Allen 官方分类**，**绝不用实测 Δ 排序** ⇒ 不循环论证。\n")
    md.append("- **CA1 只报不判**（几乎一律不可读 ⇒ 放进来会让跨 session 口径不一致）。\n")
    md.append("- **鼠 ≠ 人**：只回答「判据① 有没有判别力」，**不回填**「三个原语还是两个」。\n")
    (OUT / "exp100_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp100_summary.json'} / {OUT / 'exp100_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
