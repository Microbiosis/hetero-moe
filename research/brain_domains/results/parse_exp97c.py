"""parse_exp97c.py — exp97c（判据① 可测性自检**第三版**：C0 改成「每带用它自己相关的
二值行为变量」）的结果重建与**判决离线重算**。

## 做什么

exp97c 回答两个问题：
1. **可用 session 从 2 个回到几个**（C0 口径修正后）；
2. **跨带转分化代价是否显著为正**（判据① 有没有判别力）。

⚠️ **运行方式（2026-10-26）**：8 个主对 session 单跑一轮要 ~50 分钟，而 Job 超时上限 60 分钟
且被判决脚本**只在整轮末尾写一次结果** ⇒ 一旦超时全部丢失 ⇒ 改成
**分段 worker（每个 session 算完立刻落盘）+ 聚合器（合并后跑同一个 `decide()`）**。
⇒ ⇒ **判据/阈值/统计口径一字未动**（被判决脚本按 SHA-256 校验后原样执行；
每个 session 的随机数都由固定 seed 派生 ⇒ 分段与整轮**逐位等价**）。

本脚本：

1. **取回结果** —— 桶的读通道不稳（`hf buckets cp` 报 `xet-read-token`）⇒ 依次试：
   ① 本地已有；② 聚合器日志里的 **`===PAYLOAD===`**（完整 payload，Job 回显通道，
   **不依赖桶的读权限**）；③ 退化通道：只有 **`===REC===`**（逐 session 压缩记录）
   + **`===VERDICT===`**（云端判决）⇒ 也能离线重算；④ `hf buckets cp`（带重试）。
2. **用 exp97c 脚本里的 `decide()` 从逐 session 原始记录重算判决**（可审计、可复现）。
3. 报告**两个口径的对照**（这一轮的核心产出）：
   - **新口径**（C0′）：每带用它自己相关的二值行为变量；不可读的 session **被剔除**；
   - **旧口径**（exp97b）：所有带用同一个 4 类联合标签，不过则**全实验作废**；
   ⇒ ⇒ 直接回答「**可用 session 从 2 个回到几个**」。
4. 把逐 session 的跨带/同带代价、两套口径下的可解码性整理成表。

## 输出

- `results/exp97c-results/exp97c_perband_labels.json`（原始）
- `results/exp97c-results/exp97c_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp97c-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL_CANDIDATES = ("exp97c_final/exp97c_perband_labels.json",
                   "exp97c/exp97c_perband_labels.json")
RAW = OUT / "exp97c_perband_labels.json"
LOG = OUT / "aggregate_log.txt"


def _lines(log_path: Path):
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()


def _extract_payload(log_path: Path):
    """完整通道：聚合器回显的 `===PAYLOAD===`（单行 JSON）。"""
    for ln in _lines(log_path):
        if ln.startswith("===PAYLOAD==="):
            body = ln[len("===PAYLOAD==="):].strip()
            for cand in (body, body[body.find("{"):body.rfind("}") + 1] if "{" in body else ""):
                if not cand:
                    continue
                try:
                    return json.loads(cand)
                except Exception:
                    continue
    return None


def _extract_compact(log_path: Path):
    """退化通道：只有 `===REC===`（逐 session 压缩记录）+ `===VERDICT===` ⇒ 仍可离线重算。"""
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
    return {"config": {"source": "compact-records"},
            "per_session": recs, "verdict": verdict or {}}


def fetch(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not force:
        return RAW
    payload = _extract_payload(LOG)
    if payload:
        RAW.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                       encoding="utf-8")
        print(f"✓ 从聚合器日志抽回完整 payload（{RAW.stat().st_size} 字节）")
        return RAW
    payload = _extract_compact(LOG)
    if payload:
        RAW.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                       encoding="utf-8")
        print(f"✓ 退化通道：只有压缩记录 + 云端判决（{RAW.stat().st_size} 字节）")
        return RAW
    for rel in REL_CANDIDATES:
        for i in range(3):
            subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{rel}", str(RAW)],
                           capture_output=True, text=True)
            if RAW.exists() and RAW.stat().st_size > 0:
                print(f"✓ 取回 {RAW}（{rel}，第 {i+1} 次）")
                return RAW
            print(f"  ✗ 取回失败：{rel}（第 {i+1} 次）")
            time.sleep(6)
    raise RuntimeError("取不回 exp97c 的结果（日志里也没有、桶也读不到）")


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

    import exp97c_perband_labels as E97
    recomputed = E97.decide(per_session)

    checks = {
        "verdict": (job_verdict.get("verdict"), recomputed.get("verdict")),
        "n_sessions_primary_used": (job_verdict.get("n_sessions_primary_used"),
                                    recomputed.get("n_sessions_primary_used")),
        "n_sessions_primary_total": (job_verdict.get("n_sessions_primary_total"),
                                     recomputed.get("n_sessions_primary_total")),
        "C1.n_eff": ((job_verdict.get("C1") or {}).get("n_eff"),
                     (recomputed.get("C1") or {}).get("n_eff")),
        "C1.n_pos": ((job_verdict.get("C1") or {}).get("n_pos"),
                     (recomputed.get("C1") or {}).get("n_pos")),
        "C1.p": ((job_verdict.get("C1") or {}).get("p"),
                 (recomputed.get("C1") or {}).get("p")),
        "C1.median_delta": ((job_verdict.get("C1") or {}).get("median_delta"),
                            (recomputed.get("C1") or {}).get("median_delta")),
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

    print(f"\n预登记判决（云端写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐 session）   : {recomputed.get('verdict')}  ⇒ 一致：{same}")
    for m in mismatches:
        print("   ✗ " + m)
    c1 = recomputed.get("C1") or {}
    legacy = recomputed.get("C0_legacy_jointlabel") or {}
    print(f"新口径（C0′ 逐带各自标签）：主对 session 总数 "
          f"{recomputed.get('n_sessions_primary_total')} ⇒ 可用 "
          f"**{recomputed.get('n_sessions_primary_used')}**"
          f"（剔除 {len(recomputed.get('sessions_excluded_by_readability') or [])}）")
    print(f"旧口径（exp97b 4 类联合）：主对可用 **{legacy.get('n_primary_used')}**"
          f"（判死 {legacy.get('n_failing')}）")
    print(f"符号检验：{c1.get('n_pos')}/{c1.get('n_eff')} 正，p={c1.get('p')}，"
          f"中位 Δ={c1.get('median_delta')}")

    prim_rows = []
    for r in per_session:
        ps = r.get("pairs") or {}
        if "motor->prefrontal" in ps and "prefrontal->motor" in ps:
            rd = r.get("readable") or {}
            dec = r.get("decode") or {}
            prim_rows.append((
                r.get("session", "").split("/")[-1],
                f"{r.get('primary_stat'):+.4f}" if r.get("primary_stat") is not None else "—",
                "✓" if all(rd.get(b, {}).get("ok") for b in ("motor", "prefrontal")) else "✗",
                ", ".join(f"{b}[{v.get('var')}]:{v.get('acc', float('nan')):.2f}/p{v.get('p', float('nan')):.3f}"
                          for b, v in sorted(rd.items())),
                ", ".join(f"{b}:{dec[b]['acc']:.2f}/p{dec[b]['p']:.3f}" for b in sorted(dec)),
            ))
    prim_rows.sort(key=lambda x: x[1])

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_session": per_session,
        "n_primary_total": recomputed.get("n_sessions_primary_total"),
        "n_primary_used_new": recomputed.get("n_sessions_primary_used"),
        "n_primary_used_legacy": legacy.get("n_primary_used"),
        "sessions_excluded": recomputed.get("sessions_excluded_by_readability"),
        "note": "**鼠 ≠ 人**：本实验只回答「判据① 有没有判别力」，"
                "**不回填**「三个原语还是两个」（人 EEG 的问题）",
    }
    (OUT / "exp97c_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp97c 结果重建（C0 改成「每带用它自己相关的二值行为变量」）\n")
    md.append(f"**预登记判决（云端写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐 session 原始记录）**：`{recomputed.get('verdict')}` ⇒ "
              f"与云端一致：**{same}**\n")
    md.append("## 两个口径的对照（这一轮的核心产出）\n")
    md.append(f"- **新口径 C0′**（每带用它自己相关的二值行为变量）：主对 session 总数 "
              f"**{recomputed.get('n_sessions_primary_total')}** ⇒ 可用 "
              f"**{recomputed.get('n_sessions_primary_used')}**"
              f"（因可读性被剔除 {len(recomputed.get('sessions_excluded_by_readability') or [])}）")
    md.append(f"- **旧口径**（exp97b：所有带用同一个 4 类联合标签）：主对可用 "
              f"**{legacy.get('n_primary_used')}**（判死 {legacy.get('n_failing')}）\n")
    md.append(f"**符号检验**：{c1.get('n_pos')}/{c1.get('n_eff')} 正，p={c1.get('p')}，"
              f"中位 Δ={c1.get('median_delta')}（零差 {c1.get('n_zero')}）\n")
    md.append("## 进入主判据（CENT↔FRONT）的 session\n")
    md.append(fmt_table(prim_rows,
                        ["session", "统计量", "C0′", "新口径（各带自己的标签）", "旧口径（4 类）"]))
    md.append("\n## 诚实边界\n")
    md.append("- 排除规则**与 Δ 无关**（可读性是数据属性）⇒ 报出进入/剔除数与原因。\n")
    md.append("- **鼠 ≠ 人**：只回答「判据① 有没有判别力」，**不回填**「三个原语还是两个」。\n")
    (OUT / "exp97c_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp97c_summary.json'} / {OUT / 'exp97c_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
