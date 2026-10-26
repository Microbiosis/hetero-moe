"""parse_exp92.py — exp97b（判据① 可测性自检**第二版**，鼠·Steinmetz 2019）的结果重建与**判决离线重算**。

## 做什么

exp97b 在一个 Job 里跑完了「判据① 在同记录多区数据上有没有判别力」的移植版（码本 = 类条件均值），
判决 JSON 落在桶里（`runs/illusionbrain-align/exp97b/exp97b_transdiff_mouse.json`）。本脚本：

1. **取回 JSON** —— ⚠️ **桶的读通道不稳**（`hf buckets cp` 报 `xet-read-token`）⇒
   **三条通道依次试**：① 本地已有；② 从 `results/exp97b-results/job_log.txt` 里**抽出**
   （Job 在云端把文件内容回显到 stdout ⇒ 不依赖桶的读权限）；③ `hf buckets cp`（带重试）。
2. **用 exp97b 脚本里的 `decide()` 从逐 session 原始记录重算判决**
   （不依赖 Job 写下的 `verdict` 字段 ⇒ 判决可审计、可复现）。
3. **把 C0 的失败分成两类**（⚠️ 关键，否则会误读判决）：
   - **「不适用」**：该 session 只有一个带（或 trial 不够）⇒ **它不进入分析**（不是自检失败）；
   - **「真失败」**：进入分析的 session 里，某个带的行为解码没过（准确率 ≤ 机会 或 置换 p ≥ 0.05）。
4. 把逐 session 的跨带/同带代价、可解码准确率整理成表。

## 统计口径（与 exp89 / exp90 / exp90b / exp96 逐字一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
- **零差剔除**：差值为 0 单独计数，**不参与**符号检验。
- **统计单元 = session**（trial 不独立 ⇒ trial 级检验一律只作描述性）。

## 输出

- `results/exp97b-results/exp97b_transdiff_mouse.json`（原始）
- `results/exp91b-results/exp97b_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp97b-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp97b/exp97b_transdiff_mouse.json"
RAW = OUT / "exp97b_transdiff_mouse.json"
LOG = OUT / "job_log.txt"

SKIP_MARK = "带不足两个或 trial 不够"


def _extract_from_log(log_path: Path):
    """从 Job 日志里抽出 JSON（按 `===FILE=== <name>` 分块，只取目标文件那一块）。"""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    cur, buf = None, []
    for ln in text.splitlines():
        if ln.startswith("===FILE==="):
            parts = ln.split()
            name = parts[1] if len(parts) > 1 else ""
            if cur and name == RAW.name:
                return "\n".join(buf)
            cur, buf = name, []
        else:
            buf.append(ln)
    if cur == RAW.name:
        return "\n".join(buf)
    return None


def fetch(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not force:
        return RAW
    txt = _extract_from_log(LOG)
    if txt:
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            try:
                json.loads(txt[i:j + 1])
                RAW.write_text(txt[i:j + 1] + "\n", encoding="utf-8")
                print(f"✓ 从 Job 日志抽出 {RAW}（{RAW.stat().st_size} 字节）")
                return RAW
            except Exception:
                pass
    for i in range(4):
        subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{REL}", str(RAW)],
                       capture_output=True, text=True)
        if RAW.exists() and RAW.stat().st_size > 0:
            print(f"✓ 取回 {RAW}（第 {i+1} 次）")
            return RAW
        print(f"  ✗ 第 {i+1} 次取回失败")
        time.sleep(6)
    raise RuntimeError("取不回 exp97b 的原始 JSON（日志里也没有）")


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

    import exp97b_transdiff_mouse as E92
    recomputed = E92.decide(per_session)

    checks = {
        "verdict": (job_verdict.get("verdict"), recomputed.get("verdict")),
        "C0_pipeline_selfcheck": (job_verdict.get("C0_pipeline_selfcheck"),
                                  recomputed.get("C0_pipeline_selfcheck")),
        "C1_has_discriminative_power": (job_verdict.get("C1_has_discriminative_power"),
                                        recomputed.get("C1_has_discriminative_power")),
        "C2_degenerate_guard": (job_verdict.get("C2_degenerate_guard"),
                                recomputed.get("C2_degenerate_guard")),
        "C1.n_eff": ((job_verdict.get("C1") or {}).get("n_eff"),
                     (recomputed.get("C1") or {}).get("n_eff")),
        "C1.n_pos": ((job_verdict.get("C1") or {}).get("n_pos"),
                     (recomputed.get("C1") or {}).get("n_pos")),
        "C1.n_neg": ((job_verdict.get("C1") or {}).get("n_neg"),
                     (recomputed.get("C1") or {}).get("n_neg")),
        "C1.p": ((job_verdict.get("C1") or {}).get("p"),
                 (recomputed.get("C1") or {}).get("p")),
        "C1.median_delta": ((job_verdict.get("C1") or {}).get("median_delta"),
                            (recomputed.get("C1") or {}).get("median_delta")),
    }
    mismatches = []
    for k, (a, b) in checks.items():
        if isinstance(a, float) or isinstance(b, float):
            ok = (a is not None and b is not None
                  and (abs(float(a) - float(b)) < 1e-9
                       or (a != a and b != b)))       # NaN == NaN
        else:
            ok = (a == b)
        if not ok:
            mismatches.append(f"{k}: Job={a!r} 离线={b!r}")
    same = not mismatches

    print(f"\n预登记判决（Job 写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐 session）   : {recomputed.get('verdict')}  ⇒ 一致：{same}")
    for m in mismatches:
        print("   ✗ " + m)
    c1 = recomputed.get("C1") or {}
    print(f"C0 自检：{recomputed.get('C0_pipeline_selfcheck')}　"
          f"C1：{recomputed.get('C1_has_discriminative_power')}　"
          f"C2：{recomputed.get('C2_degenerate_guard')}")
    print(f"session 级符号检验：{c1.get('n_pos')}/{c1.get('n_eff')} 正，"
          f"p={c1.get('p')}，中位 Δ={c1.get('median_delta')}")

    # ---- C0 的失败分成「不适用」与「真失败」----
    na, real_fail = [], []
    for k, v in (recomputed.get("C0_detail") or {}).items():
        (na if (isinstance(v, str) and v == SKIP_MARK) else real_fail).append((k, v))
    print(f"\nC0：不适用（该 session 只有一个带 ⇒ 不进入分析）{len(na)} 个；"
          f"真失败（进入分析但某带解码不过）{len(real_fail)} 个")

    pair_rows = []
    for key, v in (recomputed.get("per_pair") or {}).items():
        pair_rows.append((key, v.get("n"), f"{v.get('median', float('nan')):+.4f}",
                          v.get("n_pos"), f"{v.get('p_sign', float('nan')):.4f}"))
    prim_rows = []
    for r in per_session:
        ps = r.get("pairs") or {}
        if "motor->prefrontal" in ps and "prefrontal->motor" in ps:
            dec = r.get("decode") or {}
            prim_rows.append((
                r.get("session", "").split("/")[-1],
                f"{r.get('primary_stat'):+.4f}" if r.get("primary_stat") is not None else "—",
                ps["motor->prefrontal"]["d_cross"], ps["motor->prefrontal"]["d_ctrl"],
                ps["motor->prefrontal"]["delta"],
                ps["prefrontal->motor"]["d_cross"], ps["prefrontal->motor"]["d_ctrl"],
                ps["prefrontal->motor"]["delta"],
                ", ".join(f"{b}:{dec[b]['acc']:.2f}/p{dec[b]['p']:.3f}" for b in sorted(dec)),
            ))
    prim_rows.sort(key=lambda x: x[1])

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_session": per_session,
        "per_pair": recomputed.get("per_pair"),
        "c0_not_applicable": [k for k, _ in na],
        "c0_real_failures": {k: v for k, v in real_fail},
        "note": "**鼠 ≠ 人**：本实验只回答「判据① 有没有判别力」，"
                "**不回填**「三个原语还是两个」（人 EEG 的问题）",
    }
    (OUT / "exp97b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp97b 结果重建（判据① 的**可测性自检 第二版**：鼠·Steinmetz 2019）\n")
    md.append(f"**预登记判决（Job 写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐 session 原始记录）**：`{recomputed.get('verdict')}` ⇒ "
              f"与 Job 一致：**{same}**\n")
    md.append(f"**C0 管线自检**：{recomputed.get('C0_pipeline_selfcheck')}　"
              f"**C1 判据① 有判别力**：{recomputed.get('C1_has_discriminative_power')}　"
              f"**C2 退化守卫**：{recomputed.get('C2_degenerate_guard')}\n")
    md.append(f"**session 级符号检验**：{c1.get('n_pos')}/{c1.get('n_eff')} 正，"
              f"p={c1.get('p')}，中位 Δ={c1.get('median_delta'):+.4f}（零差 {c1.get('n_zero')}）\n")
    md.append(f"**C0 失败分类**：不适用（该 session 只有一个带 ⇒ 不进入分析）**{len(na)}** 个；"
              f"真失败（进入分析但某带解码不过）**{len(real_fail)}** 个\n")
    md.append("## 进入主判据（CENT↔FRONT）的 session\n")
    md.append(fmt_table(prim_rows,
                        ["session", "统计量", "m→p 跨带", "m→p 同带", "m→p Δ",
                         "p→m 跨带", "p→m 同带", "p→m Δ", "各带解码 acc/p"]))
    md.append("\n## 所有可得方向的转分化（逐方向）\n")
    md.append(fmt_table(pair_rows, ["方向", "n session", "中位 Δ", "n 正", "符号检验 p"]))
    md.append("\n## 诚实边界\n")
    md.append("- **鼠 ≠ 人**：本实验只回答「**判据① 有没有判别力**」，"
              "**不回填**「三个原语还是两个」（人 EEG 的问题）。\n")
    md.append("- 移植引入的新环节（**共同空间** = 两带联合 PCA；**码本** = 行为类条件均值）⇒ "
              "靠**同带自对照**定标。\n")
    md.append("- **C0 判死 ⇒ 按预登记「全实验作废」** ⇒ C1 的数字**只作描述性**。\n")
    (OUT / "exp97b_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp97b_summary.json'} / {OUT / 'exp97b_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
