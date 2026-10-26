"""parse_exp91.py — exp91（判据① 的**可测性自检**，鼠·Steinmetz 2019）的结果重建与**判决离线重算**。

## 做什么

exp91 在一个 Job 里跑完了「判据① 在同记录多区数据上有没有判别力」的移植版，判决 JSON 落在桶里
（`runs/illusionbrain-align/exp91/exp91_transdiff_mouse.json`）。本脚本：

1. **取回 JSON** —— ⚠️ **桶的读通道不稳**（`hf buckets cp` 报 `xet-read-token`）⇒
   **三条通道依次试**：① 本地已有；② 从 `results/exp91-results/job_log.txt` 里**抽出**
   （Job 在云端把文件内容回显到 stdout ⇒ 不依赖桶的读权限）；③ `hf buckets cp`（带重试）。
2. **用 exp91 脚本里的 `decide()` 从逐 session 原始记录重算判决**
   （不依赖 Job 写下的 `verdict` 字段 ⇒ 判决可审计、可复现）。
3. 把逐 session 的跨带/同带代价、可解码准确率、3×3 矩阵整理成表。

## 统计口径（与 exp89 / exp90 / exp90b / exp96 逐字一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
- **零差剔除**：差值为 0 单独计数，**不参与**符号检验。
- **统计单元 = session**（trial 不独立 ⇒ trial 级检验一律只作描述性）。

## 输出

- `results/exp91-results/exp91_transdiff_mouse.json`（原始）
- `results/exp91-results/exp91_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp91-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp91/exp91_transdiff_mouse.json"
RAW = OUT / "exp91_transdiff_mouse.json"
LOG = OUT / "job_log.txt"


def _extract_from_log(log_path: Path):
    """从 Job 日志里抽出判决 JSON（Job 把文件内容回显到 stdout）。"""
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "{":
            start = i
            break
    if start is None:
        return None
    end = None
    for j in range(len(lines) - 1, start - 1, -1):
        if lines[j].strip() == "}":
            end = j
            break
    if end is None:
        return None
    return "\n".join(lines[start:end + 1]) + "\n"


def fetch(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not force:
        return RAW
    txt = _extract_from_log(LOG)
    if txt:
        try:
            json.loads(txt)
        except Exception:
            txt = None
    if txt:
        RAW.write_text(txt, encoding="utf-8")
        print(f"✓ 从 Job 日志抽出 {RAW}（{RAW.stat().st_size} 字节）")
        return RAW
    for i in range(4):
        subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{REL}", str(RAW)],
                       capture_output=True, text=True)
        if RAW.exists() and RAW.stat().st_size > 0:
            print(f"✓ 取回 {RAW}（第 {i+1} 次）")
            return RAW
        print(f"  ✗ 第 {i+1} 次取回失败")
        time.sleep(6)
    raise RuntimeError("取不回 exp91 的原始 JSON（日志里也没有）")


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

    import exp91_transdiff_mouse as E91
    recomputed = E91.decide(per_session)

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
        "C1.p": ((job_verdict.get("C1") or {}).get("p"),
                 (recomputed.get("C1") or {}).get("p")),
        "C1.median_delta": ((job_verdict.get("C1") or {}).get("median_delta"),
                            (recomputed.get("C1") or {}).get("median_delta")),
    }
    mismatches = []
    for k, (a, b) in checks.items():
        if isinstance(a, float) or isinstance(b, float):
            ok = (a is not None and b is not None and abs(float(a) - float(b)) < 1e-9)
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
          f"C1 判别力：{recomputed.get('C1_has_discriminative_power')}　"
          f"C2 退化守卫：{recomputed.get('C2_degenerate_guard')}")
    print(f"session 级符号检验：{c1.get('n_pos')}/{c1.get('n_eff')} 正，p={c1.get('p')}，"
          f"中位 Δ={c1.get('median_delta')}")

    # ---- 表 ----
    pair_rows = []
    for key, v in (recomputed.get("per_pair") or {}).items():
        pair_rows.append((key, v.get("n"), f"{v.get('median', float('nan')):+.4f}"))
    dec_rows = []
    for r in per_session:
        for b, v in (r.get("decode") or {}).items():
            acc, p = v.get("acc", float("nan")), v.get("p", float("nan"))
            dec_rows.append((r.get("session", "")[-28:], b,
                             f"{acc:.3f}" if acc == acc else "nan",
                             f"{p:.4f}" if p == p else "nan"))
    sess_rows = []
    for r in per_session:
        ps = r.get("pairs") or {}
        best = max(ps.items(), key=lambda kv: kv[1]["delta"]) if ps else None
        sess_rows.append((r.get("session", "")[-28:],
                           str(r.get("bands")),
                           r.get("n_train"), r.get("n_test"),
                           (f"{best[1]['delta']:+.4f}" if best else "—"),
                           (best[0] if best else "—")))

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_session": per_session,
        "per_pair": recomputed.get("per_pair"),
        "note": "**鼠 ≠ 人**：本实验只回答「判据① 有没有判别力」，"
                "**不回填**「三个原语还是两个」（人 EEG 的问题）",
    }
    (OUT / "exp91_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp91 结果重建（判据① 的**可测性自检**：鼠·Steinmetz 2019）\n")
    md.append(f"**预登记判决（Job 写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐 session 原始记录）**：`{recomputed.get('verdict')}` ⇒ "
              f"与 Job 一致：**{same}**\n")
    md.append(f"**C0 管线自检**：{recomputed.get('C0_pipeline_selfcheck')}　"
              f"**C1 判据① 有判别力**：{recomputed.get('C1_has_discriminative_power')}　"
              f"**C2 退化守卫**：{recomputed.get('C2_degenerate_guard')}\n")
    md.append(f"**session 级符号检验**：{c1.get('n_pos')}/{c1.get('n_eff')} 正，"
              f"p={c1.get('p')}，中位 Δ={c1.get('median_delta'):+.4f}（零差 {c1.get('n_zero')}）\n")
    md.append("## 3×3 转分化：逐方向（跨带 − 同带自对照）的中位 Δ\n")
    md.append(fmt_table(pair_rows, ["方向", "n session", "中位 Δ"]))
    md.append("\n## 逐 session 概览\n")
    md.append(fmt_table(sess_rows, ["session", "三带单元数", "train", "test", "最大 Δ", "方向"]))
    md.append("\n## 每带的行为可解码性（C0 的可读输出检查）\n")
    md.append(fmt_table(dec_rows, ["session", "带", "解码准确率", "置换 p"]))
    md.append("\n## 诚实边界\n")
    md.append("- **鼠 ≠ 人**：本实验只回答「**判据① 有没有判别力**」，"
              "**不回填**「三个原语还是两个」（人 EEG 的问题）。\n")
    md.append("- 移植引入的新环节（**共同空间** = 两带联合 PCA）⇒ 靠**同带自对照**定标。\n")
    md.append("- 联合 PCA 本身让不同带**更可比** ⇒ 方向**保守**。\n")
    (OUT / "exp91_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp91_summary.json'} / {OUT / 'exp91_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
