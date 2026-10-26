"""parse_exp90b.py — exp90b（NLB 线的**功能判据 ③**）的结果重建与**判决离线重算**。

## 做什么

exp90b 在一个 Job 里跑完了 NLB 线上的功能判据（③-CENT 伸手方向解码 / ③-FRONT 决策时程），
判决 JSON 落在桶里（`runs/illusionbrain-align/exp90b/exp90b_nlb_function.json`）。本脚本：

1. **取回 JSON** —— ⚠️ **桶的读通道不稳**（`hf buckets cp` 报 `xet-read-token`）⇒
   **三条通道依次试**：
   ① 本地已有 `results/exp90b-results/exp90b_nlb_function.json`；
   ② 从 `results/exp90b-results/job_log.txt` 里**抽出**（Job 在云端把文件内容回显到 stdout
      ⇒ 这条通道**不依赖桶的读权限**；本轮就是这么取回的）；
   ③ `hf buckets cp`（带重试）。
2. **用 exp90b 脚本里的 `decide()` 从逐 session 原始记录重算判决**
   （不依赖 Job 写下的 `verdict` 字段 ⇒ 判决可审计、可复现）；
3. 把逐 session 的解码/地板/差值、③-FRONT 的只报数字整理成表。

## 统计口径（与 exp89 / exp90 / exp96 逐字一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）。
- **零差剔除**：差值为 0（与地板持平）单独计数，**不参与**符号检验
  （否则「全 0」会被数成「全负」⇒ p 偏小 ⇒ 把「不成立」误判成「成立」）。
- **trial 级 McNemar 只作描述性**（trial 不独立 ⇒ p 偏乐观）⇒ **主证据用 session 级符号检验**。

## 输出

- `results/exp90b-results/exp90b_nlb_function.json`（原始）
- `results/exp90b-results/exp90b_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp90b-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp90b/exp90b_nlb_function.json"
RAW = OUT / "exp90b_nlb_function.json"
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
    # ② Job 日志回显（**不依赖桶的读权限**）
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
    # ③ 桶直取（读通道不稳 ⇒ 重试）
    for i in range(4):
        subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{REL}", str(RAW)],
                       capture_output=True, text=True)
        if RAW.exists() and RAW.stat().st_size > 0:
            print(f"✓ 取回 {RAW}（第 {i+1} 次）")
            return RAW
        print(f"  ✗ 第 {i+1} 次取回失败")
        time.sleep(6)
    raise RuntimeError("取不回 exp90b 的原始 JSON（日志里也没有）")


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
    per_session = data.get("per_session_cent") or []
    per_frong = data.get("per_session_frong") or []
    job_verdict = data.get("verdict") or {}

    import exp90b_nlb_function as E90B
    recomputed = E90B.decide(per_session)

    # ---- 判决比对（**判决字段逐个比**，McNemar 是描述性 ⇒ 离线拿不到 trial 级数组）----
    jc = job_verdict.get("C1_CENT_reachdir") or {}
    rc = recomputed.get("C1_CENT_reachdir") or {}
    jst = jc.get("sign_test") or {}
    rst = rc.get("sign_test") or {}
    checks = {
        "verdict": (job_verdict.get("verdict"), recomputed.get("verdict")),
        "C0_pipeline_selfcheck": (job_verdict.get("C0_pipeline_selfcheck"),
                                  recomputed.get("C0_pipeline_selfcheck")),
        "C1.pass": (jc.get("pass"), rc.get("pass")),
        "median_diff": (jc.get("median_diff"), rc.get("median_diff")),
        "sign_test.n_eff": (jst.get("n_eff"), rst.get("n_eff")),
        "sign_test.n_pos": (jst.get("n_pos"), rst.get("n_pos")),
        "sign_test.n_neg": (jst.get("n_neg"), rst.get("n_neg")),
        "sign_test.n_zero": (jst.get("n_zero"), rst.get("n_zero")),
        "sign_test.p": (jst.get("p"), rst.get("p")),
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
    print(f"理由：session 级符号检验 {rst.get('n_pos')}/{rst.get('n_eff')} 正 "
          f"p={rst.get('p')}；差值中位 {rc.get('median_diff'):+.4f}")

    cent_rows = [(s["session"], s["n_trials"], f"{s['acc']:.4f}",
                  f"{s['acc_floor']:.4f}", f"{s['diff']:+.4f}")
                 for s in per_session]
    frong_rows = []
    for s in per_frong:
        r = s.get("r_pred")
        rs = "nan" if (r is None or r != r) else f"{float(r):+.4f}"
        frong_rows.append((s["session"], s["n_trials"], rs,
                           f"{s['acc_earlylate']:.4f}", f"{s['acc_floor']:.4f}",
                           f"{s['diff']:+.4f}"))

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "mismatches": mismatches,
        "per_session_cent": per_session,
        "per_session_frong": per_frong,
        "sign_test": rst,
        "median_diff": rc.get("median_diff"),
        "C0_pipeline_selfcheck": recomputed.get("C0_pipeline_selfcheck"),
        "note_mcnemar": "trial 级 McNemar 是**描述性**的（trial 不独立 ⇒ p 偏乐观）"
                        "⇒ 主证据用 session 级符号检验；离线重算拿不到 trial 级数组 ⇒ 不重算",
        "note_morphology": "本实验是**功能判据（③）**；**形态判据（①）在现有数据上不可测**"
                           "⇒ 需要新数据（同一记录、多脑区、各有行为标签）⇒ 见 FINDINGS §三十四",
    }
    (OUT / "exp90b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp90b 结果重建（NLB 线的**功能判据 ③**）\n")
    md.append(f"**预登记判决（Job 写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐 session 原始记录）**：`{recomputed.get('verdict')}` ⇒ "
              f"与 Job 一致：**{same}**\n")
    md.append(f"**C0 管线自检**：{recomputed.get('C0_pipeline_selfcheck')}　"
              f"**C1 ③-CENT**：{rc.get('pass')}\n")
    md.append(f"**session 级符号检验**：{rst.get('n_pos')}/{rst.get('n_eff')} 正，"
              f"p={rst.get('p')}，零差 {rst.get('n_zero')}　"
              f"**差值中位**：{rc.get('median_diff'):+.4f}\n")
    md.append("## ③-CENT：逐 session 的伸手方向解码（跨 trial 5 折）\n")
    md.append(fmt_table(cent_rows, ["session", "trial 数", "解码准确率", "地板", "差值"]))
    md.append("\n## ③-FRONT：**只报，不作判决**（n=2 session ⇒ 证据极弱）\n")
    md.append(fmt_table(frong_rows,
                        ["session", "trial 数", "go_offset 预测 r", "早/晚准确率", "地板", "差值"]))
    md.append("\n## 两条诚实边界\n")
    md.append("- **trial 级 McNemar 只是描述性**（trial 不独立 ⇒ p 偏乐观）⇒ 主证据用 session 级符号检验。\n")
    md.append("- **形态判据（①）不在本实验里**：NLB 的两个区**不在同一记录** ⇒ 没有共同坐标系 ⇒ "
              "判据①**需要新数据**（同一记录、多脑区、各有行为标签）⇒ 见 FINDINGS §三十四。\n")
    (OUT / "exp90b_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"输出：{OUT / 'exp90b_summary.json'} / {OUT / 'exp90b_summary.md'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
