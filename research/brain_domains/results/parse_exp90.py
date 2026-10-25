"""parse_exp90.py — exp90（位置轴形态判据）的结果重建与**判决离线重算**。

## 做什么

exp90 在一个 Job 里训一个模型（三个原语分支）并评测 10 个被试，JSON 落在桶里
（`runs/illusionbrain-align/exp90/exp90_primitives.json`）。本脚本：

1. 取回 JSON（读通道不稳 ⇒ 重试）；
2. **用 exp90 脚本里的 `decide()` 从逐被试原始记录重算判决**
   （不依赖 Job 里写下的 `verdict` 字段 ⇒ 判决可审计、可复现）；
3. 把 3×3 转分化矩阵、接口切断代价、逐分支图像 RSA、低秩诊断整理成表；
4. 给出**与 exp94 的数值对照**（**只报告，不作判据** —— exp90 是新实现且
   有意加入了脑科学组件 ⇒ 数值本就不该逐位复现）。

## 统计口径

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）—— 与 exp89/exp94/exp96 逐字一致。
- **零差剔除**：差值为 0（与地板持平）单独计数，不参与符号检验
  （否则「10 个全 0」会被数成「全负」⇒ p=0.002 ⇒ 把「不成立」误判成「成立」）。

## 输出

- `results/exp90-results/exp90_primitives.json`（原始）
- `results/exp90-results/exp90_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp90-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp90/exp90_primitives.json"
RAW = OUT / "exp90_primitives.json"


def fetch(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not force:
        return RAW
    for i in range(4):
        r = subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{REL}", str(RAW)],
                           capture_output=True, text=True)
        if RAW.exists() and RAW.stat().st_size > 0:
            print(f"✓ 取回 {RAW}（{RAW.stat().st_size} 字节，第 {i+1} 次）")
            return RAW
        print(f"  ✗ 第 {i+1} 次取回失败：{(r.stderr or r.stdout)[:120]}")
        time.sleep(6)
    raise RuntimeError("取不回 exp90 的原始 JSON")


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
    records = data["per_subject"]
    job_verdict = data.get("verdict", {})

    import exp90_primitives as E90
    recomputed = E90.decide(records)
    same = (recomputed["verdict"] == job_verdict.get("verdict"))
    print(f"\n预登记判决（Job 写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐被试原始）  : {recomputed['verdict']}  ⇒ 一致：{same}")
    print(f"理由：{recomputed.get('why', '')}")

    ok = [r for r in records if "error" not in r]

    # ---- 3×3 转分化矩阵（逐被试中位）----
    mat_rows = []
    for A in E90.GROUPS:
        for B in E90.GROUPS:
            if A == B:
                continue
            vals = [r["transdiff"][f"{A}->{B}"] for r in ok
                    if "transdiff" in r and f"{A}->{B}" in r["transdiff"]]
            if not vals:
                continue
            v = recomputed.get("C2", {}).get(f"{A}->{B}", {})
            mat_rows.append((f"{A}→{B}", len(vals), f"{E90.med_up(vals):+.4f}",
                             f"{v.get('n_pos', '-')}/{v.get('n_eff', '-')} 正",
                             f"{v.get('p_sign', float('nan')):.4f}",
                             "✅" if v.get("pass") else "—"))
    # ---- 接口切断代价 ----
    iface_rows = []
    for key in sorted({k for r in ok for k in r.get("iface", {})}):
        vals = [r["iface"][key] for r in ok if key in r.get("iface", {})]
        v = recomputed.get("C4", {}).get(key, {})
        iface_rows.append((key, len(vals), f"{E90.med_up(vals):+.4f}",
                           f"{v.get('n_pos', '-')}/{v.get('n_eff', '-')} 正",
                           f"{v.get('p_sign', float('nan')):.4f}",
                           "✅ 有价值" if v.get("has_value") else "❌ 砍掉"))
    # ---- 逐分支图像 RSA ----
    rsa_rows = []
    for g in E90.GROUPS:
        vals = [r["img_rsa"][g]["r"] for r in ok if "img_rsa" in r and g in r["img_rsa"]]
        if vals:
            rsa_rows.append((g, len(vals), f"{E90.med_up(vals):+.4f}",
                             sum(1 for r in ok if "img_rsa" in r
                                 and g in r["img_rsa"]
                                 and r["img_rsa"][g]["p"] < 0.05)))
    # ---- 低秩诊断（只报）----
    diag_rows = []
    for g in E90.GROUPS:
        vals = [r["lowrank_diag"][g] for r in ok if "lowrank_diag" in r]
        if vals:
            diag_rows.append((g, len(vals), f"{E90.med_up(vals):.4f}"))

    summary = {
        "job_verdict": job_verdict,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "per_backbone": recomputed.get("C2"),
        "criterion_C0": recomputed.get("C0"),
        "criterion_C1": recomputed.get("C1"),
        "criterion_C3": recomputed.get("C3"),
        "criterion_C4": recomputed.get("C4"),
        "n_primitives": recomputed.get("n_primitives"),
        "interfaces_keep": recomputed.get("interfaces_keep"),
        "interfaces_drop": recomputed.get("interfaces_drop"),
        "comparable_to_exp94": recomputed.get("comparable_to_exp94"),
    }
    (OUT / "exp90_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = []
    md.append("# exp90 结果重建（位置轴形态判据：3×3 转分化 + 三条接口）\n")
    md.append(f"**预登记判决（Job 写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐被试原始记录）**：`{recomputed['verdict']}` ⇒ "
              f"与 Job 一致：**{same}**\n")
    md.append(f"**理由**：{recomputed.get('why', '')}\n")
    md.append(f"**原语个数**：{recomputed.get('n_primitives')}\n")
    md.append("## 3×3 转分化矩阵（d(A→B)：用 A 的原型集解释 B 的潜变量的残差增量）\n")
    md.append(fmt_table(mat_rows, ["方向", "n", "中位 d", "正数", "p", "敏感"]))
    md.append("\n## 接口切断代价（切断后该分支的验证残差增量）\n")
    md.append(fmt_table(iface_rows, ["接口|分支", "n", "中位", "正数", "p", "结论"]))
    md.append("\n## 逐分支的图像 RSA（位置轴第三次确认）\n")
    md.append(fmt_table(rsa_rows, ["组", "n", "r 中位", "p<0.05 的被试数"]))
    md.append("\n## 低秩诊断（只报，不作判据：低秩本身不是证据）\n")
    md.append(fmt_table(diag_rows, ["组", "n", "重建误差中位"]))
    md.append("\n## 与 exp94 的数值对照（**只报告，不作判据**）\n")
    md.append("```json\n" + json.dumps(recomputed.get("comparable_to_exp94", {}),
                                          indent=2, ensure_ascii=False) + "\n```\n")
    (OUT / "exp90_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n=== 原语个数 ===", recomputed.get("n_primitives"))
    print("=== 接口去留 ===")
    print("  保留：", recomputed.get("interfaces_keep"))
    print("  砍掉：", recomputed.get("interfaces_drop"))
    print(f"输出：{OUT/'exp90_summary.json'} / {OUT/'exp90_summary.md'}")


if __name__ == "__main__":
    main()
