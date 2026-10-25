"""parse_exp95.py — exp95（合成版 V2 vs V1）的结果重建。

## 做什么

exp95 在一个 Job 里串行跑了 **V1（复现）** 与 **V2（合成版：保留通道身份 + 恢复时间粒度）**，
JSON 落在桶里。本脚本：

1. 从桶里取回 `exp95-hybrid/exp95_trunk_hybrid.json`（读通道不稳 ⇒ 重试）；
2. 把**三向交叉验证**压成一张表：
   - **C7' ①** V1 复现 vs exp93b 已发表值（实现漂移检查）
   - **C7' ②** V1 复现 vs exp94 的 V1（**Job 间漂移**的量化）
   - **主表** V2 vs V1 的逐被试配对差 + 符号检验（三条读出线）
   - **位置轴** V1 / V2 各自的 POST/ANT 地板计数
3. 把**预登记判据**的判决原样带出（不改判据、不改数）。

## 统计口径

- **中位 = 上中位**（`sorted[n//2]`）。
- **配对优先**：V1 与 V2 同一批被试、同一采样序列训出 ⇒ 一律给逐被试配对差 + 符号检验。
- 判决**用修好的 `decide()` 从逐被试原始数据重算**（不依赖 Job 里写下的判决字段）。

## 输出

- `results/exp95-results/exp95_trunk_hybrid.json`（原始）
- `results/exp95-results/exp95_summary.{json,md}`
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp95-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp95-hybrid/exp95_trunk_hybrid.json"

CHANCE = 1.0 / 200.0


def sign_test_p(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    p = 2.0 * sum(math.comb(n, i) for i in range(min(k, m) + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


def med(xs):
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return float(s[len(s) // 2])


def fetch(rel=REL, retries=6, wait=6, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / (rel.replace("/", "__").removesuffix(".json") + ".json")
    if (not force) and dst.exists() and dst.stat().st_size > 0:
        return json.loads(dst.read_text(encoding="utf-8"))
    src = f"{BUCKET}/{rel}"
    for i in range(retries):
        try:
            r = subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                               capture_output=True, text=True, timeout=420)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                print(f"✓ 取回 {rel}")
                return json.loads(dst.read_text(encoding="utf-8"))
            print(f"  重试 {i+1}/{retries}：{r.stderr.strip()[:160]}")
        except Exception as e:
            print(f"  重试 {i+1}/{retries}：{type(e).__name__}: {e}")
        time.sleep(wait)
    raise SystemExit(f"✗ 取不回 {rel}（桶里还没有？）")


def variant_rows(ps, path):
    out = {}
    for r in ps:
        if "error" in r:
            continue
        cur = r
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and cur is not None:
            out[r["subject"]] = float(cur)
    return out


def main():
    data = fetch(force="--force" in sys.argv)
    per = data.get("per_subject") or {}
    raw_dec = data.get("decision") or {}

    # 判决**重算**（用修好的 decide()，不依赖 Job 写下的字段）
    try:
        import exp95_trunk_hybrid as E95
        dec = E95.decide(data.get("summaries") or {}, per, None)
        recomputed = True
    except Exception as e:  # pragma: no cover
        dec = raw_dec
        recomputed = False
        print(f"⚠️ 重算失败（{type(e).__name__}: {e}）⇒ 用 Job 里的原始判决")

    summ = data.get("summaries") or {}
    lines = []
    L = lines.append
    L("# exp95：合成版 V2（保留通道身份 + 恢复时间粒度）vs V1（exp94 的赢家）")
    L("")
    L("> **口径**：同一批被试、同一采样序列、同一目标、同一评测口径 ⇒ **只差「每通道几个时间 token」**。")
    L("> **参数量匹配**：V1 1,582,327 vs V2 1,592,891（比值 1.007）⇒ 差异不可能来自容量。")
    L("> **中位 = 上中位**；配对差用逐被试配对 + 双侧符号检验。")
    L("")

    # ---------- C7' 两重自一致性 ----------
    L("## C7' 自一致性（两重）")
    L("")
    L("### ① V1 复现 vs exp93b 已发表值（实现漂移）")
    L("")
    L("| 量 | exp93b 已发表 | 本实验 V1 | 容差 | 判定 |")
    L("|---|---:|---:|---:|---|")
    for k, v in (dec.get("self_consistency_exp93b") or {}).items():
        L(f"| {k} | {v['published']:.4f} | "
          f"{(v['got'] if v['got'] is not None else float('nan')):.4f} | "
          f"±{v['tol']} | {'✅' if v['ok'] else '❌'} |")
    L("")
    L("### ② V1 复现 vs exp94 的 V1（**两次 Job 之间的漂移**）")
    L("")
    L("| 量 | exp94 的 V1 | 本实验 V1 | 容差 | 判定 |")
    L("|---|---:|---:|---:|---|")
    for k, v in (dec.get("job_drift_vs_exp94") or {}).items():
        L(f"| {k} | {v['exp94']:.4f} | "
          f"{(v['got'] if v['got'] is not None else float('nan')):.4f} | "
          f"±{v['tol']} | {'✅' if v['ok'] else '❌'} |")
    L("")
    L(f"⇒ **{'通过' if dec.get('self_consistency_pass') else '未通过'}"
      f"**（未通过 ⇒ 本实验结论作废）。")
    L("")

    # ---------- 主表 ----------
    L("## 主表：V2 − V1 的逐被试配对差")
    L("")
    L("| 线 | V1（中位 / ×随机） | V2（中位 / ×随机） | 配对差中位 | V2>V1 的被试数 | 符号检验 p |")
    L("|---|---|---|---:|---:|---:|")
    pr = dec.get("paired_v2_minus_v1") or {}
    for key, path in (("zeroshot_acc1", ["zeroshot", "acc1_cos"]),
                      ("zeroshot_acc1_featavg", ["zeroshot", "acc1_featavg_cos"]),
                      ("rsa", ["rsa", "r"])):
        a = variant_rows(per.get("v1", []), path)
        b = variant_rows(per.get("v2", []), path)
        common = sorted(set(a) & set(b))
        diffs = [b[s] - a[s] for s in common]
        k, m, p = sign_test_p(diffs)
        v1m = (summ.get("v1") or {}).get(key)
        v2m = (summ.get("v2") or {}).get(key)
        r1 = f"{v1m:.4f} / {v1m/CHANCE:.1f}×" if v1m is not None else "—"
        r2 = f"{v2m:.4f} / {v2m/CHANCE:.1f}×" if v2m is not None else "—"
        L(f"| {key} | {r1} | {r2} | {med(diffs) if diffs else float('nan'):+.4f} | "
          f"{k}/{len(common)} | {p:.4f} |")
    L("")

    # ---------- 位置轴 ----------
    L("## 位置轴 × 地板：合成版有没有破坏位置轴")
    L("")
    L("| 骨干 | POST>全部随机34 | ANT<全部随机29 | POST−ANT（中位） | POST acc1 | ANT acc1 | n |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    pa = dec.get("posaxis") or {}
    for v, lab in (("v1", "V1（exp94 的赢家）"), ("v2", "V2（合成版）")):
        d = pa.get(v)
        if not d:
            L(f"| {lab} | — | — | — | — | — | — |")
            continue
        L(f"| {lab} | {d['post_above_all_rand34']}/{d['n']} | "
          f"{d['ant_below_all_rand29']}/{d['n']} | {d['post_minus_ant_median']:+.4f} | "
          f"{d['post_acc1_median']:.4f} | {d['ant_acc1_median']:.4f} | {d['n']} |")
    L("")

    # ---------- 判决 ----------
    L("## 预登记判据的判决")
    L("")
    L(f"- **H1'（合成版更好）**：`{dec.get('H1_hybrid_better')}`")
    L(f"- **H2'（位置轴不反向）**：`{dec.get('H2_posaxis_not_worse')}`")
    L(f"- **R4（读出方向一致）**：`{dec.get('readout_consistent')}` —— {dec.get('R4_note', '')}")
    L(f"- **总判决**：`{dec.get('verdict')}`")
    L("")
    L("**判决含义**：`HYBRID_BETTER` ⇒ 用 V2（合成版）；`HYBRID_REDUNDANT` ⇒ "
      "**剃刀砍掉 V2，用 V1**（并把「V1 的增益归因于通道身份」写进台账）；"
      "`HYBRID_WORSE` ⇒ 用 V1。")
    L("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp95_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "exp95_summary.json").write_text(
        json.dumps({"decision": dec, "decision_recomputed": recomputed,
                    "decision_raw_job": raw_dec.get("verdict"),
                    "summaries": summ},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {OUT/'exp95_summary.md'}")


if __name__ == "__main__":
    main()
