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
# ⚠️ exp95 的正式 Job 被**账户额度耗尽**中断（V1 跑完、V2 训练到 epoch 2 被平台取消）
#    ⇒ 结果**按变体分开落盘** ⇒ 这里逐变体取回再合并（一条命令就能在补跑后出表）
RELS = {"v1": "exp95-hybrid/v1/exp95_v1.json",
        "v2": "exp95-hybrid/v2/exp95_v2.json"}

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
    # 逐变体取回并合并（**V1 与 V2 可能是两个 Job 跑的** ⇒ 不依赖合并 JSON）
    per, summ, raw_dec = {}, {}, {}
    missing = []
    for v, rel in RELS.items():
        try:
            d = fetch(rel, force="--force" in sys.argv)
        except SystemExit:
            missing.append(v)
            continue
        per[v] = d.get("per_subject") or []
        summ[v] = d.get("summary") or {}
        if v == "v1":
            raw_dec = d.get("decision") or {}
    if not per:
        raise SystemExit("✗ 一个变体都没取到")
    if missing:
        print(f"⚠️ 缺变体 {missing} ⇒ 判决不可出（只出已有部分）")

    # 判决**重算**（用修好的 decide()，不依赖 Job 写下的字段）
    dec = {}
    recomputed = False
    if "v1" in per and "v2" in per:
        try:
            import exp95_trunk_hybrid as E95
            dec = E95.decide(summ, per, None)
            recomputed = True
        except Exception as e:  # pragma: no cover
            dec = raw_dec
            print(f"⚠️ 重算失败（{type(e).__name__}: {e}）⇒ 用 Job 里的原始判决")
    else:
        dec = raw_dec
        print("⚠️ 只有部分变体 ⇒ 不出判决（缺 H1'/H2' 所需的一侧）")

    lines = []
    L = lines.append
    L("# exp95：合成版 V2（保留通道身份 + 恢复时间粒度）vs V1（exp94 的赢家）")
    L("")
    L("> **口径**：同一批被试、同一采样序列、同一目标、同一评测口径 ⇒ **只差「每通道几个时间 token」**。")
    L("> **参数量匹配**：V1 1,582,327 vs V2 1,592,891（比值 1.007）⇒ 差异不可能来自容量。")
    L("> **中位 = 上中位**；配对差用逐被试配对 + 双侧符号检验。")
    L("")

    # ---------- C7' 两重自一致性（**只依赖 V1** ⇒ 缺 V2 也能算）----------
    L("## C7' 自一致性（两重）")
    L("")
    L("### ① V1 复现 vs exp93b 已发表值（实现漂移）")
    L("")
    L("| 量 | exp93b 已发表 | 本实验 V1 | 容差 | 判定 |")
    L("|---|---:|---:|---:|---|")
    PUB93B = {"zeroshot_acc1": (0.0589, 0.020), "zeroshot_acc1_featavg": (0.2300, 0.060),
              "rsa": (0.1273, 0.045), "zeroshot_acc5": (0.1875, 0.040)}
    PUB94V1 = {"zeroshot_acc1": (0.0626875, 0.010), "zeroshot_acc1_featavg": (0.215, 0.030),
               "rsa": (0.11921705748210504, 0.020), "zeroshot_acc5": (0.1943125, 0.025)}
    v1s = summ.get("v1") or {}
    sc_ok, drift_ok = True, True
    for k, (pub, tol) in PUB93B.items():
        got = v1s.get(k)
        ok = got is not None and abs(got - pub) <= tol
        sc_ok = sc_ok and ok
        L(f"| {k} | {pub:.4f} | "
          f"{(got if got is not None else float('nan')):.4f} | ±{tol} | {'✅' if ok else '❌'} |")
    L("")
    L("### ② V1 复现 vs exp94 的 V1（**两次 Job 之间的漂移**）")
    L("")
    L("| 量 | exp94 的 V1 | 本实验 V1 | 容差 | 判定 |")
    L("|---|---:|---:|---:|---|")
    for k, (pub, tol) in PUB94V1.items():
        got = v1s.get(k)
        ok = got is not None and abs(got - pub) <= tol
        drift_ok = drift_ok and ok
        L(f"| {k} | {pub:.4f} | "
          f"{(got if got is not None else float('nan')):.4f} | ±{tol} | {'✅' if ok else '❌'} |")
    L("")
    L(f"⇒ **{'通过' if (sc_ok and drift_ok) else '未通过'}"
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

    # ---------- 位置轴（**直接从逐被试数据算** ⇒ 缺 V2 也能出）----------
    L("## 位置轴 × 地板：合成版有没有破坏位置轴")
    L("")
    L("| 骨干 | POST>全部随机34 | ANT<全部随机29 | POST−ANT（中位） | POST acc1 | ANT acc1 | n |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    pa_by_v = {}
    for v, lab in (("v1", "V1（exp94 的赢家）"), ("v2", "V2（合成版）")):
        recs = [r for r in per.get(v, []) if "error" not in r and "posaxis" in r]
        if not recs:
            L(f"| {lab} | — | — | — | — | — | — |")
            continue
        up = sum(1 for r in recs if r["posaxis"]["verdict"]["post_above_all_rand34"])
        dn = sum(1 for r in recs if r["posaxis"]["verdict"]["ant_below_all_rand29"])
        pa = {"post_above_all_rand34": up, "ant_below_all_rand29": dn,
              "post_minus_ant_median": med(
                  [r["posaxis"]["verdict"]["post_minus_ant"] for r in recs]),
              "post_acc1_median": med(
                  [r["posaxis"]["groups"]["POST"]["acc1"] for r in recs]),
              "ant_acc1_median": med(
                  [r["posaxis"]["groups"]["ANT"]["acc1"] for r in recs]),
              "n": len(recs)}
        # ⚠️ **按变体分别存**（早先写成一个变量 ⇒ 被后一个变体覆盖 ⇒ 存进去的是 V2 的数）
        pa_by_v[v] = pa
        L(f"| {lab} | {up}/{len(recs)} | {dn}/{len(recs)} | "
          f"{pa['post_minus_ant_median']:+.4f} | "
          f"{pa['post_acc1_median']:.4f} | "
          f"{pa['ant_acc1_median']:.4f} | {len(recs)} |")
    L("")

    # ---------- 判决 ----------
    L("## 预登记判据的判决")
    L("")
    if "v1" in per and "v2" in per:
        L(f"- **H1'（合成版更好）**：`{dec.get('H1_hybrid_better')}`")
        L(f"- **H2'（位置轴不反向）**：`{dec.get('H2_posaxis_not_worse')}`")
        L(f"- **R4（读出方向一致）**：`{dec.get('readout_consistent')}` —— {dec.get('R4_note', '')}")
        L(f"- **总判决**：`{dec.get('verdict')}`")
        L("")
        L("**判决含义**：`HYBRID_BETTER` ⇒ 用 V2（合成版）；`HYBRID_REDUNDANT` ⇒ "
          "**剃刀砍掉 V2，用 V1**（并把「V1 的增益归因于通道身份」写进台账）；"
          "`HYBRID_WORSE` ⇒ 用 V1。")
    else:
        L(f"- ⚠️ **判决不可出**：缺变体 {missing} ⇒ H1'/H2' 需要 V1 与 V2 两侧。")
        L("- ⇒ **exp95 未结案**；已有的 V1 复现与上面的 C7' 漂移量化**仍然有效**。")
    L("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp95_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "exp95_summary.json").write_text(
        json.dumps({"decision": dec, "decision_recomputed": recomputed,
                    "decision_raw_job": raw_dec.get("verdict"),
                    "summaries": summ, "missing_variants": missing,
                    "self_consistency_exp93b": {
                        k: {"published": p, "got": v1s.get(k), "tol": t,
                            "ok": bool(v1s.get(k) is not None
                                       and abs(v1s[k] - p) <= t)}
                        for k, (p, t) in PUB93B.items()},
                    "job_drift_vs_exp94": {
                        k: {"exp94": p, "got": v1s.get(k), "tol": t,
                            "ok": bool(v1s.get(k) is not None
                                       and abs(v1s[k] - p) <= t)}
                        for k, (p, t) in PUB94V1.items()},
                    "posaxis_v1": pa_by_v.get("v1", {}),
                    "posaxis_v2": pa_by_v.get("v2", {}),
                    },
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {OUT/'exp95_summary.md'}")


if __name__ == "__main__":
    main()
