"""parse_exp94.py — exp94（骨干对照 × 交叉验证）的结果重建。

## 做什么

exp94 在一个 Job 里串行跑了两个变体（V0 = exp93b 现状骨干，V1 = 保留通道身份），
JSON 落在桶里。本脚本：

1. 从桶里取回 `exp94-trunk/exp94_trunk_ablation.json`（读通道不稳 ⇒ 重试）；
2. 把**交叉验证矩阵**压成一张表：每条线的 V0 / V1 / 配对差 / 符号检验；
3. 把**预登记判据**的判决原样带出（不改判据、不改数）。

## 统计口径（与本项目既有约定一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）⇒ 右偏分布会报高 30–40% ⇒ 同时给均值。
- 符号检验：双侧精确二项。
- **配对优先**：V0 与 V1 是**同一批被试、同一采样序列**训出来的 ⇒ 一律给**逐被试配对差**
  与符号检验，不只给两个中位数之差。

## 输出

- `results/exp94-results/exp94_trunk_ablation.json`（原始）
- `results/exp94-results/exp94_summary.{json,md}`（压好的表 + 判决）
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp94-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp94-trunk/exp94_trunk_ablation.json"

CHANCE = 1.0 / 200.0

# exp93b 已发表值（C7 自一致性锚点；**只用于量级检查**）
PUBLISHED_V0 = {"zeroshot_acc1": 0.0589, "zeroshot_acc1_featavg": 0.2300,
                 "rsa": 0.1273, "zeroshot_acc5": 0.1875}
TOL = {"zeroshot_acc1": 0.020, "zeroshot_acc1_featavg": 0.060,
       "rsa": 0.045, "zeroshot_acc5": 0.040}


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


def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else None


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


def variant_rows(ps, path, sub=None):
    """按路径取每个被试的值。path: list[str]"""
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
    data = fetch(force="--force" in __import__("sys").argv)
    per = data.get("per_subject") or {}

    # ⚠️ **判决重算**：Job 里跑的第一版 `decide()` 的 `paired()` 取值路径写错了
    #    （按顶层键取，而实际记录是嵌套的 `r["zeroshot"]["acc1_cos"]`）
    #    ⇒ 配对数被算成 0 ⇒ 「n_pos > n/2」判成 False ⇒ 第一版判决
    #    `H1_NOT_SUPPORTED` 是**实现 bug 的产物**，不是结论。
    # ⇒ 这里用**修好的** decide() 从逐被试原始数据重算，并**两个判决都留痕**。
    raw_dec = data.get("decision") or {}
    try:
        import exp94_trunk_ablation as E94
        dec = E94.decide(data.get("summaries") or {}, per, None)
        recomputed = True
    except Exception as e:  # pragma: no cover
        dec = raw_dec
        recomputed = False
        print(f"⚠️ 重算失败（{type(e).__name__}: {e}）⇒ 用 Job 里的原始判决")

    lines = []
    L = lines.append
    L("# exp94：骨干对照（V0 现状 vs V1 保留通道身份）× 交叉验证")
    L("")
    L("> **口径**：同一批被试、同一采样序列、同一目标（InfoNCE ↔ ViT-H-14）、")
    L("> 同一评测口径（P1 跨集 200-way 零样本检索）⇒ **只差 token 化方式**。")
    L("> **参数量匹配**：V1 不比 V0 大 ⇒ 性能差异**不可能**来自容量。")
    L("> **中位 = 上中位**；配对差用逐被试配对 + 双侧符号检验。")
    L("")
    if recomputed and raw_dec.get("verdict") != dec.get("verdict"):
        L(f"⚠️ **判决留痕**：Job 里第一版判决是 `{raw_dec.get('verdict')}`，"
          f"但那是 `paired()` **取值路径写错**（按顶层键取，而记录是嵌套的 "
          f"`r[\"zeroshot\"][\"acc1_cos\"]`）⇒ **配对数被算成 0** ⇒ "
          f"「n_pos > n/2」判成 False ⇒ **那是实现 bug 的产物，不是结论**。")
        L(f"⇒ 本表用**修好的** `decide()` 从逐被试原始数据**重算** ⇒ `{dec.get('verdict')}`。")
        L("")

    # ---------- C7 自一致性 ----------
    sc = dec.get("self_consistency") or {}
    L("## C7 自一致性（V0 复现 vs exp93b 已发表值）")
    L("")
    L("| 量 | exp93b 已发表 | 本实验 V0 | 容差 | 判定 |")
    L("|---|---:|---:|---:|---|")
    for k, v in sc.items():
        L(f"| {k} | {v['published']:.4f} | "
          f"{(v['got'] if v['got'] is not None else float('nan')):.4f} | "
          f"±{v['tol']} | {'✅' if v['ok'] else '❌'} |")
    L("")
    L(f"⇒ **{'通过' if dec.get('self_consistency_pass') else '未通过'}"
      f"**（未通过 ⇒ 本实验所有结论作废：实现漂移）。")
    L("")

    # ---------- 主表 ----------
    L("## 主表：两条读出线（200-way 零样本检索）+ RSA")
    L("")
    L("| 线 | V0（中位 / ×随机） | V1（中位 / ×随机） | 配对差中位 | V1>V0 的被试数 | 符号检验 p |")
    L("|---|---|---|---:|---:|---:|")
    rows = [
        ("C1 单试次 top-1", "zeroshot_acc1", ["zeroshot", "acc1_cos"]),
        ("C2 试次平均 top-1", "zeroshot_acc1_featavg", ["zeroshot", "acc1_featavg_cos"]),
        ("C3 RSA", "rsa", ["rsa", "r"]),
    ]
    summ = data.get("summaries") or {}
    for label, key, path in rows:
        a = variant_rows(per.get("v0", []), path)
        b = variant_rows(per.get("v1", []), path)
        common = sorted(set(a) & set(b))
        diffs = [b[s] - a[s] for s in common]
        k, m, p = sign_test_p(diffs)
        v0m = summ.get("v0", {}).get(key)
        v1m = summ.get("v1", {}).get(key)
        r0 = f"{v0m:.4f} / {v0m/CHANCE:.1f}×" if v0m is not None else "—"
        r1 = f"{v1m:.4f} / {v1m/CHANCE:.1f}×" if v1m is not None else "—"
        L(f"| {label} | {r0} | {r1} | {med(diffs) if diffs else float('nan'):+.4f} | "
          f"{k}/{len(common)} | {p:.4f} |")
    L("")

    # ---------- 位置轴交叉验证 ----------
    pa = dec.get("posaxis") or {}
    L("## C5/C6 位置轴 × 地板：**两个骨干上是否给出同向结论**")
    L("")
    L("| 骨干 | POST>全部随机34 | ANT<全部随机29 | POST−ANT（中位） | POST acc1 | ANT acc1 | POST RSA | n |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in ("v0", "v1"):
        d = pa.get(v)
        if not d:
            L(f"| {v} | — | — | — | — | — | — | — |")
            continue
        L(f"| {v} | {d['post_above_all_rand34']}/{d['n']} | "
          f"{d['ant_below_all_rand29']}/{d['n']} | "
          f"{d['post_minus_ant_median']:+.4f} | "
          f"{d['post_acc1_median']:.4f} | {d['ant_acc1_median']:.4f} | "
          f"{(d['post_rsa_median'] if d['post_rsa_median'] is not None else float('nan')):+.4f} | "
          f"{d['n']} |")
    L("")
    L(f"⇒ H2（位置轴结论的**骨干无关性**）：**{dec.get('H2_note', '—')}**")
    L("")

    # ---------- 判决 ----------
    L("## 预登记判据的判决")
    L("")
    L(f"- **H1（通道身份被抹掉是组件缺陷）**："
      f"`{dec.get('H1_channel_identity_helps')}`")
    L(f"- **H2（位置轴结论不是单一骨干的产物）**："
      f"`{dec.get('H2_posaxis_backbone_independent')}`")
    L(f"- **R4（读出方向一致）**：`{dec.get('readout_consistent')}` —— {dec.get('R4_note', '')}")
    L(f"- **总判决**：`{dec.get('verdict')}`")
    L("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp94_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "exp94_summary.json").write_text(
        json.dumps({"decision": dec, "decision_recomputed": recomputed,
                    "decision_raw_job": raw_dec.get("verdict"),
                    "summaries": summ, "self_consistency": sc,
                    "published_v0": PUBLISHED_V0, "tol": TOL},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {OUT/'exp94_summary.md'}")


if __name__ == "__main__":
    main()
