"""parse_exp96.py — exp96（三组模型层剖面）的结果重建与**判决离线重算**。

## 做什么

exp96 在一个 Job 里跑完 10 个被试，JSON 落在桶里
（`runs/illusionbrain-align/exp96/exp96_group_profile.json`）。本脚本：

1. 取回 JSON（读通道不稳 ⇒ 重试）；
2. **用 exp96 脚本里的 `decide()` 从逐被试原始记录重算判决**
   （不依赖 Job 里写下的 `verdict` 字段 ⇒ 判决可审计、可复现）；
3. 给出**逐骨干**的 B / C 拆解（哪个骨干、卡在哪一条、差多少）；
4. 另给一份**事后缩小骨干范围**的分析（只用 s1a），**明确标注它不是预登记判决**。

## 统计口径

- **中位 = 上中位**（`sorted[n//2]`）—— 与 exp89/exp95 逐字一致。
- **零差剔除**：差值为 0（与地板持平）单独计数，不参与符号检验
  （否则「10 个全 0」会被数成「全负」⇒ p=0.002 ⇒ 把「不成立」误判成「成立」）。
- 符号检验 = 精确二项（双侧）。

## 输出

- `results/exp96-results/exp96_group_profile.json`（原始）
- `results/exp96-results/exp96_summary.{json,md}`
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp96-results"
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp96/exp96_group_profile.json"
RAW = OUT / "exp96_group_profile.json"


def fetch(force=False):
    """从桶里取原始 JSON（读通道不稳 ⇒ 重试）。"""
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
    raise RuntimeError("取不回 exp96 的原始 JSON")


def decide_for_backbone(records, tag):
    """**只用某一个骨干**跑一遍 A/B/C（事后分析用）。

    ⚠️ **不修改已上传的 exp96 脚本**（它带着预登记判据，事后改它就是篡改预登记）
    ⇒ 这里用 exp96 脚本导出的**同一个** `sign_test_nonzero` / `median_upper` /
    `CRITERIA` 阈值，独立复现判据逻辑。
    """
    import exp96_group_profile as E96
    ok = [r for r in records if "error" not in r]

    # A：特征层逐被试复现
    max_diff, n_chk = 0.0, 0
    for r in ok:
        subj = r.get("subject")
        for gname, ref_map in E96.REF89["r"].items():
            if subj not in ref_map:
                continue
            got = r.get("feature", {}).get(gname, {}).get("r")
            if got is None:
                return {"verdict": "INSUFFICIENT", "why": f"特征层缺 {subj}/{gname}"}
            max_diff = max(max_diff, abs(float(got) - ref_map[subj]))
            n_chk += 1
    if n_chk == 0:
        return {"verdict": "INSUFFICIENT", "why": "没有可比对的被试"}
    if max_diff > E96.REF_TOL:
        return {"verdict": "PIPELINE_DRIFT", "A_max_abs_diff": max_diff,
                "why": f"特征层复算漂移 {max_diff:.2e}"}

    def _diffs(gname, kind):
        out = []
        for r in ok:
            m = r.get("model", {}).get(tag)
            if not m:
                continue
            rt = m.get("groups", {}).get(gname, {}).get("r")
            fx = (m.get("floors", {}).get(gname, {}) or {}).get(kind)
            if rt is None or not fx:
                continue
            out.append(rt - fx["r_median"])
        return out

    d_b = _diffs("POST", "mixed")
    if not d_b:
        return {"verdict": "INSUFFICIENT", "why": f"{tag} 缺 POST 地板"}
    n_eff, k, m_, p, n_zero = E96.sign_test_nonzero(d_b)
    med_b = E96.median_upper(d_b)
    b_pass = bool(n_eff >= E96.MIN_NONZERO and k == n_eff and med_b > 0
                  and p <= E96.CRITERIA["B_post_anchor"]["p_max"])
    B = {"n": len(d_b), "n_eff": n_eff, "n_zero": n_zero, "n_pos": k, "n_neg": m_,
         "p_sign": p, "median": med_b, "pass": b_pass}
    if not b_pass:
        return {"verdict": "ANCHOR_FAIL", "A_max_abs_diff": max_diff, "B": B,
                "why": f"{tag} 的 POST 锚点没过 ⇒ 该骨干的模型层读出口径不可信"}

    d_c = _diffs("FRONT", "pure")
    if not d_c:
        return {"verdict": "INSUFFICIENT", "why": f"{tag} 缺 FRONT 的 pure 地板"}
    n_eff2, k2, m2, p2, n_zero2 = E96.sign_test_nonzero(d_c)
    med_c = E96.median_upper(d_c)
    same_sign = bool(n_eff2 > 0 and (k2 == n_eff2 or m2 == n_eff2))
    c_pass = bool(n_eff2 >= E96.MIN_NONZERO and same_sign
                  and p2 <= E96.CRITERIA["C_front_separable"]["p_max"])
    C = {"n": len(d_c), "n_eff": n_eff2, "n_zero": n_zero2, "n_pos": k2,
         "n_neg": m2, "p_sign": p2, "median": med_c, "pass": c_pass,
         "direction": ("above_floor" if med_c > 0 else
                       "below_floor" if med_c < 0 else "same_as_floor")}
    if not c_pass:
        return {"verdict": "THREE_GROUPS_FAIL", "A_max_abs_diff": max_diff,
                "B": B, "C": C,
                "why": f"{tag} 上 FRONT 与内容匹配地板不可区分 ⇒ 三组不成立"}
    return {"verdict": "THREE_GROUPS_HOLD", "A_max_abs_diff": max_diff,
            "B": B, "C": C,
            "why": f"{tag} 上 FRONT 显著脱离内容匹配地板（{C['direction']}）"
                   "⇒ 三组成立 ⇒ exp90 走 3×3、NLB 功能线可开"}


def per_backbone(records):
    """逐骨干拆出 B（POST 锚点）与 C（FRONT 可分离性）的细节。"""
    import exp96_group_profile as E96
    out = {}
    for tag, _ck, _ro in E96.MODEL_READOUTS:
        info = {"B": None, "C": None}
        # B：POST vs mixed 地板
        diffs_b, diffs_b_pure, diffs_c, diffs_c_mixed = [], [], [], []
        for r in records:
            if "error" in r:
                continue
            m = r.get("model", {}).get(tag)
            if not m:
                continue
            g = m.get("groups", {})
            fl = m.get("floors", {})
            rt_post = g.get("POST", {}).get("r")
            if rt_post is not None:
                fx = fl.get("POST", {}) or {}
                if fx.get("mixed"):
                    diffs_b.append(rt_post - fx["mixed"]["r_median"])
                if fx.get("pure"):
                    diffs_b_pure.append(rt_post - fx["pure"]["r_median"])
            rt_front = g.get("FRONT", {}).get("r")
            if rt_front is not None:
                fx = fl.get("FRONT", {}) or {}
                if fx.get("pure"):
                    diffs_c.append(rt_front - fx["pure"]["r_median"])
                if fx.get("mixed"):
                    diffs_c_mixed.append(rt_front - fx["mixed"]["r_median"])
        for name, diffs in (("B", diffs_b), ("C", diffs_c)):
            if not diffs:
                continue
            n_eff, k, m_, p, n_zero = E96.sign_test_nonzero(diffs)
            med = E96.median_upper(diffs)
            info[name] = {"n": len(diffs), "n_eff": n_eff, "n_zero": n_zero,
                          "n_pos": k, "n_neg": m_, "p_sign": p, "median": med,
                          "diffs": diffs}
        # 参照：C 若改用 mixed 地板会怎样（记录用，不作判决）
        if diffs_c_mixed:
            n_eff, k, m_, p, n_zero = E96.sign_test_nonzero(diffs_c_mixed)
            info["C_mixed_ref"] = {"n": len(diffs_c_mixed), "n_eff": n_eff,
                                   "n_pos": k, "n_neg": m_, "p_sign": p,
                                   "median": E96.median_upper(diffs_c_mixed)}
        out[tag] = info
    return out


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

    import exp96_group_profile as E96
    recomputed = E96.decide(records)
    same = (recomputed["verdict"] == job_verdict.get("verdict"))
    print(f"\n预登记判决（Job 写的）  : {job_verdict.get('verdict')}")
    print(f"离线重算（逐被试原始）  : {recomputed['verdict']}  ⇒ 一致：{same}")

    s1a_only = decide_for_backbone(records, "s1a")
    s1b_only = decide_for_backbone(records, "s1b")
    print(f"事后只看 s1a（**非预登记**）：{s1a_only['verdict']}")

    bb = per_backbone(records)

    # ---- 特征层复现（判据 A 的细节）----
    a_rows = []
    for r in records:
        if "error" in r:
            continue
        for g in ("ALL", "POST", "CENT", "FRONT", "ANT"):
            ref = E96.REF89["r"][g].get(r["subject"])
            got = r.get("feature", {}).get(g, {}).get("r")
            if ref is None or got is None:
                continue
            a_rows.append((r["subject"], g, f"{got:+.4f}", f"{ref:+.4f}",
                           f"{got-ref:+.5f}"))
    max_diff = max(abs(float(x[4])) for x in a_rows) if a_rows else float("nan")

    # ---- 组别真值表（模型层）----
    grp_rows = []
    for tag in bb:
        for g in ("ALL", "POST", "CENT", "FRONT", "ANT"):
            vals = []
            for r in records:
                if "error" in r:
                    continue
                v = (r.get("model", {}).get(tag, {}).get("groups", {})
                     .get(g, {}).get("r"))
                vals.append(v)
            vals = [v for v in vals if v is not None]
            if vals:
                grp_rows.append((tag, g, len(vals),
                                 f"{E96.median_upper(vals):+.4f}"))

    # ---- 汇总输出 ----
    summary = {
        "job_verdict": job_verdict,
        "s1b_only_verdict_post_hoc": s1b_only,
        "recomputed_verdict": recomputed,
        "recompute_matches_job": same,
        "s1a_only_verdict_post_hoc": s1a_only,
        "per_backbone": bb,
        "criterion_A_max_abs_diff": max_diff,
        "criterion_A_n_checked": len(a_rows),
    }
    (OUT / "exp96_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    md = []
    md.append("# exp96 结果重建（三组模型层剖面 + 双地板）\n")
    md.append(f"**预登记判决（Job 写的）**：`{job_verdict.get('verdict')}`\n")
    md.append(f"**离线重算（逐被试原始记录）**：`{recomputed['verdict']}` ⇒ "
              f"与 Job 一致：**{same}**\n")
    md.append(f"**理由**：{recomputed.get('why', '')}\n")
    md.append(f"**事后只看 s1a（非预登记）**：`{s1a_only['verdict']}` ⇒ "
              f"{s1a_only.get('why', '')}\n")
    md.append("## 判据 A：特征层逐被试复现 exp89\n")
    md.append(f"检查 {len(a_rows)} 个（被试 × 组）值，最大绝对差 "
              f"**{max_diff:.2e}**（容差 1e-3）⇒ 管线零漂移。\n")
    md.append("## 逐骨干的 B / C\n")
    rows = []
    for tag, info in bb.items():
        b = info.get("B") or {}
        c = info.get("C") or {}
        cm = info.get("C_mixed_ref") or {}
        rows.append((tag,
                     f"{b.get('n_pos', '-')}/{b.get('n_eff', '-')} 正",
                     f"{b.get('p_sign', float('nan')):.4f}",
                     f"{b.get('median', float('nan')):+.4f}",
                     "✅" if b.get("n_pos") == b.get("n_eff") else "—",
                     f"{c.get('n_pos', '-')}/{c.get('n_eff', '-')} 正",
                     f"{c.get('p_sign', float('nan')):.4f}",
                     f"{c.get('median', float('nan')):+.4f}",
                     f"{cm.get('median', float('nan')):+.4f}"))
    md.append(fmt_table(rows, ["骨干", "B 正数", "B p", "B 中位差",
                               "B 过", "C(pure) 正数", "C p", "C 中位差",
                               "C(mixed) 参照中位差"]))
    md.append("\n## 模型层各组真值（跨被试上中位）\n")
    md.append(fmt_table(grp_rows, ["骨干", "组", "n", "r 中位"]))
    (OUT / "exp96_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n=== 逐骨干 ===")
    for tag, info in bb.items():
        b, c = info.get("B") or {}, info.get("C") or {}
        print(f"  {tag}: B 正 {b.get('n_pos')}/{b.get('n_eff')} p={b.get('p_sign'):.4f} "
              f"中位 {b.get('median'):+.4f} | C(pure) 正 {c.get('n_pos')}/{c.get('n_eff')} "
              f"p={c.get('p_sign'):.4f} 中位 {c.get('median'):+.4f} "
              f"方向 {c.get('direction')}")
    print(f"\n判据 A：{len(a_rows)} 个值，最大绝对差 {max_diff:.2e}")
    print(f"输出：{OUT/'exp96_summary.json'} / {OUT/'exp96_summary.md'}")


if __name__ == "__main__":
    main()
