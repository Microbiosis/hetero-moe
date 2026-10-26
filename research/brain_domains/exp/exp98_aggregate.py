#!/usr/bin/env python3
"""exp98_aggregate.py — 把各分片的**逐 session 原始记录**合并，跑**同一个** `decide()`。

## 做什么

1. 读 `/domains/runs/illusionbrain-align/exp98/shard*/exp98_posneg_control.json`
   ⇒ 取每片的 `per_session`（**原始记录**，不是摘要）。
2. 按 session 名去重合并（分片互不相交 ⇒ 正常就是 32 条）。
3. 用**被判决脚本里的 `decide()`**（import，不复制）重算判决 ⇒ **阈值/统计口径的唯一出处**。
4. 写 `/domains/runs/illusionbrain-align/exp98/exp98_posneg_control.json`
   （= 最终判决）并回显 `===REC===` / `===VERDICT===`（**离线重算的回显通道**）。

⚠️ **判据一字未动**：本脚本只做「合并 + 调用同一个 `decide()`」，
不重新实现任何阈值或统计口径 ⇒ ⇒ 分片与整轮**逐位等价**。
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUN = pathlib.Path("/domains/runs/illusionbrain-align/exp98b")
JUDGE = "/domains/scripts/exp98_posneg_control.py"


def main():
    import argparse
    import importlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/domains/runs/illusionbrain-align/exp98b")
    ap.add_argument("--judge", default=JUDGE, help="被判决脚本（默认 = exp98 的）")
    ap.add_argument("--out-name", default="exp98_posneg_control.json",
                    help="最终结果文件名（默认 = exp98 的）")
    args = ap.parse_args()
    RUN = pathlib.Path(args.run)
    jdir = str(pathlib.Path(args.judge).parent)
    if jdir not in sys.path:
        sys.path.insert(0, jdir)
    mod = pathlib.Path(args.judge).stem
    E98 = importlib.import_module(mod)

    shards = sorted(RUN.glob(f"shard*/{args.out_name}"))
    print(f"分片结果 {len(shards)} 个", flush=True)
    per_session, seen, criteria = [], set(), None
    for sp in shards:
        d = json.loads(sp.read_text(encoding="utf-8"))
        if criteria is None:
            # ⚠️ 判据描述**取被判决脚本自己写的**（不在这儿重抄 ⇒ 一条事实只写一处）
            criteria = d.get("criteria")
        for r in d.get("per_session") or []:
            k = str(r.get("session"))
            if k in seen:
                continue
            seen.add(k)
            per_session.append(r)
    print(f"合并后逐 session 记录 {len(per_session)} 条", flush=True)
    verdict = E98.decide(per_session)
    payload = {"config": {"source": f"aggregate:{mod}", "n_shards": len(shards)},
               "criteria": criteria,
               "per_session": per_session, "verdict": verdict}
    (RUN / args.out_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for r in per_session:
        print("===REC===" + json.dumps(r, ensure_ascii=False, default=str), flush=True)
    print("===VERDICT===" + json.dumps(verdict, ensure_ascii=False, default=str), flush=True)
    print(f"\n=== 判决：{verdict['verdict']} ===", flush=True)
    print(f"  进入主判据：{verdict['n_sessions_primary_used']} / {verdict['n_sessions_total']}",
          flush=True)
    print(f"  符号检验：{verdict['C1']['n_pos']}/{verdict['C1']['n_eff']} 正，"
          f"p={verdict['C1']['p']:.4f}，中位 Δ={verdict['C1']['median_delta']:+.4f}", flush=True)
    c1 = verdict.get("C1") or {}
    if c1.get("same_median_pooled") is not None:
        print(f"  同组中位 Δ={c1['same_median_pooled']:+.4f}；"
              f"跨组中位 Δ={c1['cross_median_pooled']:+.4f}", flush=True)
    else:
        print(f"  正对照中位 Δ={verdict.get('C1_pos_median')}；"
              f"负对照中位 Δ={verdict.get('C1_neg_median')}", flush=True)
    print(f"  解释：{verdict['C1_interpretation']}", flush=True)
    print(f"  C2 退化守卫：{verdict['C2_degenerate_guard']}", flush=True)
    print(f"结果写入 {RUN}/{args.out_name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
