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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/domains/runs/illusionbrain-align/exp98b")
    args = ap.parse_args()
    RUN = pathlib.Path(args.run)
    if JUDGE not in sys.path:
        sys.path.insert(0, str(pathlib.Path(JUDGE).parent))
    import exp98_posneg_control as E98

    shards = sorted(RUN.glob("shard*/exp98_posneg_control.json"))
    print(f"分片结果 {len(shards)} 个", flush=True)
    per_session, seen = [], set()
    for sp in shards:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for r in d.get("per_session") or []:
            k = str(r.get("session"))
            if k in seen:
                continue
            seen.add(k)
            per_session.append(r)
    print(f"合并后逐 session 记录 {len(per_session)} 条", flush=True)
    verdict = E98.decide(per_session)
    payload = {"config": {"source": "exp98_aggregate", "n_shards": len(shards)},
               "criteria": {
                   "C0prime": "逐带可读性：能把刺激身份（118 张自然图像，空白屏已剔除）解码到显著高于机会"
                              "（acc>1/118 且置换 p<0.05）；主对三区任一不过 ⇒ 该 session 剔除（排除规则与 Δ 无关）",
                   "C1": "正/负对照分离：逐 session 统计量 = Δ(VISp↔VISam) − Δ(VISp↔VISl)；"
                         "session 级符号检验 ≥8 session 且 p≤0.05 且中位>0",
                   "C2": "退化守卫：主判据统计量不得是非零常数",
                   "C3": "只报：其余带对（VISp↔VISrl/VISal/CA1/VISpm）、每带解码准确率",
               },
               "per_session": per_session, "verdict": verdict}
    (RUN / "exp98_posneg_control.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for r in per_session:
        print("===REC===" + json.dumps(r, ensure_ascii=False, default=str), flush=True)
    print("===VERDICT===" + json.dumps(verdict, ensure_ascii=False, default=str), flush=True)
    print(f"\n=== 判决：{verdict['verdict']} ===", flush=True)
    print(f"  进入主判据：{verdict['n_sessions_primary_used']} / {verdict['n_sessions_total']}",
          flush=True)
    print(f"  符号检验：{verdict['C1']['n_pos']}/{verdict['C1']['n_eff']} 正，"
          f"p={verdict['C1']['p']:.4f}，中位 Δ={verdict['C1']['median_delta']:+.4f}", flush=True)
    print(f"  正对照中位 Δ={verdict['C1_pos_median']:+.4f}；"
          f"负对照中位 Δ={verdict['C1_neg_median']:+.4f}", flush=True)
    print(f"  解释：{verdict['C1_interpretation']}", flush=True)
    print(f"  C2 退化守卫：{verdict['C2_degenerate_guard']}", flush=True)
    print(f"结果写入 {RUN}/exp98_posneg_control.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
