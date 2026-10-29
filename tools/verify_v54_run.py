#!/usr/bin/env python3
"""v54 远端产物**本地独立复核** (不重新训练, 只复算)。

为什么必须有这一步
------------------
全网格跑在 HF Space 上。Space 自己判自己"通过", 与 v53 的处境是一样的:
产物说 n_classes=8、代码是 6, 没人发现。要把那句话钉死, 必须有一个**在另一台机器上、
用本地代码**读同一份产物并重算判决的东西。

它做的五件事
------------
    C1 代码血缘  产物里的 code_fp / predictions_sha == 本地算出来的
    C2 网格完备  36 格 × 5 seeds 一个不缺 (缺就 FAIL, 不"差不多")
    C3 配对真实  同一 (arm, D, seed) 下三档的 init_sha 必须逐位相同
    C4 判决复算  用本地判据重算 E1–E6, 与产物里的 verdicts 逐条对拍
    C5 闸门一致  产物声称的 all_pass == 本地重算的 all_pass

用法
----
    python tools/verify_v54_run.py                          # 从任务桶取
    python tools/verify_v54_run.py --path local/metrics.json   # 检本地文件
    python tools/verify_v54_run.py --prefix v54-86e0f4e --bucket redmingwei/hetero-moe-jobs

退出码: 0 全部一致 / 1 有不一致 / 2 产物缺失或未完成
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def local_fingerprint() -> dict:
    from examples.run_v54_evaluator_swap import code_fingerprint
    return code_fingerprint()


def fetch(prefix: str, bucket: str, token=None) -> dict:
    from huggingface_hub import download_bucket_files
    import tempfile
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "metrics.json")
    download_bucket_files(bucket, files=[(f"{prefix}/metrics.json", dst)],
                          token=token, raise_on_missing_files=False)
    if not os.path.exists(dst):
        raise SystemExit(f"⛔ 桶里没有 {prefix}/metrics.json (可能仍是 completeness 未过 → 只有 metrics.partial.json)")
    return json.load(open(dst, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--bucket", default="redmingwei/hetero-moe-jobs")
    ap.add_argument("--prefix", default="v54-86e0f4e")
    a = ap.parse_args()

    print("=" * 78)
    print("v54 远端产物 · 本地独立复核")
    print("=" * 78)

    m = json.load(open(a.path, encoding="utf-8")) if a.path else fetch(a.prefix, a.bucket)
    fails, warns = [], []

    def chk(cid, ok, detail, warn=False):
        st = "  ok " if ok else (" warn" if warn else " FAIL")
        print(f"[{st}] {cid} {detail}")
        if not ok:
            (warns if warn else fails).append(cid)

    # ---- C1 代码血缘 ----
    lf = local_fingerprint()
    chk("C1a", m.get("code_fingerprint", {}).get("_combined") == lf["_combined"],
        f"code_fp 产物={m.get('code_fingerprint', {}).get('_combined')} 本地={lf['_combined']}")
    from examples.run_v54_evaluator_swap import _predictions_sha
    chk("C1b", m.get("predictions_sha") == _predictions_sha(),
        f"判据指纹 产物={m.get('predictions_sha')} 本地={_predictions_sha()}")

    # ---- C2 网格完备 ----
    from research.aligner.v54_evaluator_swap.integrity import expected_grid, completeness
    cfg = m.get("config", {})
    from research.aligner.v54_evaluator_swap.audit import ARMS
    grid = expected_grid(cfg.get("d_shareds", []), cfg.get("evaluators", []),
                         list(ARMS), cfg.get("seeds", 0), cfg.get("seed_base", 0))
    comp = completeness(m.get("results", {}), grid)
    chk("C2", comp["ok"], f"网格 {comp['n_cells_have']}/{comp['n_cells_want']} 格, "
                          f"观测 {comp['n_obs_have']}/{comp['n_obs_want']}"
                          + (f"; 缺={comp['missing_cells'][:4]}" if comp["missing_cells"] else ""))

    # ---- C3 配对真实 (产物级) ----
    res = m.get("results", {})
    bad_pair, n_pair = [], 0
    keys = [k for k in res if k.endswith(":learned_lift")] + \
           [k for k in res if k.endswith(":frozen_random")]
    by_arm_seed: dict[tuple, dict] = {}
    for k in keys:
        dtag, ev, arm = k.split(":")
        for r in res[k].get("rows", []):
            by_arm_seed.setdefault((dtag, arm, r.get("seed")), {})[ev] = r.get("init_sha")
    for (dtag, arm, seed), evs in by_arm_seed.items():
        vals = {v for v in evs.values() if v}
        n_pair += 1
        if len(vals) != 1:
            bad_pair.append({"d": dtag, "arm": arm, "seed": seed, "init_sha": evs})
    chk("C3", not bad_pair, f"配对检查 {n_pair} 组 (arm,D,seed), 不一致 {len(bad_pair)} 组"
                            + (f" 例: {bad_pair[0]}" if bad_pair else ""))

    # ---- C4 判决复算 ----
    import examples.run_v54_evaluator_swap as R
    v_local, audit_local, pass_local = R.judge(res)
    mism = []
    for dtag, vv in (m.get("verdicts") or {}).items():
        for c, val in vv.items():
            lv = v_local.get(dtag, {}).get(c)
            if bool(val) != bool(lv) and not (val is None and lv is None):
                mism.append({"d": dtag, "c": c, "artifact": val, "local": lv})
    chk("C4", not mism, f"判决逐条对拍: {len(mism)} 处不一致" + (f" 例: {mism[:3]}" if mism else ""))

    # ---- C5 闸门一致 ----
    chk("C5", bool(m.get("all_pass")) == bool(pass_local),
        f"all_pass 产物={m.get('all_pass')} 本地重算={pass_local} "
        f"(完备={audit_local['_gate']['grid_complete']} 无未决={audit_local['_gate']['no_undecided']} "
        f"全真={audit_local['_gate']['all_true']})")

    print("\n" + "=" * 78)
    print(f"判决 (本地重算):")
    for dtag, vv in v_local.items():
        cells = []
        for c, val in vv.items():
            a = audit_local[dtag][c]
            tag = "过" if val is True else ("未过" if val is False else "无结论")
            cells.append(f"{c}={tag}({a['n_true']}/{a['n_expected']})")
        print(f"  {dtag}: " + "  ".join(cells))
    print(f"\n结果: {len(fails)} FAIL / {len(warns)} WARN / {len(fails) + 9 - len(warns)} PASS")
    if fails:
        print("→ 产物与本地复算不一致, 该产物不得作为结论依据")
        return 1
    print("→ 远端产物在本地代码下逐条复现 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
