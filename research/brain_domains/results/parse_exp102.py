#!/usr/bin/env python3
r"""parse_exp102.py — exp102 判决的**离线重算**：读逐被试原始记录 ⇒ 重算出 C0/C1/C1b/C2/C3。

## 为什么要有它

项目纪律：**判决必须能用逐被试/逐 session 的原始记录离线重算**（不能只在云端算一次就信）。
⇒ ⇒ 本脚本**不碰任何模型**，只读 `results/exp102-results/per_subject.csv`，
用**与云端同一套函数**（`exp/exp102_gen.py` 的 `judge()` ⇒ 同一套阈值与统计口径）重算一遍
⇒ ⇒ **两边必须逐字一致**（不一致 ⇒ 有一边的实现错了 ⇒ 停下来查）。

## 用法

```powershell
python research\brain_domains\results\parse_exp102.py \
    --csv research\brain_domains\results\exp102-results\per_subject.csv
python research\brain_domains\results\parse_exp102.py --self-test
```
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent / "exp"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import exp102_gen as G  # noqa: E402


def load_records(csv_path, loss_summary=None):
    """读逐被试原始记录 ⇒ 记录列表（**字段名以 CSV 表头为准**）。

    ⚠️ C0 的「训练损失下降」是**整轮量**（不在逐被试表里）⇒ 从 `loss_summary.json` 读回来
    ⇒ ⇒ 否则离线重算会误判成 C0_FAIL（这正是这个脚本要抓的那类不一致）。
    """
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = {"subject": row.get("subject"), "ok": True}
            for k in ("own2", "floor2", "own50", "floor50", "cross_subject2",
                      "cond2_same_clip", "clip_sim", "img_pair_dist",
                      "per_concept_recall"):
                v = row.get(k)
                if v is None or v == "":
                    continue
                try:
                    rec[k] = float(v)
                except ValueError:
                    pass
            if "n_images" in row and row["n_images"]:
                try:
                    rec["n_images"] = int(row["n_images"])
                except ValueError:
                    pass
            if loss_summary:
                rec["loss_summary"] = loss_summary
            out.append(rec)
    return out


def self_test():
    """用**合成记录**验证：离线重算与云端判决函数是同一套（不是各写一遍）。"""
    fail = 0

    def chk(name, cond):
        nonlocal fail
        print(("✓ " if cond else "✗ ") + name)
        if not cond:
            fail += 1

    recs = G._fake_records()
    v = G.judge(recs)
    chk("合成记录 ⇒ 判决 = CONDITION_DRIVES_GENERATION",
        v.get("verdict") == "CONDITION_DRIVES_GENERATION")
    chk("C1 的 p 与 n 都在", "C1" in v and "two_way_minus_floor" in v.get("C1", {}))
    chk("C1b 只报不改判（不在判决键里）", "C1b_report" in v and "C1b_pass" not in v)
    chk("C3 有传递效率", v.get("C3_report", {}).get("transfer_efficiency") is not None)
    v2 = G.judge(G._fake_records(own=0.50, floor=0.52))
    chk("own2 ≤ 地板 ⇒ CONDITION_DOES_NOT_DRIVE_GENERATION",
        v2.get("verdict") == "CONDITION_DOES_NOT_DRIVE_GENERATION")
    v3 = G.judge(G._fake_records(n=5))
    chk("被试 < 8 ⇒ DIRECTION_RIGHT_UNDERPOWERED",
        v3.get("verdict") == "DIRECTION_RIGHT_UNDERPOWERED")
    # CSV 往返：写出来再读回来 ⇒ 判决不变
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "per_subject.csv"
        cols = ["subject", "n_images", "own2", "floor2", "own50", "floor50",
                "cross_subject2", "cond2_same_clip", "clip_sim", "img_pair_dist",
                "per_concept_recall"]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        back = load_records(p, loss_summary=recs[0].get("loss_summary"))
        v4 = G.judge(back)
        chk("CSV 往返后判决不变",
            v4.get("verdict") == v.get("verdict")
            and v4["C1"]["two_way_minus_floor"]["p_sign"] == v["C1"]["two_way_minus_floor"]["p_sign"])
    print(f"\n{'全部通过' if fail == 0 else f'{fail} 项失败'}")
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent /
                                         "exp102-results" / "per_subject.csv"))
    ap.add_argument("--verdict", default=str(Path(__file__).resolve().parent /
                                             "exp102-results" / "verdict.json"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not Path(args.csv).exists():
        print(f"找不到 {args.csv} ⇒ 先跑云端 judge step")
        return 1
    loss_summary = None
    lp = Path(args.csv).parent / "loss_summary.json"
    if lp.exists():
        loss_summary = json.loads(lp.read_text(encoding="utf-8"))
    recs = load_records(args.csv, loss_summary=loss_summary)
    v = G.judge(recs)
    print(f"逐被试记录 {len(recs)} 条")
    for r in recs:
        print(f"  {r['subject']}: own2={r.get('own2')} floor2={r.get('floor2')} "
              f"own50={r.get('own50')} floor50={r.get('floor50')} "
              f"cross2={r.get('cross_subject2')} cond2={r.get('cond2_same_clip')}")
    print("\n=== 离线重算 ===")
    print(json.dumps(v, indent=2, ensure_ascii=False, default=str))

    vp = Path(args.verdict)
    if vp.exists():
        cloud = json.loads(vp.read_text(encoding="utf-8")).get("verdict", {})
        same = (cloud.get("verdict") == v.get("verdict")
                and json.dumps(cloud.get("C1"), sort_keys=True, default=str)
                == json.dumps(v.get("C1"), sort_keys=True, default=str))
        print(f"\n与云端 verdict.json 一致？{'✓ 是' if same else '✗ 否 ⇒ 两边实现不一致 ⇒ 要查'}")
        if not same:
            print("  云端:", json.dumps(cloud, ensure_ascii=False, default=str)[:800])
            print("  本地:", json.dumps(v, ensure_ascii=False, default=str)[:800])
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
