#!/usr/bin/env python3
"""merge_alignment.py — 把各被试的 `alignment_sub-XX.json` 合成最终表。

## 为什么需要它

「7 基模脑对齐表」在 cpu-upgrade 上要跑十几小时 ⇒ 实际执行方式是
**10 个被试 × 2 趟（stable / dev 两版 braindecode）= 20 个并行 Job**，
每个 Job 只写自己的 `alignment_sub-XX.json`（多个 Job 写同一个文件会互相覆盖）。
这个脚本负责把它们合回来，出最终表。

⚠️ **去重**：两趟都算了「630-d 原始特征」基线行（它不依赖基模）⇒
合并时同一被试的同一行会出现两次 ⇒ 按 (被试, 行名) 去重，保留先到的。

用法：

    python merge_alignment.py --dir <stable目录> --dir <dev目录> [--out <输出目录>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
for _p in (str(EXP_DIR), str(EXP_DIR.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_brain_alignment import (BASELINE_ROW, render_table,  # noqa: E402
                                  summarize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True,
                    help="含 alignment_sub-*.json 的目录（可给多个）")
    ap.add_argument("--out", default="", help="输出目录（默认第一个 --dir）")
    ap.add_argument("--order", default="",
                    help="行顺序，逗号分隔（默认：基线行在前，其余按首次出现顺序）")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dir]
    out = Path(args.out) if args.out else dirs[0]
    out.mkdir(parents=True, exist_ok=True)

    # ⚠️ **按 (被试, 行名) 合并，不能按被试去重**：
    #    两趟跑的是**不同的模型**（stable 跑 4 个、dev 跑 2 个），
    #    同一被试在两趟里的行是**互补**的 —— 按被试去重会把 dev 趟的行整条丢掉。
    by_subject = {}
    for d in dirs:
        files = sorted(d.glob("alignment_sub-*.json"))
        print(f"── {d}：{len(files)} 个文件")
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  ✗ {f.name} 读失败：{e}")
                continue
            for rec in data.get("per_subject") or []:
                sub = rec.get("subject")
                cur = by_subject.setdefault(sub, {"subject": sub, "rows": [],
                                                  "n_train_images": rec.get("n_train_images")})
                for row in rec.get("rows") or []:
                    names = [r.get("name") for r in cur["rows"]]
                    if row.get("name") in names:
                        continue          # 同一行的重复（两趟都算了基线）⇒ 留先到的
                    cur["rows"].append(row)
                print(f"  ✓ {f.parent.name}/{f.name}: {sub} "
                      f"(累计 {len(cur['rows'])} 行)")

    per_subject = [by_subject[k] for k in sorted(by_subject)]

    if not per_subject:
        print("✗ 没有可合并的被试记录")
        return 1

    agg = summarize(per_subject)
    order = ([BASELINE_ROW] if BASELINE_ROW in agg else [])
    if args.order:
        order = [x.strip() for x in args.order.split(",") if x.strip()]
    else:
        order += [n for n in agg if n not in order]

    md = render_table(agg, order=order)
    (out / "alignment_table.md").write_text(md, encoding="utf-8")
    (out / "alignment_full.json").write_text(json.dumps(
        {"schema_version": "1.0", "script": "merge_alignment",
         "n_subjects": len(per_subject), "summary": agg,
         "per_subject": per_subject},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"7 基模脑对齐表（{len(per_subject)} 个被试）")
    print("=" * 100)
    print(md)
    print(f"\n输出：{out/'alignment_table.md'} / {out/'alignment_full.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
