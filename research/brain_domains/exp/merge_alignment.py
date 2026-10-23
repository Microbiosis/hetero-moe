#!/usr/bin/env python3
"""merge_alignment.py — 把各被试的 `alignment_sub-XX.json` 合成最终表。

## 为什么需要它

「7 基模脑对齐表」在 cpu-upgrade 上要跑十几小时 ⇒ 实际执行方式是
**10 个被试 = 10 个并行 Job**，每个 Job 只写自己的 `alignment_sub-XX.json`
（多个 Job 写同一个文件会互相覆盖）。
这个脚本负责把它们合回来，出最终表。

用法：

    python merge_alignment.py --dir <结果目录> [--out <输出目录>]

输入：`alignment_sub-*.json`（每个含 `per_subject: [ {...} ]`）
输出：`alignment_table.md` + `alignment_full.json`
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
    ap.add_argument("--dir", required=True, help="含 alignment_sub-*.json 的目录")
    ap.add_argument("--out", default="", help="输出目录（默认与 --dir 相同）")
    ap.add_argument("--order", default="",
                    help="行顺序，逗号分隔（默认：基线行在前，其余按首次出现顺序）")
    args = ap.parse_args()

    src = Path(args.dir)
    out = Path(args.out) if args.out else src
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("alignment_sub-*.json"))
    if not files:
        print(f"✗ {src} 里没有 alignment_sub-*.json")
        return 1
    per_subject = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ✗ {f.name} 读失败：{e}")
            continue
        recs = d.get("per_subject") or []
        per_subject.extend(recs)
        print(f"  ✓ {f.name}: {len(recs)} 个被试")
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
