#!/usr/bin/env python3
"""unify_formats — 把 framework/ 下的表格统一为 parquet（CSV 转出后删除）。

理由：数据集的使用者是模型不是人，表格只需要一种格式。
每个 CSV 先读出来写成同名 parquet（覆盖可能已存在的旧 parquet，保证与 CSV 内容一致），
确认写出成功后才删除 CSV。归档表 sources_v1 转成 parquet 放进 superseded/。
"""
import sys
from pathlib import Path

import pandas as pd

FW = Path("/domains/framework")
SUP = FW / "superseded"

rows = []
failed = []

for csv in sorted(FW.glob("*.csv")):
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        failed.append((csv.name, f"read: {e}"))
        continue

    if csv.stem == "sources_v1":
        SUP.mkdir(parents=True, exist_ok=True)
        out = SUP / "sources_v1.parquet"
    else:
        out = FW / (csv.stem + ".parquet")

    try:
        df.to_parquet(out, index=False)
    except Exception as e:
        failed.append((csv.name, f"write: {e}"))
        continue

    back = pd.read_parquet(out)
    if len(back) != len(df):
        failed.append((csv.name, f"rows mismatch {len(df)} vs {len(back)}"))
        continue

    csv.unlink()
    rows.append((csv.name, out.name, len(df)))

print(f"converted={len(rows)} failed={len(failed)}")
for a, b, n in rows:
    print(f"  {a:32s} -> {b:34s} rows={n}")
for a, why in failed:
    print(f"  FAIL {a}: {why}")

print("\n--- framework/ 顶层剩余扩展名 ---")
from collections import Counter
c = Counter(p.suffix for p in FW.iterdir() if p.is_file())
for k, v in c.most_common():
    print(f"  {k or '(none)'}: {v}")

sys.exit(1 if failed else 0)
