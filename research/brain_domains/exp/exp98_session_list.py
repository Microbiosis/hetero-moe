"""exp98_session_list.py — 从可得性扫描日志里排出 session 清单（预登记用）。

用法：python research/brain_domains/exp/exp98_session_list.py
"""
import json
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "results" / "exp98-scan.log"
NEED = ("VISp", "VISl", "VISam")


def load():
    txt = LOG.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in re.finditer(r"===S===(\{.*?\})\r?\n", txt, re.S):
        try:
            rows.append(json.loads(m.group(1)))
        except Exception:
            pass
    rows.sort(key=lambda r: r["session"])
    return rows


def main():
    rows = load()
    print(f"sessions={len(rows)}（按 session 编号升序）")
    ok_prefix = []
    for i, r in enumerate(rows):
        a = r.get("areas") or {}
        good = all(a.get(x, 0) >= 20 for x in NEED)
        if good:
            ok_prefix.append(r["session"])
        print(f"  {i+1:2d} {r['session']:16s} VISp={a.get('VISp',0):4d} "
              f"VISl={a.get('VISl',0):4d} VISam={a.get('VISam',0):4d} "
              f"VISrl={a.get('VISrl',0):4d} VISal={a.get('VISal',0):4d} "
              f"CA1={a.get('CA1',0):4d} {'OK' if good else '--'}")
    for n in (8, 10, 12, 16, 20, 24):
        k = sum(1 for r in rows[:n]
                if all((r.get("areas") or {}).get(x, 0) >= 20 for x in NEED))
        print(f"前 {n:2d} 个里三区齐备的：{k}")
    print("首个可用 session 列表：")
    print(json.dumps(ok_prefix, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
