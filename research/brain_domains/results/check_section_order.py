"""check_section_order.py — 检查 FINDINGS 的 H2 章节**编号是否递增**。

## 为什么需要

2026-10-24 晚：我在 §二十五 后面插入「§二十七」时用错了锚点 ⇒
文件里出现了 **二十五 → 二十七 → 二十六** 的错位 ⇒
读者按编号找会找错，而且**没有任何守卫会报**。

⇒ 这个检查是**结构级**的（不是文字级）⇒ 便宜 ⇒ 放进守卫链。

## 判据

所有形如 `## <中文数字>、` 的 H2 标题，编号必须**严格递增**；
非编号的 H2（如 `## 0. 这份规范解决什么`）不参与比较，但会原样保留。

## 用法

    python research/brain_domains/results/check_section_order.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FINDINGS = ROOT / "FINDINGS_TIER0_RSA.md"

CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
      "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14,
      "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
      "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
      "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29, "三十": 30}


def num(title):
    m = re.match(r"^(二十[一二三四五六七八九]|[一二三四五六七八九十]+)、", title)
    return CN.get(m.group(1)) if m else None


def main():
    if not FINDINGS.exists():
        print(f"✗ 找不到 {FINDINGS}")
        return 1
    text = FINDINGS.read_text(encoding="utf-8")
    seq = []
    for m in re.finditer(r"^## (.*)$", text, re.M):
        n = num(m.group(1))
        if n is not None:
            seq.append((n, m.group(1)[:40]))
    bad = []
    for i in range(1, len(seq)):
        if seq[i][0] <= seq[i - 1][0]:
            bad.append(f"「{seq[i - 1][1]}」({seq[i - 1][0]}) 之后是 "
                       f"「{seq[i][1]}」({seq[i][0]})")
    if bad:
        print("✗ 章节编号顺序错位：")
        for b in bad:
            print("   -", b)
        return 1
    print(f"✓ 章节顺序正确（{len(seq)} 个编号章节，递增）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
