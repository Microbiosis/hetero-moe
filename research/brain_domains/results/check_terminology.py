"""check_terminology.py — 措辞限定词守卫（`TERMINOLOGY.md` 的机器可判部分）。

## 为什么需要

2026-10-24 的一次纠正暴露了一类**不会被数字守卫发现**的错误：
把**左右轴**（裂脑实验）的证据当成了**前后轴**的先例，
以及把「两个原语」当成了脑的**部件清单**。
⇒ 这类错误靠人记不住 ⇒ 写成规则（`TERMINOLOGY.md`）⇒ 再写成**机械检查**。

## 机械检查三条

1. **限定词不得丢失**：凡出现「两原语 / 两个原语 / 一对原语 / 分原语」的行，
   **同一行**必须出现「位置轴」；否则报错并给出 文件:行号。
2. **左右轴的引用必须带轴标签**：凡出现「裂脑」的行，同一行必须出现「左右轴」。
3. **轴表不能少行**：`BRAIN_STRUCTURE_SPEC.md` §1.4 的轴表必须同时有
   「谱系轴 / 位置轴 / 左右轴 / 功能轴」；`MODEL_FORM.md` 的轴清单必须有「左右轴」。

## 豁免

`TERMINOLOGY.md` 本身**豁免**：它是「旧表述」的登记表，故意逐字保留裸表述。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FILES = ["FINDINGS_TIER0_RSA.md", "INTEGRATION.md",
         "BRAIN_STRUCTURE_SPEC.md", "MODEL_FORM.md"]
EXEMPT = {"TERMINOLOGY.md"}

PRIMITIVE_TERMS = ("两原语", "两个原语", "一对原语", "分原语")
LATERAL_TERMS = ("裂脑",)
AXES = ("谱系轴", "位置轴", "左右轴", "功能轴", "层/柱轴")
# 「位置轴」的两个同义写法都算数（规则本身用「位置轴」，正文有时写「前后轴」）
POSITION_QUALIFIERS = ("位置轴", "前后轴")


def main():
    problems = []
    for name in FILES:
        fp = ROOT / name
        if not fp.exists():
            problems.append(f"缺文件 {name}")
            continue
        lines = fp.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if (any(t in line for t in PRIMITIVE_TERMS)
                    and not any(q in line for q in POSITION_QUALIFIERS)):
                problems.append(
                    f"[限定词] {name}:{i} 出现「原语」类表述但同行没有「位置轴/前后轴」"
                    f"：{line.strip()[:90]}")
            if any(t in line for t in LATERAL_TERMS) and "左右轴" not in line:
                problems.append(
                    f"[轴标签] {name}:{i} 出现「裂脑」但同行没有「左右轴」"
                    f"：{line.strip()[:90]}")

    # --- 轴台账：结构检查（不只是找词）---
    spec = ROOT / "BRAIN_STRUCTURE_SPEC.md"
    if spec.exists():
        text = spec.read_text(encoding="utf-8")
        i = text.find("### 1.2")
        j = text.find("\n### ", i + 1) if i >= 0 else -1
        sec = text[i: j if j > 0 else len(text)] if i >= 0 else ""
        if not sec:
            problems.append("BRAIN_STRUCTURE_SPEC.md 里找不到「### 1.2 四条轴的台账」")
        else:
            for ax in AXES:
                if ax not in sec:
                    problems.append(f"[轴台账] §1.2 缺「{ax}」这一行")
            for col in ("证据状态", "不能借来干什么"):
                if col not in sec:
                    problems.append(f"[轴台账] §1.2 缺「{col}」这一列")
            for ax, need in (("位置轴", "实测是真轴"), ("发育轴", "缺 pNE 侧数据"),
                             ("左右轴", "不可归因")):
                if ax in sec and need not in sec:
                    problems.append(f"[轴台账] §1.2 的「{ax}」行没写明状态（应含「{need}」）")
    mf = ROOT / "MODEL_FORM.md"
    if mf.exists():
        if "左右轴" not in mf.read_text(encoding="utf-8"):
            problems.append("[轴表] MODEL_FORM.md 的轴清单缺「左右轴」")

    if problems:
        print("✗ 措辞限定词问题：")
        for p in problems:
            print("   -", p)
        return 1
    print(f"✓ 措辞合格（原语表述都带「位置轴/前后轴」；裂脑引用都带「左右轴」；"
          f"轴台账有 {len(AXES)} 条轴 + 证据状态 + 不可借用列）｜豁免：{', '.join(sorted(EXEMPT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
