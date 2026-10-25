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

    # --- R4（读出同口径）：规则本身要在，且台账里被限定过的结论不能把限定词删掉 ---
    term = ROOT / "TERMINOLOGY.md"
    if term.exists():
        ttext = term.read_text(encoding="utf-8")
        if "### R4" not in ttext:
            problems.append("[R4] TERMINOLOGY.md 缺「### R4 · 跨模型/跨配置的对比：读出必须同口径」")
        else:
            for need in ("读出类型相同", "还有哪个因子也变了", "同一批被试"):
                if need not in ttext:
                    problems.append(f"[R4] TERMINOLOGY.md 的 R4 缺要素「{need}」")
        if "### R5" not in ttext:
            problems.append("[R5] TERMINOLOGY.md 缺「### R5 · 跨 Job 的速度比较必须同 flavor、同争用、同口径」")
        else:
            for need in ("flavor 相同", "争用相同", "口径相同", "前向"):
                if need not in ttext:
                    problems.append(f"[R5] TERMINOLOGY.md 的 R5 缺要素「{need}」")
    else:
        problems.append("缺文件 TERMINOLOGY.md")

    # A18 / A13 这两条是**被口径限定过**的结论 ⇒ 限定词被删掉就要报错
    # ⚠️ 检查点必须选**在该行里只出现一次**的特征串：
    #    否则自测的「只替换第一处」会命中不到 ⇒ 守卫形同虚设。
    integ = ROOT / "INTEGRATION.md"
    if integ.exists():
        itext = integ.read_text(encoding="utf-8")
        for aid, need, label in (("**A18**", "读出口径", "A18 必须带「读出口径」限定"),
                                 ("**A13**", "n_train=16540", "A13 必须带「n_train=16540」限定")):
            line = next((l for l in itext.splitlines() if aid in l), "")
            if not line:
                problems.append(f"[台账] INTEGRATION.md 里找不到 {aid} 那一行")
            elif need not in line:
                problems.append(f"[台账] {label}")
            elif line.count(need) != 1:
                problems.append(
                    f"[台账] {aid} 行的检查点「{need}」出现 {line.count(need)} 次"
                    f"（自测会命中不到 ⇒ 请换成单次出现的串）")

    # FINDINGS 里 A18 的那条「修正表」行也必须带读出限定（它在 §二十九·七）
    fnd = ROOT / "FINDINGS_TIER0_RSA.md"
    if fnd.exists():
        ftext = fnd.read_text(encoding="utf-8")
        a18 = [l for l in ftext.splitlines() if "**A18**" in l and l.startswith("|")]
        if not a18:
            problems.append("[台账] FINDINGS 里找不到 A18 的修正表那一行")
        else:
            hits = [l for l in a18 if "要加读出限定" in l]
            if not hits:
                problems.append("[台账] FINDINGS 的 A18 修正表行缺「要加读出限定」")
            elif len(hits) > 1:
                problems.append(
                    f"[台账] FINDINGS 的 A18 行有 {len(hits)} 处含「要加读出限定」"
                    f"（自测会命中不到 ⇒ 请用单次出现的串）")

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
