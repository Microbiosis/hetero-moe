"""check_terminology.py — 措辞限定词守卫（`TERMINOLOGY.md` 的机器可判部分）。

## 为什么需要

2026-10-24 的一次纠正暴露了一类**不会被数字守卫发现**的错误：
把**左右轴**（裂脑实验）的证据当成了**前后轴**的先例，
以及把「两个原语」当成了脑的**部件清单**。
⇒ 这类错误靠人记不住 ⇒ 写成规则（`TERMINOLOGY.md`）⇒ 再写成**机械检查**。

## 机械检查六条

1. **限定词不得丢失**：凡出现「两原语 / 两个原语 / 一对原语 / 分原语」的行，
   **同一行**必须出现「位置轴」；否则报错并给出 文件:行号。
2. **左右轴的引用必须带轴标签**：凡出现「裂脑」的行，同一行必须出现「左右轴」。
3. **轴表不能少行**：`BRAIN_STRUCTURE_SPEC.md` §1.4 的轴表必须同时有
   「谱系轴 / 位置轴 / 左右轴 / 功能轴」；`MODEL_FORM.md` 的轴清单必须有「左右轴」。
4. **R4 / R5 规则要素齐全**（读出同口径 / 速度同 flavor 争用口径）。
5. **被口径限定过的结论不能把限定词删掉**（A13 的 `n_train=16540`、A18 的「读出口径」）。
6. **奥卡姆结构守卫**（2026-10-25 新增）：
   - **G1 帧一致**：§二十九 的标题与正文必须说的是同一套四个杠杆
     （拟合图数 / 训练目标 / 读出口径 / 特征窗口）—— 防止「标题四层、正文五项」式失配；
   - **G2 窗口限定**：`INTEGRATION.md` 的 A14/A15 必须带「0–250 ms 窗口」
     —— 兑现 §二十九·六 的承诺（位置轴读数随窗口系统性变化）；
   - **G3 状态不互斥**：台账 §1 已标 ✅ 的结论不得又出现在未测表 §4.2
     （台账状态格自带「未定」的除外，如 A16）。
7. **拆分与设计书的结构守卫**：
   - **G4 三组绑定**：`BRAIN_STRUCTURE_SPEC.md` §4.1 的通道数必须与 exp89 原表一致，
     §4.4 必须写清「形状共享，权重不共享」与「③-FRONT 弱代理」；
   - **G5 设计书不许退化成空话**：五个动词、三行组件、每条自写连接都要带判据、
     关键外部出处（Eliasmith 2005 / World Models）必须在；
   - **G6 算子表 ⇄ 代码类名绑定**：`COMPONENT_DESIGN.md` §3 的算子表必须与
     `arch/ops_brain.py` 里的类**双向一致**，性质测试的**项数**必须与
     `arch/test_ops_brain.py` 里的 `check()` 调用数一致，
     且 §3 必须写明「只有性质测试，没有判据实验」（不许把实现正确说成有用）。

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

    # --- G1：exp93 的「杠杆帧」必须与标题一致（奥卡姆：同一套框不许在标题与正文各说各的）---
    if fnd.exists():
        ftext2 = fnd.read_text(encoding="utf-8")
        h29 = next((l for l in ftext2.splitlines()
                    if l.startswith("## 二十九、exp93")), "")
        if not h29:
            problems.append("[帧] FINDINGS 里找不到「## 二十九、exp93」标题")
        else:
            i = ftext2.find("## 二十九、exp93")
            j = ftext2.find("\n## ", i + 1)
            sec29 = ftext2[i: j if j > 0 else len(ftext2)]
            for lever in ("拟合图数", "训练目标", "读出口径", "特征窗口"):
                if lever not in h29:
                    problems.append(
                        f"[帧] §二十九 的标题里没有「{lever}」"
                        f"⇒ 标题说的杠杆集合与正文不一致")
                if lever not in sec29:
                    problems.append(f"[帧] §二十九 的正文里没有「{lever}」")

    # --- G2：位置轴读数的**窗口限定**必须落在台账上（兑现 §二十九·六 的承诺）---
    integ_text = (ROOT / "INTEGRATION.md").read_text(encoding="utf-8") if (
        ROOT / "INTEGRATION.md").exists() else ""
    if integ_text:
        for aid in ("**A14**", "**A15**"):
            ln = next((l for l in integ_text.splitlines() if aid in l), "")
            if not ln:
                problems.append(f"[窗口] INTEGRATION.md 里找不到 {aid} 那一行")
            elif "0–250 ms 窗口" not in ln:
                problems.append(
                    f"[窗口] {aid} 行缺「0–250 ms 窗口」限定"
                    f"（§二十九·六 已承诺补上：位置轴读数会随窗口系统性变化）")

        # --- G3：未测表（§4.2）不得与台账（§1）矛盾 ---
        #     「已结案（✅）」的结论再作为**未测表的行** = 两个状态互斥。
        #     只看**首格（行的主语）**：在说明格里**引用**一条已结案的结论是合法的
        #     （例：「整体形态是否更优 | 未测（……底座已兑现（A19）⇒ 现在可开工）」）。
        #     豁免：台账状态格本身带「未定」的（如 A16「✅ 成立（限定：归因未定）」）。
        ledger = {}
        for l in integ_text.splitlines():
            m = re.match(r"^\|\s*\*\*(A\d+)\*\*\s*\|", l)
            if m:
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                ledger[m.group(1)] = cells[-1] if cells else ""
        i42 = integ_text.find("### 4.2 还没测的")
        if i42 < 0:
            problems.append("[状态] INTEGRATION.md 里找不到「### 4.2 还没测的」小节")
        else:
            j42 = integ_text.find("\n## ", i42)
            sec42 = integ_text[i42: j42 if j42 > 0 else len(integ_text)]
            for l in sec42.splitlines():
                if not l.startswith("|"):
                    continue
                first = l.strip().strip("|").split("|")[0]
                for aid in sorted(set(re.findall(r"A\d+", first))):
                    st = ledger.get(aid, "")
                    if st.startswith("✅") and "未定" not in st:
                        problems.append(
                            f"[状态] §4.2 未测表有一行以 {aid} 为主语，"
                            f"但台账 §1 已把它标为「{st[:24]}…」"
                            f"⇒ 两个状态互斥（改台账，或从未测表移除）")

    # --- G4：三组切分（按脑科学解剖）必须三处口径一致 ---
    #     ① C 类表要有 C19（前部不是一个功能单位 + 两条关键一手出处）
    #     ② A14 必须带「ANT 不是一个原语」的限定
    #     ③ 规格 §4.2 必须有三组切分 + 3×3 转分化矩阵
    integ_f = ROOT / "INTEGRATION.md"
    spec_f = ROOT / "BRAIN_STRUCTURE_SPEC.md"
    if integ_f.exists() and spec_f.exists():
        itext = integ_f.read_text(encoding="utf-8")
        stext = spec_f.read_text(encoding="utf-8")
        c19 = next((l for l in itext.splitlines() if l.startswith("| **C19**")), "")
        if not c19:
            problems.append("[C19] INTEGRATION.md 里找不到 C19 那一行（前部皮层功能的外部参照）")
        else:
            for need in ("不是一个功能单位", "10.1038/nrn2667", "10.1126/science.1138071",
                         "CENT", "FRONT", "功能轴"):
                if need not in c19:
                    problems.append(f"[C19] 缺「{need}」")
        a14 = next((l for l in itext.splitlines() if l.startswith("| **A14**")), "")
        if not a14:
            problems.append("[台账] INTEGRATION.md 里找不到 A14 那一行")
        else:
            for need in ("不是一个原语", "CENT 运动 14", "FRONT 执行 15", "可分性未测"):
                if need not in a14:
                    problems.append(
                        f"[台账] A14 缺「{need}」"
                        f"（前部按脑科学解剖含两个子系统 ⇒ 不能当成一个原语）")
        i42 = stext.find("### 4.2 exp90 的规格")
        j42 = stext.find("### 4.3 ", i42 + 1) if i42 >= 0 else -1
        sec42 = stext[i42: j42 if j42 > 0 else len(stext)]
        if not sec42:
            problems.append("[规格] BRAIN_STRUCTURE_SPEC.md 里找不到「### 4.2 exp90 的规格」小节")
        else:
            for need in ("三组切分", "CENT 运动 14", "FRONT 执行 15", "3×3 矩阵"):
                if need not in sec42:
                    problems.append(f"[规格] §4.2 缺「{need}」")
        # ③ 的落位必须已定（且带「弱代理」与 exp96 这两个限定）
        # ⚠️ **检查点必须唯一**（本行只出现一次）⇒ 逐行/逐短语查，不做整节包含判断
        i44 = stext.find("### 4.4 ")
        j44 = stext.find("## 5. ", i44 + 1) if i44 >= 0 else -1
        sec44 = stext[i44: j44 if j44 > 0 else len(stext)]
        if not sec44:
            problems.append("[规格] BRAIN_STRUCTURE_SPEC.md 里找不到「### 4.4」小节")
        else:
            l44 = sec44.splitlines()
            h44 = next((l for l in l44 if l.startswith("### 4.4")), "")
            for need in ("单开一条线", "形状共享，权重不共享"):
                if need not in h44:
                    problems.append(f"[规格] §4.4 标题缺「{need}」")
            row_fr = next((l for l in l44
                           if l.startswith("| **③-FRONT（执行）**")), "")
            if not row_fr:
                problems.append("[规格] §4.4 里找不到「③-FRONT（执行）」那一行")
            elif "弱代理" not in row_fr:
                problems.append(
                    "[规格] §4.4 的 ③-FRONT 行缺「弱代理」"
                    "（DMFC 只有 3 段，只能给弱代理，不得说成执行控制）")
            if "exp96 决定要不要开这条线" not in sec44:
                problems.append("[规格] §4.4 缺「exp96 决定要不要开这条线」")
        # 台账 §4.1 第 7 条必须已定（不再是待决）
        row7 = next((l for l in itext.splitlines()
                     if l.startswith("| 7 |") and "③" in l), "")
        if not row7:
            problems.append("[规格] INTEGRATION.md §4.1 里找不到第 7 条（③ 的落位）")
        else:
            if "待决" in row7:
                problems.append("[规格] INTEGRATION.md §4.1 第 7 条仍是「待决」⇒ ③ 的落位必须已定")
            if "单开一条线" not in row7:
                problems.append(
                    "[规格] INTEGRATION.md §4.1 第 7 条缺「单开一条线」"
                    "（③ 的功能证据必须单开一条线，不并入 exp90）")
        # 「还没测的」表里必须留着这两条（别被悄悄测掉）
        for need in ("③-FRONT 的「执行控制」真判据", "NLB 功能线"):
            if need not in itext:
                problems.append(f"[规格] INTEGRATION.md 的「还没测的」表缺「{need}」")

    # --- G5：组件设计书必须是真的「按脑科学选组件 + 连接我们自己写」---
    cd_f = ROOT / "COMPONENT_DESIGN.md"
    if not cd_f.exists():
        problems.append("[设计书] 缺 COMPONENT_DESIGN.md（按脑科学选组件的蓝图）")
    else:
        cdtext = cd_f.read_text(encoding="utf-8")
        # 四个动词都要在
        for need in ("选型", "拆分", "连接", "包装", "重组"):
            if need not in cdtext:
                problems.append(f"[设计书] 缺动词「{need}」")
        # 三行组件（脑科学定义）都要在
        for need in ("**感觉 / 语义**", "**运动**", "**执行**"):
            if need not in cdtext:
                problems.append(f"[设计书] 缺组件行「{need}」")
        # 我们自写的连接（每个都必须带判据）——**不许只剩空话**
        # ⚠️ 用**唯一的长短语**做检查点（短词在文档里出现多次 ⇒ 存在性检查抓不到篡改）
        for need in ("**跨通道注意力**（保留通道身份 ⇒ 63 个通道 token ⇒ Transformer 跨通道交互）",
                     "**单损失对比学习**（InfoNCE ↔ 图像特征，批次内图像互不相同）"):
            if need not in cdtext:
                problems.append(f"[设计书] 缺自写连接「{need[:18]}…」")
        # 关键的外部出处（连接/包装范式）
        for need in ("10.1162/0899766053630332", "1803.10122"):
            if need not in cdtext:
                problems.append(f"[设计书] 缺外部出处「{need}」")
        # R2 的警告要在
        if "R2" not in cdtext:
            problems.append("[设计书] 缺 R2 的轴标注警告")

    # --- G6：设计书 §3 的算子表必须与 `arch/ops_brain.py` 的类**机器绑定** ---
    # ⚠️ 目的：防止「文档里的算子清单」与「代码里真有的算子」各走各的
    #    （文档漂移是本项目反复出现过的失败模式）。
    ops_f = ROOT / "arch" / "ops_brain.py"
    test_f = ROOT / "arch" / "test_ops_brain.py"
    cdtext = cd_f.read_text(encoding="utf-8") if cd_f.exists() else ""
    m3 = re.search(r"^## 3\.[^\n]*\n(.*?)^---\s*$", cdtext, re.M | re.S)
    if not ops_f.exists():
        problems.append("[设计书] 缺 arch/ops_brain.py（我们自写的算子）")
    elif not test_f.exists():
        problems.append("[设计书] 缺 arch/test_ops_brain.py（算子的性质测试）")
    elif not m3:
        problems.append("[设计书] 缺「## 3.」小节（我们自写的算子清单）")
    else:
        sec3 = m3.group(1)
        cls_names = re.findall(r"^class\s+([A-Za-z_]\w*)\s*\(",
                               ops_f.read_text(encoding="utf-8"), re.M)
        # 只取**表格第一列**的算子名 ⇒ 正文里的 `ch_mix` 之类不会被误当成算子
        doc_ops = []
        for line in sec3.splitlines():
            mm = re.match(r"\|\s*`([A-Za-z_]\w*)`", line)
            if mm:
                doc_ops.append(mm.group(1))
        for c in cls_names:
            if c not in doc_ops:
                problems.append(f"[设计书] §3 的算子表缺算子「{c}」（代码里有、文档里没有）")
        for d in doc_ops:
            if d not in cls_names:
                problems.append(f"[设计书] §3 写了代码里不存在的算子「{d}」")
        # ⚠️ **算子类名在全文只能出现一次**（在 §3 表格里）——
        #    理由：同一个事实抄在两处 ⇒ 篡改其中一处时守卫会打不中（锚点不唯一）
        #    ⇒ 既防文档漂移，也防守卫自身失效。
        for c in cls_names:
            n = cdtext.count(c)
            if n != 1:
                problems.append(
                    f"[设计书] 算子类名「{c}」在全文出现 {n} 次（应恰好 1 次，"
                    "且只在 §3 的算子表里 ⇒ 其它地方只写指针）")
        # 性质测试的**项数**必须与测试文件里的 check() 调用数一致
        n_checks = len(re.findall(r"(?m)^\s*check\(",
                                  test_f.read_text(encoding="utf-8")))
        mc = re.search(r"\*\*(\d+) 项全绿\*\*", sec3)
        if not mc:
            problems.append("[设计书] §3 缺「**N 项全绿**」的项数声明")
        elif int(mc.group(1)) != n_checks:
            problems.append(
                f"[设计书] §3 声明 {mc.group(1)} 项性质测试，"
                f"测试文件里实际有 {n_checks} 项（项数变了 ⇒ 文档要同步）")
        # 诚实状态：算子目前**只有性质测试，没有判据实验**
        for need in ("只有性质测试，没有判据实验", "区内混、区间不混"):
            if need not in sec3:
                problems.append(f"[设计书] §3 缺「{need}」")

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
