"""check_guards_selftest.py — **守卫的反向自测**：证明两个守卫真能抓漂移。

## 为什么需要

守卫如果抓不住漂移，它就只是个「一直在说 OK」的摆设。
⇒ 用**篡改 → 必须报错 → 还原 → 必须通过**的循环来证明它有效。
（这一步抓到过两个真问题：① 守卫只搜整节 ⇒ 同一个数在别处出现就永远通过；
② 标记串没跟着归一化排版减号 ⇒ 守卫误报「找不到行」。）

## 用法

    python research/brain_domains/results/check_guards_selftest.py

退出码 0 = 全部篡改都被抓到；非 0 = 有守卫抓不住（**守卫本身失效**）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
G_NUM = str(HERE / "check_findings_consistency.py")
G_TERM = str(HERE / "check_terminology.py")
G_ORD = str(HERE / "check_section_order.py")

# (守卫, 文件, 原串, 改串, 说明)
# ⚠️ 每个用例都必须**打在被检查的位置**上：
#    同一个数在文档里可能出现多处（表里、散文里、台账里），
#    打到没被检查的那处 ⇒ 守卫当然不会报错（那是**用例写错**，不是守卫失效）。
CASES = [
    # --- 数字守卫：位置轴（exp89），打在被检查的表行 / 配对差行上 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| **POST 后部** | 34 | **+0.0628** |", "| **POST 后部** | 34 | **+0.0699** |",
     "exp89 后部中位（表行）"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "= 中位 +0.0525，10/10 为正（符号检验 p = 0.0020）",
     "= 中位 +0.0525，10/10 为正（符号检验 p = 0.0099）",
     "exp89 后−前差 p（配对差行）"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**−0.0134，1/10 为正（p = 0.0215）**",
     "**−0.0133，1/10 为正（p = 0.0215）**", "exp90 左右配对差中位"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "+0.0393", "+0.0399", "exp90 RIGHT RSA"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "0.0059（零 0.0050）", "0.0059（零 0.0051）",
     "exp90 跨侧 acc1"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "对齐/错位 = 1.107", "对齐/错位 = 1.207",
     "exp90 对齐/错位比"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "中位 **+0.098**", "中位 **+0.199**",
     "左右轴 后部幅度差"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "| +0.0221 (10/10) | +0.0014 (**0/10**) |",
     "| +0.0222 (10/10) | +0.0014 (**0/10**) |", "exp91a bandpow 后部读出"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.0213，10/10，p=0.0020**",
     "**+0.0213，10/10，p=0.0099**", "exp91a bandpow 后−前 p"),
    # --- exp92 三支线（§二十八·二）---
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| S1a（纯自监督） | +0.0352 | **+0.0566** |",
     "| S1a（纯自监督） | +0.0352 | **+0.0666** |", "exp92-A S1a 后部"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| 后部残余（后部 R−L − 前部 R−L） | +0.0868 | 9/10 | 0.0215 |",
     "| 后部残余（后部 R−L − 前部 R−L） | +0.0868 | 9/10 | 0.0915 |",
     "exp92-B 后部残余 p"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| **630-d 手工特征** | **0.0082** | 1.6× |",
     "| **630-d 手工特征** | **0.0092** | 1.6× |", "exp92-C 手工档位"),
    # --- 章节顺序（结构级）---
    (G_ORD, "FINDINGS_TIER0_RSA.md", "## 二十六、左右轴（外侧轴）",
     "## 二十九、左右轴（外侧轴）", "章节编号顺序错位"),
    # --- R4（读出同口径）：新规则与被限定过的台账行 ---
    (G_TERM, "TERMINOLOGY.md", "**读出类型相同**", "**读出类型随便**",
     "措辞：R4 的要素被删"),
    (G_TERM, "TERMINOLOGY.md", "还有哪个因子也变了", "别的都不用管",
     "措辞：R4 的「混淆因子」要素被删"),
    (G_TERM, "INTEGRATION.md", "是「读出口径」造成的", "是「模型能力」造成的",
     "台账：A18 的读出限定被删"),
    (G_TERM, "INTEGRATION.md", "n_train=16540）：单试次 **0.0321**（6.4× 随机）",
     "n_train=2000）：单试次 **0.0321**（6.4× 随机）",
     "台账：A13 的 n_train 限定被改"),
    (G_TERM, "TERMINOLOGY.md", "**flavor 相同**", "**机器随便**",
     "措辞：R5 的 flavor 要素被删"),
    (G_TERM, "TERMINOLOGY.md", "**争用相同**", "**随便跑**",
     "措辞：R5 的争用要素被删"),
    (G_TERM, "FINDINGS_TIER0_RSA.md",
     "⇒ **要加读出限定**：在**离散码读出**下成立", "⇒ **一直成立**：在**离散码读出**下成立",
     "台账：A18 的读出限定在 FINDINGS 里被删"),
    # --- 措辞守卫 ---
    (G_TERM, "INTEGRATION.md", "最小模型 = 位置轴上的两个原语", "最小模型 = 两个原语",
     "措辞：原语表述缺轴限定"),
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "不能借来干什么", "借用注意",
     "措辞：轴台账缺「不能借来干什么」列"),
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "缺 pNE 侧数据", "缺别的数据",
     "措辞：轴台账的发育轴没写明状态"),
    # --- 奥卡姆审查（2026-10-25）新增的三项结构守卫 ---
    (G_TERM, "FINDINGS_TIER0_RSA.md",
     "拟合图数 / 训练目标 / 读出口径 / 特征窗口 四个杠杆",
     "拟合图数 / 训练目标 / 特征窗口 三个杠杆",
     "结构：§二十九 的杠杆帧与正文失配"),
    (G_TERM, "INTEGRATION.md", "**0–250 ms 窗口**", "**信号窗**",
     "台账：A14 的窗口限定被删"),
    (G_TERM, "INTEGRATION.md", "同样带「0–250 ms 窗口」限定", "同样带「信号窗」限定",
     "台账：A15 的窗口限定被删"),
    (G_TERM, "INTEGRATION.md",
     "| 「脑结构模型」的整体形态是否更优 | **未测**",
     "| 「脑结构模型」的整体形态是否更优（A18） | **未测**",
     "台账：已结案的结论又出现在未测表"),
    # --- exp94：骨干对照 × 交叉验证（数字守卫）---
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| **C2 试次平均 top-1** | 0.2100（42.0×） | 0.2150（43.0×） |",
     "| **C2 试次平均 top-1** | 0.2100（42.0×） | 0.2250（43.0×） |",
     "exp94：主表 V1 的试次平均被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| V1（保留通道身份）| **10/10** | **10/10** | **+0.0595** |",
     "| V1（保留通道身份）| **10/10** | **10/10** | **+0.0695** |",
     "exp94：位置轴 V1 的 POST−ANT 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| zeroshot acc1 | 0.0589 | **0.0573** |",
     "| zeroshot acc1 | 0.0589 | **0.0673** |",
     "exp94：自一致性表 V0 值被改"),
    (G_NUM, "INTEGRATION.md", "配对差 +0.0047、10/10、p=0.0020",
     "配对差 +0.0047、10/10、p=0.0200",
     "exp94：台账 A20 的 p 值被改"),
]


def run(guard):
    r = subprocess.run([sys.executable, guard], capture_output=True, text=True)
    return r.returncode, r.stdout


def main():
    # 基线：两个守卫都必须先通过
    base_ok = True
    for g, name in ((G_NUM, "数字守卫"), (G_TERM, "措辞守卫")):
        code, out = run(g)
        if code != 0:
            print(f"✗ 基线就不过关（{name}）：\n{out}")
            base_ok = False
    if not base_ok:
        return 1
    print("✓ 基线：两个守卫都通过")

    bad = 0
    for guard, fname, old, new, label in CASES:
        p = ROOT / fname
        orig = p.read_text(encoding="utf-8")
        if old not in orig:
            print(f"✗ {label}：锚点没找到（守卫/文档已变）")
            bad += 1
            continue
        try:
            p.write_text(orig.replace(old, new, 1), encoding="utf-8")
            code, out = run(guard)
            caught = code != 0
            print(f"{'✓' if caught else '✗'} {label}：篡改后退出码 {code}"
                  f"{'（抓到了）' if caught else '（没抓到 ⇒ 守卫失效！）'}")
            if not caught:
                bad += 1
        finally:
            p.write_text(orig, encoding="utf-8")
        code2, _ = run(guard)
        if code2 != 0:
            print(f"✗ {label}：还原后守卫没过（退出码 {code2}）")
            bad += 1

    if bad:
        print(f"\n✗ {bad} 处失效 ⇒ 守卫不可信")
        return 1
    print(f"\n✓ 全部 {len(CASES)} 处篡改都被抓到，且都能还原")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
