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
    # --- exp97b 三支线（§二十八·二）---
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| S1a（纯自监督） | +0.0352 | **+0.0566** |",
     "| S1a（纯自监督） | +0.0352 | **+0.0666** |", "exp97b-A S1a 后部"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| 后部残余（后部 R−L − 前部 R−L） | +0.0868 | 9/10 | 0.0215 |",
     "| 后部残余（后部 R−L − 前部 R−L） | +0.0868 | 9/10 | 0.0915 |",
     "exp97b-B 后部残余 p"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| **630-d 手工特征** | **0.0082** | 1.6× |",
     "| **630-d 手工特征** | **0.0092** | 1.6× |", "exp97b-C 手工档位"),
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
    # --- exp95：V1 复现 + Job 间漂移 + 位置轴「9–10/10」的降级表述 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| zeroshot acc1 | 0.0589 | 0.0627 | **0.0622** | 0.0004 |",
     "| zeroshot acc1 | 0.0589 | 0.0627 | **0.0722** | 0.0004 |",
     "exp95：V1 复现值被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| **exp95 的 V1 复现** | **9/10** | **10/10** | +0.0568 |",
     "| **exp95 的 V1 复现** | **10/10** | **10/10** | +0.0568 |",
     "exp95：位置轴 POST 侧的 9/10 被改成 10/10"),
    (G_NUM, "INTEGRATION.md", "**POST 侧 9–10/10**", "**POST 侧 10/10**",
     "exp95：台账 A14 的「9–10/10」降级表述被删"),
    (G_NUM, "FINDINGS_TIER0_RSA.md",
     "| **C1 单试次 top-1** | **0.0622**（12.4×）| 0.0619（12.4×）|",
     "| **C1 单试次 top-1** | **0.0622**（12.4×）| 0.0719（12.4×）|",
     "exp95：V2 的 C1 值被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "`HYBRID_WORSE`", "`HYBRID_BETTER`",
     "exp95：判决字符串被改（与 JSON 双向绑定）"),
    # --- 三组切分（按脑科学解剖）---
    (G_TERM, "INTEGRATION.md", "| **C19** |", "| **C18** |",
     "三组：C19（前部不是一个功能单位的外部参照）被删"),
    (G_TERM, "INTEGRATION.md", "「前部（ANT 29）」不是一个原语", "「前部（ANT 29）」就是一个原语",
     "三组：A14 的「ANT 不是一个原语」限定被删"),
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "**三组切分**", "**两组合并**",
     "三组：规格 §4.2 的三组切分被删"),
    (G_NUM, "BRAIN_STRUCTURE_SPEC.md", "| **CENT** 运动 | 14 |", "| **CENT** 运动 | 15 |",
     "三组：§4.1 分组表的 CENT 通道数被改"),
    (G_NUM, "BRAIN_STRUCTURE_SPEC.md", "| **FRONT** 执行 | 15 |", "| **FRONT** 执行 | 14 |",
     "三组：§4.1 分组表的 FRONT 通道数被改"),
    # --- ③ 的落位（单开一条功能线）---
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "（**形状共享，权重不共享**）",
     "（**形状共享，权重共享**）",
     "③：§4.4 标题的「形状共享、权重不共享」被改"),
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "**弱代理**，**不是**「执行控制」",
     "**强证据**，**不是**「执行控制」",
     "③：§4.4 把 DMFC 的弱代理说成强证据"),
    (G_TERM, "INTEGRATION.md", "③ 的功能证据单开一条线", "③ 的功能证据并入 exp90",
     "③：台账 §4.1 第 7 条的落位被改"),
    (G_TERM, "INTEGRATION.md", "③-FRONT 的「执行控制」真判据", "③-FRONT 的真判据",
     "③：「还没测的」表里那条空白被删"),
    # --- G5：组件设计书不许退化成空话 ---
    (G_TERM, "COMPONENT_DESIGN.md",
     "**跨通道注意力**（保留通道身份 ⇒ 63 个通道 token ⇒ Transformer 跨通道交互）",
     "**跨通道池化**（保留通道身份 ⇒ 63 个通道 token ⇒ Transformer 跨通道交互）",
     "设计书：我们自写的连接「跨通道注意力」被改"),
    (G_TERM, "COMPONENT_DESIGN.md", "10.1162/0899766053630332", "10.1162/00000000",
     "设计书：包装范式的外部出处（Eliasmith 2005）被改"),
    (G_TERM, "COMPONENT_DESIGN.md", "| **运动**<br>（CENT 14ch）|", "| **动作**<br>（CENT 14ch）|",
     "设计书：组件行「运动」被改"),
    # --- G6：算子表 ⇄ 代码类名 ⇄ 性质测试项数 ---
    (G_TERM, "COMPONENT_DESIGN.md", "`RegionLocalLowRank`", "`RegionLocalLowRanks`",
     "设计书：§3 的算子名与代码类名脱钩"),
    (G_TERM, "COMPONENT_DESIGN.md", "**33 项全绿**", "**34 项全绿**",
     "设计书：§3 的性质测试项数与测试文件脱钩"),
    (G_TERM, "COMPONENT_DESIGN.md", "exp90 的判据实验因模型塌缩而全实验作废",
     "exp90 的判据实验全部通过",
     "设计书：§3 把「因模型塌缩而全实验作废」说成「全部通过」"),
    # --- exp96：数字绑定 + **两份判决都要留痕** ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "中位 **−0.0348** ✅", "中位 **−0.0349** ✅",
     "exp96：§三十二·三 的 C 中位差被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "5/10 正、5/10 负，p=1.0000，中位 +0.0023",
     "6/10 正、4/10 负，p=1.0000，中位 +0.0023",
     "exp96：§三十二·三 的 s1b 锚点计数被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "① 预登记判决 = `ANCHOR_FAIL`",
     "① 预登记判决 = `THREE_GROUPS_HOLD`",
     "exp96：预登记判决被改成事后判决（篡改留痕）"),
    (G_NUM, "INTEGRATION.md", "两份判决都在", "都在",
     "exp96：台账 A21 的「两份判决」留痕被删"),
    # --- exp90：判决 / 根因 / 「C2 假阳性」留痕 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "预登记判决 = `PIPELINE_SELFCHECK_FAIL`",
     "预登记判决 = `THREE_GROUPS_HOLD`",
     "exp90：预登记判决被改成别的"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "（实测：**ln(256) = 5.5452**，30 个 epoch 一动不动）",
     "（实测：**ln(512) = 5.5452**，30 个 epoch 一动不动）",
     "exp90：根因里的具体数字被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**C2（置换敏感性）会给出假阳性**",
     "**C2（置换敏感性）没问题**",
     "exp90：C2 假阳性那条留痕被删"),
    (G_NUM, "INTEGRATION.md", "PIPELINE_SELFCHECK_FAIL", "THREE_GROUPS_HOLD",
     "exp90：台账 A22 的判决字符串被改"),
    # --- exp90b：NLB 功能判据③ 的判决 + 「形态判据①需要新数据」这条留痕 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "C1_CENT_OK", "C1_CENT_FAIL",
     "exp90b：§三十四 的判决字符串被改"),
    (G_NUM, "INTEGRATION.md", "`C1_CENT_OK`", "`C1_CENT_FAIL`",
     "exp90b：台账 A22 的 exp90b 判决被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "两个数据集各缺一半", "两个数据集互不缺口",
     "exp90b：「各缺一半」这条留痕被删"),
    # --- §三十五（收新数据摸底）：注册源 id + 「收新数据先缓」这条结论 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "DANDI:000017", "DANDI:000999",
     "摸底：Steinmetz 的注册源 id 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**收新数据先缓**", "**立即收新数据**",
     "摸底：「收新数据先缓」这条结论被改"),
    # --- §三十六（exp97a/exp97b）：判决与关键数字 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 = `C0_FAIL`**", "**判决 = `C1_OK`**",
     "exp97b：判决字符串被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**p = 0.0703**", "**p = 0.0003**",
     "exp97b：符号检验的 p 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 `C0_FAIL`**", "**判决 `C1_OK`**",
     "exp97a：判决字符串被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "p=0.7266", "p=0.0266",
     "exp97a：符号检验的 p 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**exp97c（零新数据）**", "**exp93z（零新数据）**",
     "挖底：下一步的实验编号被改"),
    # --- §三十八（exp97c：C0′ 逐带各自标签）：判决 + 核心数字 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 = `INSUFFICIENT_SESSIONS`**",
     "**判决 = `C1_OK`**",
     "exp97c：判决字符串被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**p = 0.2500**", "**p = 0.0500**",
     "exp97c：符号检验的 p 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**可用 session = 3 个**", "**可用 session = 8 个**",
     "exp97c：可用 session 数被改（这一轮的核心产出）"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**中位 Δ = +3.2523**", "**中位 Δ = -3.2523**",
     "exp97c：中位 Δ 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**3/3 正**", "**1/3 正**",
     "exp97c：符号检验的正向计数被改"),
    # --- §三十九（exp98：Allen VC-NP 正/负对照）：判决 + 核心数字 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 = `C1_FAIL`**",
     "**判决 = `C1_OK`**",
     "exp98：判决字符串被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**p = 0.0963**", "**p = 0.0063**",
     "exp98：符号检验的 p 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**5/18 正**", "**18/5 正**",
     "exp98：符号检验的正向计数被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**中位 Δ = -2.0845**", "**中位 Δ = +2.0845**",
     "exp98：中位 Δ 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**进入主判据 = 18 个（共 32）**",
     "**进入主判据 = 28 个（共 32）**",
     "exp98：进入主判据的 session 数被改（这一轮的核心产出）"),
    # --- §四十（exp100：组内 vs 跨组）：判决 + 核心数字 ---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 = `C1_OK`**（组内 vs 跨组）",
     "**判决 = `C1_FAIL`**（组内 vs 跨组）",
     "exp100：判决字符串被改（判据① 第一次通过验收的那个判决）"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**26/30 正**", "**30/26 正**",
     "exp100：符号检验的正向计数被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**p = 0.0001**", "**p = 0.0002**",
     "exp100：符号检验的 p 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**中位 Δ = +7.4021**", "**中位 Δ = -7.4021**",
     "exp100：中位 Δ 被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**进入主判据 = 30 个（共 32）**（组内 vs 跨组）",
     "**进入主判据 = 28 个（共 32）**（组内 vs 跨组）",
     "exp100：进入主判据的 session 数被改"),
    # --- §四十一（exp90c：判据② 接口的因果价值）---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 = `INTERFACE_VALUE_OK`**",
     "**判决 = `NO_INTERFACE_VALUE`**",
     "exp90c：判决字符串被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.0476**", "**+0.0076**",
     "exp90c：POST 的 200-way 中位被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.1227**", "**+0.2227**",
     "exp90c：POST 的图像 RSA 中位被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+2.1002**", "**-2.1002**",
     "exp90c：top-down 接口的中位被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "−1.0636", "+1.0636",
     "exp90c：背侧流接口的中位被改（全角负号 ⇒ 守卫按归一化后的 ASCII 串比对）"),
    # --- §四十二（exp101：②③ 线第一个里程碑 · 脑侧条件的可辨识性）---
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**判决 = `CONDITION_USABLE`**",
     "**判决 = `CONDITION_TOO_WEAK`**",
     "exp101：判决字符串被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.0589**", "**+0.0089**",
     "exp101：200-way 条件自检中位被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.7809**", "**+0.2809**",
     "exp101：2-way 中位被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.1283**", "**-0.1283**",
     "exp101：50-way 判决量（与机会的差）被改"),
    (G_NUM, "FINDINGS_TIER0_RSA.md", "**+0.2809**", "**+0.0809**",
     "exp101：2-way 判决量（与机会的差）被改"),
]


def run(guard):
    r = subprocess.run([sys.executable, guard], capture_output=True, text=True)
    return r.returncode, r.stdout


def write_retry(p, text, tries=5, wait=0.4):
    """**写盘要重试**：Windows 上偶发 `OSError: [Errno 22]`（文件句柄竞争），
    而这个脚本的写入是**破坏性**的（篡改）⇒ 写失败 ⇒ 文件可能停在坏状态 ⇒ 必须重试。"""
    import time
    last = None
    for _ in range(tries):
        try:
            p.write_text(text, encoding="utf-8")
            return True
        except OSError as e:      # 瞬时占用/句柄竞争
            last = e
            time.sleep(wait)
    raise RuntimeError(f"写 {p} 连续 {tries} 次失败：{last}")


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
            write_retry(p, orig.replace(old, new, 1))
            code, out = run(guard)
            caught = code != 0
            print(f"{'✓' if caught else '✗'} {label}：篡改后退出码 {code}"
                  f"{'（抓到了）' if caught else '（没抓到 ⇒ 守卫失效！）'}")
            if not caught:
                bad += 1
        finally:
            # ⚠️ **还原必须重试且必须成功**（写失败 ⇒ 文件停在篡改状态 ⇒ 整份文档是坏的）
            try:
                write_retry(p, orig)
            except Exception as e:
                print(f"✗✗ {label}：**还原失败** ⇒ {fname} 可能停在篡改状态！"
                      f" 请 `git checkout -- {fname}` 恢复。原错误：{e}")
                return 1
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
