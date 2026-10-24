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
    # --- 措辞守卫 ---
    (G_TERM, "INTEGRATION.md", "最小模型 = 位置轴上的两个原语", "最小模型 = 两个原语",
     "措辞：原语表述缺轴限定"),
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "不能借来干什么", "借用注意",
     "措辞：轴台账缺「不能借来干什么」列"),
    (G_TERM, "BRAIN_STRUCTURE_SPEC.md", "缺 pNE 侧数据", "缺别的数据",
     "措辞：轴台账的发育轴没写明状态"),
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
