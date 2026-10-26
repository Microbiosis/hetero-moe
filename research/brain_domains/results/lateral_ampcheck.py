"""lateral_ampcheck.py — **左右差异的归因检查**：是「脑的右半球优势」还是「记录链的左右偏倚」？

## 问题

`lateral_sitepairs.py` 在我们数据里查到一个**跨 10 个被试一致**的「右 > 左」偏差
（28 对同位点配对，10/10 被试，p=0.0020）。
但**这个偏差来自哪里**，我们分不清：

- **脑**：视觉/物体加工的右半球优势（文献里成立）；
- **设备**：放大器/接地/电极阻抗/参考接法的左右系统偏倚（同样会 10/10 一致）。

⇒ 判别办法（**免费**，用 exp89 已有的逐通道诱发幅度）：

| 若 | 则是 |
|---|---|
| **前部**（没有视觉诱发信号的地方）也系统性偏右 | **全局增益/记录链偏倚** ⇒ 偏差不可归因于脑 |
| 只有**后部**（O/PO/P/T，真正有视觉诱发信号的地方）偏右 | 更像**功能性的右半球优势** ⇒ 偏差可归因于脑 ⇒ 左右轴在**这个数据里**可测 |

## 输入

`results/posaxis-results/exp89_posaxis_bucket.json`（**云端**那份，含逐通道 `amp_te`；
日志重建的那份没有逐通道幅度 ⇒ 本脚本需要云端那份，用 `fetch_posaxis.py` 取）

## 输出

`results/lateral-results/lateral_ampcheck.{json,md}`
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "posaxis-results" / "exp89_posaxis_bucket.json"
OUT = HERE / "lateral-results"

POST = {"O", "PO", "P", "T"}          # 带图像语义 + 视觉诱发的位点族
ANT = {"F", "FC", "AF", "FT", "C", "CP"}   # 没有视觉诱发信号的位置


def pre(name):
    m = re.match(r"^(.*?)(\d+)$", str(name).strip())
    if not m:
        return (str(name).upper(), 0)
    return (m.group(1).upper(), int(m.group(2)))


def sign_test(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan"), float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    p = 2.0 * sum(math.comb(n, i) for i in range(min(k, m) + 1)) / (2 ** n)
    s = sorted(d)
    return k, m, float(min(1.0, p)), s[len(s) // 2]


def main():
    if not SRC.exists():
        print(f"✗ 缺 {SRC.name}（桶里那份；用 fetch_posaxis.py 取，"
              f"或改文件名）")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for r in d.get("per_subject", []):
        pos = r.get("position") or {}
        amp = pos.get("amp_te")
        names = r.get("ch_names")
        if not amp or not names:
            continue

        def grp(prefixes, side):
            vals = []
            for i, nm in enumerate(names):
                pr, dg = pre(nm)
                if pr not in prefixes or dg == 0:
                    continue
                if side == "L" and dg % 2 == 1:
                    vals.append(amp[i])
                elif side == "R" and dg % 2 == 0:
                    vals.append(amp[i])
            return sum(vals) / len(vals) if vals else float("nan")

        rows.append({
            "subject": r.get("subject"),
            "post_L": grp(POST, "L"), "post_R": grp(POST, "R"),
            "ant_L": grp(ANT, "L"), "ant_R": grp(ANT, "R"),
        })
    if not rows:
        print("✗ 没有可用的逐通道幅度")
        return 1

    dp = [x["post_R"] - x["post_L"] for x in rows]
    da = [x["ant_R"] - x["ant_L"] for x in rows]
    kp, mp, pp, mdp = sign_test(dp)
    ka, ma, pa, mda = sign_test(da)
    out = {"n_subjects": len(rows), "rows": rows,
           "posterior": {"n_pos": kp, "n_neg": mp, "p_sign": pp,
                         "median_diff": mdp, "diffs": dp},
           "anterior": {"n_pos": ka, "n_neg": ma, "p_sign": pa,
                        "median_diff": mda, "diffs": da}}
    (OUT / "lateral_ampcheck.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 左右差异的归因检查：脑的优势 vs 记录链偏倚", ""]
    L.append(f"（数据：exp89 云端 JSON 的逐通道视觉诱发幅度，{len(rows)} 个被试；"
             f"峰峰值，48–500 ms）")
    L.append("")
    L.append("| 被试 | 后部左 | 后部右 | 后部差(R−L) | 前部左 | 前部右 | 前部差(R−L) |")
    L.append("|---|---|---|---|---|---|---|")
    for x in rows:
        L.append(f"| {x['subject']} | {x['post_L']:.3f} | {x['post_R']:.3f} "
                 f"| {x['post_R']-x['post_L']:+.3f} | {x['ant_L']:.3f} "
                 f"| {x['ant_R']:.3f} | {x['ant_R']-x['ant_L']:+.3f} |")
    L.append("")
    L.append(f"- **后部**（O/PO/P/T，真有视觉诱发信号）：右>左 "
             f"**{kp}/{kp+mp}**，p = {pp:.4f}，差中位 {mdp:+.3f}")
    L.append(f"- **前部**（F/FC/AF/FT/C/CP，**没有**视觉诱发信号）：右>左 "
             f"**{ka}/{ka+ma}**，p = {pa:.4f}，差中位 {mda:+.3f}")
    L.append("")
    L.append("## 读法")
    L.append("")
    if pa < 0.05:
        L.append("⇒ **前部也系统性偏右**（前部没有视觉诱发信号可偏）"
                 "⇒ 这个偏差更可能是**全局的左右增益/记录链偏倚**"
                 "⇒ ⇒ 左右差异**不可归因于脑** ⇒ 左右轴**不能**进我们的形态主张。")
    elif pp < 0.05:
        L.append("⇒ 偏倚只出现在**有视觉诱发信号**的后部 ⇒ 更像**功能性的右半球优势**"
                 "⇒ ⇒ 左右轴在**这个数据里可测**，但仍需排除记录链偏倚（需乳突参考/双侧独立采集）。")
    else:
        L.append("⇒ 两处都不显著 ⇒ 左右差异**既不显著也不可归因**"
                 "⇒ ⇒ 左右轴**不能**进我们的形态主张。")
    L.append("")
    (OUT / "lateral_ampcheck.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"写入 {OUT}/lateral_ampcheck.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
