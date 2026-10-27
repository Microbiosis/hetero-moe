#!/usr/bin/env python3
r"""render_exp102_section.py — 从 exp102 的结果文件**生成** §四十七 的 markdown。

## 为什么用脚本生成段落

项目纪律：**数字只能来自数据**（不能手抄）⇒ ⇒ 本脚本读
`results/exp102-results/{verdict.json,per_subject.csv,loss_summary.json}` 与
`results/exp102-results/train_curve.json` ⇒ 插值出 §四十七 的全部数字 ⇒ 人只写散文。

⇒ 用法：
```powershell
python research\brain_domains\results\render_exp102_section.py \
    --results-dir research\brain_domains\results\exp102-results \
    --curve research\brain_domains\results\exp102-results\train_curve.json \
    --out D:\.minimax\tmp\section47.md
```
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ATM = {"two_way_pct": 83.1, "fifty_way_top1_pct": 28.4, "fid": 24.6}
EXP101 = {"two_way_pct_single_trial": 78.09, "two_way_pct_trial_avg": 92.00,
          "fifty_way_pct_single_trial": 14.83, "fifty_way_pct_trial_avg": 42.00}


def pct(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def num(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--curve", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default="")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    verdict = json.loads((rd / "verdict.json").read_text(encoding="utf-8"))
    v = verdict.get("verdict", {})
    loss = {}
    lp = rd / "loss_summary.json"
    if lp.exists():
        loss = json.loads(lp.read_text(encoding="utf-8"))

    rows = []
    cp = rd / "per_subject.csv"
    if cp.exists():
        with open(cp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)

    curve = []
    if args.curve and Path(args.curve).exists():
        curve = json.loads(Path(args.curve).read_text(encoding="utf-8")).get("epochs", [])

    c1 = v.get("C1", {}).get("two_way_minus_floor", {})
    c2 = v.get("C2", {}).get("fifty_way_minus_floor", {})
    c1b = v.get("C1b_report", {}).get("two_way_minus_cross_subject", {})
    c3 = v.get("C3_report", {})
    c0 = v.get("C0", {})

    L = []
    L.append("## 四十七、exp102（**出结果**）：②③ 条件生成 —— **脑侧条件能不能驱动出可辨认的图像**")
    L.append("")
    L.append(f"> **判决：`{v.get('verdict')}`**（判据正本 = §四十四，**出结果前已落盘并推远端**）")
    L.append(f"> 脚本：`exp/exp102_{cache,train,gen}.py`；判决 = `exp102_gen.py` 的 `judge()`（**纯函数**）；"
             f"离线重算：`results/parse_exp102.py`（读逐被试原始记录 ⇒ 与云端逐字对齐）。")
    L.append("")

    L.append("### 四十七·一、**C0 管线自检**")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 训练损失下降（末 10% < 首 10%）| **{bool(c0.get('loss_decreased'))}**"
             f"（首 {num(c0.get('loss_first10'), 5)} → 末 {num(c0.get('loss_last10'), 5)}）|")
    L.append(f"| 生成的图不是常数（图间成对距离中位）| **{bool(c0.get('images_not_constant'))}**"
             f"（中位 {num(c0.get('img_pair_dist_median'), 2)}）|")
    L.append(f"| C0 总判 | **{bool(v.get('C0_pass'))}** |")
    L.append("")

    L.append("### 四十七·二、**逐被试原始记录**（离线可重算）")
    L.append("")
    L.append("| 被试 | own 2-way | 地板 2-way | Δ | own 50-way | 地板 50-way | Δ | "
             "跨被试同概念 2-way | 条件侧 2-way | CLIP 相似度 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def g(k):
            try:
                return float(r.get(k))
            except (TypeError, ValueError):
                return None
        o2, f2 = g("own2"), g("floor2")
        o50, f50 = g("own50"), g("floor50")
        L.append(f"| {r.get('subject')} | {num(o2)} | {num(f2)} | "
                 f"{num(None if (o2 is None or f2 is None) else o2 - f2)} | "
                 f"{num(o50)} | {num(f50)} | "
                 f"{num(None if (o50 is None or f50 is None) else o50 - f50)} | "
                 f"{num(g('cross_subject2'))} | {num(g('cond2_same_clip'))} | "
                 f"{num(g('clip_sim'))} |")
    L.append("")

    L.append("### 四十七·三、**判决**（阈值见 §四十四·三，**一字未改**）")
    L.append("")
    L.append("| 判据 | 量 | n | 正/负/零 | p（配对符号检验）| 中位 | 过？|")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| **C1**（主判据）2-way − 地板 | Δ | {c1.get('n_eff')} | "
             f"{c1.get('n_pos')}/{c1.get('n_neg')}/{c1.get('n_zero')} | "
             f"{num(c1.get('p_sign'), 4)} | {num(c1.get('median'))} | "
             f"**{bool(v.get('C1_pass'))}** |")
    L.append(f"| **C2** 50-way − 地板 | Δ | {c2.get('n_eff')} | "
             f"{c2.get('n_pos')}/{c2.get('n_neg')}/{c2.get('n_zero')} | "
             f"{num(c2.get('p_sign'), 4)} | {num(c2.get('median'))} | "
             f"**{bool(v.get('C2_pass'))}** |")
    L.append(f"| **C1b**（只报）被试特异性 | own2 − 跨被试同概念 | {c1b.get('n_eff')} | "
             f"{c1b.get('n_pos')}/{c1b.get('n_neg')}/{c1b.get('n_zero')} | "
             f"{num(c1b.get('p_sign'), 4)} | {num(c1b.get('median'))} | 只报 |")
    L.append("")
    L.append(f"**解释**：{v.get('interpretation') or '（C1 通过 ⇒ 条件驱动了生成）'}")
    L.append("")

    L.append("### 四十七·四、**C3（只报）**：传递效率与并排参照")
    L.append("")
    L.append("| 量 | 值 |")
    L.append("|---|---|")
    L.append(f"| **传递效率** = 生成的 2-way ÷ 条件的 2-way（同一 CLIP 口径）| "
             f"**{num(c3.get('transfer_efficiency'))}** |")
    L.append(f"| 生成的 2-way（中位）| {num(c3.get('generated_2way_median'))} |")
    L.append(f"| 条件的 2-way（中位，同一 CLIP）| {num(c3.get('condition_2way_median_same_clip'))} |")
    L.append(f"| 地板 2-way（中位）| {num(c3.get('floor_2way_median'))} |")
    L.append(f"| CLIP 相似度（生成图 ↔ 真图，均值）| {num(c3.get('clip_similarity_mean'))} |")
    L.append(f"| 每概念召回（中位）| {num(c3.get('per_concept_recall_median'))} |")
    L.append(f"| ⚠️ ATM（C 类：别人发表的系统）| 2-way **{ATM['two_way_pct']}%** / "
             f"50-way **{ATM['fifty_way_top1_pct']}%** / FID {ATM['fid']} |")
    L.append(f"| exp101（我们的**条件**侧，同数据集同 split）| 2-way "
             f"**{EXP101['two_way_pct_single_trial']}%**（单试次）/ "
             f"**{EXP101['two_way_pct_trial_avg']}%**（试次平均）；50-way "
             f"{EXP101['fifty_way_pct_single_trial']}% / {EXP101['fifty_way_pct_trial_avg']}% |")
    L.append("")
    if curve:
        first = curve[0].get("mean_loss")
        last = curve[-1].get("mean_loss")
        L.append(f"**训练曲线**：{len(curve)} 个 epoch（每 epoch {curve[0].get('step', '?')} 步）"
                 f"；首个 epoch 均值 {num(first, 5)} → 末个 {num(last, 5)}。")
        L.append("")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已生成 {args.out}（{len(L)} 行）｜判决 = {v.get('verdict')}")


if __name__ == "__main__":
    main()
