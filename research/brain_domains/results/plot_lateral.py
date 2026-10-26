"""plot_lateral.py — 左右轴（外侧轴）结果图。

## 两张图

- **左：同位点左右配对差**（每个点 = 一个位点对，如 O1/O2）
  x 轴按解剖前后排（额极 → 枕），y 轴 = 该对的「左 − 右」RSA 差中位
  ⇒ 一眼看出：**带图像语义的后部位点（O/PO/P）右侧更强**，而**前部也偏右**
  （后者就是「不可归因」的来源）。
- **右：归因检查**（每个点 = 一个被试）
  后部（O/PO/P/T）的右−左幅度差 vs 前部（F/FC/AF/FT/C/CP）的右−左幅度差
  ⇒ **两处都偏右** ⇒ 差异不只出现在「有视觉信号」的地方 ⇒ 来源分不清。

## 输入

`results/lateral-results/lateral_sitepairs.json`、`lateral_ampcheck.json`
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "lateral-results"
OUT = RES / "lateral_axis.png"

# 位点族 → 颜色（后部一族用暖色，前部用冷色）
FAM_COLOR = {"O": "#d1495b", "PO": "#d1495b", "P": "#e08e45", "T": "#e8b04b",
             "C": "#3d5a80", "CP": "#4f7ca8", "F": "#5b8fb9", "FC": "#7fa8c9",
             "AF": "#9dbdd4", "FT": "#9dbdd4", "TP": "#6b8fb0", "Fp": "#9dbdd4"}
FAM_LABEL = {"O": "枕", "PO": "枕顶", "P": "顶", "T": "颞", "C": "中央",
             "CP": "中央顶", "F": "额", "FC": "额中央", "AF": "额极",
             "FT": "额颞", "TP": "顶颞", "Fp": "额极"}
# 前→后 的位点顺序
ORDER = ["Fp", "AF", "F", "FC", "FT", "C", "CP", "TP", "P", "PO", "T", "O"]


def site_prefix(name):
    m = re.match(r"^(.*?)(\d+)$", str(name).strip())
    return m.group(1).upper() if m else str(name).upper()


def pick_font():
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [fam]
            return fam
        except Exception:
            continue
    return None


def main():
    if not (RES / "lateral_sitepairs.json").exists():
        print("✗ 缺 lateral_sitepairs.json")
        return 1
    sp = json.loads((RES / "lateral_sitepairs.json").read_text(encoding="utf-8"))
    am = json.loads((RES / "lateral_ampcheck.json").read_text(encoding="utf-8"))
    print("字体：", pick_font())

    per_pair = sp["per_pair"]
    rows = []
    for key, e in per_pair.items():
        l, r = key.split("/")
        rows.append((site_prefix(l), l, r, e))
    rows.sort(key=lambda t: (ORDER.index(t[0]) if t[0] in ORDER else 99, t[1]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(wspace=0.28)

    # ---- 左：逐位点对的左右差 ----
    ax = axes[0]
    x = np.arange(len(rows))
    ys = [e["median_diff"] for _, _, _, e in rows]
    cols = [FAM_COLOR.get(fam, "#888") for fam, _, _, _ in rows]
    ax.bar(x, ys, color=cols, width=0.66, zorder=3)
    ax.axhline(0, color="#333", lw=0.9)
    for xi, (_, l, r, e) in zip(x, rows):
        ax.text(xi, e["median_diff"] + (0.0008 if e["median_diff"] >= 0 else -0.0008),
                f"{l}/{r}", rotation=90, fontsize=5.6, ha="center",
                va="bottom" if e["median_diff"] >= 0 else "top", color="#333")
    ax.set_xticks([])
    ax.set_ylabel("同位点配对差（RSA，左 − 右）", fontsize=9.5)
    ax.set_title("同位点左右配对：28 个位点对（横轴按前→后排）", fontsize=10.5)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    agg = sp.get("per_subject_aggregate", {})
    if "all_pairs" in agg:
        e = agg["all_pairs"]
        ax.text(0.985, 0.955,
                f"逐被试聚合：{e['n_pos']}/{e['n_pos']+e['n_neg']} 偏左"
                f"（即 {e['n_pos']+e['n_neg']-e['n_pos']}/"
                f"{e['n_pos']+e['n_neg']} 偏右），p = {e['p_sign']:.4f}",
                transform=ax.transAxes, fontsize=8.8, color="#111", ha="right",
                va="top",
                bbox=dict(fc="#fff8e1", ec="#999", lw=0.6))
    # 图例（放到坐标区外，别压柱子）
    seen, handles = set(), []
    for fam in ORDER:
        if fam in seen or fam not in FAM_COLOR:
            continue
        seen.add(fam)
        handles.append(plt.Line2D([], [], color=FAM_COLOR[fam], lw=6,
                                  label=FAM_LABEL.get(fam, fam)))
    ax.legend(handles=handles, fontsize=7.2, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), framealpha=0.95)

    # ---- 右：归因检查 ----
    ax2 = axes[1]
    rws = am["rows"]
    xs = np.arange(len(rws))
    dp = [r["post_R"] - r["post_L"] for r in rws]
    da = [r["ant_R"] - r["ant_L"] for r in rws]
    ax2.plot(xs, dp, "o-", color="#d1495b", lw=1.6, ms=6, label="后部 O/PO/P/T（有视觉诱发信号）")
    ax2.plot(xs, da, "s-", color="#3d5a80", lw=1.6, ms=6, label="前部 F/FC/AF/FT/C/CP（无视觉诱发信号）")
    ax2.axhline(0, color="#333", lw=0.9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([r["subject"].replace("sub-", "") for r in rws], fontsize=8.5)
    ax2.set_xlabel("被试", fontsize=9.5)
    ax2.set_ylabel("诱发幅度的 右 − 左", fontsize=9.5)
    ax2.set_title("归因检查：前部（无视觉信号）也偏右 => 来源不可归因", fontsize=10.5)
    ax2.grid(alpha=0.22)
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)
    # 前部的量级小 ~10 倍 ⇒ 单独标出来，并给它一个可读的注记
    ax2.text(0.985, 0.52,
             f"后部 中位 {am['posterior']['median_diff']:+.3f}"
             f"（右>左 {am['posterior']['n_pos']}/{am['posterior']['n_pos']+am['posterior']['n_neg']}）\n"
             f"前部 中位 {am['anterior']['median_diff']:+.3f}"
             f"（右>左 {am['anterior']['n_pos']}/{am['anterior']['n_pos']+am['anterior']['n_neg']}）"
             f"\n前部量级小 ~10 倍，但方向一致（5/5 为正）\n"
             f"=> 差异不只出现在「有视觉信号」的地方",
             transform=ax2.transAxes, fontsize=8.3, color="#111", ha="right",
             va="center",
             bbox=dict(fc="#fff8e1", ec="#999", lw=0.6))

    fig.suptitle("左右轴（外侧轴）：测得到系统性左右差异，但来源不可归因 => 不进形态主张",
                 fontsize=12)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"✓ 写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
