"""plot_readout.py — exp91a 图：**三种读出下，位置轴的组级剖面**。

- 左图：每种读出下 ALL / 后部 / 前部 / 枕叶 的 RSA 中位（分组条形）
- 右图：**后 − 前**的配对差（10/10 为正？p 值多少）三种读出并排

⇒ 一眼看出：**方向在读出之间一致**，而**绝对值随读出变**（频带更弱、协方差几乎无信号）⇒
「更强的读出」≠「对本题更敏感」。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE / "readout-results" / "exp91a_readout.json"
OUT = HERE / "readout-results" / "readout_axis.png"

READOUTS = [("amp630", "amp630\n逐通道幅度（基线）"),
            ("bandpow", "bandpow\n频带功率"),
            ("covpat", "covpat\n空间模式协方差")]
GROUPS = ["ALL", "POST", "OCC", "ANT"]
G_LABEL = {"ALL": "全部 63", "POST": "后部 34", "OCC": "枕叶 8", "ANT": "前部 29"}
G_COLOR = {"ALL": "#888", "POST": "#d1495b", "OCC": "#e08e45", "ANT": "#3d5a80"}


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
    if not SRC.exists():
        print("✗ 缺 exp91a_readout.json")
        return 1
    d = json.loads(SRC.read_text(encoding="utf-8"))
    s = d.get("summary", {})
    print("字体：", pick_font())

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(wspace=0.26)

    # ---- 左：组级剖面（三种读出）----
    ax = axes[0]
    w = 0.2
    for k, (ro, _) in enumerate(READOUTS):
        e = s.get(ro) or {}
        xs = np.arange(len(GROUPS)) + (k - 1) * w
        vals = [e.get(g, {}).get("r_median", np.nan) for g in GROUPS]
        ax.bar(xs, vals, width=w * 0.92, color=[G_COLOR[g] for g in GROUPS],
               alpha=1.0 - 0.18 * k, edgecolor="#333", linewidth=0.4, zorder=3,
               label=ro)
        for xi, v in zip(xs, vals):
            if v == v:
                ax.text(xi, v + 0.0012, f"{v:+.4f}", ha="center", fontsize=6.2,
                        rotation=90, va="bottom")
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels([G_LABEL[g] for g in GROUPS], fontsize=9.5)
    ax.set_ylabel("RSA（指标 A，越高越贴图像语义）", fontsize=9.5)
    ax.set_title("三种读出下的位置轴组级剖面（10 个被试的中位）\n"
                 "每组内从左到右 = amp630 / bandpow / covpat", fontsize=10)
    ax.axhline(0, color="#333", lw=0.9)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    # 图例按**组**配色（读出由组内位置表示）
    handles = [plt.Line2D([], [], color=G_COLOR[g], lw=7, label=G_LABEL[g])
               for g in GROUPS]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.92,
              ncol=2)

    # ---- 右：后 − 前 的配对差 ----
    ax2 = axes[1]
    ros = [ro for ro, _ in READOUTS if (s.get(ro) or {}).get("_POST_minus_ANT")]
    med = [(s[ro]["_POST_minus_ANT"]["median"]) for ro in ros]
    npos = [(s[ro]["_POST_minus_ANT"]["n_pos"]) for ro in ros]
    tot = [(s[ro]["_POST_minus_ANT"]["n"]) for ro in ros]
    ps = [(s[ro]["_POST_minus_ANT"]["p_sign"]) for ro in ros]
    y = np.arange(len(ros))
    ax2.barh(y, med, color=["#d1495b" if m > 0 else "#3d5a80" for m in med],
             height=0.5, zorder=3, edgecolor="#333", linewidth=0.4)
    ax2.axvline(0, color="#333", lw=0.9)
    for yi, (m, a, b, p) in enumerate(zip(med, npos, tot, ps)):
        ax2.text(m + (0.002 if m >= 0 else -0.002), yi, f"{m:+.4f}  ({a}/{b} 为正, p={p:.4f})",
                 va="center", ha="left" if m >= 0 else "right", fontsize=8.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([ro for ro in ros], fontsize=9)
    ax2.set_xlabel("后部 − 前部（配对差，RSA）", fontsize=9.5)
    ax2.set_title("分工方向在读出之间一致", fontsize=10.5)
    ax2.grid(axis="x", alpha=0.22, zorder=0)
    ax2.set_xlim(min(min(med) * 1.8, -0.02), max(max(med) * 2.4, 0.06))

    fig.suptitle("exp91a 读出强度检验：更弱的读出（频带）也给出同样的分工方向；"
                 "协方差读出几乎无信号（对码是平方的、丢符号）", fontsize=11.5)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"✓ 写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
