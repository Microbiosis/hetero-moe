"""plot_posaxis.py — exp89 位置轴结果图（头皮图 + 组级条形图）。

## 图的内容

- **左图：逐通道语义对齐的头皮图**（63 电极，色 = 该通道单独的 RSA 中位）
  ⇒ 一眼看出「位置轴」是不是真轴（后部亮、前部暗）。
- **右图：通道组消融**（ALL / 后部 POST / 前部 ANT / 枕 OCC / 顶 PARI / 颞 TEMP /
  中 CENT / 额 FRONT 的 RSA 中位），叠加**同尺寸随机通道组**的零分布带
  与**打乱通道顺序**后「后−前」差的零分布。

## 输入

`results/posaxis-results/exp89_posaxis.json`
（桶里下载的那份有逐通道曲线；日志重建的那份没有 ⇒ 左图会跳过）

## 用法

    python research/brain_domains/results/plot_posaxis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge  # noqa: F401

HERE = Path(__file__).resolve().parent
RES = HERE / "posaxis-results"
JSON = RES / "exp89_posaxis.json"
PNG = RES / "exp89_posaxis.png"

# ---- 10-10 标准电极位置（头心为原点、鼻朝上，单位=头半径）----
POS = {
    "Fp1": (-0.29, 0.83), "Fp2": (0.29, 0.83),
    "AF7": (-0.55, 0.62), "AF3": (-0.19, 0.68), "AFz": (0.0, 0.72),
    "AF4": (0.19, 0.68), "AF8": (0.55, 0.62),
    "F7": (-0.72, 0.36), "F5": (-0.40, 0.42), "F3": (-0.22, 0.44),
    "F1": (-0.09, 0.44), "Fz": (0.0, 0.44), "F2": (0.09, 0.44),
    "F4": (0.22, 0.44), "F6": (0.40, 0.42), "F8": (0.72, 0.36),
    "FT9": (-0.86, 0.06), "FT7": (-0.68, 0.10), "FT8": (0.68, 0.10),
    "FT10": (0.86, 0.06),
    "FC5": (-0.40, 0.22), "FC3": (-0.22, 0.26), "FC1": (-0.09, 0.27),
    "FCz": (0.0, 0.27), "FC2": (0.09, 0.27), "FC4": (0.22, 0.26),
    "FC6": (0.40, 0.22),
    "T7": (-0.76, -0.22), "T8": (0.76, -0.22),
    "C5": (-0.36, -0.10), "C3": (-0.20, -0.12), "C1": (-0.09, -0.12),
    "Cz": (0.0, -0.12), "C2": (0.09, -0.12), "C4": (0.20, -0.12),
    "C6": (0.36, -0.10),
    "TP9": (-0.72, -0.42), "TP7": (-0.55, -0.40), "TP8": (0.55, -0.40),
    "TP10": (0.72, -0.42),
    "CP5": (-0.34, -0.30), "CP3": (-0.19, -0.28), "CP1": (-0.09, -0.27),
    "CPz": (0.0, -0.27), "CP2": (0.09, -0.27), "CP4": (0.19, -0.28),
    "CP6": (0.34, -0.30),
    "P7": (-0.66, -0.56), "P5": (-0.38, -0.50), "P3": (-0.20, -0.48),
    "P1": (-0.09, -0.47), "Pz": (0.0, -0.47), "P2": (0.09, -0.47),
    "P4": (0.20, -0.48), "P6": (0.38, -0.50), "P8": (0.66, -0.56),
    "PO7": (-0.52, -0.72), "PO3": (-0.26, -0.68), "POz": (0.0, -0.67),
    "PO4": (0.26, -0.68), "PO8": (0.52, -0.72),
    "O1": (-0.32, -0.84), "Oz": (0.0, -0.86), "O2": (0.32, -0.84),
}

GROUP_CN = {"OCC": "枕叶", "PARI": "顶叶", "TEMP": "颞叶", "CENT": "中央",
            "FRONT": "额叶", "POST": "后部", "ANT": "前部", "ALL": "全部"}
ORDER = ["ALL", "POST", "ANT", "OCC", "PARI", "TEMP", "CENT", "FRONT"]
JSON2 = RES / "per_channel_curve.json"


def pick_font():
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "Source Han Sans SC", "WenQuanYi Micro Hei"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [fam]
            return fam
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


def scalp(ax, names, vals, title):
    """头皮图：鼻子朝上，耳朵在左右。"""
    ax.set_aspect("equal")
    ax.axis("off")
    # 头
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.4, ec="#444"))
    # 鼻子
    ax.plot([-0.09, 0, 0.09], [0.98, 1.16, 0.98], color="#444", lw=1.4)
    # 耳朵
    for s in (-1, 1):
        ax.add_patch(Wedge((s * 1.0, 0.0), 0.22, -70 if s > 0 else 110,
                           70 if s > 0 else 250, width=0.10,
                           fill=False, lw=1.2, ec="#888"))
    # 后/前分界（把 34 个后部通道与 29 个前部通道分开）
    th = np.linspace(0.15, np.pi - 0.15, 120)
    ax.plot(0.62 * np.cos(th), 0.62 * np.sin(th), color="#666", lw=1.1, ls="--")
    ax.text(0.0, -1.12, "后部 34（枕/顶/颞）", fontsize=8, ha="center", color="#555")
    ax.text(0.0, 1.20, "前部 29（额/中央）", fontsize=8, ha="center", color="#555")
    v = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(v).any():
        ax.text(0, 0, "无逐通道数据", ha="center", va="center")
        return
    lo, hi = float(np.nanpercentile(v, 5)), float(np.nanmax(v))
    norm = plt.Normalize(vmin=min(lo, 0.0), vmax=hi if hi > 0 else 1e-6)
    cmap = plt.get_cmap("RdBu_r")
    for nm, val in zip(names, v):
        if nm not in POS or not np.isfinite(val):
            continue
        x, y = POS[nm]
        ax.scatter([x], [y], c=[val], cmap=cmap, norm=norm, s=110,
                   edgecolors="#222", linewidths=0.6, zorder=3)
        ax.text(x, y - 0.13, nm, fontsize=5.0, ha="center", color="#333")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.030, pad=0.01, location="left")
    cb.set_label("单通道 RSA", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def main():
    if not JSON.exists():
        print(f"✗ 没有 {JSON}（先跑 fetch_posaxis.py 或 parse_posaxis_from_logs.py）")
        return 1
    data = json.loads(JSON.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    per_subject = data.get("per_subject", [])
    fam = pick_font()
    if fam:
        print(f"字体：{fam}")
    else:
        print("⚠ 没找到中文字体，图内用英文标签")

    fig = plt.figure(figsize=(14.2, 5.8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.30)

    # ---- 左：头皮图 ----
    ax1 = fig.add_subplot(gs[0, 0])
    names, curve = None, None
    for r in per_subject:
        if r.get("per_channel_rsa") and r.get("ch_names"):
            names = r["ch_names"]
            curve = np.asarray(r["per_channel_rsa"], dtype=np.float64)
            break
    if curve is not None and len(per_subject) > 1:
        allc = np.array([r["per_channel_rsa"] for r in per_subject
                         if r.get("per_channel_rsa")])
        curve = np.median(allc, axis=0)
    # 曲线也可以来自 exp89b（专门算逐通道曲线的那个 Job）
    if curve is None and JSON2.exists():
        d2 = json.loads(JSON2.read_text(encoding="utf-8"))
        c2 = d2.get("median_curve")
        n2 = d2.get("ch_names")
        if c2 and n2:
            names, curve = n2, np.asarray(c2, dtype=np.float64)
            print(f"曲线来自 {JSON2.name}")
    ttl = ("逐通道语义对齐（63 电极，跨被试中位）"
           if curve is not None else "逐通道语义对齐（无数据）")
    scalp(ax1, names or [], curve if curve is not None else [], ttl)
    if curve is not None:
        ax1.text(0.5, -0.06, "后部（枕/顶/颞）亮、前部（额/中）暗：位置轴是真轴",
                 transform=ax1.transAxes, ha="center", va="top", fontsize=8.5,
                 color="#333")

    # ---- 右：组级条形 ----
    ax2 = fig.add_subplot(gs[0, 1])
    gs_ = [g for g in ORDER if g in summary and "r_median" in summary[g]]
    vals = [summary[g]["r_median"] for g in gs_]
    colors = []
    for g in gs_:
        if g == "ALL":
            colors.append("#888")
        elif g in ("POST", "OCC", "PARI", "TEMP"):
            colors.append("#d1495b")
        else:
            colors.append("#3d5a80")
    x = np.arange(len(gs_))
    bars = ax2.bar(x, vals, color=colors, width=0.62, zorder=3)
    # 同尺寸随机通道组零分布带
    rn = summary.get("random_group_null_summary", {})
    for k, (xi, g) in enumerate(zip(x, gs_)):
        if g in rn:
            m = rn[g]["median_of_null_medians"]
            ax2.plot([xi - 0.31, xi + 0.31], [m, m], color="#222", lw=1.8,
                     ls="--", zorder=4)
            # 标注重叠 ⇒ 交替上下错开
            dy = 0.0022 if k % 2 == 0 else -0.0042
            va = "bottom" if k % 2 == 0 else "top"
            ax2.text(xi, m + dy, "同尺寸随机组中位", fontsize=7,
                     ha="center", va=va, color="#222")
    # 打乱通道顺序的「后−前」差零分布
    nul = summary.get("POST_minus_ANT_null")
    if nul:
        ax2.axhspan(nul["null_median"], nul["null_max"], color="#999", alpha=0.18,
                    zorder=1, label="打乱通道顺序后的差值零分布")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.002),
                 f"{v:+.4f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([GROUP_CN.get(g, g) for g in gs_], fontsize=9)
    ax2.set_xlabel("通道组（后部 = 枕+顶+颞；前部 = 额+中央）", fontsize=8.5)
    ax2.set_ylabel("RSA（指标 A，越高越贴图像语义）", fontsize=9)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_title("通道组消融（630-d 原始特征，同基线表协议）", fontsize=10)
    ax2.grid(axis="y", alpha=0.25, zorder=0)
    ax2.legend(fontsize=7.5, loc="upper right")
    pma = summary.get("POST_minus_ANT")
    if pma:
        ax2.text(0.02, 0.04,
                 f"后部 − 前部：{pma['median']:+.4f}（{pma['n_pos']}/{pma['n']} 为正，"
                 f"p={pma['p_sign']:.4f}）",
                 transform=ax2.transAxes, fontsize=9, color="#111",
                 bbox=dict(fc="#fff8e1", ec="#999", lw=0.6))

    fig.suptitle("exp89 位置轴三测量（零训练）：后部带语义、前部带多少？",
                 fontsize=12)
    fig.savefig(PNG, bbox_inches="tight")
    print(f"✓ 写入 {PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
