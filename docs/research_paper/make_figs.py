"""重绘论文的两张**数据图** (可复现)。

为什么需要它
------------
论文的图一直没有进版本库 (docs/research_paper/figs/ 与 make_figs.py 都被
gitignore), 图源 (.mmd/.svg) 也从未提交 → 换台机器就编译不出带图的 PDF。
现在做两件事:
  1. 把从已提交 PDF 里**原样抽出**的四张矢量图入库 (tools/extract_paper_figs.py),
     保证任何机器都能编译出带图的 PDF;
  2. 把这两张**数据图**的绘制过程写成脚本 (本文件), 使其可复现、可修改。
     两张示意图 (heterogeneity_spectrum / alignment_ladder) 是手工排版的
     示意图, 保留抽出版。

数字的来源 (逐个标注, 便于核对):
  fig_alignment  ← 论文 Table 3 (对齐器阶梯, 5-seed 均值 fuse MSE)
      v10: linear 1.0436 / mean 0.4753 / attn 0.3448
      v14: linear 1.1409 / attn 0.3376 / mixture 0.2575
      v15: attn 0.3220 / mixture 0.2534 / nested 0.2645
  fig_distill    ← 论文 Table 6 (chimera 蒸馏, 5-seed 均值)
      v13 MLP student: teacher 0.3245 / student-distilled 0.3076 / student-no-distill 0.4390
      v16 TinyBERT student: teacher 0.2764 / student-distilled 0.1380
      v21 BERT+ViT student: teacher 0.3248 / student-distilled 0.1620
      v17 多教师: single 0.325 / best-single 0.329 / multi-2 0.376 / multi-3 0.347 /
                  multi-3-weighted 0.338

用法: python3 docs/research_paper/make_figs.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")

# ---- fig_alignment: 对齐器阶梯 (三个受控对比) ----
ALIGN = [
    ("(a) v10: projection vs. semantic", ["linear", "mean", "attn"], [1.0436, 0.4753, 0.3448]),
    ("(b) v14: attention vs. mixture", ["linear", "attn", "mixture"], [1.1409, 0.3376, 0.2575]),
    ("(c) v15: mixture vs. nested", ["attn", "mixture", "nested"], [0.3220, 0.2534, 0.2645]),
]
ALIGN_COLORS = [["#808080", "#8ec9f2", "#1f6fb4"], ["#808080", "#1f6fb4", "#1aa160"],
                ["#1f6fb4", "#1aa160", "#d99a00"]]


def fig_alignment():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    for ax, (title, labels, vals), colors in zip(axes, ALIGN, ALIGN_COLORS):
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("fuse MSE", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, max(vals) * 1.28)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_alignment.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("写出", out)


# ---- fig_distill: 单教师蒸馏 / 多教师退化 ----
DISTILL_A = {
    "v13\nMLP student": (0.3245, 0.4390, 0.3076),
    "v16\nTinyBERT student": (0.2764, None, 0.1380),
    "v21\nBERT+ViT student": (0.3248, None, 0.1620),
}
DISTILL_B = {
    "single": 0.325, "best\nsingle": 0.329, "multi-2": 0.376,
    "multi-3": 0.347, "multi-3\nweighted": 0.338,
}
C_TEACHER, C_NODIST, C_DIST = "#d99a00", "#cc5500", "#1aa160"


def fig_distill():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    xs = list(DISTILL_A)
    w = 0.26
    for i, x in enumerate(xs):
        t, nd, d = DISTILL_A[x]
        if t is not None:
            ax1.bar(i - w, t, w, color=C_TEACHER, label="teacher (fusion ensemble)" if i == 0 else None)
        if nd is not None:
            ax1.bar(i, nd, w, color=C_NODIST, label="student, no distillation" if i == 0 else None)
        ax1.bar(i + w, d, w, color=C_DIST, label="student, distilled" if i == 0 else None)
        for off, v in ((-w, t), (0, nd), (w, d)):
            if v is not None:
                ax1.text(i + off, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax1.set_xticks(range(len(xs)))
    ax1.set_xticklabels(xs, fontsize=6.5)
    ax1.set_title("(a) single-teacher distillation", fontsize=8)
    ax1.set_ylabel("fuse MSE", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=5.6, frameon=False, loc="upper center", ncol=1)
    ax1.set_ylim(0, 0.62)

    names = list(DISTILL_B)
    vals = [DISTILL_B[n] for n in names]
    colors = [C_DIST if i < 2 else C_NODIST for i in range(len(names))]
    bars = ax2.bar(names, vals, color=colors, width=0.62)
    ax2.set_title("(b) v17: multi-teacher regression", fontsize=8)
    ax2.tick_params(labelsize=6.5)
    ax2.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=6)
    ax2.set_ylim(0, max(vals) * 1.22)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_distill.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("写出", out)


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig_alignment()
    fig_distill()


if __name__ == "__main__":
    main()
