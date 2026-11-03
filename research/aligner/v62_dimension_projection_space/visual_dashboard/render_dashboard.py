"""v62.3 仪表盘版 —— 形式沿用旧版 moe_mechanism_full.mp4，内容换成 v62.3 真实数据。

七幕 + 片头 + 总结 + 片尾，进度条与标题块与旧版一致。
帧数与旧版对齐（1370 帧 / 45.7 秒）。
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_dash as SD
from style_dash import (BG, FG, DIM, GOLD, ICE, GREEN, RED, VIOLET, ECOL,
                        seg, band, head, canvas, title_block, progress)
import acts

# 中文字体必须在画之前配好（否则全是豆腐块）
SD.setup_fonts()

OUT = os.environ.get('DASH_OUT', '/workspace/v62-viz/data/dash_frames')
DATA = os.environ.get('DASH_DATA', '/workspace/v62-viz/data/v62_dash.json')
METRICS = os.environ.get('DASH_METRICS',
                          '/workspace/v62-viz/data/v62_metrics.json')
os.makedirs(OUT, exist_ok=True)

SD.SLOT_COL = {'Q': '#C9E2AE', 'K': '#A6CF8C', 'V': '#83C167', 'O': '#77B05D'}

# 幕区间（与旧版一致的重叠淡入淡出）
# 幕区间：**主体内部**的时间比例（0..1，不含片头）。
# 注意 band() 用的是这个比例，而各幕内部元素用的是幕内时间 —— 两套，别混。
A1 = (0.000, 0.235)
A2 = (0.215, 0.400)
A3 = (0.385, 0.545)
A4 = (0.530, 0.680)
A5 = (0.665, 0.815)
A6 = (0.800, 0.930)
A7 = (0.915, 1.000)

F_INTRO = 60           # 片头帧数
F_A1 = 420             # ① 前向
F_A2 = 340             # ② 投影矩阵
F_A3 = 270             # ③ 槽门控
F_A4 = 270             # ④ 专家路由
F_A5 = 250             # ⑤ 对齐
F_A6 = 180             # ⑥ 逐 token 路由
F_A7 = 180             # ⑦ 总结
TOTAL = F_INTRO + F_A1 + F_A2 + F_A3 + F_A4 + F_A5 + F_A6 + F_A7

NAMES = ['① 前向', '② 投影', '③ 槽门控', '④ 专家', '⑤ 对齐', '⑥ 路由', '⑦ 总结']
COLS = [GOLD, ICE, VIOLET, RED, GREEN, GOLD, GREEN]
RANGES = [A1, A2, A3, A4, A5, A6, A7]
SPANS = [(F_INTRO, F_INTRO + F_A1), (F_INTRO + F_A1, F_INTRO + F_A1 + F_A2),
         (F_INTRO + F_A1 + F_A2, F_INTRO + F_A1 + F_A2 + F_A3),
         (F_INTRO + F_A1 + F_A2 + F_A3, F_INTRO + F_A1 + F_A2 + F_A3 + F_A4),
         (F_INTRO + F_A1 + F_A2 + F_A3 + F_A4, F_INTRO + F_A1 + F_A2 + F_A3 + F_A4 + F_A5),
         (F_INTRO + F_A1 + F_A2 + F_A3 + F_A4 + F_A5, F_INTRO + F_A1 + F_A2 + F_A3 + F_A4 + F_A5 + F_A6),
         (F_INTRO + F_A1 + F_A2 + F_A3 + F_A4 + F_A5 + F_A6, TOTAL)]

MAIN = '异构 MoE 内部机制'
SUB = '真实算子 · 真实前向 · v62.3 · 4 功能槽 × 1024 维 · 3 模态'


def draw_intro(fi):
    t = fi / max(1, F_INTRO - 1)
    fig, ax = canvas()
    title_block(ax, MAIN, SUB)
    a = seg(t, 0.05, 0.55)
    ax.text(50, 34.0, '维度投影空间', color=GOLD, fontsize=22, ha='center',
            va='center', alpha=a, weight='bold')
    ax.text(50, 27.0, '空间 = Σ_s S_s —— 由功能槽张成，不是由一个大小的数决定',
            color=DIM, fontsize=11, ha='center', va='center', alpha=a)
    b = seg(t, 0.60, 0.95)
    if b > 0.01:
        for k, (nm, cc) in enumerate(zip(NAMES, COLS)):
            x = 8.0 + k * 12.0
            aa = b * seg(t, 0.60 + k * 0.04, 0.74 + k * 0.04)
            ax.text(x, 11.0, nm[0], color=cc, fontsize=17, ha='center',
                    alpha=aa, weight='bold')
            ax.text(x, 7.5, nm[2:], color=DIM, fontsize=9, ha='center', alpha=aa)
    fig.savefig('%s/f%04d.png' % (OUT, fi), facecolor=BG, dpi=100)
    plt.close(fig)


def draw_main(fi):
    """fi: 主体的全局帧号（0..TOTAL-F_INTRO-1）"""
    # 定位到哪一幕
    gi = None
    for k, (a, b) in enumerate(SPANS):
        if a <= fi < b:
            gi = (k, fi - a, b - a)
            break
    if gi is None:
        return
    k, local, ln = gi
    t = local / max(1, ln - 1)          # 幕内时间（各幕内部元素用）
    tg = fi / max(1, TOTAL - F_INTRO - 1)  # 主体全局时间（band / progress 用）
    fig, ax = canvas()
    title_block(ax, MAIN, SUB)
    fn = [acts.act1, acts.act2, acts.act3, acts.act4, acts.act5, acts.act6, acts.act7][k]
    fn(ax, t, band(tg, RANGES[k]))
    progress(ax, tg, NAMES, RANGES, COLS)
    fig.savefig('%s/f%04d.png' % (OUT, fi + F_INTRO), facecolor=BG, dpi=100)
    plt.close(fig)


def main():
    with open(DATA) as f:
        data = json.load(f)
    with open(METRICS) as f:
        metrics = json.load(f)
    # energy_keep 不在 metrics.json 里 —— 它是 capture 时实测的
    # （投影后范数 / 投影前范数），在 v62_capture.npz 里。
    # energy_keep（投影后/投影前范数比）实测于 capture 时，存在 v62_meta.json
    _meta = json.load(open(os.environ.get('DASH_META',
                                          '/workspace/v62-viz/data/v62_meta.json')))
    metrics['_energy_keep'] = [float(x) for x in _meta['energy_keep']]
    acts.load(data, metrics)
    print('data n=%d  total=%d' % (data['n'], TOTAL), flush=True)
    for fi in range(F_INTRO):
        draw_intro(fi)
    for fi in range(TOTAL - F_INTRO):
        draw_main(fi)
        if fi % 100 == 0:
            print('  frame', fi, flush=True)
    print('done', TOTAL, flush=True)


if __name__ == '__main__':
    main()
