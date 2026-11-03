"""仪表盘风格 —— 沿用 moe_mechanism_full.mp4（旧版七幕）的视觉语言。

从 hetero-moe-demo/part_a.py 提取的「形式」层：
  · 左上固定标题块（17pt bold + 9.5pt 副标题）
  · 每幕左上幕名（head()，13pt bold）
  · 底部章节进度条（彩色段，当前幕高亮）
  · 片头的七幕目录
配色与旧版完全一致。

幕的**内容**换成 v62.3 真实数据；形式不变。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BG = '#0a0c10'; FG = '#e8eaed'; DIM = '#5b6572'
GOLD = '#F0873B'; ICE = '#64E3F6'; GREEN = '#35D0BA'; RED = '#CF3655'; VIOLET = '#b085f5'
ECOL = [GOLD, ICE, GREEN, VIOLET]


def setup_fonts():
    for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if os.path.exists(f):
            fm.fontManager.addfont(f)
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


import os  # noqa: E402


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    """旧版的淡入淡出：a 处起、b 处止。"""
    return sm((t - a) / max(1e-6, b - a)) if t < b else 1.0


def band(t, rng, fade=0.022):
    """在幕区间内淡入淡出（与旧版一致）。"""
    a, b = rng
    return seg(t, a, a + fade) * (1 - seg(t, b - fade, b))


def head(ax, txt, col, a):
    """左上角幕名 —— 旧版原样。"""
    ax.text(2.5, 54.0, txt, color=col, fontsize=13, weight='bold', alpha=a, va='center')


def vstrip(ax, x, y, w, h, vec, color, alpha, maxabs=None):
    """竖条向量条 —— 旧版原样。"""
    n = len(vec)
    m = maxabs if maxabs else max(1e-6, np.abs(vec).max())
    bw = w / n
    for k in range(n):
        v = vec[k] / m
        al = min(1.0, 0.10 + abs(v) * 0.9) * alpha
        c = color if v >= 0 else RED
        ax.add_patch(plt.Rectangle((x + k * bw, y), bw * 0.82, h, color=c, alpha=al, lw=0))


def heat(ax, x, y, w, h, M, color, alpha, vmin=None, vmax=None, cmap=None):
    """热力图 —— 用矩形网格画（与旧版一致的风格，不依赖 imshow 的插值）。

    M: [r, c] 数值。归一化到 0..1 后按亮度上色。
    """
    M = np.asarray(M, dtype=float)
    r, c = M.shape
    lo = vmin if vmin is not None else np.nanmin(M)
    hi = vmax if vmax is not None else np.nanmax(M)
    rng = max(1e-9, hi - lo)
    cw, chh = w / c, h / r
    base = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float) if isinstance(color, str) else np.asarray(color, float)
    for i in range(r):
        for j in range(c):
            v = (M[i, j] - lo) / rng
            if not np.isfinite(v):
                continue
            rgb = base * (0.12 + 0.88 * v)
            ax.add_patch(plt.Rectangle((x + j * cw, y + (r - 1 - i) * chh),
                                       cw * 0.9, chh * 0.9,
                                       color=tuple(np.clip(rgb, 0, 255) / 255),
                                       alpha=alpha, lw=0))


def progress(ax, t, names, ranges, cols, prg=None):
    """底部章节进度条 —— 旧版原样（当前幕高亮 0.85，其余 0.18）。"""
    if prg is None:
        prg = seg(t, 0.01, 0.03) * 0.9
    if prg <= 0.01:
        return
    for k, (rg, nm, cc) in enumerate(zip(ranges, names, cols)):
        x0 = 3.0 + k * (94.0 / len(names))
        w = 94.0 / len(names) - 2.0
        active = rg[0] <= t <= rg[1]
        ax.add_patch(plt.Rectangle((x0, 0.9), w, 0.85,
                                   color=cc, alpha=(0.85 if active else 0.18) * prg, lw=0))
        ax.text(x0 + w / 2, 2.4, nm, color=cc if active else DIM,
                fontsize=9, ha='center', alpha=prg)


def canvas():
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')
    return fig, ax


def title_block(ax, main, sub):
    """左上固定标题块 —— 旧版原样。"""
    ax.text(2.5, 60.0, main, color=FG, fontsize=17, weight='bold', va='center')
    ax.text(2.5, 57.2, sub, color=DIM, fontsize=9.5, va='center')
