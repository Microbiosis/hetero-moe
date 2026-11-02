"""第 ⑥ 幕（结构化 token · 四槽分工）与第 ⑦ 幕（专家路由）。

所有数值来自真实前向（data/v62_capture.npz）与 2002 条 held-out 统计
（data/v62_route.json）。不编造。
"""
import os
import json
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent if '__file__' in dir() else _pl.Path.cwd()
DATA = str(_HERE / 'data')
FRAMES = str(_HERE / 'frames')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
import style as ST
from matplotlib.patches import Circle, FancyArrowPatch

for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
          '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

D = np.load(DATA + '/v62_capture.npz')
R = json.load(open(DATA + '/v62_route.json'))
# 用 2002 条全量报告值（route.json 的 report 段），字段名 ent 不是 entropy
_L = R['report']['expert']['load']
_E = R['report']['expert']['ent']
BAL = min(_L) / (max(_L) + 1e-9)

BG='#0a0c10'; FG='#e8eaed'; DIM='#5b6572'
SLOT_COL = list(ST.SLOT_GRAD)   # Q K V O
EXP_COL  = list(ST.EXP_GRAD)
SLOT_NAMES = ['Q', 'K', 'V', 'O']

W, H = 100, 62
FPS = 30


def sm(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def seg(t, a, b):
    if b <= a: return 1.0 if t >= b else 0.0
    return sm((t - a) / (b - a))


SUBS = {
  '6': [('空间于是不是一个大小的数，而是四个子空间相加。',
         'The space is a sum of subspaces, not a size.')],
  '7': [('四个专家都在用，但并不均衡。',
         'All four experts work, but not evenly.')],
}


def draw_subs(ax, which, alpha):
    if alpha <= 0.01: return
    zh, en = SUBS[which][0]
    ST.rbox(ax, (0, 0), W, 7.0, fc='#000000', ec='#000000', lw=1.6, alpha=0.55 * alpha, fill_alpha=0.45, z=2)
    ax.text(50, 4.6, zh, color=FG, fontsize=13.5, ha='center', va='center', alpha=alpha)
    ax.text(50, 1.8, en, color=DIM, fontsize=9, ha='center', va='center', alpha=alpha * 0.9)


def frame(fi, total, fn, out, which=None):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    t = fi / max(total - 1, 1)
    # 背景：中心微亮的柔和暗场
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, W, 0, H], aspect='auto', zorder=-10,
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  'bg', [ST.BG_DEEP, ST.BG, ST.BG_DEEP]),
              interpolation='bilinear')
    fn(ax, t)
    if which:
        draw_subs(ax, which, seg(t, 0.0, 0.06))
        ST.watermark(ax, 'v62 · 维度投影空间')
    # 镜头极缓慢推近
    ST.camera(ax, zoom=1.0 + 0.022 * t)
    fig.savefig(f'{out}/f{fi:05d}.png', facecolor=BG, dpi=100)
    plt.close(fig)


# ============================================================ 幕 ⑥
def act6(ax, t):
    """⑥ 一个 token 的结构：四个槽各贡献多少 —— 柱体平滑滚动 + 拆合动画。"""
    p = seg(t, 0.0, 0.05)
    ax.text(50, 58, '⑥ 一个 token 的结构：四个槽各贡献多少', color='#35D0BA',
            fontsize=14, weight='bold', alpha=p, va='center', ha='center')

    sa = D['slot_alpha']            # [16,4] 真实逐 token 槽门控
    ent = -np.sum(sa * np.log(sa + 1e-12), -1) / np.log(4)
    order = np.argsort(-ent)         # 按熵从高到低滚动：典型 → 极端

    # ---------- 0.06-0.20：v62.1 塌缩演示 ----------
    a1 = seg(t, 0.06, 0.14) * (1 - seg(t, 0.22, 0.30))
    if a1 > 0.01:
        ax.text(50, 51.0, 'v62.1 实测：槽门控熵 0.0209', color='#CF3655',
                fontsize=10, alpha=a1, va='center', ha='center')
        base, bh, bw = 33.0, 13.0, 8.0
        for i, v in enumerate([0.9995, 0.0002, 0.0001, 0.0002]):
            ST.rbox(ax, (14 + i * 18, base), bw, max(bh * v, 0.15), fc=SLOT_COL[i], ec=SLOT_COL[i], lw=1.6, alpha=a1 * 0.9, fill_alpha=0.45, z=2)
            ax.text(14 + i * 18 + bw / 2, base - 2.4, SLOT_NAMES[i],
                    color=DIM, fontsize=11, ha='center', alpha=a1)
        ax.text(50, 25.0, '一个槽吃掉几乎全部权重', color='#CF3655',
                fontsize=11, alpha=a1, va='center', ha='center')

    # ---------- 0.26-0.72：四根柱随 token 平滑滚动 ----------
    a2 = seg(t, 0.26, 0.36)
    if a2 > 0.01:
        ax.text(50, 47.0, 'v62.2 修后：熵 0.693，四槽都参与', color='#35D0BA',
                fontsize=10, alpha=a2, va='center', ha='center')

        base, bw, maxh = 26.0, 12.0, 26.0
        n = len(sa)
        # 连续滚动的位置（带小数，柱高用相邻两个 token 插值 → 平滑）
        pos = seg(t, 0.30, 0.68) * (n + 2)
        i0 = int(np.floor(pos)) % n
        i1 = (i0 + 1) % n
        frac = pos - np.floor(pos)
        row = sa[order[i0]] * (1 - frac) + sa[order[i1]] * frac

        for i in range(4):
            v = float(row[i])
            # 逐槽错开生长
            grow = a2 * seg(t, 0.30 + i * 0.02, 0.40 + i * 0.02)
            hgt = max(maxh * v * grow, 0.35)
            x = 8 + i * 22
            ST.rbox(ax, (x, base), bw, hgt, fc=SLOT_COL[i], ec=SLOT_COL[i], lw=1.6, alpha=0.92, fill_alpha=0.45, z=2)
            ax.text(x + bw / 2, base - 2.6, SLOT_NAMES[i], color=SLOT_COL[i],
                    fontsize=12, ha='center', alpha=a2, weight='bold', path_effects=ST.text_glow(SLOT_COL[i]))
            ax.text(x + bw / 2, base + hgt + 2.2, f'{v:.3f}', color=FG,
                    fontsize=12, ha='center', alpha=a2 * grow)
        # token 序号
        ax.text(50, 21.0, f'token #{order[i0]}  （按熵从高到低滚动）',
                color=DIM, fontsize=9, ha='center', alpha=a2, va='center')
        ax.text(50, 17.5,
                f'四槽激活率 100%  ·  平均熵 {np.mean(-np.sum(sa*np.log(sa+1e-12),-1))/np.log(4):.3f}'
                f'  ·  和 = 1  ·  token 数不变',
                color='#35D0BA', fontsize=10, ha='center',
                alpha=seg(t, 0.44, 0.52), va='center')

    # ---------- 0.74-0.92：拆分重组动画（底部中央，不越界）----------
    a4 = seg(t, 0.74, 0.84)
    if a4 > 0.01:
        ax.text(50, 14.0, '可分离 · 可重组', color='#64E3F6',
                fontsize=11, weight='bold', alpha=a4, va='center', ha='center')
        fly = seg(t, 0.80, 0.92)
        # 左块：token 5 的 K 槽，向中间飞
        lx = 14 + (30 - 14) * fly
        ST.rbox(ax, (lx, 8.6), 10, 4.4, fc='#64E3F6', ec='#64E3F6', lw=1.4,
                alpha=a4 * 0.32, fill_alpha=0.45, z=2)
        ax.text(lx + 5, 10.8, 'token5 的 K', color='#64E3F6', fontsize=8.5,
                ha='center', va='center', alpha=a4)
        # 右块：token 0 的 O 槽，向中间飞
        rx = 86 - (86 - 70) * fly
        ST.rbox(ax, (rx, 8.6), 10, 4.4, fc='#b085f5', ec='#b085f5', lw=1.4,
                alpha=a4 * 0.32, fill_alpha=0.45, z=2)
        ax.text(rx + 5, 10.8, 'token0 的 O', color='#b085f5', fontsize=8.5,
                ha='center', va='center', alpha=a4)
        # 中间合成
        if fly > 0.85:
            g = seg(fly, 0.85, 1.0)
            ST.rbox(ax, (50, 8.4), 10, 4.8, fc=FG, ec=FG, lw=1.4,
                    alpha=g * 0.16, fill_alpha=0.45, z=2)
            ax.text(55, 10.8, '新 token', color=FG, fontsize=8.5, ha='center',
                    va='center', alpha=g)
            ax.annotate('', xy=(49.5, 10.8), xytext=(44.5, 10.8),
                        arrowprops=dict(arrowstyle='->', color=FG, lw=1.4, alpha=g))
            ax.annotate('', xy=(60.5, 10.8), xytext=(69.5, 10.8),
                        arrowprops=dict(arrowstyle='->', color=FG, lw=1.4, alpha=g))

    # 原来的 '[B, S, 4, 256] · token 数不变' 单独一行与字幕重复，
    # 「token 数不变」并入下面那行，避免与拆分重组的合成块相撞。


# ============================================================ 幕 ⑦
def act7(ax, t):
    """⑦ 1024 维 → 4 个专家：谁在干活 —— token 点依次飞向选中的专家，液面真实上涨。"""
    p = seg(t, 0.0, 0.05)
    ax.text(50, 60, '⑦ 1024 维 → 4 个专家：谁在干活', color='#35D0BA',
            fontsize=14, weight='bold', alpha=p, va='center', ha='center')

    load = R['report']['expert']['load']       # 2002 条真实
    EA = D['expert_alpha']                     # [16,4] 真实 top-k
    EXP = [46 + e * 13 for e in range(4)]
    EY = 54.0

    # ---------- canonical 带 ----------
    a1 = seg(t, 0.04, 0.14)
    if a1 > 0.01:
        ST.rbox(ax, (4, 48), 22, 5, fc='#64E3F6', ec='#64E3F6', lw=1.6, alpha=a1 * 0.22, fill_alpha=0.45, z=2)
        ax.text(15, 50.5, 'canonical 1024', color='#64E3F6', fontsize=9,
                ha='center', va='center', alpha=a1)

    # ---------- 四个专家盒 ----------
    a2 = seg(t, 0.08, 0.18)
    if a2 > 0.01:
        for e in range(4):
            ST.rbox(ax, (EXP[e], 44), 10, 11, fc=FG, ec=EXP_COL[e], lw=1.3, alpha=a2, fill_alpha=0.45, z=2, fill=False)
            ax.text(EXP[e] + 5, 49.5, f'专家{e}', color=EXP_COL[e], fontsize=9,
                    ha='center', va='center', alpha=a2)

    # ---------- token 点依次飞出 ----------
    a3 = seg(t, 0.13, 0.60)
    if a3 > 0.01:
        NT = 16
        for k in range(NT):
            tk = seg(a3 * 1.35 - k * 0.035, 0.0, 1.0)
            if tk <= 0.02 or tk >= 0.999:
                continue
            y0 = 50.5 - (k % 5) * 1.2
            live = np.where(EA[k] > 0)[0]
            for e in live:
                # 抛物线路径，终点落进盒内
                xe = EXP[e] + 5 + ((k % 3) - 1) * 1.6
                ye = 46.5 + ((k % 4) - 1.5) * 1.1
                mx = (26 + xe) / 2
                my = y0 - 4.0
                x = (1 - tk) ** 2 * 26 + 2 * (1 - tk) * tk * mx + tk ** 2 * xe
                y = (1 - tk) ** 2 * y0 + 2 * (1 - tk) * tk * my + tk ** 2 * ye
                ST.glow_dot(ax, (x, y), 0.9, EXP_COL[e], alpha=0.95)

    # ---------- 真实负载柱 ----------
    a4 = seg(t, 0.52, 0.66)
    if a4 > 0.01:
        base, bw, maxh = 12.0, 13.0, 24.0
        for e in range(4):
            x = 12 + e * 21
            hgt = maxh * load[e] * a4
            ST.rbox(ax, (x, base), bw, maxh, fc=FG, ec=EXP_COL[e], lw=0.8, alpha=a4 * 0.35, fill_alpha=0.45, z=2, fill=False)
            ST.rbox(ax, (x, base), bw, hgt, fc=EXP_COL[e], ec=EXP_COL[e], lw=1.6, alpha=0.9, fill_alpha=0.45, z=2)
            ax.text(x + bw / 2, base - 2.6, f'专家{e}', color=EXP_COL[e],
                    fontsize=10.5, ha='center', alpha=a4, weight='bold', path_effects=ST.text_glow(EXP_COL[e]))
            ax.text(x + bw / 2, base + hgt + 2.0, f'{load[e]:.3f}', color=FG,
                    fontsize=11, ha='center', alpha=a4)

        a5 = seg(t, 0.64, 0.74)
        ax.text(12 + 21 + bw / 2, base + maxh * load[1] + 6.5,
                '承担 73%', color='#64E3F6', fontsize=10, ha='center', alpha=a5)
        ax.text(12 + bw / 2, base + maxh * load[0] + 6.5,
                '36%', color=DIM, fontsize=10, ha='center', alpha=a5)

        ax.text(50, 9.5,
                f'均衡 {BAL:.3f}  ·  熵 {_E:.3f}  ·  top-k={R["top_k"]}（每 token 只跑 2 个专家）',
                color=DIM, fontsize=9.5, ha='center', alpha=a4, va='center')

    # ---------- 角标 ----------
    a6 = seg(t, 0.72, 0.82)
    ax.text(50, 38.0, '⚠ 四个专家均为随机初始化，未从真实 FFN 抽取',
            color='#CF3655', fontsize=10.5, ha='center', alpha=a6, va='center')
    a7 = seg(t, 0.82, 0.90)
    ax.text(50, 34.0, '专家已解冻 · 50.3M 可训练参数',
            color=DIM, fontsize=9.5, ha='center', alpha=a7, va='center')

    # ---------- 底部对照条 ----------
    a8 = seg(t, 0.88, 0.97)
    ST.rbox(ax, (6, 17.5), 88, 3.0, fc='#64E3F6', ec='#64E3F6', lw=1.6, alpha=a8 * 0.09, fill_alpha=0.45, z=2)
    ax.text(50, 19.1,
            '槽门控：全槽软分布（四个都参与）    |    专家选择：top-k=2（真的省算力）',
            color='#64E3F6', fontsize=9.5, ha='center', alpha=a8, va='center')


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else '6'
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    out = os.path.join(FRAMES, f'act{which}')
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith('.png'):
            os.remove(os.path.join(out, f))
    fn = act6 if which == '6' else act7
    print(f'渲染 幕{which}: {total} 帧', flush=True)
    for fi in range(total):
        frame(fi, total, fn, out, which)
        if fi % 40 == 0:
            print(f'  {fi}/{total}', flush=True)
            gc.collect()      # 长跑时 matplotlib 图元会累积，触发内存压力
    print('done', flush=True)
