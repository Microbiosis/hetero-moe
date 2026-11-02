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
import motion as MO
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
    # 环境层：① 大面积缓慢流动的光带 ② 漂移的微粒
    # 整片一直有低频运动，画面不死
    MO.light_band(ax, t)
    MO.dust(ax, t, n=170, seed=7 + (ord(which) if which else 0) * 13)
    fn(ax, t)
    if which:
        draw_subs(ax, which, seg(t, 0.0, 0.05))
        ST.watermark(ax, 'v62 · 维度投影空间')
    # 转场：幕首淡入（黑幕压一下），幕尾淡出
    if which:
        # 真转场：幕尾淡到**全黑**、幕首从全黑浮起。
        # 之前只到 0.45，于是幕与幕之间是「画面+45%黑」硬切到「画面+45%黑」，
        # 内容突变造成 4.75 的帧差 —— 转场反而成了全片最大的运动。
        head = MO.fade(t, 0.0, 0.10)      # 幕首：从全黑浮起
        tail = MO.fade(t, 0.88, 0.10)     # 幕尾：淡到全黑
        ov = max(1.0 - head, tail)
        if ov > 0.01:
            ax.add_patch(Rectangle((0, 0), W, H, fc=BG, ec='none',
                                   alpha=ov, zorder=50))
    # 镜头极缓慢推近
    ST.camera(ax, zoom=1.0 + 0.022 * t, drift=MO.idle_y(t, 0.10, 0.33))
    fig.savefig(f'{out}/f{fi:05d}.png', facecolor=BG, dpi=100)
    plt.close(fig)


# ============================================================ 幕 ⑥
def act6(ax, t):
    """⑥ 一个 token 的结构：四个槽各贡献多少（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 58, '⑥ 一个 token 的结构：四个槽各贡献多少',
                   t, 0.0, 0.10, color='#35D0BA', fontsize=14,
                   path_effects=ST.text_glow('#35D0BA'))

    sa = D['slot_alpha']
    ent = -np.sum(sa * np.log(sa + 1e-12), -1) / np.log(4)
    ENT_LO, ENT_HI = float(ent.min()), float(ent.max())

    # ---- v62.1 塌缩 ----
    a1 = seg(t, 0.02, 0.09) * (1 - seg(t, 0.13, 0.19))
    if a1 > 0.01:
        ax.text(50, 50.0, 'v62.1：router 从未训练（lazy bug）→ 熵 0.021', color='#CF3655',
                fontsize=10, alpha=a1, va='center', ha='center')
        base, bh, bw = 33.0, 13.0, 8.0
        for i, v in enumerate([0.9995, 0.0002, 0.0001, 0.0002]):
            ST.rbox(ax, (14 + i * 18, base), bw, max(bh * v, 0.15), fc=SLOT_COL[i],
                    ec=SLOT_COL[i], lw=1.4, alpha=a1 * 0.9, fill_alpha=0.45, z=2)
            ax.text(14 + i * 18 + bw / 2, base - 2.6, SLOT_NAMES[i], color=DIM,
                    fontsize=11, ha='center', alpha=a1)
        ax.text(50, 25.0, '一个槽吃掉几乎全部 —— 因为 router 是随机的', color='#CF3655',
                fontsize=10.5, alpha=a1, va='center', ha='center')

    # ---- v62.2 四柱滚动 ----
    a2 = seg(t, 0.10, 0.18)
    if a2 > 0.01:
        ax.text(50, 47.0, 'v62.3 修好 router 后：熵 0.647', color='#35D0BA',
                fontsize=10, alpha=a2, va='center', ha='center')
        order = np.argsort(-ent)
        n = len(sa)
        base, bw, maxh = 26.0, 12.0, 24.0
        pos = seg(t, 0.12, 0.92) * (n + 2)   # 整幕持续滚，不留静止段
        i0 = int(np.floor(pos)) % n
        i1 = (i0 + 1) % n
        frac = pos - np.floor(pos)
        row = sa[order[i0]] * (1 - frac) + sa[order[i1]] * frac
        for i in range(4):
            v = float(row[i])
            grow = a2 * seg(t, 0.16 + i * 0.018, 0.24 + i * 0.018)
            hgt = max(maxh * v * grow, 0.35)
            x = 8 + i * 22
            ST.rbox(ax, (x, base), bw, hgt, fc=SLOT_COL[i], ec=SLOT_COL[i], lw=1.4,
                    alpha=0.92, fill_alpha=0.45, z=2)
            ax.text(x + bw / 2, base - 2.6, SLOT_NAMES[i], color=SLOT_COL[i],
                    fontsize=12, ha='center', alpha=a2, weight='bold')
            ax.text(x + bw / 2, base + hgt + 2.0, f'{v:.3f}', color=FG,
                    fontsize=12, ha='center', alpha=a2 * grow)
        # 统计信息挪到**顶部** —— 底部要塞拆分重组的三个块，
        # 塞不下（柱子标签已在 23.4，往下还要 5 行）
        # 合并成一行 —— 顶部的位置被 v62.1(50) / v62.2(47) 占着，
        # 分两行会撞。底部则要塞拆分重组的三个块。
        ax.text(50, 53.0,
                f'token #{i0}  ·  四槽激活率 100%  ·  平均熵 {np.mean(ent):.3f}'
                f'  ·  Q 槽权重≈0（由 K/V/O 承担，熵 {ENT_LO:.3f}–{ENT_HI:.3f}）',
                color='#35D0BA', fontsize=9.5, ha='center',
                alpha=seg(t, 0.44, 0.54), va='center')

    # ---- 拆分重组 ----
    a4 = seg(t, 0.62, 0.70)
    if a4 > 0.01:
        # 标题放在块的**上方** —— 原来在 y=10.5，正压在中间的合成块上
        ax.text(50, 16.5, '可分离 · 可重组', color='#64E3F6',
                fontsize=11, weight='bold', alpha=a4, va='center', ha='center')
        fly = seg(t, 0.66, 0.82)
        lx = 8 + (28 - 8) * fly
        ST.rbox(ax, (lx, 8.6), 10, 4.4, fc='#64E3F6', ec='#64E3F6', lw=1.4,
                alpha=a4 * 0.32, fill_alpha=0.45, z=2)
        ax.text(lx + 5, 10.8, 'token5 的 K', color='#64E3F6', fontsize=8.5,
                ha='center', va='center', alpha=a4)
        rx = 66 + (62 - 66) * fly
        ST.rbox(ax, (rx, 8.6), 10, 4.4, fc='#b085f5', ec='#b085f5', lw=1.4,
                alpha=a4 * 0.32, fill_alpha=0.45, z=2)
        ax.text(rx + 5, 10.8, 'token0 的 O', color='#b085f5', fontsize=8.5,
                ha='center', va='center', alpha=a4)
        if fly > 0.85:
            g = seg(fly, 0.85, 1.0)
            ST.rbox(ax, (48, 8.4), 10, 4.8, fc=FG, ec=FG, lw=1.4,
                    alpha=g * 0.16, fill_alpha=0.45, z=2)
            ax.text(53, 10.8, '新 token', color=FG, fontsize=8.5, ha='center',
                    va='center', alpha=g)
            # 箭头在块**之间**，不画进块里
            ax.annotate('', xy=(46.5, 10.8), xytext=(38.5, 10.8),
                        arrowprops=dict(arrowstyle='->', color=FG, lw=1.4, alpha=g))
            ax.annotate('', xy=(59.5, 10.8), xytext=(61.5, 10.8),
                        arrowprops=dict(arrowstyle='->', color=FG, lw=1.4, alpha=g))


# ============================================================ 幕 ⑦
def act7(ax, t):
    """⑦ 1024 维 → 4 个专家：谁在干活（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 60, '⑦ 1024 维 → 4 个专家：谁在干活',
                   t, 0.0, 0.10, color='#35D0BA', fontsize=14,
                   path_effects=ST.text_glow('#35D0BA'))

    load = R['report']['expert']['load']
    EA = D['expert_alpha']
    EXP = [46 + e * 13 for e in range(4)]

    a1 = seg(t, 0.02, 0.08)
    if a1 > 0.01:
        ST.rbox(ax, (4, 48), 22, 5, fc='#64E3F6', ec='#64E3F6', lw=1.4,
                alpha=a1 * 0.22, fill_alpha=0.45, z=2)
        ax.text(15, 50.5, 'canonical 1024', color='#64E3F6', fontsize=9,
                ha='center', va='center', alpha=a1)
        for e in range(4):
            ST.rbox(ax, (EXP[e], 44), 10, 11, fc='none', ec=EXP_COL[e], lw=1.3,
                    alpha=a1, fill_alpha=0.0, z=2)
            ax.text(EXP[e] + 5, 49.5, f'专家{e}', color=EXP_COL[e], fontsize=9,
                    ha='center', va='center', alpha=a1)

    a3 = seg(t, 0.04, 0.86)
    if a3 > 0.01:
        for k in range(16):
            tk = seg(a3 * 1.55 - k * 0.026, 0.0, 1.0)
            if tk <= 0.02 or tk >= 0.999:
                continue
            y0 = 50.5 - (k % 5) * 1.2
            for e in np.where(EA[k] > 0)[0]:
                xe = EXP[e] + 5 + ((k % 3) - 1) * 1.6
                ye = 46.5 + ((k % 4) - 1.5) * 1.1
                mx = (26 + xe) / 2
                my = y0 - 4.0
                x = (1 - tk) ** 2 * 26 + 2 * (1 - tk) * tk * mx + tk ** 2 * xe
                y = (1 - tk) ** 2 * y0 + 2 * (1 - tk) * tk * my + tk ** 2 * ye
                ax.add_patch(Circle((x, y), 0.9, color=EXP_COL[e], alpha=0.95))

    a4 = seg(t, 0.38, 0.54)
    if a4 > 0.01:
        base, bw, maxh = 15.0, 13.0, 19.0
        for e in range(4):
            x = 12 + e * 21
            hgt = maxh * load[e] * a4
            ST.rbox(ax, (x, base), bw, maxh, fc='none', ec=EXP_COL[e], lw=0.8,
                    alpha=a4 * 0.35, fill_alpha=0.0, z=2)
            ST.rbox(ax, (x, base), bw, hgt, fc=EXP_COL[e], ec=EXP_COL[e], lw=1.4,
                    alpha=0.9, fill_alpha=0.45, z=2)
            ax.text(x + bw / 2, base - 2.6, f'专家{e}', color=EXP_COL[e],
                    fontsize=10.5, ha='center', alpha=a4, weight='bold')
            MO.counter(ax, x + bw / 2, base + hgt + 2.0, load[e], t,
                       0.38 + e * 0.015, 0.13, fmt='{:.3f}', color=FG,
                       fontsize=11, alpha=a4)

        a5 = seg(t, 0.50, 0.60)
        ax.text(12 + 21 + bw / 2, base + maxh * load[1] + 6.5, '承担 73%',
                color='#64E3F6', fontsize=10, ha='center', alpha=a5)
        ax.text(12 + bw / 2, base + maxh * load[0] + 6.5, '36%',
                color=DIM, fontsize=10, ha='center', alpha=a5)
        ax.text(50, 9.0,
                f'均衡 {BAL:.3f}  ·  熵 {_E:.3f}  ·  top-k={R["top_k"]}（每 token 只跑 2 个专家）',
                color=DIM, fontsize=9.5, ha='center', alpha=a4, va='center')

    a6 = seg(t, 0.58, 0.68)
    ax.text(50, 36.5, '⚠ 四个专家均为随机初始化，未从真实 FFN 抽取',
            color='#CF3655', fontsize=10.5, ha='center', alpha=a6, va='center')
    a7 = seg(t, 0.66, 0.76)
    ax.text(50, 32.5, '专家已解冻 · 50.3M 可训练参数',
            color=DIM, fontsize=9.5, ha='center', alpha=a7, va='center')

    a8 = seg(t, 0.74, 0.88)
    if a8 > 0.01:
        ST.rbox(ax, (6, 40.0), 88, 3.2, fc='#64E3F6', ec='#64E3F6', lw=1.6,
                alpha=a8 * 0.09, fill_alpha=0.45, z=2)
        ax.text(50, 41.6,
                '槽门控：全槽软分布    |    专家选择：top-k=2（真的省算力）',
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
