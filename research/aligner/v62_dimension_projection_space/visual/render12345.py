"""第 ①–⑤ 幕：真实尺度 / 四视角 / 两两对齐 / 三模型崩 / 环路闭合。

数字来源：
  - 维度、能量保留、槽门控      → data/v62_capture.npz + v62_meta.json（真实前向）
  - 对齐 CKA、秩扫描、环路一致性 → research/aligner/v62_*/results/metrics.json（实测）
"""
import gc
import os
import numpy as np
import os
import json
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

ST.setup_fonts()          # 中文字体：手机沙箱与 Kaggle 路径不同，见 style.py
# 画布尺寸可配置：1600×1000 是发布版，1280×800 是省内存的云端版
_FW, _FH = (int(x) for x in os.environ.get('FIG_SIZE', '1600x1000').split('x'))
FIG_SIZE = (_FW / 100.0, _FH / 100.0)
FIG_DPI = int(os.environ.get('FIG_DPI', '100'))

D = np.load(DATA + '/v62_capture.npz')
import json
M = json.load(open(DATA + '/v62_meta.json'))
MM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics.json')))
AL = MM['alignment']
RS = MM['rank_sweep']['by_slot']['K']
CON = MM['consistency']

# 视觉风格见 style.py（借鉴 3b1b/manim 色板）
BG=ST.BG; FG=ST.WHITE; DIM=ST.GREY_C
GOLD=ST.GOLD; ICE=ST.BLUE; GREEN=ST.GREEN; RED=ST.RED; VIOLET=ST.PURPLE
MOD_COL = {'Qwen': ST.MOD_GRAD[0], 'BERT': ST.MOD_GRAD[1], 'ViT': ST.MOD_GRAD[2]}
MOD_DIM = dict(zip(['Qwen','BERT','ViT'], M['dims_native']))
MOD_EN = dict(zip(['Qwen','BERT','ViT'], M['energy_keep']))
SLOT_NAMES = ['Q','K','V','O']
SLOT_COL = list(ST.SLOT_GRAD)
W, H = 100, 62


def sm(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def seg(t, a, b):
    if b <= a: return 1.0 if t >= b else 0.0
    return sm((t - a) / (b - a))


def frame(fi, total, fn, out, which=None):
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG)
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
    fig.savefig(f'{out}/f{fi:05d}.png', facecolor=BG, dpi=FIG_DPI)
    plt.close(fig)



# ============================================================ 字幕
SUBS = {
  '1': [('三个底座的宽度差 3.5 倍。', 'Three backbones, widths 3.5× apart.')],
  '2': [('一个 token 分成四个视角，各有自己的子空间。',
         'One token, four views, four separate subspaces.')],
  '3': [('两两之间，本来就能对齐到 0.99。', 'Any two of them already align to 0.99.')],
  '4': [('但三个一起，一个变换就垮了。', 'Three at once — one transform collapses.')],
  '5': [('让三条路径自洽，0.99 就回来了。',
         'Make the three paths consistent, and 0.99 returns.')],
  '6': [('空间于是不是一个大小的数，而是四个子空间相加。',
         'The space is a sum of subspaces, not a size.')],
  '7': [('四个专家都在用，但并不均衡。',
         'All four experts work, but not evenly.')],
  '8': [('645,190,802 参数，三个真实骨干。',
         '645,190,802 parameters. Three real backbones.')],
}


def draw_subs(ax, which, alpha):
    """底部字幕条：中文 + 英文。"""
    if alpha <= 0.01:
        return
    zh, en = SUBS[which][0]
    ST.rbox(ax, (0, 0), W, 7.0, fc='#000000', ec='#000000', lw=1.6, alpha=0.55 * alpha, fill_alpha=0.45, z=2)
    ax.text(50, 4.6, zh, color=FG, fontsize=13.5, ha='center', va='center', alpha=alpha)
    ax.text(50, 1.8, en, color=DIM, fontsize=9, ha='center', va='center', alpha=alpha * 0.9)

# ============================================================ ① 真实尺度
def act1(ax, t):
    """① 三个底座宽度差 3.5 倍。

    左半：底座宽度按真实比例（896 : 256 : 768），各自 P_m 正交投影 → 256（等长）。
    右半：点明「256 不是空间」—— 空间 = 四槽张成 = 1024（机制在 ② 幕展开）。
    底  ：能量保留，说明投影是等距的。
    """
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 60, '① 三个底座宽度差 3.5 倍',
                   t, 0.0, 0.10, color=GOLD, fontsize=14,
                   path_effects=ST.text_glow(GOLD))

    MODELS = ['Qwen', 'BERT', 'ViT']
    MAXD = 896
    LEN_MAX = 26.0
    X_IN, X_GATE = 4.0, 32.0
    LEN_OUT = LEN_MAX * 256 / MAXD          # 出口段一律 256 → 长度相同

    # ---- 闸门 P ----
    ag = seg(t, 0.03, 0.09)
    if ag > 0.01:
        ST.rbox(ax, (X_GATE, 28), 4.0, 20, fc=FG, ec=FG, lw=1.6,
                alpha=ag * 0.10, fill_alpha=0.30, z=2)
        ax.text(X_GATE + 2.0, 50.5, 'P', color=FG, fontsize=12,
                ha='center', alpha=ag, weight='bold', path_effects=ST.text_glow(FG))
        ax.text(X_GATE + 2.0, 25.5, '正交投影 · 保范数', color=DIM, fontsize=7.5,
                ha='center', va='center', alpha=ag)

    # ---- 三条底座带子（真实比例），依次流过 ----
    for i, m in enumerate(MODELS):
        t0 = 0.05 + i * 0.045
        a = seg(t, t0, t0 + 0.13)
        if a <= 0.01:
            continue
        y = 45 - i * 7 + MO.idle_y(t, 0.20, 0.55, i * 0.31)
        wlen = LEN_MAX * MOD_DIM[m] / MAXD
        head = seg(a, 0.0, 0.62)
        ST.rbox(ax, (X_IN, y), max(wlen * head, 0.01), 3.8,
                fc=MOD_COL[m], ec=MOD_COL[m], lw=1.5, alpha=0.85,
                fill_alpha=0.45, z=2)
        ax.text(X_IN, y + 5.4, f'{m}', color=MOD_COL[m], fontsize=10,
                va='center', alpha=a, weight='bold')
        ax.text(X_IN + 6.0, y + 5.4, f'{MOD_DIM[m]} 维', color=MOD_COL[m],
                fontsize=9, va='center', alpha=a)
        ax.text(X_IN + 13.0, y + 5.4, f'（Qwen 的 {MOD_DIM[m]/MAXD:.2f}×）',
                color=DIM, fontsize=8, va='center', alpha=a * 0.95)
        if i > 0:   # Qwen 长度的基准虚线
            ax.plot([X_IN, X_IN + LEN_MAX], [y + 1.9, y + 1.9], color=DIM,
                    lw=0.8, ls=(0, (3, 4)), alpha=0.32 * a, zorder=1)
        # 出口：各自 256，等长
        if head > 0.60:
            b = seg(a, 0.58, 0.90)
            ST.rbox(ax, (X_GATE + 5.0, y), LEN_OUT * b, 3.8, fc=MOD_COL[m],
                    ec=MOD_COL[m], lw=1.5, alpha=0.85, fill_alpha=0.45, z=2)
            ax.text(X_GATE + 5.0 + LEN_OUT + 1.0, y + 1.9, '256', color=MOD_COL[m],
                    fontsize=8.5, va='center', alpha=a)
        if head > 0.4:
            ax.annotate('', xy=(X_GATE + 4.4, y + 1.9), xytext=(X_GATE + 0.8, y + 1.9),
                        arrowprops=dict(arrowstyle='->', color=MOD_COL[m],
                                        lw=1.1, alpha=a * min(1, (head - 0.4) * 4)))
        # 光点从幕一开始就跑 —— 幕的前段（元素还没建好）也不静止
        MO.flow(ax, [(X_IN, y + 1.9),
                     (X_IN + wlen * head, y + 1.9),
                     (X_GATE, y + 1.9),
                     (X_GATE + 5.0 + LEN_OUT * head, y + 1.9)],
                t, n=6, color=MOD_COL[m], r=0.40, alpha=a * 0.9, speed=0.55)

    # ---- 右半：256 不是空间 ----
    aq = seg(t, 0.26, 0.44)
    if aq > 0.01:
        ax.text(74, 44.0, '但 256 只是每个模态的入口', color=DIM, fontsize=10,
                ha='center', alpha=aq)
        ax.text(74, 38.0, '空间 = Σ_s S_s = 4 × 256 = 1024', color=GREEN,
                fontsize=13, ha='center', alpha=aq, weight='bold')
        ax.text(74, 32.5, '→ ② 幕', color=ICE, fontsize=10, ha='center', alpha=aq)
        ax.text(74, 25.5, '融合层专家：1024 → 4096 → 1024', color=GOLD,
                fontsize=9.5, ha='center', alpha=aq)

    # ---- 能量保留 ----
    ae = seg(t, 0.56, 0.70)
    if ae > 0.01:
        for i, m in enumerate(MODELS):
            x = 18 + i * 18
            ax.text(x, 17.0, m, color=MOD_COL[m], fontsize=9, ha='center', alpha=ae)
            ax.text(x, 13.2, f'保留能量 {MOD_EN[m]*100:.0f}%', color=FG, fontsize=9.5,
                    ha='center', alpha=ae)
    ax.text(74, 13.2, 'BERT 100%（正交可逆，零损失）· Qwen/ViT 高于随机预期',
            color=DIM, fontsize=8, ha='center', alpha=seg(t, 0.62, 0.74))
    ax.text(74, 9.5, '投影等距：‖Pᵀh‖ = ‖h‖，丢的是与保留子空间正交的分量',
            color=DIM, fontsize=8, ha='center', alpha=seg(t, 0.70, 0.84))


# ============================================================ ② 四视角
def act2(ax, t):
    """② 一个 token 分成四个视角 —— 柱体裂开成四条轨道（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 58.5, '② 一个 token 分成四个视角',
                   t, 0.0, 0.10, color=ICE, fontsize=14,
                   path_effects=ST.text_glow(ICE))

    a1 = seg(t, 0.02, 0.10)
    if a1 > 0.01:
        ST.rbox(ax, (7, 30), 4.5, 20 * a1, fc=FG, ec=FG, lw=1.5,
                alpha=0.35, fill_alpha=0.30, z=2)
        ax.text(9.2, 27.5, 'token', color=FG, fontsize=9.5, ha='center', alpha=a1)

    a2 = seg(t, 0.08, 0.28)
    if a2 > 0.01:
        grow = seg(a2, 0.0, 1.0)
        split = seg(a2, 0.12, 0.70)
        for i, sl in enumerate(SLOT_NAMES):
            y_end = 46 - i * 8 + MO.idle_y(t, 0.22, 0.48, i * 0.27)
            y_now = 40 + (y_end - 40) * split
            x_now = 16 + (72 - 16) * grow
            ax.plot([11.5, x_now], [40, y_now], color=SLOT_COL[i], lw=1.2,
                    alpha=a2 * 0.75 * (1 - seg(t, 0.24, 0.34)))
            ST.rbox(ax, (16, y_now - 2.2), max(x_now - 16, 0.01), 4.4,
                    fc=SLOT_COL[i], ec=SLOT_COL[i], lw=1.5, alpha=0.8,
                    fill_alpha=0.42, z=2)
            ax.text(18, y_now + 5.0, f'{sl}  子空间（256 维）', color=SLOT_COL[i],
                    fontsize=9.5, va='center', alpha=a2)

    a3 = seg(t, 0.12, 0.92)
    if a3 > 0.01:
        for i, sl in enumerate(SLOT_NAMES):
            y = 46 - i * 8
            for k in range(3):
                ph = (a3 * 2.4 + k / 3.0) % 1.0
                ax.add_patch(Circle((18 + (68 - 18) * ph, y), 0.85,
                                    color=SLOT_COL[i], alpha=a3 * 0.9))

    a4 = seg(t, 0.44, 0.58)
    if a4 > 0.01:
        ax.text(50, 24.0, '四条轨道各有自己的投影矩阵，互不相交',
                color=ICE, fontsize=11, ha='center', alpha=a4, va='center')
        ax.text(50, 19.5, '不是把一个 256 维空间切成四块',
                color=DIM, fontsize=9.5, ha='center', alpha=a4, va='center')
    a5 = seg(t, 0.56, 0.72)
    ax.text(50, 14.5, '空间 = Σ_s S_s  =  4 × 256  =  1024 维',
            color=GREEN, fontsize=13, ha='center', alpha=a5, va='center',
            weight='bold', path_effects=ST.text_glow(GREEN))
    ax.text(50, 9.0, 'token 数不变 —— 合并发生在模型轴，不在 token 轴',
            color=DIM, fontsize=9, ha='center', alpha=seg(t, 0.68, 0.86), va='center')


# ============================================================ ③ 两两对齐
def act3(ax, t):
    """③ 两两之间，本来就能对齐 —— 三座桥依次接通（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 58.5, '③ 两两之间，本来就能对齐',
                   t, 0.0, 0.10, color=GREEN, fontsize=14,
                   path_effects=ST.text_glow(GREEN))
    ax.text(50, 54.5, 'Q 槽', color=DIM, fontsize=9, ha='center',
            alpha=seg(t, 0.02, 0.07), va='center')

    tgt = {'Qwen×BERT': AL['pairwise']['Q']['Qwen×BERT'],
           'Qwen×ViT':  AL['pairwise']['Q']['Qwen×ViT'],
           'BERT×ViT':  AL['pairwise']['Q']['BERT×ViT']}
    BR = [('Qwen', 'BERT', 'Qwen×BERT', 20, 44, 48, 48),
          ('Qwen', 'ViT',  'Qwen×ViT',  20, 76, 34, 34),
          ('BERT', 'ViT',  'BERT×ViT',  44, 76, 20, 20)]
    for _m1, _m2, _k, _x1, _x2, _y1, _y2 in BR:
        assert _y1 == _y2, '桥必须是水平的：%s' % _k
    for _mm, _pts in (('Qwen', [(20, 48), (20, 34)]),
                      ('BERT', [(44, 48), (44, 20)]),
                      ('ViT',  [(76, 34), (76, 20)])):
        _ys = sorted(y for _x, y in _pts)
        ax.plot([_pts[0][0]] * 2, _ys, color=MOD_COL[_mm], lw=0.9,
                ls=(0, (2, 3)), alpha=0.45, zorder=1)

    for i, (m1, m2, key, x1, x2, y1, y2) in enumerate(BR):
        t0 = 0.06 + i * 0.045
        a = seg(t, t0, t0 + 0.13)
        if a <= 0.01:
            continue
        _dy = MO.idle_y(t, 0.22, 0.5, i * 0.29)
        for (xx, yy, mm) in ((x1, y1, m1), (x2, y2, m2)):
            ax.add_patch(Circle((xx, yy + _dy), 2.0, color=MOD_COL[mm], alpha=a * 0.85))
            ax.text(xx, yy + _dy - 3.6, mm, color=MOD_COL[mm], fontsize=8.5,
                    ha='center', va='center', alpha=a)
        # out_back：桥到位时轻微过冲 —— follow-through
        g = MO.ease((t - t0) / 0.13, 'out_back') if t >= t0 else 0.0
        x_end = x1 + (x2 - x1) * g
        ax.plot([x1, x_end], [y1 + _dy, y1 + _dy], color=GREEN, lw=2.4, alpha=0.95)
        # 端点持续呼吸 —— 桥接完也不停
        _pr = 1.0 + 0.18 * np.sin(2 * np.pi * (2.2 * t + i * 0.7))
        ax.add_patch(Circle((x_end, y1 + _dy), 0.75 * _pr, color=GREEN, alpha=a))
        # 光点从桥的**全长**上跑，且从幕一开始就跑 ——
        # 之前等 a>0.55 才启动，幕的前段（元素还没建好）就成了静止段
        if t > 0.02:
            MO.flow(ax, [(x1, y1 + _dy), (x2, y1 + _dy)], t, n=5,
                    color='#BFFFC0', r=0.38, alpha=0.95, speed=0.7,
                    phase=i * 0.31)
        MO.counter(ax, (x1 + x2) / 2, y1 + 3.4 + _dy, tgt[key], t, t0 + 0.05, 0.16,
                   fmt='{:.4f}', color=GREEN, fontsize=11.5, alpha=a,
                   path_effects=ST.text_glow(GREEN))

    a4 = seg(t, 0.25, 0.38)
    if a4 > 0.01:
        for m1, m2, key, x1, x2, y1, y2 in BR:
            ax.plot([x1, x2], [y1, y1], color=GREEN, lw=3.2, alpha=a4)

    a5 = seg(t, 0.36, 0.50)
    ax.text(50, 12.5, '三座桥全绿 —— 看起来一切顺利',
            color=GREEN, fontsize=11.5, ha='center', alpha=a5, va='center')
    ax.text(50, 8.8, '   '.join(f'{k} {v:.4f}' for k, v in tgt.items()),
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.44, 0.62), va='center')


# ============================================================ ④ 三模型崩
def act4(ax, t):
    """④ 三个一起，垮了 —— 流穿越透镜时逐块被剪开（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 58.5, '④ 三个一起，垮了',
                   t, 0.0, 0.10, color=RED, fontsize=14,
                   path_effects=ST.text_glow(RED))

    X0, X1_ = 10.0, 88.0
    LX, LW = 46.0, 7.0
    NSEG, SPREAD, SHEAR = 8, 0.85, 7.5
    MODELS = ['Qwen', 'BERT', 'ViT']

    a1 = seg(t, 0.02, 0.10)
    if a1 > 0.01:
        ST.rbox(ax, (LX, 26), LW, 24, fc=FG, ec=FG, lw=1.5,
                alpha=a1 * 0.10, fill_alpha=0.30, z=2)
        ax.text(LX + LW / 2, 51.5, '单一变换', color=FG, fontsize=10,
                ha='center', alpha=a1)

    a2 = seg(t, 0.06, 0.58)
    if a2 > 0.01:
        ST.rbox(ax, (LX, 26), LW, 24, fc=FG, ec=FG, lw=1.5,
                alpha=a2 * 0.10, fill_alpha=0.30, z=2)
        # 三拍速度：快 → 慢 → 快（张弛对比，而不是匀速推到底）
        _u = a2 * 1.18
        if _u < 0.45:
            _f = MO.ease(_u / 0.45, 'out_quad') * 0.52
        else:
            _f = 0.52 + MO.ease((_u - 0.45) / 0.55, 'in_out_cubic') * 0.48
        front = _f
        for i, m in enumerate(MODELS):
            y0 = 44 - i * 7
            dirn = i - 1
            for k in range(NSEG):
                u = k / (NSEG - 1)
                prog = front - (1 - u) * SPREAD
                if prog <= 0.001:
                    continue
                x = X0 + (X1_ - X0) * prog
                over = max(0.0, min(1.0, (prog - (LX - X0) / (X1_ - X0))
                                    / max(1e-6, 1 - (LX - X0) / (X1_ - X0))))
                y = y0 + dirn * SHEAR * (over ** 0.8)
                # 渐进长大：原来 prog>0 就整块出现，一帧冒出好几个 → 4.75 的帧差突变
                _g = min(1.0, max(0.0, prog / 0.09))
                if _g <= 0.02:
                    continue
                ST.rbox(ax, (x + 1.3 * (1 - _g), y + 2.1 * (1 - _g)),
                        2.6 * _g, 4.2 * _g, fc=MOD_COL[m], ec=MOD_COL[m],
                        lw=0.8, r=0.12, alpha=0.9 * _g, fill_alpha=0.95, z=2)
                if 0.02 < over < 0.92 and dirn != 0:
                    ax.plot([x, x], [y0, y], color=RED, lw=1.1,
                            alpha=0.5 * (1 - abs(over - 0.45) * 2))
                if k == 0:
                    ax.text(x + 3.6, y + 2.2, m, color=MOD_COL[m], fontsize=9,
                            va='center', alpha=0.95)
        # 光点沿三条流持续向右跑 —— 覆盖「流已走完」的静止段
        for i, m in enumerate(MODELS):
            y0 = 44 - i * 7
            MO.flow(ax, [(X0, y0), (X1_, y0)], t, n=6, color=MOD_COL[m],
                    r=0.38, alpha=0.75, speed=0.65, phase=i * 0.33)
        if a2 > 0.85:
            ax.text(74, 22.0, '三股被剪开了 —— 因为 router 是随机的', color=RED,
                    fontsize=10, ha='center', alpha=seg(a2, 0.85, 1.0), va='center')

    a3 = seg(t, 0.40, 0.52)
    if a3 > 0.01:
        ax.text(20, 21.0, '两两', color=DIM, fontsize=10, ha='right', alpha=a3, va='center')
        MO.counter(ax, 22, 21.0, 0.9893, t, 0.40, 0.14, fmt='{:.4f}',
                   color=GREEN, fontsize=13, ha='left', alpha=a3,
                   path_effects=ST.text_glow(GREEN))
        ax.text(20, 15.5, '三模型', color=DIM, fontsize=10, ha='right', alpha=a3, va='center')
        MO.counter(ax, 38, 15.5, 0.6767, t, 0.42, 0.14, fmt='{:.4f}',
                   color=RED, fontsize=13, ha='left', alpha=a3,
                   path_effects=ST.text_glow(RED))
        ax.annotate('', xy=(36, 15.5), xytext=(36, 21.0),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=1.4, alpha=a3))

    a4 = seg(t, 0.50, 0.64)
    if a4 > 0.01:
        # 画在流**下方**且更淡 —— 之前压在 Qwen 流的头一个方块上
        ST.rbox(ax, (9.5, 37.6), 22, 3.4, fc='none', ec=MOD_COL['Qwen'], lw=0.7,
                alpha=a4 * 0.42, fill_alpha=0.0, z=1)
        ax.text(20.5, 39.3, 'Qwen 退出', color=DIM, fontsize=8.5, ha='center',
                va='center', alpha=a4 * 0.75)
        ax.text(20, 10.0, '去掉 Qwen 后', color=DIM, fontsize=10, ha='right',
                alpha=a4, va='center')
        MO.counter(ax, 22, 10.0, 0.9879, t, 0.52, 0.12, fmt='{:.4f}',
                   color=GREEN, fontsize=13, ha='left', alpha=a4,
                   path_effects=ST.text_glow(GREEN))

    a5 = seg(t, 0.64, 0.80)
    ax.text(76, 11.0, '不是 Qwen 的 GQA 结构有问题\n而是一个变换要同时满足三对约束',
            color=ICE, fontsize=9.5, ha='center', alpha=a5, va='center')


# ============================================================ ⑤ 环路闭合
def act5(ax, t):
    """⑤ 环路一闭合，三个就回来了（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 58.5, '⑤ 环路一闭合，三个就回来了',
                   t, 0.0, 0.10, color=VIOLET, fontsize=14,
                   path_effects=ST.text_glow(VIOLET))

    tri = [(30, 44), (18, 26), (42, 26)]
    labels = ['Qwen', 'BERT', 'ViT']
    a1 = seg(t, 0.02, 0.12)
    _dy5 = MO.idle_y(t, 0.24, 0.45, 0.13)
    if a1 > 0.01:
        for (x, y), lb, c in zip(tri, labels, [GOLD, ICE, GREEN]):
            ax.add_patch(Circle((x, y + _dy5), 3.4, color=c, alpha=a1 * 0.85))
            ax.text(x, y, lb, color='#0a0c10', fontsize=8.5, ha='center',
                    va='center', alpha=a1, weight='bold')

    edges = [(0, 1), (1, 2), (0, 2)]
    for i, (ia, ib) in enumerate(edges):
        t0 = 0.08 + i * 0.045
        a = seg(t, t0, t0 + 0.11)
        if a <= 0.01:
            continue
        (x1, y1), (x2, y2) = tri[ia], tri[ib]
        ax.plot([x1, x2], [y1, y2], color=VIOLET, lw=2.0, alpha=a)
        if t > 0.02:
            MO.flow(ax, [(x1, y1 + _dy5), (x2, y2 + _dy5)], t, n=4,
                    color='#E6C6FF', r=0.36, alpha=0.9, speed=0.6, phase=i * 0.4)

    a2 = seg(t, 0.24, 0.38)
    a2b = a2 * (1 - seg(t, 0.46, 0.56))
    if a2b > 0.01:
        ax.text(50, 21.0, '同一个终点，走法不同 ⇒ 结果不同',
                color=RED, fontsize=10.5, ha='center', alpha=a2b, va='center')
        ax.text(50, 16.5, f'缝宽 {CON["two_path_diff"]:.4f}',
                color=RED, fontsize=9.5, ha='center', alpha=a2b, va='center')

    a3 = seg(t, 0.34, 0.48)
    if a3 > 0.01:
        ax.text(50, 50.0, 'A_ac  ≡  A_bc ∘ A_ab', color=VIOLET, fontsize=13.5,
                ha='center', alpha=a3, va='center')
        ax.text(50, 45.0, '环路一致性约束：三条路径必须自洽',
                color=DIM, fontsize=9.5, ha='center', alpha=a3, va='center')

    GAP = 4.2
    a4 = seg(t, 0.46, 0.66)
    tighten = a4
    if a4 > 0.01:
        for (x, y), lb, c in zip(tri, labels, [GOLD, ICE, GREEN]):
            ax.add_patch(Circle((x, y), 3.4 + 0.5 * tighten, color=c, alpha=0.9))
            ax.text(x, y, lb, color='#0a0c10', fontsize=8.5, ha='center',
                    va='center', alpha=0.95, weight='bold')
        for ia, ib in edges:
            (x1, y1), (x2, y2) = tri[ia], tri[ib]
            ax.plot([x1, x2], [y1, y2], color=VIOLET, lw=2.0 + 1.6 * tighten,
                    alpha=0.95, path_effects=ST.line_glow(VIOLET, lw=5, alpha=0.4))
        vx, vy = tri[2]
        g = GAP * (1 - tighten)
        m1 = (vx + g * 0.28, vy - g * 0.5)
        m2 = (vx + g * 0.28, vy + g * 0.5)
        for (mx, my), mc in ((m1, ICE), (m2, GOLD)):
            ST.rbox(ax, (mx - 1.3, my - 1.3), 2.6, 2.6, fc=mc, ec=mc, lw=1.2,
                    alpha=0.95, fill_alpha=0.9, z=5)
        if tighten > 0.7:
            ax.text(vx + 9.0, vy, '两个结果合到一处', color=GREEN, fontsize=9.5,
                    ha='left', va='center', alpha=seg(tighten, 0.7, 1.0))

    a5 = seg(t, 0.62, 0.78)
    if a5 > 0.01:
        MO.counter(ax, 26, 10.0, 0.6767, t, 0.62, 0.12, fmt='{:.4f}',
                   color=RED, fontsize=13, ha='center', alpha=a5)
        ax.annotate('', xy=(44, 10.0), xytext=(34, 10.0),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8, alpha=a5))
        MO.counter(ax, 50, 10.0, 0.9925, t, 0.66, 0.12, fmt='{:.4f}',
                   color=GREEN, fontsize=15, ha='center', alpha=a5,
                   path_effects=ST.text_glow(GREEN))
        ax.text(72, 10.0, '三模型', color=DIM, fontsize=10, ha='center', alpha=a5, va='center')
    ax.text(50, 13.5, f'一致性损失 {CON["loss_before"]:.6f} → {CON["loss_after"]:.6f}',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.66, 0.86), va='center')


# ============================================================ ⑧ 收尾
def act8(ax, t):
    """⑧ 收尾（紧凑时间线）。"""
    p = MO.pop(t, 0.0, 0.10, 'out_back')
    MO.pop_text(ax, 50, 55, 'v62 · 维度投影空间',
                   t, 0.0, 0.10, color=GREEN, fontsize=16,
                   path_effects=ST.text_glow(GREEN))

    rows = [
        ('两两对齐',   '0.9879 – 0.9898', GREEN),
        ('三模型',     '0.6767  →  0.9925', GREEN),
        ('空间',       '256  →  Σ = 1024 维', ICE),
        ('专家',       '均衡 0.612  ·  四个都在用', GOLD),
        ('槽门控',     'Q 槽权重 ≈ 0（router 自己学的）', RED),
    ]
    for i, (k, v, c) in enumerate(rows):
        a = seg(t, 0.03 + i * 0.06, 0.13 + i * 0.06)
        if a <= 0.01:
            continue
        y = 47 - i * 5.5
        slide = (1 - a) * 8
        ax.text(28 - slide, y, k, color=DIM, fontsize=11, ha='right', alpha=a, va='center')
        ax.text(32 - slide, y, v, color=c, fontsize=13, ha='left', alpha=a,
                va='center', weight='bold')

    a5 = seg(t, 0.40, 0.54)
    ax.text(50, 15.0, '645,190,802 参数  ·  held-out 0.9076  ·  三个真实骨干',
            color=FG, fontsize=10.5, ha='center', alpha=a5, va='center')
    a6 = seg(t, 0.52, 0.86)
    if a6 > 0.01:
        txt = '空间不是一个大小的数，而是按功能槽张成的一组子空间之和'
        ax.text(50, 9.0, txt[:int(len(txt) * min(1.0, a6 * 1.25))],
                color=ICE, fontsize=12, ha='center', alpha=a6, va='center')
        MO.flow(ax, [(14, 9.0), (86, 9.0)], t, n=8, color='#BFD8FF',
                r=0.30, alpha=a6 * 0.8, speed=0.35)


if __name__ == '__main__':
    import sys
    which = sys.argv[1]; total = int(sys.argv[2])
    # 输出目录可由第三个参数指定（render_all.py 用它渲染到临时目录，
    # 校验通过后才原子替换正式目录 —— 否则子脚本会直接写正式目录）
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(FRAMES, f'act{which}')
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith('.png'):
            os.remove(os.path.join(out, f))
    fn = {'1': act1, '2': act2, '3': act3, '4': act4, '5': act5, '8': act8}[which]
    print(f'渲染 幕{which}: {total} 帧', flush=True)
    for fi in range(total):
        frame(fi, total, fn, out, which)
        # 每帧后强制回收：matplotlib 的 artist（每帧 200+ 个 patch）
        # 不主动 GC 会累积，Kaggle CPU kernel 只有 ~4GB 内存时会 OOM。
        # 本地 14GB 够用，所以这个坑只在云端暴露。
        plt.close('all')
        gc.collect()
        if fi % 40 == 0:
            print(f'  {fi}/{total}', flush=True)
    print('done', flush=True)
