"""第 ①–⑤ 幕：真实尺度 / 四视角 / 两两对齐 / 三模型崩 / 环路闭合。

数字来源：
  - 维度、能量保留、槽门控      → data/v62_capture.npz + v62_meta.json（真实前向）
  - 对齐 CKA、秩扫描、环路一致性 → research/aligner/v62_*/results/metrics.json（实测）
"""
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
from matplotlib.patches import Circle, FancyArrowPatch

for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
          '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    """① 三个底座的宽度差 3.5 倍 —— 三条带子依次流过 P 闸门。"""
    p = seg(t, 0.0, 0.05)
    ax.text(50, 60, '① 三个底座的宽度差 3.5 倍', color=GOLD, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center', path_effects=ST.text_glow(GOLD))

    MODELS = ['Qwen', 'BERT', 'ViT']
    MAXD = 896
    SLEN = 46.0                 # 满宽条子的长度
    X_IN, X_GATE, X_OUT = 8.0, 56.0, 66.0

    # ---- 闸门 ----
    ag = seg(t, 0.05, 0.15)
    if ag > 0.01:
        ST.rbox(ax, (X_GATE, 30), 4.5, 24, fc=FG, ec=FG, lw=1.6, alpha=ag * 0.16, fill_alpha=0.45, z=2)
        ax.text(X_GATE + 2.25, 55.5, 'P', color=FG, fontsize=13,
                ha='center', alpha=ag, weight='bold', path_effects=ST.text_glow(FG))

    # ---- 每条带子占 1/3 时间依次通过 ----
    for i, m in enumerate(MODELS):
        t0 = 0.08 + i * 0.16
        a = seg(t, t0, t0 + 0.14)
        if a <= 0.01:
            continue
        y = 46 - i * 8
        wlen = SLEN * MOD_DIM[m] / MAXD
        # 入口段：带子从左长出
        head = seg(a, 0.0, 0.55)                     # 0→1 表示走到闸门
        x_end = X_IN + (X_GATE - X_IN) * head
        ST.rbox(ax, (X_IN, y), max(x_end - X_IN, 0.01), 4.4, fc=MOD_COL[m], ec=MOD_COL[m], lw=1.6, alpha=0.85, fill_alpha=0.45, z=2)
        # 模态名与维度放在同一行（放在带子下方会压到下一条带子）
        ax.text(X_IN + 2, y + 6.8, f'{m}  {MOD_DIM[m]}', color=MOD_COL[m],
                fontsize=10, va='center', alpha=a)
        # 出口段：等宽 256
        if head > 0.62:
            b = seg(a, 0.55, 0.9)
            wout = SLEN * 256 / MAXD
            ST.rbox(ax, (X_OUT, y), wout * b, 4.4, fc=MOD_COL[m], ec=MOD_COL[m], lw=1.6, alpha=0.85, fill_alpha=0.45, z=2)
            ax.text(X_OUT + wout * b + 1.6, y + 2.2, '256', color=MOD_COL[m],
                    fontsize=9.5, va='center', alpha=a)
        # 闸门处的箭头
        if head > 0.4:
            ax.annotate('', xy=(X_GATE + 5.5, y + 2.2), xytext=(X_GATE - 2, y + 2.2),
                        arrowprops=dict(arrowstyle='->', color=MOD_COL[m],
                                        lw=1.2, alpha=a * min(1, (head - 0.4) * 4)))

    # ---- 能量保留 ----
    ae = seg(t, 0.58, 0.72)
    if ae > 0.01:
        # 三行一起出现（逐个淡入会让未出现的行看着像残留）
        for i, m in enumerate(MODELS):
            x = 22 + i * 20
            ax.text(x, 21, m, color=MOD_COL[m], fontsize=9.5, ha='center', alpha=ae)
            ax.text(x, 17, f'保留能量 {MOD_EN[m]*100:.0f}%', color=FG, fontsize=10,
                    ha='center', alpha=ae)
    ax.text(50, 13.0, 'BERT 的能量一分不丢，Qwen 丢掉 46%，ViT 丢掉 41%',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.74, 0.86), va='center')
    ax.text(50, 9.0, '三个模态的序列长度也不同：文本 128，图像 197',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.82, 0.95), va='center')


# ============================================================ ② 四视角
def act2(ax, t):
    """② 一个 token 分成四个视角 —— 柱体裂开成四条并行轨道。"""
    p = seg(t, 0.0, 0.05)
    ax.text(50, 60, '② 一个 token 分成四个视角', color=ICE, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center', path_effects=ST.text_glow(ICE))

    # ---- token 柱 ----
    a1 = seg(t, 0.06, 0.20)
    if a1 > 0.01:
        ST.rbox(ax, (7, 30), 4.5, 20 * a1, fc=FG, ec=FG, lw=1.6, alpha=0.35, fill_alpha=0.45, z=2)
        ax.text(9.2, 27.5, 'token', color=FG, fontsize=9.5, ha='center', alpha=a1)

    # ---- 裂开：四条轨道从一点分出并伸长 ----
    a2 = seg(t, 0.20, 0.48)
    grow = seg(a2, 0.0, 1.0)
    split = seg(a2, 0.15, 0.75)
    for i, sl in enumerate(SLOT_NAMES):
        y_end = 46 - i * 8
        y_start = 40                              # 从 token 柱中部出发
        y_now = y_start + (y_end - y_start) * split
        x_now = 16 + (72 - 16) * grow
        # 轨道完全展开后连接线淡出，否则四根斜线交叉显乱
        conn = a2 * 0.75 * (1 - seg(t, 0.42, 0.53))
        ax.plot([11.5, x_now], [40, y_now], color=SLOT_COL[i], lw=1.2, alpha=conn)
        ST.rbox(ax, (16, y_now - 2.2), max(x_now - 16, 0.01), 4.4, fc=SLOT_COL[i], ec=SLOT_COL[i], lw=1.6, alpha=a2 * 0.75, fill_alpha=0.45, z=2)
        ax.text(18, y_now + 5.2, f'{sl}  子空间（256 维）', color=SLOT_COL[i],
                fontsize=9.5, va='center', alpha=a2)

    # ---- 轨道上流动的亮点（token 内容流过）----
    a3 = seg(t, 0.44, 0.74)
    if a3 > 0.01:
        for i, sl in enumerate(SLOT_NAMES):
            y = 46 - i * 8
            for k in range(3):
                ph = (a3 * 2.2 + k / 3.0) % 1.0
                x = 18 + (68 - 18) * ph
                ax.add_patch(Circle((x, y), 0.85, color=SLOT_COL[i],
                                    alpha=a3 * 0.9))

    # ---- 收束 ----
    a4 = seg(t, 0.72, 0.86)
    if a4 > 0.01:
        ax.text(50, 21, '四条轨道各有自己的投影矩阵，互不相交',
                color=ICE, fontsize=11, ha='center', alpha=a4, va='center')
        ax.text(50, 16.5, '不是把一个 256 维空间切成四块',
                color=DIM, fontsize=9.5, ha='center', alpha=a4, va='center')
    a5 = seg(t, 0.80, 0.92)
    ax.text(50, 12.5, '空间 = Σ_s S_s  =  4 × 256  =  1024 维',
            color=GREEN, fontsize=13, ha='center', alpha=a5, va='center', weight='bold', path_effects=ST.text_glow(GREEN))
    ax.text(50, 8.8, 'token 数不变 —— 合并发生在模型轴，不在 token 轴',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.86, 0.97), va='center')


# ============================================================ ③ 两两对齐
def act3(ax, t):
    """③ 两两之间，本来就能对齐 —— 三座桥依次接通，数值同步爬升。

    每座桥独立成一条水平带（两端用小圆点标出模态名），
    避免桥横穿模态方块。
    """
    p = seg(t, 0.0, 0.05)
    ax.text(50, 60, '③ 两两之间，本来就能对齐', color=GREEN, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center', path_effects=ST.text_glow(GREEN))
    ax.text(50, 55.5, 'Q 槽', color=DIM, fontsize=9.5, ha='center',
            alpha=seg(t, 0.04, 0.10), va='center')

    tgt = {'Qwen×BERT': AL['pairwise']['Q']['Qwen×BERT'],
           'Qwen×ViT':  AL['pairwise']['Q']['Qwen×ViT'],
           'BERT×ViT':  AL['pairwise']['Q']['BERT×ViT']}
    # 每座桥：两端的 (x, y) 与标签
    BR = [('Qwen', 'BERT', 'Qwen×BERT', 20, 44, 48, 44),
          ('Qwen', 'ViT',  'Qwen×ViT',  20, 76, 32, 32),
          ('BERT', 'ViT',  'BERT×ViT',  44, 76, 16, 16)]

    for i, (m1, m2, key, x1, x2, y1, y2) in enumerate(BR):
        t0 = 0.10 + i * 0.12
        a = seg(t, t0, t0 + 0.11)
        # 端点圆
        if a > 0.01:
            for (xx, yy, mm) in ((x1, y1, m1), (x2, y2, m2)):
                ST.glow_dot(ax, (xx, yy), 2.0, MOD_COL[mm], alpha=a * 0.85)
                ax.text(xx, yy - 4.2, mm, color=MOD_COL[mm], fontsize=8.5,
                        ha='center', va='center', alpha=a)
        # 桥生长
        if a > 0.01:
            g = seg(a, 0.15, 1.0)
            xm = x1 + (x2 - x1) * g
            ax.plot([x1, xm], [y1, y1], color=GREEN, lw=2.6, alpha=0.95)
            ST.glow_dot(ax, (xm, y1), 0.75, GREEN, alpha=a)
            ax.text((x1 + x2) / 2, y1 + 3.0, f'{tgt[key] * g:.4f}', color=GREEN,
                    fontsize=11.5, ha='center', alpha=a, weight='bold', path_effects=ST.text_glow(GREEN))

    # ---- 全部接通 ----
    a4 = seg(t, 0.52, 0.66)
    if a4 > 0.01:
        for m1, m2, key, x1, x2, y1, y2 in BR:
            ax.plot([x1, x2], [y1, y1], color=GREEN, lw=3.2, alpha=a4)

    a5 = seg(t, 0.64, 0.76)
    ax.text(50, 11.5, '三座桥全绿 —— 看起来一切顺利',
            color=GREEN, fontsize=11.5, ha='center', alpha=a5, va='center')
    ax.text(50, 8.8,
            '   '.join(f'{k} {v:.4f}' for k, v in tgt.items()),
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.72, 0.86), va='center')


# ============================================================ ④ 三模型崩
def act4(ax, t):
    """④ 三个一起，垮了 —— 实时动态。

    三条模态流各自是一个「波」：整体从左边缘推进，前端先穿过透镜。
    穿过透镜的部分被剪切（越过越多偏得越多），于是看到的是
    **正在被撕开**，而不是切好的静态图。
    """
    p = seg(t, 0.0, 0.05)
    ax.text(50, 59, '④ 三个一起，垮了', color=RED, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center', path_effects=ST.text_glow(RED))

    X0, X1 = 10.0, 88.0          # 流的起点 / 终点
    LX, LW = 46.0, 7.0           # 透镜
    P_LENS = (LX - X0) / (X1 - X0)
    NSEG = 9                     # 每条流的 token 块数
    SPREAD = 0.34                # 流的长度（占全程比例）
    SHEAR = 7.5                  # 完全错开时的 y 位移
    MODELS = ['Qwen', 'BERT', 'ViT']

    # ---------------- 阶段 A：透镜 + 三股待入 ----------------
    a1 = seg(t, 0.04, 0.18)
    # 阶段 A 的箭头与标签随流推进淡出（否则与流里的模态名重复）
    fade_a = 1.0 - seg(t, 0.20, 0.40)
    if a1 > 0.01 and fade_a > 0.01:
        ST.rbox(ax, (LX, 26), LW, 26, fc=FG, ec=FG, lw=1.6, alpha=a1 * 0.16, fill_alpha=0.45, z=2)
        ax.text(LX + LW / 2, 53.5, '单一变换', color=FG, fontsize=10.5,
                ha='center', alpha=a1)
        for i, m in enumerate(MODELS):
            y = 44 - i * 7
            ax.annotate('', xy=(LX - 2, y + 2.2), xytext=(X0 + 6, y + 2.2),
                        arrowprops=dict(arrowstyle='->', color=MOD_COL[m],
                                        lw=1.3, alpha=a1 * 0.8 * fade_a))
            ax.text(X0, y + 5.8, m, color=MOD_COL[m], fontsize=9.5,
                    ha='left', alpha=a1 * fade_a)

    # ---------------- 阶段 B：流推进 + 实时剪开 ----------------
    a2 = seg(t, 0.18, 0.58)
    if a2 > 0.01:
        front = seg(a2 * 1.12, 0.0, 1.0)      # 流前端的整体位置 0→1
        ST.rbox(ax, (LX, 26), LW, 26, fc=FG, ec=FG, lw=1.6, alpha=a2 * 0.14, fill_alpha=0.45, z=2)
        ST.rbox(ax, (LX, 26), LW, 26, fc=FG, ec=FG, lw=1.0, alpha=a2 * 0.32, fill_alpha=0.45, z=2, fill=False)
        ax.text(LX + LW / 2, 53.5, '单一变换', color=FG, fontsize=10.5,
                ha='center', alpha=a2)

        for i, m in enumerate(MODELS):
            y0 = 44 - i * 7
            dirn = i - 1                          # Qwen 上、BERT 不动、ViT 下
            for k in range(NSEG):
                u = k / (NSEG - 1)               # 1 = 前端，0 = 后端
                pos = front - (1 - u) * SPREAD    # 每个块在流中的世界位置
                if pos < 0.002:
                    continue
                x = X0 + (X1 - X0) * pos
                over = max(0.0, min(1.0, (pos - P_LENS) / max(1e-6, 1 - P_LENS)))
                y = y0 + dirn * SHEAR * (over ** 0.8)
                ST.rbox(ax, (x, y), 3.2, 4.4, fc=MOD_COL[m], ec=MOD_COL[m], lw=0.8, r=0.12, alpha=0.88, fill_alpha=0.95, z=2)
                # 正在被撕的地方画一道红痕
                if 0.03 < over < 0.90 and dirn != 0:
                    ax.plot([x, x], [y0, y], color=RED, lw=1.2,
                            alpha=0.55 * (1 - abs(over - 0.45) * 2))
                if k == 0:                        # 前端方块旁标模态名
                    ax.text(x + 3.6, y + 2.2, m, color=MOD_COL[m],
                            fontsize=9.5, va='center', alpha=0.95)

        if a2 > 0.80:
            ax.text(74, 21, '三股被剪开了', color=RED, fontsize=10.5,
                    ha='center', alpha=seg(a2, 0.80, 1.0), va='center')

    # ---------------- 阶段 C：数字翻转 ----------------
    a3 = seg(t, 0.60, 0.72)
    if a3 > 0.01:
        ax.text(20, 22.0, '两两', color=DIM, fontsize=10, ha='right', alpha=a3, va='center')
        ax.text(22, 22.0, '0.9893', color=GREEN, fontsize=13, ha='left', alpha=a3,
                va='center', weight='bold', path_effects=ST.text_glow(GREEN))
        ax.text(20, 17.0, '三模型', color=DIM, fontsize=10, ha='right', alpha=a3, va='center')
        ax.text(38, 17.0, '0.6767', color=RED, fontsize=13, ha='left', alpha=a3,
                va='center', weight='bold', path_effects=ST.text_glow(RED))
        ax.annotate('', xy=(36, 7.0), xytext=(36, 17.0),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=1.4, alpha=a3))

    # ---------------- 阶段 D：去掉 Qwen，其余重新对齐 ----------------
    a4 = seg(t, 0.74, 0.86)
    if a4 > 0.01:
        refocus = seg(a4, 0.22, 0.78)
        for i, m in enumerate(MODELS):
            y0 = 44 - i * 7
            dirn = i - 1
            if m == 'Qwen':
                # 淡出
                ST.rbox(ax, (X0, 41.8), 26, 4.4, fc=BG, ec=BG, lw=1.6, alpha=a4 * 0.94, fill_alpha=0.45, z=2)
                ST.rbox(ax, (X0, 41.8), 26, 4.4, fc=FG, ec=MOD_COL['Qwen'], lw=0.8, alpha=a4 * 0.35, fill_alpha=0.45, z=2, fill=False)
                ax.text(X0 + 13, 44.0, 'Qwen 退出', color=DIM, fontsize=9.5,
                        ha='center', va='center', alpha=a4)
                continue
            for k in range(NSEG):
                u = k / (NSEG - 1)
                pos = 1.0 - (1 - u) * SPREAD
                x = X0 + (X1 - X0) * pos
                over = max(0.0, min(1.0, (pos - P_LENS) / max(1e-6, 1 - P_LENS)))
                y = y0 + dirn * SHEAR * (over ** 0.8) * (1 - refocus)
                ST.rbox(ax, (x, y), 3.2, 4.4, fc=MOD_COL[m], ec=MOD_COL[m], lw=0.8, r=0.12, alpha=0.88, fill_alpha=0.95, z=2)
        ax.text(20, 12.0, '去掉 Qwen 后', color=DIM, fontsize=10, ha='right',
                alpha=a4, va='center')
        ax.text(22, 12.0, '0.9879', color=GREEN, fontsize=13, ha='left',
                alpha=a4, va='center', weight='bold', path_effects=ST.text_glow(GREEN))

    a5 = seg(t, 0.88, 0.96)
    ax.text(50, 8.8, '不是 Qwen 的 GQA 结构有问题 —— 是一个变换要同时满足三对约束',
            color=ICE, fontsize=10, ha='center', alpha=a5, va='center')


# ============================================================ ⑤ 环路闭合
def act5(ax, t):
    """⑤ 环路一闭合，三个就回来了 —— 动态。

    几何上三条路径的终点本来就是同一个节点，所以「缝」不能画成几何缺口。
    准确的说法是：**两条路径到达同一节点，但携带的结果不同**。
    因此画两个结果标记（折线路径的结果 / 直达边的结果）在节点旁错开，
    施加环路一致性约束后两者合拢 —— 那是「被拉到一起」，不是「缺口补上」。
    """
    p = seg(t, 0.0, 0.05)
    ax.text(50, 59, '⑤ 环路一闭合，三个就回来了', color=VIOLET, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center', path_effects=ST.text_glow(VIOLET))

    tri = [(30, 44), (18, 26), (42, 26)]     # Qwen, BERT, ViT
    labels = ['Qwen', 'BERT', 'ViT']
    cols = [GOLD, ICE, GREEN]
    edges = [(0, 1), (1, 2), (0, 2)]          # A→B, B→C, A→C

    # ---------------- 三节点 ----------------
    a1 = seg(t, 0.06, 0.20)
    if a1 > 0.01:
        for (x, y), lb, c in zip(tri, labels, cols):
            ST.glow_dot(ax, (x, y), 3.4, c, alpha=a1 * 0.85)
            ax.text(x, y, lb, color='#0a0c10', fontsize=8.5, ha='center',
                    va='center', alpha=a1, weight='bold')

    # ---------------- 三条边依次画出 ----------------
    for i, (ia, ib) in enumerate(edges):
        a = seg(t, 0.20 + i * 0.055, 0.31 + i * 0.055)
        if a <= 0.01:
            continue
        (x1, y1), (x2, y2) = tri[ia], tri[ib]
        ax.plot([x1, x2], [y1, y2], color=VIOLET, lw=2.0, alpha=a)

    # ---------------- 两个结果标记（错开 = 缝）----------------
    # 折线路径 A→B→C 的结果 vs 直达边 A→C 的结果
    GAP = 4.2
    a2 = seg(t, 0.38, 0.52)
    tighten = seg(t, 0.64, 0.80)              # 约束收紧
    gap = GAP * (1 - tighten)

    vx, vy = tri[2]                            # ViT 节点
    if a2 > 0.01:
        # 两个结果标记放在 ViT 节点**右侧**，垂直错开（避开节点本身）
        # 合拢目标是 ViT 节点本身 —— 语义是「两个结果在同一点重合」，
        # 停在节点外侧会看起来像「结果飞出去了」
        m1 = (vx + gap * 0.28, vy - gap * 0.5)
        m2 = (vx + gap * 0.28, vy + gap * 0.5)
        ST.rbox(ax, (m1[0] - 1.1, m1[1] - 1.1), 2.2, 2.2, fc=ICE, ec=ICE, lw=1.6, alpha=a2 * 0.9, fill_alpha=0.45, z=2)
        ST.rbox(ax, (m2[0] - 1.1, m2[1] - 1.1), 2.2, 2.2, fc=GOLD, ec=GOLD, lw=1.6, alpha=a2 * 0.9, fill_alpha=0.45, z=2)
        # 从 ViT 节点连到各自的标记
        ax.plot([vx, m1[0]], [vy, m1[1]], color=ICE, lw=1.2, alpha=a2 * 0.8)
        ax.plot([vx, m2[0]], [vy, m2[1]], color=GOLD, lw=1.2, alpha=a2 * 0.8)
        lab_a = a2 * (1 - seg(t, 0.64, 0.74))   # 收紧时淡出，避免与合拢提示撞
        ax.text(m1[0] + 2.0, m1[1], '走两条边', color=ICE, fontsize=9,
                ha='left', va='center', alpha=lab_a)
        ax.text(m2[0] + 2.0, m2[1], '走一条直达', color=GOLD, fontsize=9,
                ha='left', va='center', alpha=lab_a)

    # 缝的文字（收紧时淡出）
    a2b = a2 * (1 - seg(t, 0.66, 0.78))
    if a2b > 0.01:
        ax.text(50, 21.0, '同一个终点，走法不同 ⇒ 结果不同',
                color=RED, fontsize=10.5, ha='center', alpha=a2b, va='center')
        ax.text(50, 17.0, f'缝宽 {CON["two_path_diff"]:.4f}',
                color=RED, fontsize=9.5, ha='center', alpha=a2b, va='center')

    # ---------------- 约束公式 ----------------
    a3 = seg(t, 0.52, 0.62)
    if a3 > 0.01:
        ax.text(50, 53.0, 'A_ac  ≡  A_bc ∘ A_ab', color=VIOLET, fontsize=13.5,
                ha='center', alpha=a3, va='center')
        ax.text(50, 48.8, '环路一致性约束：三条路径必须自洽',
                color=DIM, fontsize=9.5, ha='center', alpha=a3, va='center')

    # ---------------- 收紧：两个标记合拢 + 边发光 ----------------
    if tighten > 0.01:
        for i, (ia, ib) in enumerate(edges):
            (x1, y1), (x2, y2) = tri[ia], tri[ib]
            ax.plot([x1, x2], [y1, y2], color=VIOLET,
                lw=2.0 + 1.6 * tighten, alpha=0.95,
                path_effects=ST.line_glow(VIOLET, lw=5, alpha=0.4))
        if a2 > 0.01:
            m1 = (vx + gap * 0.28, vy - gap * 0.5)
            m2 = (vx + gap * 0.28, vy + gap * 0.5)
            ST.rbox(ax, (m1[0] - 1.1, m1[1] - 1.1), 2.2, 2.2, fc=ICE, ec=ICE, lw=1.6, alpha=0.9, fill_alpha=0.45, z=2)
            ST.rbox(ax, (m2[0] - 1.1, m2[1] - 1.1), 2.2, 2.2, fc=GOLD, ec=GOLD, lw=1.6, alpha=0.9, fill_alpha=0.45, z=2)
            ax.plot([vx, m1[0]], [vy, m1[1]], color=ICE, lw=1.2, alpha=0.85)
            ax.plot([vx, m2[0]], [vy, m2[1]], color=GOLD, lw=1.2, alpha=0.85)
            if tighten > 0.75:
                ax.text(vx + 8.0, vy - 4.6, '两个结果合到一处', color=GREEN,
                        fontsize=10, ha='left', va='center',
                        alpha=seg(tighten, 0.75, 1.0))

    # ---------------- 结果 ----------------
    a5 = seg(t, 0.82, 0.92)
    if a5 > 0.01:
        ax.text(26, 9.0, '0.6767', color=RED, fontsize=13, ha='center', alpha=a5, va='center')
        ax.annotate('', xy=(44, 9.0), xytext=(34, 9.0),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8, alpha=a5))
        ax.text(50, 9.0, '0.9925', color=GREEN, fontsize=15, ha='center',
                alpha=a5, va='center', weight='bold', path_effects=ST.text_glow(GREEN))
        ax.text(72, 9.0, '三模型', color=DIM, fontsize=10, ha='center', alpha=a5, va='center')
    ax.text(50, 13.0, f'一致性损失 {CON["loss_before"]:.6f} → {CON["loss_after"]:.6f}',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.88, 0.97), va='center')


# ============================================================ ⑧ 收尾
def act8(ax, t):
    """⑧ 收尾 —— 逐行淡入，最后一句从中心展开。"""
    p = seg(t, 0.0, 0.05)
    ax.text(50, 56, 'v62 · 维度投影空间', color=GREEN, fontsize=16,
            weight='bold', alpha=p, ha='center', va='center', path_effects=ST.text_glow(GREEN))

    rows = [
        ('两两对齐',   '0.9879 – 0.9898', GREEN),
        ('三模型',     '0.6767  →  0.9925', GREEN),
        ('空间',       '256  →  Σ = 1024 维', ICE),
        ('专家',       '4 个都在用  ·  均衡 0.492', GOLD),
    ]
    for i, (k, v, c) in enumerate(rows):
        # 逐行从左滑入
        a = seg(t, 0.08 + i * 0.11, 0.20 + i * 0.11)
        if a <= 0.01:
            continue
        y = 47 - i * 7
        slide = (1 - a) * 8
        ax.text(28 - slide, y, k, color=DIM, fontsize=11, ha='right', alpha=a, va='center')
        ax.text(32 - slide, y, v, color=c, fontsize=13, ha='left', alpha=a,
                va='center', weight='bold', path_effects=ST.text_glow(c))

    a5 = seg(t, 0.56, 0.68)
    ax.text(50, 15, '645,190,802 参数  ·  held-out 0.9101  ·  三个真实骨干',
            color=FG, fontsize=10.5, ha='center', alpha=a5, va='center')

    # 末句：从中心展开
    a6 = seg(t, 0.68, 0.90)
    if a6 > 0.01:
        txt = '空间不是一个大小的数，而是按功能槽张成的一组子空间之和'
        w = len(txt) * 0.62
        ax.text(50, 8.5, txt[:int(len(txt) * min(1.0, a6 * 1.3))],
                color=ICE, fontsize=12, ha='center', alpha=a6, va='center')


if __name__ == '__main__':
    import sys
    which = sys.argv[1]; total = int(sys.argv[2])
    out = os.path.join(FRAMES, f'act{which}')
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith('.png'):
            os.remove(os.path.join(out, f))
    fn = {'1': act1, '2': act2, '3': act3, '4': act4, '5': act5, '8': act8}[which]
    print(f'渲染 幕{which}: {total} 帧', flush=True)
    for fi in range(total):
        frame(fi, total, fn, out, which)
        if fi % 40 == 0:
            print(f'  {fi}/{total}', flush=True)
            gc.collect()      # 长跑时 matplotlib 图元会累积，触发内存压力
    print('done', flush=True)
