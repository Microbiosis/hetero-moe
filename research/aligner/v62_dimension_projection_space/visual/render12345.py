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
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

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

BG='#0a0c10'; FG='#e8eaed'; DIM='#5b6572'
GOLD='#F0873B'; ICE='#64E3F6'; GREEN='#35D0BA'; RED='#CF3655'; VIOLET='#b085f5'
MOD_COL = {'Qwen': GOLD, 'BERT': ICE, 'ViT': GREEN}
MOD_DIM = dict(zip(['Qwen','BERT','ViT'], M['dims_native']))
MOD_EN = dict(zip(['Qwen','BERT','ViT'], M['energy_keep']))
SLOT_NAMES = ['Q','K','V','O']
SLOT_COL = [GOLD, ICE, GREEN, VIOLET]
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
    fn(ax, t)
    if which:
        draw_subs(ax, which, seg(t, 0.0, 0.06))
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
    ax.add_patch(Rectangle((0, 0), W, 7.0, color='#000000', alpha=0.55 * alpha))
    ax.text(50, 4.6, zh, color=FG, fontsize=13.5, ha='center', va='center', alpha=alpha)
    ax.text(50, 1.8, en, color=DIM, fontsize=9, ha='center', va='center', alpha=alpha * 0.9)

# ============================================================ ① 真实尺度
def act1(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 59, '① 三个底座的宽度差 3.5 倍', color=GOLD, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center')

    # 三条按真实比例的带子
    a1 = seg(t, 0.06, 0.18)
    if a1 > 0.01:
        maxd = 896
        for i, m in enumerate(['Qwen', 'BERT', 'ViT']):
            y = 46 - i * 7
            wlen = 46 * MOD_DIM[m] / maxd
            ax.add_patch(Rectangle((8, y), wlen * a1, 3.6, color=MOD_COL[m], alpha=0.85))
            # 标签放在条子上方：放在右端会与固定位置的箭头重叠
            ax.text(8 + wlen * a1 + 1.5, y + 5.4, f'{m} {MOD_DIM[m]}',
                    color=MOD_COL[m], fontsize=10.5, va='center', alpha=a1)

    # P 闸门
    a2 = seg(t, 0.28, 0.38)
    if a2 > 0.01:
        ax.add_patch(Rectangle((58, 32), 4, 22, color=FG, alpha=a2 * 0.16))
        ax.text(60, 55.5, 'P', color=FG, fontsize=14, ha='center', alpha=a2, weight='bold')
        ax.text(60, 29.5, '正交投影 · 冻结', color=DIM, fontsize=8.5, ha='center', alpha=a2)
        for i, m in enumerate(['Qwen', 'BERT', 'ViT']):
            y = 46 - i * 7
            ax.annotate('', xy=(58, y + 1.8), xytext=(56, y + 1.8),
                        arrowprops=dict(arrowstyle='->', color=DIM, lw=1, alpha=a2))

    # 出口三条等宽
    a3 = seg(t, 0.42, 0.56)
    if a3 > 0.01:
        for i, m in enumerate(['Qwen', 'BERT', 'ViT']):
            y = 46 - i * 7
            wlen = 46 * 256 / 896
            ax.add_patch(Rectangle((66, y), wlen * a3, 3.6, color=MOD_COL[m], alpha=0.85))
            ax.text(66 + wlen * a3 + 1.2, y + 5.4, '256',
                    color=MOD_COL[m], fontsize=10.5, va='center', alpha=a3)
            ax.annotate('', xy=(66, y + 1.8), xytext=(62, y + 1.8),
                        arrowprops=dict(arrowstyle='->', color=FG, lw=1.2, alpha=a3))

    # 能量保留
    a4 = seg(t, 0.62, 0.74)
    if a4 > 0.01:
        for i, m in enumerate(['Qwen', 'BERT', 'ViT']):
            x = 22 + i * 20
            ax.text(x, 22, f'{m}', color=MOD_COL[m], fontsize=9.5, ha='center', alpha=a4)
            ax.text(x, 18, f'保留能量 {MOD_EN[m]*100:.0f}%', color=FG,
                    fontsize=10, ha='center', alpha=a4)
    ax.text(50, 12.5, 'BERT 的能量一分不丢，Qwen 丢掉 46%，ViT 丢掉 41%',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.74, 0.86), va='center')
    ax.text(50, 4, '三个模态的序列长度也不同：文本 128，图像 197',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.80, 0.92), va='center')


# ============================================================ ② 四视角
def act2(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 59, '② 一个 token 分成四个视角', color=ICE, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center')

    # 左侧 token 柱
    a1 = seg(t, 0.08, 0.20)
    if a1 > 0.01:
        ax.add_patch(Rectangle((6, 34), 4, 18, color=FG, alpha=a1 * 0.35))
        ax.text(8, 29, 'token', color=FG, fontsize=10, ha='center', alpha=a1)

    # 四条轨道
    for i, s in enumerate(SLOT_NAMES):
        a = seg(t, 0.22 + i * 0.06, 0.34 + i * 0.06)
        if a <= 0.01: continue
        y = 46 - i * 8
        ax.add_patch(Rectangle((18, y), 58, 4.4, color=SLOT_COL[i], alpha=a * 0.16))
        ax.add_patch(Rectangle((18, y), 58 * a, 4.4, color=SLOT_COL[i], alpha=0.55))
        ax.text(20, y + 6.2, f'{s}  子空间（256 维）', color=SLOT_COL[i],
                fontsize=10, va='center', alpha=a)
        ax.annotate('', xy=(18, y + 2.2), xytext=(10, 43),
                    arrowprops=dict(arrowstyle='->', color=DIM, lw=0.9, alpha=a * 0.7))

    a2 = seg(t, 0.56, 0.66)
    if a2 > 0.01:
        ax.text(50, 22, '四条轨道各有自己的投影矩阵，互不相交',
                color=ICE, fontsize=11.5, ha='center', alpha=a2, va='center')
        ax.text(50, 17.5, '不是把一个 256 维空间切成四块',
                color=DIM, fontsize=10, ha='center', alpha=a2, va='center')
    a3 = seg(t, 0.72, 0.84)
    ax.text(50, 11, '空间 = Σ_s S_s  =  4 × 256  =  1024 维',
            color=GREEN, fontsize=13, ha='center', alpha=a3, va='center', weight='bold')
    ax.text(50, 5, 'token 数不变 —— 合并发生在模型轴，不在 token 轴',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.80, 0.92), va='center')


# ============================================================ ③ 两两对齐
def act3(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 59, '③ 两两之间，本来就能对齐', color=GREEN, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center')
    ax.text(50, 55, 'Q 槽', color=DIM, fontsize=10, ha='center',
            alpha=seg(t, 0.04, 0.10), va='center')

    pairs = [('Qwen', 'BERT', 'Qwen×BERT', 30),
             ('Qwen', 'ViT',  'Qwen×ViT',  24),
             ('BERT', 'ViT',  'BERT×ViT',  18)]
    for i, (m1, m2, key, y) in enumerate(pairs):
        a = seg(t, 0.12 + i * 0.10, 0.26 + i * 0.10)
        if a <= 0.01: continue
        v = AL['pairwise']['Q'][key]
        x1 = 18 if m1 == 'Qwen' else (32 if m1 == 'BERT' else 46)
        x2 = 18 if m2 == 'Qwen' else (32 if m2 == 'BERT' else 46)
        ax.add_patch(Rectangle((x1, y), 10, 5, color=MOD_COL[m1], alpha=a * 0.3))
        ax.text(x1 + 5, y + 2.5, m1, color=MOD_COL[m1], fontsize=9.5,
                ha='center', va='center', alpha=a)
        ax.add_patch(Rectangle((x2, y), 10, 5, color=MOD_COL[m2], alpha=a * 0.3))
        ax.text(x2 + 5, y + 2.5, m2, color=MOD_COL[m2], fontsize=9.5,
                ha='center', va='center', alpha=a)
        # 桥
        ax.plot([x1 + 10, x2], [y + 2.5, y + 2.5], color=GREEN, lw=2.2, alpha=a)
        # 标签放在**桥的中点**正上方（桥从 x1+10 连到 x2），否则会压在盒子上
        ax.text((x1 + 10 + x2) / 2, y + 6.4, f'{v:.4f}', color=GREEN,
                fontsize=11, ha='center', alpha=a, weight='bold')

    a4 = seg(t, 0.50, 0.60)
    ax.text(50, 8, '三座桥全绿 —— 看起来一切顺利',
            color=GREEN, fontsize=11.5, ha='center', alpha=a4, va='center')
    ax.text(50, 3.5, 'Qwen×BERT 0.9893   Qwen×ViT 0.9883   BERT×ViT 0.9879',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.56, 0.68), va='center')


# ============================================================ ④ 三模型崩
def act4(ax, t):
    """④ 三个一起，垮了 —— 实时动态。

    三条模态流各自是一个「波」：整体从左边缘推进，前端先穿过透镜。
    穿过透镜的部分被剪切（越过越多偏得越多），于是看到的是
    **正在被撕开**，而不是切好的静态图。
    """
    p = seg(t, 0.0, 0.05)
    ax.text(50, 59, '④ 三个一起，垮了', color=RED, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center')

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
        ax.add_patch(Rectangle((LX, 26), LW, 26, color=FG, alpha=a1 * 0.16))
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
        ax.add_patch(Rectangle((LX, 26), LW, 26, color=FG, alpha=a2 * 0.14))
        ax.add_patch(Rectangle((LX, 26), LW, 26, fill=False, edgecolor=FG,
                               lw=1.0, alpha=a2 * 0.32))
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
                ax.add_patch(Rectangle((x, y), 3.2, 4.4, color=MOD_COL[m],
                                       alpha=0.62))
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
        ax.text(20, 17.0, '两两', color=DIM, fontsize=10, ha='right', alpha=a3, va='center')
        ax.text(22, 17.0, '0.9893', color=GREEN, fontsize=13, ha='left', alpha=a3,
                va='center', weight='bold')
        ax.text(20, 7.0, '三模型', color=DIM, fontsize=10, ha='right', alpha=a3, va='center')
        ax.text(38, 7.0, '0.6767', color=RED, fontsize=13, ha='left', alpha=a3,
                va='center', weight='bold')
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
                ax.add_patch(Rectangle((X0, 41.8), 26, 4.4, color=BG, alpha=a4 * 0.94))
                ax.add_patch(Rectangle((X0, 41.8), 26, 4.4, fill=False,
                                       edgecolor=MOD_COL['Qwen'], lw=0.8,
                                       alpha=a4 * 0.35, linestyle='--'))
                ax.text(X0 + 13, 44.0, 'Qwen 退出', color=DIM, fontsize=9.5,
                        ha='center', va='center', alpha=a4)
                continue
            for k in range(NSEG):
                u = k / (NSEG - 1)
                pos = 1.0 - (1 - u) * SPREAD
                x = X0 + (X1 - X0) * pos
                over = max(0.0, min(1.0, (pos - P_LENS) / max(1e-6, 1 - P_LENS)))
                y = y0 + dirn * SHEAR * (over ** 0.8) * (1 - refocus)
                ax.add_patch(Rectangle((x, y), 3.2, 4.4, color=MOD_COL[m],
                                       alpha=0.62))
        ax.text(20, 12.0, '去掉 Qwen 后', color=DIM, fontsize=10, ha='right',
                alpha=a4, va='center')
        ax.text(22, 12.0, '0.9879', color=GREEN, fontsize=13, ha='left',
                alpha=a4, va='center', weight='bold')

    a5 = seg(t, 0.88, 0.96)
    ax.text(50, 4, '不是 Qwen 的 GQA 结构有问题 —— 是一个变换要同时满足三对约束',
            color=ICE, fontsize=10, ha='center', alpha=a5, va='center')


# ============================================================ ⑤ 环路闭合
def act5(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 59, '⑤ 环路一闭合，三个就回来了', color=VIOLET, fontsize=14,
            weight='bold', alpha=p, ha='center', va='center')

    # 三角形
    tri = [(30, 44), (18, 26), (42, 26)]     # A, B, C
    labels = ['Qwen', 'BERT', 'ViT']
    a1 = seg(t, 0.08, 0.20)
    if a1 > 0.01:
        for (x, y), lb, c in zip(tri, labels, [GOLD, ICE, GREEN]):
            ax.add_patch(Circle((x, y), 3.4, color=c, alpha=a1 * 0.85))
            ax.text(x, y, lb, color='#0a0c10', fontsize=8.5, ha='center',
                    va='center', alpha=a1, weight='bold')

    # 三条边
    edges = [(0, 1, 'A_Qwen→BERT'), (1, 2, 'A_BERT→ViT'), (0, 2, 'A_Qwen→ViT')]
    for i, (ia, ib, _) in enumerate(edges):
        a = seg(t, 0.20 + i * 0.06, 0.32 + i * 0.06)
        if a <= 0.01: continue
        (x1, y1), (x2, y2) = tri[ia], tri[ib]
        ax.plot([x1, x2], [y1, y2], color=VIOLET, lw=2.0, alpha=a)

    # 缝
    a2 = seg(t, 0.40, 0.52)
    if a2 > 0.01:
        ax.text(50, 19, '走两条边 ≠ 走一条直达 —— 三角形有道缝',
                color=RED, fontsize=10.5, ha='center', alpha=a2, va='center')
        ax.text(50, 14, f'缝宽 {CON["two_path_diff"]:.4f}',
                color=RED, fontsize=9.5, ha='center', alpha=a2, va='center')

    # 约束
    a3 = seg(t, 0.54, 0.64)
    if a3 > 0.01:
        ax.text(50, 20.5, 'A_ac  ≡  A_bc ∘ A_ab', color=VIOLET, fontsize=13,
                ha='center', alpha=a3, va='center')
        ax.text(50, 15.5, '环路一致性约束：三条路径必须自洽',
                color=DIM, fontsize=9.5, ha='center', alpha=a3, va='center')

    # 闭合
    a4 = seg(t, 0.66, 0.78)
    if a4 > 0.01:
        for (x, y), lb, c in zip(tri, labels, [GOLD, ICE, GREEN]):
            ax.add_patch(Circle((x, y), 3.4 + 0.5 * a4, color=c, alpha=0.9))
            ax.text(x, y, lb, color='#0a0c10', fontsize=8.5, ha='center',
                    va='center', alpha=a4, weight='bold')
        for i, (ia, ib, _) in enumerate(edges):
            (x1, y1), (x2, y2) = tri[ia], tri[ib]
            ax.plot([x1, x2], [y1, y2], color=VIOLET, lw=2.6, alpha=a4)

    # 结果
    a5 = seg(t, 0.80, 0.90)
    if a5 > 0.01:
        ax.text(26, 8, '0.6767', color=RED, fontsize=13, ha='center', alpha=a5, va='center')
        ax.annotate('', xy=(44, 8), xytext=(34, 8),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8, alpha=a5))
        ax.text(50, 8, '0.9925', color=GREEN, fontsize=15, ha='center',
                alpha=a5, va='center', weight='bold')
        ax.text(72, 8, '三模型', color=DIM, fontsize=10, ha='center', alpha=a5, va='center')
    ax.text(50, 3, f'一致性损失 {CON["loss_before"]:.6f} → {CON["loss_after"]:.6f}',
            color=DIM, fontsize=9.5, ha='center', alpha=seg(t, 0.86, 0.96), va='center')




# ============================================================ ⑧ 收尾
def act8(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 55, 'v62 · 维度投影空间', color=GREEN, fontsize=16,
            weight='bold', alpha=p, ha='center', va='center')

    rows = [
        ('两两对齐',   '0.9879 – 0.9898', GREEN),
        ('三模型',     '0.6767  →  0.9925', GREEN),
        ('空间',       '256  →  Σ = 1024 维', ICE),
        ('专家',       '4 个都在用  ·  均衡 0.492', GOLD),
    ]
    for i, (k, v, c) in enumerate(rows):
        a = seg(t, 0.10 + i * 0.12, 0.24 + i * 0.12)
        if a <= 0.01: continue
        y = 46 - i * 7
        ax.text(28, y, k, color=DIM, fontsize=11, ha='right', alpha=a, va='center')
        ax.text(32, y, v, color=c, fontsize=13, ha='left', alpha=a, va='center',
                weight='bold')

    a5 = seg(t, 0.62, 0.74)
    ax.text(50, 14, '645,190,802 参数  ·  held-out 0.9101  ·  三个真实骨干',
            color=FG, fontsize=10.5, ha='center', alpha=a5, va='center')

    a6 = seg(t, 0.80, 0.92)
    ax.text(50, 7, '空间不是一个大小的数，而是按功能槽张成的一组子空间之和',
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
        if fi % 40 == 0: print(f'  {fi}/{total}', flush=True)
    print('done', flush=True)
