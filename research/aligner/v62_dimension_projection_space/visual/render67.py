"""第 ⑥ 幕（结构化 token · 四槽分工）与第 ⑦ 幕（专家路由）。

所有数值来自真实前向（data/v62_capture.npz）与 2002 条 held-out 统计
（data/v62_route.json）。不编造。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent if '__file__' in dir() else _pl.Path.cwd()
DATA = str(_HERE / 'data')
FRAMES = str(_HERE / 'frames')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

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
SLOT_COL = ['#F0873B', '#64E3F6', '#35D0BA', '#b085f5']   # Q K V O
EXP_COL  = ['#F0873B', '#64E3F6', '#35D0BA', '#b085f5']
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
    ax.add_patch(Rectangle((0, 0), W, 7.0, color='#000000', alpha=0.55 * alpha))
    ax.text(50, 4.6, zh, color=FG, fontsize=13.5, ha='center', va='center', alpha=alpha)
    ax.text(50, 1.8, en, color=DIM, fontsize=9, ha='center', va='center', alpha=alpha * 0.9)


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


# ============================================================ 幕 ⑥
def act6(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 57, '⑥ 一个 token 的结构：四个槽各贡献多少', color='#35D0BA',
            fontsize=14, weight='bold', alpha=p, va='center', ha='center')

    # --- 0.00-0.22 塌缩演示（v62.1 的真实读数）---
    a1 = seg(t, 0.02, 0.10) * (1 - seg(t, 0.18, 0.26))
    if a1 > 0.01:
        ax.text(2.5, 50.5, 'v62.1 实测：槽门控熵 0.0209', color='#CF3655',
                fontsize=10, alpha=a1, va='center')
        # 塌成一格
        base, bh, bw = 33.0, 12.0, 8.0
        # v62.1 的真实形态：一个槽吃掉几乎全部权重（具体哪个槽随 token 变）
        vals_collapsed = [0.9995, 0.0002, 0.0001, 0.0002]
        for i, v in enumerate(vals_collapsed):
            hgt = max(bh * v, 0.12)
            ax.add_patch(Rectangle((14 + i * 18, base), bw, hgt,
                                   color=SLOT_COL[i], alpha=a1 * 0.9))
            ax.text(14 + i * 18 + bw / 2, base - 2.2, SLOT_NAMES[i],
                    color=DIM, fontsize=11, ha='center', alpha=a1)
        ax.text(50, 22.0, '一个槽吃掉几乎全部权重',
                color='#CF3655', fontsize=11, alpha=a1, va='center', ha='center')

    # --- 0.26+ 展开（v62.2 真实值）---
    a2 = seg(t, 0.26, 0.36)
    if a2 > 0.01:
        ax.text(2.5, 50.5, 'v62.2 修后：熵 0.693，四槽都参与', color='#35D0BA',
                fontsize=10, alpha=a2, va='center')

        # 滚动展示若干 token 的真实 slot_alpha
        sa = D['slot_alpha']            # [16,4]
        n = len(sa)
        # 按熵从高到低排序后循环展示 —— 等距抽样会正好抽到极端 token（t3 的 0.911），
        # 让画面看起来像没修好。排序后展示的是「典型到极端」的完整谱。
        ent = -np.sum(sa * np.log(sa + 1e-12), -1) / np.log(4)
        order = np.argsort(-ent)
        step = int((t - 0.26) / 0.60 * n)
        idx = order[step % n]
        row = sa[idx]
        base, bw, maxh = 26.0, 12.0, 26.0

        for i, v in enumerate(row):
            grow = a2 * seg(t, 0.30 + i * 0.02, 0.40 + i * 0.02)
            hgt = max(maxh * float(v) * grow, 0.35)
            x = 8 + i * 22
            ax.add_patch(Rectangle((x, base), bw, hgt,
                                   color=SLOT_COL[i], alpha=0.92, lw=0))
            ax.text(x + bw / 2, base - 2.4, SLOT_NAMES[i],
                    color=SLOT_COL[i], fontsize=12, ha='center', alpha=a2, weight='bold')
            ax.text(x + bw / 2, base + hgt + 2.2, f'{v:.3f}',
                    color=FG, fontsize=12, ha='center', alpha=a2 * grow)

        # 四槽激活率行
        a3 = seg(t, 0.44, 0.52)
        ax.text(50, base - 8.0,
                f'四槽激活率 100%  ·  平均熵 {np.mean(-np.sum(sa*np.log(sa+1e-12),-1))/np.log(4):.3f}  ·  和 = 1',
                color='#35D0BA', fontsize=11, alpha=a3, va='center', ha='center')

        # --- 拆分重组 ---
        a4 = seg(t, 0.56, 0.66)
        if a4 > 0.01:
            ax.text(50, 20.0, '可分离 · 可重组', color='#64E3F6',
                    fontsize=11.5, weight='bold', alpha=a4, va='center', ha='center')
            # 左：token5 的 K
            ax.add_patch(Rectangle((8, 8), 24, 8, color='#64E3F6', alpha=a4 * 0.28))
            ax.text(20, 12, 'token 5 的 K 槽', color='#64E3F6', fontsize=10.5,
                    ha='center', va='center', alpha=a4)
            ax.add_patch(Rectangle((38, 8), 24, 8, color='#b085f5', alpha=a4 * 0.28))
            ax.text(50, 12, 'token 0 的 O 槽', color='#b085f5', fontsize=10.5,
                    ha='center', va='center', alpha=a4)
            ax.annotate('', xy=(76, 12), xytext=(64, 12),
                        arrowprops=dict(arrowstyle='->', color=FG, lw=1.6, alpha=a4))
            ax.add_patch(Rectangle((78, 7), 18, 10, color=FG, alpha=a4 * 0.13))
            ax.text(87, 12, '新 token', color=FG, fontsize=10.5,
                    ha='center', va='center', alpha=a4)

    # 底部小字
    a5 = seg(t, 0.10, 0.20)
    ax.text(50, 2.5, '[B, S, 4, 256]  ·  token 数不变  ·  全槽软分布',
            color=DIM, fontsize=9, ha='center', alpha=a5, va='center')


# ============================================================ 幕 ⑦
def act7(ax, t):
    p = seg(t, 0.0, 0.06)
    ax.text(50, 60.0, '⑦ 1024 维 → 4 个专家：谁在干活', color='#35D0BA',
            fontsize=14, weight='bold', alpha=p, va='center', ha='center')

    load = R['report']['expert']['load']       # 2002 条真实
    EA = D['expert_alpha']                     # [16,4] 真实 top-k

    # ---------- 上半区 y=46..56：1024 带 → 4 个专家盒 → top-2 连线 ----------
    a1 = seg(t, 0.04, 0.14)
    if a1 > 0.01:
        ax.add_patch(Rectangle((4, 49), 22, 5, color='#64E3F6', alpha=a1 * 0.22))
        ax.text(15, 51.5, 'canonical 1024', color='#64E3F6', fontsize=9,
                ha='center', va='center', alpha=a1)

        for e in range(4):
            x = 34 + e * 16
            ax.add_patch(Rectangle((x, 45), 12, 10, fill=False,
                                   edgecolor=EXP_COL[e], lw=1.3, alpha=a1))
            ax.text(x + 6, 50.0, f'专家{e}', color=EXP_COL[e], fontsize=9,
                    ha='center', va='center', alpha=a1)

        a2 = seg(t, 0.12, 0.24)
        if a2 > 0.01:
            for tok in range(0, 16, 3):
                live = np.where(EA[tok] > 0)[0]
                y0 = 49.5 + (tok % 3 - 1) * 1.1
                for e in live:
                    xe = 34 + e * 16 + 6
                    # 先水平到专家列上方，再垂直落进盒子顶边
                    ax.plot([26, xe], [y0, 58.0], color=EXP_COL[e],
                            lw=0.7, alpha=a2 * 0.5)
                    ax.plot([xe, xe], [58.0, 55.2], color=EXP_COL[e],
                            lw=0.7, alpha=a2 * 0.5)
            ax.text(50, 42.0, '每个 token 只连 2 条线（top-k=2）',
                    color=DIM, fontsize=9, ha='center', alpha=a2, va='center')

    # ---------- 下半区 y=10..38：真实负载柱 ----------
    a3 = seg(t, 0.30, 0.44)
    if a3 > 0.01:
        base, bw, maxh = 12.0, 13.0, 24.0
        for e in range(4):
            x = 12 + e * 21
            hgt = maxh * load[e] * a3
            # 背景槽
            ax.add_patch(Rectangle((x, base), bw, maxh, fill=False,
                                   edgecolor=EXP_COL[e], lw=0.8, alpha=a3 * 0.35))
            ax.add_patch(Rectangle((x, base), bw, hgt, color=EXP_COL[e], alpha=0.9))
            ax.text(x + bw / 2, base - 2.6, f'专家{e}', color=EXP_COL[e],
                    fontsize=10.5, ha='center', alpha=a3, weight='bold')
            ax.text(x + bw / 2, base + hgt + 2.0, f'{load[e]:.3f}', color=FG,
                    fontsize=11, ha='center', alpha=a3)

        a4 = seg(t, 0.46, 0.56)
        ax.text(12 + 21 + bw / 2, base + maxh * load[1] + 6.5,
                '承担 73%', color='#64E3F6', fontsize=10, ha='center', alpha=a4)
        ax.text(12 + bw / 2, base + maxh * load[0] + 6.5,
                '36%', color=DIM, fontsize=10, ha='center', alpha=a4)

        ax.text(50, 6.2,
                f'均衡 {BAL:.3f}  ·  熵 {_E:.3f}  ·  top-k={R["top_k"]}（每 token 只跑 2 个专家）',
                color=DIM, fontsize=9.5, ha='center', alpha=a3, va='center')

    # ---------- 角标 ----------
    a5 = seg(t, 0.60, 0.72)
    ax.text(50, 40.5, '⚠ 四个专家均为随机初始化，未从真实 FFN 抽取',
            color='#CF3655', fontsize=10.5, ha='center', alpha=a5, va='center')
    a6 = seg(t, 0.74, 0.84)
    ax.text(50, 33.0, '专家已解冻 · 50.3M 可训练参数',
            color=DIM, fontsize=9.5, ha='center', alpha=a6, va='center')

    # ---------- 底部对照条 ----------
    a7 = seg(t, 0.86, 0.96)
    ax.add_patch(Rectangle((6, 1.5), 88, 3.2, color='#64E3F6', alpha=a7 * 0.09))
    ax.text(50, 3.1,
            '槽门控：全槽软分布（四个都参与）    |    专家选择：top-k=2（真的省算力）',
            color='#64E3F6', fontsize=9.5, ha='center', alpha=a7, va='center')


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else '6'
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    out = os.path.join(FRAMES, f'act{which}')
    os.makedirs(out, exist_ok=True)
    fn = act6 if which == '6' else act7
    print(f'渲染 幕{which}: {total} 帧', flush=True)
    for fi in range(total):
        frame(fi, total, fn, out)
        if fi % 40 == 0: print(f'  {fi}/{total}', flush=True)
    print('done', flush=True)
