"""动态感与节奏的工具集。

按 motion design 的原则补三样当前缺的东西：

1. **环境层（Staging / 层次）** —— 背景微粒场一直缓慢流动，画面不再"死"。
2. **Follow-through（跟随惯性）** —— 元素到位后不立刻停，带轻微回弹/余韵。
3. **节奏对比（Timing & Pacing）** —— 多种缓动曲线，以及幕间转场。

Manim（3b1b/manim）的对应做法：3Blue1Brown 的画面永远有一层缓慢飘动的
星尘做衬底，主体元素到位时用 ease_out 而非线性停住。
"""
import numpy as np


# ---------------- 缓动曲线 ----------------
def ease(t, kind='smooth'):
    """t∈[0,1] → 缓动后的值。"""
    t = float(np.clip(t, 0.0, 1.0))
    if kind == 'smooth':
        return t * t * (3 - 2 * t)
    if kind == 'linear':
        return t
    if kind == 'out_cubic':
        return 1 - (1 - t) ** 3
    if kind == 'in_cubic':
        return t ** 3
    if kind == 'in_out_cubic':
        return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    if kind == 'out_quint':
        return 1 - (1 - t) ** 5
    if kind == 'out_back':          # 弹出带回弹（follow-through）
        c1, c3 = 1.70158, 2.70158
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
    if kind == 'out_elastic':       # 弹性
        if t == 0 or t == 1:
            return t
        c4 = 2 * np.pi / 3
        return 2 ** (-10 * t) * np.sin((t * 10 - 0.75) * c4) + 1
    if kind == 'out_expo':
        return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)
    return t * t * (3 - 2 * t)


def pop(t, t0, dur=0.35, kind='out_back'):
    """在 [t0, t0+dur] 内弹入，之后保持 1。"""
    if t < t0:
        return 0.0
    return ease((t - t0) / dur if dur > 0 else 1.0, kind)


def fade(t, t0, dur=0.25, kind='smooth'):
    """在 [t0, t0+dur] 内淡入。"""
    if t < t0:
        return 0.0
    return ease((t - t0) / dur if dur > 0 else 1.0, kind)


def pulse(t, freq=1.0, phase=0.0, lo=0.55, hi=1.0):
    """呼吸：0..1 之间的连续脉动，给静态元素一点生命。"""
    s = 0.5 + 0.5 * np.sin(2 * np.pi * (freq * t + phase))
    return lo + (hi - lo) * s


# ---------------- 背景微粒场 ----------------
def dust(ax, t, n=110, seed=7, W=100, H=62, r=0.13, alpha=0.62):
    """缓慢飘动的微粒（3Blue1Brown 式的星尘衬底）。

    粒子位置由 (seed, 初始位置) 决定，随时间做小幅度漂移 ——
    整层始终在动，但速度远慢于主体，不会抢注意力。
    """
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-6, W + 6, n)
    y0 = rng.uniform(-4, H + 4, n)
    vx = rng.uniform(-1.6, 1.6, n)
    vy = rng.uniform(-1.1, 1.1, n)
    sz = rng.uniform(0.5, 1.9, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    x = x0 + vx * t * 2.2
    y = y0 + vy * t * 1.5 + np.sin(2 * np.pi * t * 0.25 + ph) * 0.9
    tw = 0.55 + 0.45 * np.sin(2 * np.pi * t * 0.55 + ph)
    for i in range(n):
        ax.add_patch(plt_circle(x[i], y[i], r * sz[i] * 0.5,
                                alpha * tw[i] * 0.42))


def plt_circle(x, y, r, a):
    from matplotlib.patches import Circle
    return Circle((x, y), r, fc='#9FB4CC', ec='none', alpha=a, zorder=0)


# ---------------- 数字滚动 ----------------
def counter(ax, x, y, val, t, t0, dur=0.5, fmt='{:.4f}', color='#FFFFFF',
            fontsize=12, ha='center', va='center', alpha=1.0, weight='bold',
            path_effects=None):
    """数字从 0 滚到目标值（而不是突然出现）—— 数据的连续变化更有说服力。"""
    k = ease((t - t0) / dur if dur > 0 else 1.0, 'out_quint') if t >= t0 else 0.0
    v = val * k
    txt = fmt.format(v)
    kw = {}
    if path_effects is not None:
        kw['path_effects'] = path_effects
    ax.text(x, y, txt, color=color, fontsize=fontsize, ha=ha, va=va,
            alpha=alpha, weight=weight, **kw)


def pop_text(ax, x, y, txt, t, t0=0.0, dur=0.10, color='#FFFFFF',
             fontsize=14, ha='center', va='center', weight='bold',
             path_effects=None, z=6):
    """弹出式文字入场。

    out_back 会过冲到 1.08 —— alpha 不能超过 1（matplotlib 会报错），
    所以过冲交给**字号**承担：字从 88% 长到 100%，视觉上就是"弹出来"。
    """
    k = ease((t - t0) / dur, 'out_back') if t >= t0 else 0.0
    a = float(np.clip(k, 0.0, 1.0))
    fs = fontsize * (0.88 + 0.12 * min(k, 1.08))
    kw = {}
    if path_effects is not None:
        kw['path_effects'] = path_effects
    ax.text(x, y, txt, color=color, fontsize=fs, ha=ha, va=va, alpha=a,
            weight=weight, zorder=z, **kw)


# ---------------- 转场 ----------------
def transition(ax, t, n_acts, i, dur=0.06):
    """幕间的转场遮罩：幕首淡入 + 幕尾淡出（黑）。

    i 是当前幕序号（0 起）。返回 (alpha, 黑幕强度)。
    """
    head = fade(t, 0.0, dur)
    tail = 1.0 - fade(t, 1.0 - dur, dur)
    return head, (1 - head) if t < dur else 0.0
