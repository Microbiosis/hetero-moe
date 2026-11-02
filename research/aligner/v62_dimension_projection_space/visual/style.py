"""视觉风格 —— 借鉴 3b1b/manim（ManimCE 0.22.1）的色板与图形语言。

色值全部取自 manim/utils/color/manim_colors.py（PyPI sdist 实测，非凭印象）：

  BLUE_C  #58C4DD    TEAL_C  #5CD0B3    GREEN_C #83C167   YELLOW_C #F7D96F
  GOLD_C  #F0AC5F    RED_C   #FC6255    MAROON_C #C55F73   PURPLE_C #CAA3E8
  GREY_A  #DDDDDD    GREY_B  #BBBBBB    GREY_C  #888888   GREY_D  #444444
  GREY_E  #222222    BLACK   #000000

借鉴的三条原则：
1. **同色相深浅梯度**，而不是四个互不相干的高饱和色相。
   Manim 每个色相给 A/B/C/D/E 五档；四个槽用同一色相的四档，整体协调。
2. **描边 + 半透明填充**（3B1B 的图形语汇），不是实心色块。
3. **排版分层**：主标题 / 幕号 / 数据标注 / 出处水印，字号与灰度拉开。
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

# ---------------- Manim 色板 ----------------
BG        = '#14161C'      # 冷调深灰（Manim GREY_E #222222 略冷化）
BG_DEEP   = '#0C0E13'
WHITE     = '#FFFFFF'
GREY_A    = '#DDDDDD'
GREY_B    = '#BBBBBB'
GREY_C    = '#888888'
GREY_D    = '#444444'

BLUE      = '#58C4DD'
BLUE_A    = '#C7E9F1'
BLUE_D    = '#29ABCA'
TEAL      = '#5CD0B3'
GREEN     = '#83C167'
YELLOW    = '#F7D96F'
GOLD      = '#F0AC5F'
RED       = '#FC6255'
MAROON    = '#C55F73'
PURPLE    = '#CAA3E8'

# 四个槽：Manim 绿系的 A/B/C/D 四档深浅（同色相梯度 = 协调）
SLOT_GRAD = ['#C9E2AE', '#A6CF8C', '#83C167', '#77B05D']
# 三个模态：Manim 的暖金 / 浅蓝 / 紫 —— 与槽的绿系不撞，且各自有身份
MOD_GRAD  = [GOLD, BLUE_A, PURPLE]
# 四个专家：Manim 暖色组（RED/GOLD/MAROON/PURPLE），与模态色分属不同场景
EXP_GRAD  = [RED, GOLD, MAROON, PURPLE]

# 强调数字用 GOLD（Manim 常用黄/金标关键量）
ACCENT   = GOLD
NEGATIVE = RED

# ---------------- 排版层级 ----------------
FS_TITLE   = 19      # 幕标题
FS_SUB     = 10.5    # 副标题 / 槽位说明
FS_NUM     = 12.5    # 关键数字
FS_LABEL   = 9.5     # 标注
FS_MICRO   = 8       # 水印 / 出处

def glow(lw=8, color=None, alpha=0.55):
    """外发光：先用背景色多次描边叠出光晕。"""
    c = color or BG
    return [pe.Stroke(linewidth=lw, foreground=c, alpha=alpha),
            pe.Stroke(linewidth=lw * 0.55, foreground=c, alpha=alpha * 0.8),
            pe.Normal()]

def breathing_glow(color, t, freq=1.6, base=1.8, amp=0.9):
    """呼吸辉光：强度随时间脉动，让静态元素也有生命。"""
    import math
    k = base + amp * (0.5 + 0.5 * math.sin(2 * math.pi * freq * t))
    return text_glow(color, lw=k, alpha=0.30 + 0.25 * (k - base) / max(amp, 1e-6))


def text_glow(color, lw=1.8, alpha=0.45):
    """文字辉光：同色细描边（在字下方），模拟 3B1B 的发光数字。

    宽度必须**远小于字号**，否则描边会盖住字本身糊成一团。
    11.5pt 的字用 1.8pt 的描边刚好。
    """
    return [pe.Stroke(linewidth=lw, foreground=color, alpha=alpha),
            pe.Stroke(linewidth=lw * 0.45, foreground=color, alpha=alpha * 0.8),
            pe.Normal()]


def line_glow(color, lw=4, alpha=0.6):
    return [pe.Stroke(linewidth=lw, foreground=color, alpha=alpha),
            pe.Stroke(linewidth=lw * 0.45, foreground=color, alpha=alpha),
            pe.Normal()]


def glow_dot(ax, xy, r, fc, alpha=1.0, z=3, ring=0.42):
    """发光节点：半透明实心 + 同色粗描边（3B1B 的"能量点"）。

    实心圆在深底上显得死板；描边圆 + 内芯更像流动的数据。
    """
    x, y = xy
    ax.add_patch(plt.Circle((x, y), r, fc=to_rgba(fc, alpha * 0.85),
                            ec=to_rgba(fc, alpha), lw=r * ring, zorder=z,
                            path_effects=line_glow(fc, lw=max(r * 1.6, 4), alpha=0.5)))
    ax.add_patch(plt.Circle((x, y), r * 0.32, fc=to_rgba('#FFFFFF', alpha * 0.9),
                            ec='none', zorder=z + 1))
    return None


def rbox(ax, xy, w, h, fc, ec=None, lw=2.0, r=0.9, alpha=1.0,
         fill_alpha=0.45, z=2, shadow=False, fill=True):
    """圆角 + 描边 + 半透明填充（3B1B 的图形语汇）。

    xy 是 (x, y) 元组，写法与 Rectangle((x, y), w, h, ...) 一致。
    """
    x, y = xy
    ec = ec or fc
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.5, y - 0.6), w, h,
                     boxstyle=f'round,pad=0,rounding_size={r}',
                     fc='none', ec='none', alpha=0.35, zorder=z - 1))
    # FancyBboxPatch 没有 fill_alpha 参数 —— 填充透明度用 RGBA 表达
    face = to_rgba(fc, fill_alpha) if fill else 'none'
    edge = to_rgba(ec, alpha)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=face, ec=edge, lw=lw, zorder=z,
                       path_effects=glow(lw=max(lw * 2.2, 6)) if alpha > 0.85 else None)
    ax.add_patch(p)
    return p

def title(ax, zh, en=None, sub=None, color=WHITE):
    """统一的幕标题排版。"""
    ax.text(50, 58.5, zh, color=color, fontsize=FS_TITLE, weight='bold',
            ha='center', va='center')
    if en:
        ax.text(50, 54.6, en, color=GREY_C, fontsize=FS_LABEL,
                ha='center', va='center')

def watermark(ax, text, alpha=1.0):
    ax.text(97, 3.2, text, color=GREY_D, fontsize=FS_MICRO,
            ha='right', va='center', alpha=alpha)

def camera(ax, zoom=1.0, cx=50, cy=31):
    """镜头推近：把画布按 zoom 放大并平移，使中心保持在 (cx, cy)。"""
    if zoom == 1.0:
        return
    ax.set_xlim(cx - 50 / zoom, cx + 50 / zoom)
    ax.set_ylim(cy - 31 / zoom, cy + 31 / zoom)
