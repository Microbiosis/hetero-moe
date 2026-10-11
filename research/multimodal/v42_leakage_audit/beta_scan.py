"""v42.0 — 多模态蒸馏泄漏审计的算子集 (β_h 连续过渡扫描).

审计对象
--------
``v21_multimodal`` 的学生损失是::

    loss = 0.7 · MSE(y, teacher) + 0.3 · MSE(y, target)

这与 ``v16_bert_student`` 的结构**完全相同**, 即 v32 判定为泄漏源的那个系数:
``w_teacher = 0.7 ⇒ β_h = 0.3``。v21 声称"多模态学生蒸馏后比教师优 50.5%"
(学生 0.1578 vs 教师 0.3190)。v32 已在 v16 上证明: **把 β_h 压到 0, 学生回到教师水平,
全部表观裕度归零。** 本模块提供把该检验移植到 v21 所需的算子。

两种可能, 必须能区分
---------------------
- **线性插值**: 效应与 β_h 成正比, 关系是一条直线。⇒ 所谓"蒸馏增益"其实是对目标项的
  逐步加权, **没有机制**。
- **阈值机制**: 关系在某处有折点, 例如 β_h=0.3 以下无效、以上才激活。
  ⇒ 存在真实的中间机制, 裕度不是纯插值。

区分它们的关键不是拟合一条直线, 而是看**残差与折点是否大于种子噪声** ——
这是 v40 之后本项目强制遵守的纪律(均值/中位数口径必须先过稳健性)。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

MIN_USABLE_NOISE = 1e-9
# 折点判据: 二阶差分的峰值集中度。平滑曲率下该值≈1, 孤立折点通常 ≳3。
# 这是**独立于噪声容差的显式阈值** —— 两者量纲不同, 不可互相推导。
KINK_RATIO_THRESHOLD = 3.0

__all__ = [
    "blend_loss",
    "beta_from_weight",
    "weight_from_beta",
    "linearity_report",
    "linearity_verdict",
    "MIN_USABLE_NOISE",
    "KINK_RATIO_THRESHOLD",
]


def beta_from_weight(w_teacher: float) -> float:
    """蒸馏权重 → 泄漏系数 β_h。``β_h = 1 − w_teacher``。"""
    return 1.0 - float(w_teacher)


def weight_from_beta(beta_h: float) -> float:
    """泄漏系数 β_h → 蒸馏权重 ``w_teacher``。"""
    return 1.0 - float(beta_h)


def blend_loss(student, teacher, target, beta_h: float, teacher_weights=None):
    """参数化蒸馏损失: ``(1−β_h)·MSE(y, y_T) + β_h·MSE(y, t)``。

    - ``beta_h = 0`` : 纯蒸馏, 无直接目标项 ⇒ 泄漏通道关闭
    - ``beta_h = 1`` : 纯目标回归, 教师完全不起作用 ⇒ 捷径全开

    ``teacher_weights`` 是逐样本权重(可广播), 用于排除某些样本带教师信息的情形;
    缺省为 None(每个样本同等)。
    """
    import torch.nn.functional as F

    t_teacher = teacher.detach()
    d_teacher = F.mse_loss(student, t_teacher, reduction="none")
    d_target = F.mse_loss(student, target, reduction="none")
    if teacher_weights is not None:
        w = teacher_weights
        while w.dim() < d_teacher.dim():
            w = w.unsqueeze(-1)
        d_teacher = d_teacher * w
        d_target = d_target * (1.0 - w)
    return (1.0 - float(beta_h)) * d_teacher.mean() + float(beta_h) * d_target.mean()


def _ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx


def linearity_report(betas: Sequence[float], means: Sequence[float],
                     noise: float) -> Dict[str, float]:
    """给出一条 β_h → MSE 曲线的线性诊断。

    参数
    ----
    betas : 扫描点
    means : 每个点的均值
    noise : 种子间噪声尺度(同 β_h 下不同 seed 的标准差), 用于判断残差是否**超过噪声**

    返回 ``slope / intercept / r2 / max_second_diff / max_abs_residual /
    residual_over_noise / monotone``。
    """
    xs, ys = list(map(float, betas)), list(map(float, means))
    slope, inter = _ols(xs, ys)
    pred = [slope * x + inter for x in xs]
    resid = [y - p for y, p in zip(ys, pred)]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0

    # 二阶差分: 若存在阈值折点, 曲线的"弯折"会在这里显形
    second = [ys[i + 1] - 2 * ys[i] + ys[i - 1] for i in range(1, len(ys) - 1)]
    max_second = max((abs(v) for v in second), default=0.0)
    max_res = max((abs(r) for r in resid), default=0.0)
    # 折点必须是**局部**尖峰: 某一个二阶差分显著大于其余。
    # 若全部二阶差分同号且量级相当, 那是系统性曲率(平滑凸/凹衰减), 不是阈值。
    abs_second = [abs(v) for v in second] or [0.0]
    mean_second = sum(abs_second) / len(abs_second)
    kink_ratio = max_second / mean_second if mean_second > 0 else 0.0
    same_sign = (all(v >= 0 for v in second) or all(v <= 0 for v in second)) if second else True
    noise = max(float(noise), 1e-12)

    return {
        "slope": slope,
        "intercept": inter,
        "r2": r2,
        "max_abs_residual": max_res,
        "max_second_diff": max_second,
        "noise": noise,
        "residual_over_noise": max_res / noise,
        "second_over_noise": max_second / noise,
        "kink_ratio": kink_ratio,
        "second_same_sign": same_sign,
        "monotone": all(ys[i + 1] <= ys[i] for i in range(len(ys) - 1)),
    }


def linearity_verdict(report: Dict[str, float], tol: float = 1.0) -> str:
    """把 :func:`linearity_report` 的读数翻译成裁决。

    裁决规则(刻意保守):
    - 残差 **不超过**噪声 (``residual_over_noise ≤ tol``) ⇒ 无可检出的非线性 ⇒ ``linear``
    - 曲率超过噪声但**二阶差分同号且量级相当** (``kink_ratio`` 小) ⇒ ``smooth_curvature``
      —— 系统性凸/凹衰减, **不是阈值**
    - 曲率超过噪声且某一处二阶差分**孤立突出** (``kink_ratio`` 大) ⇒ ``threshold``
    - 其余 ⇒ ``inconclusive``, **不下结论**

    第三种结果是本项目 v40 之后必须保留的选项: 读不出结构就说读不出,
    不要拿一条拟合线冒充机制。

    **无重复样本 ⇒ 噪声尺度未定义 ⇒ 任何比值都是除零产物, 必须拒绝下结论。**
    单 seed 时 noise=0, 残差/噪声会算出 1e10 量级的假信号, 从而把任意一条曲线
    误判成 threshold。这条护栏是实测踩出来的: 第一次冒烟 (1 seed) 就误判了。
    """
    if report["noise"] <= MIN_USABLE_NOISE:
        return "inconclusive"
    if report["residual_over_noise"] <= tol:
        return "linear"
    # 有真实曲率, 但要分清是"折点"还是"平滑曲率"
    if report["second_over_noise"] <= tol:
        return "inconclusive"
    if report["kink_ratio"] >= KINK_RATIO_THRESHOLD:
        return "threshold"             # 某一处孤立尖峰
    return "smooth_curvature"         # 系统性凸/凹, 量级相当 ⇒ 不是阈值
