"""v42.0 — β_h 泄漏审计算子单元测试 (8 项)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V42.0 β_h 泄漏审计算子 — 单元测试 (10 项)"

import torch
import torch.nn.functional as F

from research.multimodal.v42_leakage_audit import (
    KINK_RATIO_THRESHOLD,
    MIN_USABLE_NOISE,
    beta_from_weight,
    blend_loss,
    linearity_report,
    linearity_verdict,
    weight_from_beta,
)


def test_beta_weight_roundtrip():
    """1. β_h 与 w_teacher 互为逆映射"""
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert abs(weight_from_beta(beta_from_weight(w)) - w) < 1e-12
    assert abs(beta_from_weight(0.7) - 0.3) < 1e-12, "v21/v16 的 0.7 权重对应 β_h=0.3"
    print("[PASS] test_beta_weight_roundtrip")
    return True


def test_blend_endpoints():
    """2. β_h=0 纯蒸馏 / β_h=1 纯目标, 两端与定义一致"""
    s = torch.randn(2, 3, 4)
    t_teacher, t_target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    pure_d = blend_loss(s, t_teacher, t_target, 0.0)
    pure_t = blend_loss(s, t_teacher, t_target, 1.0)
    assert torch.allclose(pure_d, F.mse_loss(s, t_teacher), atol=1e-6)
    assert torch.allclose(pure_t, F.mse_loss(s, t_target), atol=1e-6)
    print("[PASS] test_blend_endpoints")
    return True


def test_blend_is_affine_in_beta():
    """3. 损失对 β_h 仿射 —— 这正是"线性插值"的可证伪前提"""
    s = torch.randn(2, 3, 4)
    t_teacher, t_target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    a = blend_loss(s, t_teacher, t_target, 0.0).item()
    b = blend_loss(s, t_teacher, t_target, 1.0).item()
    for beta in (0.1, 0.37, 0.83):
        got = blend_loss(s, t_teacher, t_target, beta).item()
        assert abs(got - (a + beta * (b - a))) < 1e-5, beta
    print("[PASS] test_blend_is_affine_in_beta")
    return True


def test_blend_detaches_teacher():
    """4. 教师不接收梯度"""
    s = torch.randn(2, 3, 4, requires_grad=True)
    t_teacher = torch.randn(2, 3, 4, requires_grad=True)
    blend_loss(s, t_teacher, torch.randn(2, 3, 4), 0.7).backward()
    assert t_teacher.grad is None, "教师输出不应有梯度"
    assert s.grad is not None and s.grad.abs().sum() > 0
    print("[PASS] test_blend_detaches_teacher")
    return True


def test_report_recovers_known_line():
    """5. 对一条已知直线, 诊断应给出 R²=1 且近零残差"""
    xs = [i / 10 for i in range(11)]
    ys = [0.30 - 0.14 * x for x in xs]
    r = linearity_report(xs, ys, noise=0.01)
    assert r["r2"] > 0.9999, r
    assert abs(r["slope"] + 0.14) < 1e-6, r
    assert linearity_verdict(r) == "linear"
    print("[PASS] test_report_recovers_known_line")
    return True


def test_smooth_curvature_is_not_threshold():
    """6a. 系统性凸衰减(二阶差分同号且量级相当)不得判成 threshold"""
    xs = [i / 10 for i in range(11)]
    ys = [0.33 * (1 - i / 10) ** 2 for i in range(11)]   # 平滑凸衰减
    r = linearity_report(xs, ys, noise=0.002)
    assert r["second_same_sign"] is True, r
    assert r["kink_ratio"] < KINK_RATIO_THRESHOLD, r
    assert linearity_verdict(r) == "smooth_curvature", r
    print("[PASS] test_smooth_curvature_is_not_threshold")
    return True


def test_report_detects_threshold():
    """6. 对一条有折点的曲线, 应判为 threshold"""
    xs = [i / 10 for i in range(11)]
    ys = [0.30] * 5 + [0.30 - 0.14 * (i - 4) for i in range(5, 11)]
    r = linearity_report(xs, ys, noise=0.005)
    assert linearity_verdict(r) == "threshold", r
    assert r["second_over_noise"] > 1.0
    print("[PASS] test_report_detects_threshold")
    return True


def test_verdict_can_be_inconclusive():
    """7. 噪声压过残差时必须允许"不下结论" —— v40 之后的强制选项"""
    xs = [i / 10 for i in range(11)]
    ys = [0.30, 0.28, 0.31, 0.29, 0.30, 0.28, 0.31, 0.29, 0.30, 0.28, 0.30]
    r = linearity_report(xs, ys, noise=0.20)      # 噪声远大于效应
    assert linearity_verdict(r) == "linear", "噪声大时残差自然不显著"
    r2 = linearity_report(xs, ys, noise=0.001)    # 同样的数据, 噪声小
    assert linearity_verdict(r2) != "linear", "同样的曲线, 低噪声下应暴露非线性"
    print("[PASS] test_verdict_can_be_inconclusive")
    return True


def test_verdict_refuses_when_noise_undefined():
    """7b. 无重复样本(noise=0)时必须拒绝下结论 —— 不得把除零产物当机制"""
    xs = [i / 10 for i in range(11)]
    ys = [0.30 - 0.14 * x for x in xs]
    r0 = linearity_report(xs, ys, noise=0.0)          # 单 seed 的情形
    assert r0["noise"] <= MIN_USABLE_NOISE, "无重复样本时噪声应落在不可用区间"
    assert linearity_verdict(r0) == "inconclusive", "噪声未定义时不得给出 linear/threshold"
    # 即便曲线真的强非线性, 没有重复样本也不能声称是折点
    kink = [0.30] * 5 + [0.30 - 0.14 * (i - 4) for i in range(5, 11)]
    rk = linearity_report(xs, kink, noise=0.0)
    assert linearity_verdict(rk) == "inconclusive", "有折点也需重复样本支撑"
    print("[PASS] test_verdict_refuses_when_noise_undefined")
    return True


def test_report_noise_scaling():
    """8. 同一曲线, 噪声放大 10× 时残差比应同步缩小 —— 证明噪声尺度真的参与判定"""
    xs = [i / 10 for i in range(11)]
    ys = [0.30 - 0.14 * x + (0.01 if i % 2 else -0.01) for i, x in enumerate(xs)]
    hi = linearity_report(xs, ys, noise=0.5)
    lo = linearity_report(xs, ys, noise=0.005)
    assert hi["residual_over_noise"] < lo["residual_over_noise"]
    assert abs(hi["slope"] - lo["slope"]) < 1e-9, "斜率不应依赖噪声尺度"
    print("[PASS] test_report_noise_scaling")
    return True


TESTS = [
    test_beta_weight_roundtrip,
    test_blend_endpoints,
    test_blend_is_affine_in_beta,
    test_blend_detaches_teacher,
    test_report_recovers_known_line,
    test_smooth_curvature_is_not_threshold,
    test_report_detects_threshold,
    test_verdict_can_be_inconclusive,
    test_verdict_refuses_when_noise_undefined,
    test_report_noise_scaling,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
