"""v40.0 — 分派式多教师算子单元测试 (12 项).

本文件是权威副本。根目录 ``tests/`` 下的转发层若存在也只是薄转发层 —— 本代刻意**不**添加,
理由见 MANIFEST._note(新增转发层会改变 run.sh test 的套件计数)。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V40.0 分派式多教师算子 — 单元测试 (12 项)"

import torch
import torch.nn.functional as F

from research.distillation.v40_dispatch import (
    consensus_assign,
    cyclic_assign,
    dispatch_loss,
    oracle_assign,
    pairwise_disagreement,
    teacher_mean,
    teacher_target_mse,
    weighted_loss,
)


def _two_conflicting():
    """构造两个方向相反的教师, 学生在中间 —— 加权和会互相抵消."""
    student = torch.zeros(1, 1, 4)
    return student, {"up": torch.ones(1, 1, 4), "down": -torch.ones(1, 1, 4)}


def test_weighted_loss_equals_manual_weighted_sum():
    """1. weighted_loss 等于手工加权和 (v17 distill.py:51 的逐行等价)"""
    s = torch.zeros(2, 3, 4)
    t = {"a": torch.ones(2, 3, 4), "b": torch.ones(2, 3, 4) * 2}
    out = weighted_loss(s, t, weights={"a": 0.25, "b": 0.75})
    expected = 0.25 * 1.0 + 0.75 * 4.0
    assert torch.allclose(out["total"], torch.tensor(expected), atol=1e-6)
    assert torch.allclose(out["a"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out["b"], torch.tensor(4.0), atol=1e-6)
    print("[PASS] test_weighted_loss_equals_manual_weighted_sum")
    return True


def test_weighted_loss_default_equal_weights():
    """2. 默认权重等分"""
    s = torch.zeros(1, 2, 4)
    t = {"a": torch.ones(1, 2, 4), "b": torch.ones(1, 2, 4)}
    out = weighted_loss(s, t)
    # 等权: 0.5*1 + 0.5*1 = 1
    assert torch.allclose(out["total"], torch.tensor(1.0), atol=1e-6)
    print("[PASS] test_weighted_loss_default_equal_weights")
    return True


def test_weighted_loss_rejects_shape_mismatch():
    """3. 教师与学生形状不符时报错"""
    s = torch.zeros(1, 2, 4)
    try:
        weighted_loss(s, {"a": torch.ones(1, 2, 5)})
    except ValueError:
        print("[PASS] test_weighted_loss_rejects_shape_mismatch")
        return True
    raise AssertionError("形状不符时应当抛 ValueError")


def test_dispatch_all_same_teacher_equals_that_teacher_loss():
    """4. 分派到同一教师时, 等于只用该教师 (与 weighted 单教师等价)"""
    s = torch.randn(3, 2, 4)
    t = {"a": torch.randn(3, 2, 4), "b": torch.randn(3, 2, 4)}
    dispatched = dispatch_loss(s, t, ["a", "a", "a"])
    solo = weighted_loss(s, {"a": t["a"]})
    assert torch.allclose(dispatched, solo["total"], atol=1e-6)
    print("[PASS] test_dispatch_all_same_teacher_equals_that_teacher_loss")
    return True


def test_dispatch_equals_mean_of_per_sample_losses():
    """5. dispatch_loss 等于逐样本单教师 loss 的均值"""
    s = torch.randn(4, 2, 8)
    t = {"a": torch.randn(4, 2, 8), "b": torch.randn(4, 2, 8)}
    assign = ["a", "b", "a", "b"]
    manual = sum(F.mse_loss(s[i], t[assign[i]][i]) for i in range(4)) / 4
    assert torch.allclose(dispatch_loss(s, t, assign), manual, atol=1e-6)
    print("[PASS] test_dispatch_equals_mean_of_per_sample_losses")
    return True


def test_weighted_dilutes_where_dispatch_does_not():
    """6. 核心构造: 相反方向的两个教师下, 加权和被抵消, 分派不被抵消"""
    s, t = _two_conflicting()
    # 加权和: 0.5*(0-1)^2 + 0.5*(0-(-1))^2 = 0.5 + 0.5 = 1.0
    w = weighted_loss(s, t)["total"]
    # 分派到 "up": (0-1)^2 = 1.0;  分派到 "down": (0-(-1))^2 = 1.0 —— 同样 1.0
    d = dispatch_loss(s, t, ["up"])
    assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(d, torch.tensor(1.0), atol=1e-6)

    # 真正的差别在梯度: 加权和把两个相反方向的梯度摊平
    gs = s.clone().requires_grad_(True)
    weighted_loss(gs, t)["total"].backward()
    g_weighted = gs.grad.clone()

    gs2 = s.clone().requires_grad_(True)
    dispatch_loss(gs2, t, ["up"]).backward()
    g_dispatch = gs2.grad.clone()

    # 加权和的梯度幅度严格小于单教师分派 —— 这就是"信号稀释"
    assert g_dispatch.abs().sum() > g_weighted.abs().sum() * 1.9, (
        f"分派梯度 {g_dispatch.abs().sum():.4f} 应显著大于加权梯度 {g_weighted.abs().sum():.4f}"
    )
    print("[PASS] test_weighted_dilutes_where_dispatch_does_not")
    return True


def test_dispatch_rejects_bad_assign():
    """7. assign 长度不符 / 未知教师 均报错"""
    s = torch.randn(3, 2, 4)
    t = {"a": torch.randn(3, 2, 4)}
    for bad, exc in ((["a", "a"], ValueError), (["a", "a", "zzz"], KeyError)):
        try:
            dispatch_loss(s, t, bad)
        except exc:
            continue
        raise AssertionError(f"{bad} 应当抛 {exc.__name__}")
    print("[PASS] test_dispatch_rejects_bad_assign")
    return True


def test_oracle_assign_picks_nearest_teacher():
    """8. oracle 分派选中与目标最近的那个教师"""
    t = {
        "far": torch.ones(3, 1, 4),
        "near": torch.zeros(3, 1, 4),
        "mid": torch.full((3, 1, 4), 0.5),
    }
    target = torch.zeros(3, 1, 4)
    assign = oracle_assign(t, target)
    assert assign == ["near", "near", "near"], assign
    print("[PASS] test_oracle_assign_picks_nearest_teacher")
    return True


def test_oracle_is_strict_upper_bound_vs_any_dispatch():
    """9. oracle 分派的 loss 不大于任何其他分派 (上界性质)"""
    s = torch.zeros(4, 1, 4)
    t = {
        "a": torch.full((4, 1, 4), 0.2),
        "b": torch.full((4, 1, 4), 5.0),
        "c": torch.full((4, 1, 4), -3.0),
    }
    target = torch.zeros(4, 1, 4)
    oracle = dispatch_loss(s, t, oracle_assign(t, target)).item()
    for cand in (cyclic_assign(t), consensus_assign(t), ["a"] * 4, ["b"] * 4, ["c"] * 4):
        assert dispatch_loss(s, t, cand).item() >= oracle - 1e-6, cand
    print("[PASS] test_oracle_is_strict_upper_bound_vs_any_dispatch")
    return True


def test_consensus_assign_is_target_free_and_deterministic():
    """10. 共识分派不读目标 (签名无 target) 且确定性"""
    import inspect
    assert "target" not in inspect.signature(consensus_assign).parameters, \
        "consensus_assign 不得接收目标 —— 无泄漏是它的全部意义"
    t = {
        "a": torch.randn(5, 2, 8),
        "b": torch.randn(5, 2, 8),
        "c": torch.randn(5, 2, 8),
    }
    assert consensus_assign(t) == consensus_assign(t)
    assert len(consensus_assign(t)) == 5
    print("[PASS] test_consensus_assign_is_target_free_and_deterministic")
    return True


def test_cyclic_assign_is_zero_information_baseline():
    """11. 轮转分派对教师质量无感 —— 零信息对照"""
    t = {"good": torch.randn(4, 1, 4), "bad": torch.randn(4, 1, 4)}
    assert cyclic_assign(t) == ["good", "bad", "good", "bad"]
    print("[PASS] test_cyclic_assign_is_zero_information_baseline")
    return True


def test_diagnostics_separate_consensus_from_disagreement():
    """12. 诊断量: 完全一致的教师 disagreement=0; 共识目标可比单教师更好"""
    same = {"a": torch.randn(3, 1, 4), "b": torch.randn(3, 1, 4)}
    # 把 b 复制成 a
    same["b"] = same["a"].clone()
    assert pairwise_disagreement(same) < 1e-12

    # 两个互补教师 + 无偏目标: 均值应当比任一单教师更接近目标
    t = {"a": torch.ones(2, 1, 4), "b": -torch.ones(2, 1, 4)}
    target = torch.zeros(2, 1, 4)
    mse = teacher_target_mse(t, target)
    assert mse["__consensus_mean__"] < mse["a"] and mse["__consensus_mean__"] < mse["b"], mse
    assert torch.allclose(teacher_mean(t), torch.zeros(2, 1, 4), atol=1e-6)
    print("[PASS] test_diagnostics_separate_consensus_from_disagreement")
    return True


TESTS = [
    test_weighted_loss_equals_manual_weighted_sum,
    test_weighted_loss_default_equal_weights,
    test_weighted_loss_rejects_shape_mismatch,
    test_dispatch_all_same_teacher_equals_that_teacher_loss,
    test_dispatch_equals_mean_of_per_sample_losses,
    test_weighted_dilutes_where_dispatch_does_not,
    test_dispatch_rejects_bad_assign,
    test_oracle_assign_picks_nearest_teacher,
    test_oracle_is_strict_upper_bound_vs_any_dispatch,
    test_consensus_assign_is_target_free_and_deterministic,
    test_cyclic_assign_is_zero_information_baseline,
    test_diagnostics_separate_consensus_from_disagreement,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
