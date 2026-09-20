"""V17.0 — MultiTeacherDistiller 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research.distillation.v17_multi_teacher import MultiTeacherDistiller


def test_distiller_default_equal_weights():
    """1. 默认权重等分"""
    distiller = MultiTeacherDistiller(["v10_attn", "v13_mixture"])
    assert distiller.weights["v10_attn"] == 0.5
    assert distiller.weights["v13_mixture"] == 0.5
    print("[PASS] test_distiller_default_equal_weights")
    return True


def test_distiller_custom_weights():
    """2. 自定义权重正确"""
    distiller = MultiTeacherDistiller(["a", "b", "c"], weights={"a": 0.5, "b": 0.3, "c": 0.2})
    assert distiller.weights["a"] == 0.5
    assert distiller.weights["b"] == 0.3
    assert distiller.weights["c"] == 0.2
    print("[PASS] test_distiller_custom_weights")
    return True


def test_distiller_total_loss_components():
    """3. total loss = 加权和, 各教师独立 loss 都返回"""
    distiller = MultiTeacherDistiller(["a", "b"], weights={"a": 0.6, "b": 0.4})
    s = {"a": torch.zeros(2, 4, 16), "b": torch.zeros(2, 4, 16)}
    t = {"a": torch.ones(2, 4, 16), "b": torch.ones(2, 4, 16) * 2}
    losses = distiller(s, t)
    # 各 loss 应是 mse(student=0, teacher)
    assert "loss_a" in losses and "loss_b" in losses
    assert "total" in losses
    # mse(0, 1) = 1, mse(0, 2) = 4
    assert torch.allclose(losses["loss_a"], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(losses["loss_b"], torch.tensor(4.0), atol=1e-5)
    # total = 0.6*1 + 0.4*4 = 0.6 + 1.6 = 2.2
    assert torch.allclose(losses["total"], torch.tensor(2.2), atol=1e-5), \
        f"total={losses['total'].item()}"
    print("[PASS] test_distiller_total_loss_components")
    return True


def test_distiller_gradient_flows_to_student():
    """4. 蒸馏损失反向时学生输出有梯度"""
    distiller = MultiTeacherDistiller(["t1", "t2"])
    s = {
        "t1": torch.randn(1, 4, 8, requires_grad=True),
        "t2": torch.randn(1, 4, 8, requires_grad=True),
    }
    t = {"t1": torch.zeros(1, 4, 8), "t2": torch.zeros(1, 4, 8)}
    losses = distiller(s, t)
    losses["total"].backward()
    assert s["t1"].grad is not None and s["t1"].grad.abs().sum() > 0
    assert s["t2"].grad is not None and s["t2"].grad.abs().sum() > 0
    print("[PASS] test_distiller_gradient_flows_to_student")
    return True


def test_distiller_teacher_detach():
    """5. 教师输出不接收梯度 (detach 正确)"""
    distiller = MultiTeacherDistiller(["t"])
    teacher_out = torch.randn(1, 4, 8, requires_grad=True)
    student_out = torch.randn(1, 4, 8, requires_grad=True)
    losses = distiller({"t": student_out}, {"t": teacher_out})
    losses["total"].backward()
    assert teacher_out.grad is None or teacher_out.grad.abs().sum() == 0, \
        "教师输出不应接收梯度"
    print("[PASS] test_distiller_teacher_detach")
    return True


def test_distiller_weight_normalization():
    """6. 权重未归一化时也能工作 (用户负责归一化)"""
    # 不归一化: 权重和为 2.0
    distiller = MultiTeacherDistiller(["t"], weights={"t": 2.0})
    s = {"t": torch.zeros(1, 4, 4)}
    t = {"t": torch.ones(1, 4, 4)}
    losses = distiller(s, t)
    # mse(0, 1) = 1, total = 2.0 * 1 = 2.0
    assert torch.allclose(losses["total"], torch.tensor(2.0), atol=1e-5)
    print("[PASS] test_distiller_weight_normalization")
    return True


def test_distiller_three_teachers():
    """7. 三教师蒸馏 (v10/v13/v15) 总损失等于加权和"""
    distiller = MultiTeacherDistiller(["v10", "v13", "v15"])
    s = {n: torch.zeros(1, 4, 8) for n in ["v10", "v13", "v15"]}
    t = {n: torch.ones(1, 4, 8) for n in ["v10", "v13", "v15"]}
    losses = distiller(s, t)
    expected = sum(distiller.weights.values()) * 1.0   # 每个 mse=1
    assert torch.allclose(losses["total"], torch.tensor(expected), atol=1e-5)
    print("[PASS] test_distiller_three_teachers")
    return True


def test_distiller_zero_weights():
    """8. 权重为 0 的教师不影响总损失"""
    distiller = MultiTeacherDistiller(["active", "frozen"],
                                       weights={"active": 1.0, "frozen": 0.0})
    s = {"active": torch.zeros(1, 4, 4), "frozen": torch.zeros(1, 4, 4)}
    t = {"active": torch.ones(1, 4, 4), "frozen": torch.ones(1, 4, 4) * 100}
    losses = distiller(s, t)
    # 只有 active 教师贡献: 1.0 * 1 = 1.0
    assert torch.allclose(losses["total"], torch.tensor(1.0), atol=1e-5)
    print("[PASS] test_distiller_zero_weights")
    return True


TESTS = [
    test_distiller_default_equal_weights,
    test_distiller_custom_weights,
    test_distiller_total_loss_components,
    test_distiller_gradient_flows_to_student,
    test_distiller_teacher_detach,
    test_distiller_weight_normalization,
    test_distiller_three_teachers,
    test_distiller_zero_weights,
]


if __name__ == "__main__":
    print("=== V17.0 MultiTeacherDistiller — 单元测试 (8 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    import sys
    sys.exit(0 if passed == len(TESTS) else 1)