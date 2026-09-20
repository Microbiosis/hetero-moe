"""V19.0 — C-1 根因诊断工具单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research.central_diagnostics.v19_c1_analysis import DiagnosticRunner, CapacityComparison, LongTrainingComparison


def test_capacity_comparison_output_shape():
    """1. CapacityComparison 输出形状正确"""
    model = CapacityComparison(d_shared=32, num_experts=3, student_capacity=[16])
    x = torch.randn(2, 8, 32)
    y, alpha = model(x)
    assert y.shape == x.shape
    assert alpha.shape == (2, 8, 3)
    print("[PASS] test_capacity_comparison_output_shape")
    return True


def test_capacity_comparison_gradient_flow():
    """2. CapacityComparison 所有参数梯度可达 (需让 alpha 也参与 loss).

    注意: U.grad 需要 c 非零才能传播 (因为 z = W·x + U·c, dL/dU = grad_z · c^T).
    测试用随机初始化 c 验证整条链.
    """
    torch.manual_seed(0)
    model = CapacityComparison(d_shared=32, num_experts=3, student_capacity=[16, 8])
    # 手动让 c 非零, 验证 U.grad 也能传播
    with torch.no_grad():
        model.c.data = torch.randn_like(model.c) * 0.1
    x = torch.randn(1, 4, 32)
    y, alpha = model(x)
    target_alpha = torch.zeros_like(alpha)
    target_alpha[..., 0] = 1.0
    loss = y.sum() + F.mse_loss(alpha, target_alpha)
    loss.backward()
    # 学生
    for p in model.student.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    # 中枢
    assert model.c.grad is not None and model.c.grad.abs().sum() > 0
    assert model.U.grad is not None and model.U.grad.abs().sum() > 0, \
        f"U.grad={model.U.grad}"
    # 路由器
    assert model.W.grad is not None and model.W.grad.abs().sum() > 0
    print("[PASS] test_capacity_comparison_gradient_flow")
    return True


def test_long_training_output_shape():
    """3. LongTrainingComparison 输出形状正确"""
    model = LongTrainingComparison(d_shared=32, num_experts=3)
    x = torch.randn(2, 8, 32)
    s = model.student(x)
    z = F.linear(x, model.W) + model.U @ model.c
    assert s.shape == x.shape
    assert z.shape == (2, 8, 3)
    print("[PASS] test_long_training_output_shape")
    return True


def test_diagnostic_runner_record():
    """4. DiagnosticRunner.record 能正确存储结果"""
    runner = DiagnosticRunner()
    runner.record("H1", "U-train-baseline", 1.5)
    runner.record("H1", "U-frozen", 0.5)
    runner.record("H3", "100-steps", 1.2)
    runner.record("H3", "1000-steps", 0.8)
    assert "H1" in runner.results
    assert "H3" in runner.results
    assert len(runner.results["H1"]) == 2
    assert len(runner.results["H3"]) == 2
    print("[PASS] test_diagnostic_runner_record")
    return True


def test_diagnostic_runner_summary():
    """5. DiagnosticRunner.summary 生成可读报告"""
    runner = DiagnosticRunner()
    runner.record("H1", "U-train-baseline", 1.5)
    runner.record("H1", "U-frozen", 0.5)
    report = runner.summary()
    assert "H1" in report
    assert "U-train-baseline" in report
    assert "fuse=1.5000" in report
    print("[PASS] test_diagnostic_runner_summary")
    return True


def test_capacity_comparison_c_init_zero():
    """6. CapacityComparison 初始化: c=0 时 broadcast = 0"""
    model = CapacityComparison(d_shared=32, num_experts=3, student_capacity=[16])
    x = torch.randn(1, 4, 32)
    y1, _ = model(x)
    # 手动让 c=0 (默认就是 0)
    assert model.c.abs().sum() == 0
    # 此时 z = W·x + U·0 = W·x, alpha 应等于 softmax(W·x)
    z_expected = F.linear(x, model.W)
    alpha_expected = F.softmax(z_expected, dim=-1)
    _, alpha = model(x)
    assert torch.allclose(alpha, alpha_expected, atol=1e-5), \
        "c=0 时 alpha 应等于 softmax(W·x)"
    print("[PASS] test_capacity_comparison_c_init_zero")
    return True


def test_capacity_comparison_various_student_sizes():
    """7. 不同学生容量都能运行"""
    for cap in [[8], [16, 8], [32, 16, 8]]:
        model = CapacityComparison(d_shared=32, num_experts=3, student_capacity=cap)
        x = torch.randn(1, 4, 32)
        y, alpha = model(x)
        assert y.shape == x.shape
        n_params = sum(p.numel() for p in model.student.parameters())
        assert n_params > 0
    print("[PASS] test_capacity_comparison_various_student_sizes")
    return True


def test_diagnostic_runner_multiple_hypotheses():
    """8. DiagnosticRunner 支持多个假设独立追踪"""
    runner = DiagnosticRunner()
    for hyp in ["H1", "H2", "H3", "H4", "H5"]:
        for cond in ["A", "B", "C"]:
            runner.record(hyp, cond, float(hyp[1]) + 0.1 * len(cond))
    assert len(runner.results) == 5
    assert all(len(runner.results[h]) == 3 for h in runner.results)
    print("[PASS] test_diagnostic_runner_multiple_hypotheses")
    return True


TESTS = [
    test_capacity_comparison_output_shape,
    test_capacity_comparison_gradient_flow,
    test_long_training_output_shape,
    test_diagnostic_runner_record,
    test_diagnostic_runner_summary,
    test_capacity_comparison_c_init_zero,
    test_capacity_comparison_various_student_sizes,
    test_diagnostic_runner_multiple_hypotheses,
]


if __name__ == "__main__":
    print("=== V19.0 C-1 根因诊断工具 — 单元测试 (8 项) ===")
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