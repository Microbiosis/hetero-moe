"""V13.0 — MixtureAligner + StudentModel + distill_loss 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research.aligner.v13_mixture import MixtureAligner, StudentModel, distill_loss


def test_mixture_aligner_output_shape():
    """1. MixtureAligner 输出形状正确"""
    aligner = MixtureAligner(modal_dims=[16, 32, 24], d_shared=64, n_shared_experts=4)
    h_list = [torch.randn(2, 8, 16), torch.randn(2, 8, 32), torch.randn(2, 8, 24)]
    out = aligner(h_list)
    assert out.shape == (2, 8, 64), f"shape {out.shape}"
    print("[PASS] test_mixture_aligner_output_shape")
    return True


def test_mixture_aligner_heterogeneous_dims():
    """2. MixtureAligner 处理异构维度 (BERT 312, Llama 768, ViT 192)"""
    aligner = MixtureAligner(modal_dims=[312, 768, 192], d_shared=256, n_shared_experts=4, num_heads=4)
    h_list = [torch.randn(1, 4, 312), torch.randn(1, 4, 768), torch.randn(1, 4, 192)]
    out = aligner(h_list)
    assert out.shape == (1, 4, 256)
    print("[PASS] test_mixture_aligner_heterogeneous_dims")
    return True


def test_mixture_aligner_gradient_flow():
    """3. MixtureAligner 参数梯度可达"""
    aligner = MixtureAligner(modal_dims=[16, 16, 16], d_shared=32, n_shared_experts=2, num_heads=4)
    h_list = [torch.randn(1, 4, 16) for _ in range(3)]
    out = aligner(h_list)
    out.sum().backward()
    n_with_grad = sum(1 for p in aligner.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in aligner.parameters())
    assert n_with_grad > 0, "no parameter received gradient"
    assert n_with_grad >= n_total * 0.8, f"too few: {n_with_grad}/{n_total}"
    print(f"[PASS] test_mixture_aligner_gradient_flow ({n_with_grad}/{n_total} params with grad)")
    return True


def test_student_model_output_shape():
    """4. StudentModel 输出形状正确"""
    student = StudentModel(d_in=312, d_out=256, lora_rank=8)
    x = torch.randn(2, 4, 312)
    y = student(x)
    assert y.shape == (2, 4, 256)
    print("[PASS] test_student_model_output_shape")
    return True


def test_student_lora_param_savings():
    """5. StudentModel LoRA 节省参数 (可训练 < 总参数 20%)"""
    student = StudentModel(d_in=512, d_out=256, lora_rank=8, hidden=128)
    trainable = student.trainable_param_count()
    total = student.total_param_count()
    ratio = trainable / total
    assert ratio < 0.2, f"LoRA 占比过高: {ratio:.2%} ({trainable}/{total})"
    print(f"[PASS] test_student_lora_param_savings (trainable {ratio:.2%})")
    return True


def test_student_only_lora_trainable():
    """6. StudentModel 只有 LoRA 参数可训练 (底座冻结)"""
    student = StudentModel(d_in=312, d_out=256, lora_rank=8)
    for n, p in student.named_parameters():
        if "A" in n or "B" in n:
            assert p.requires_grad, f"{n} 应可训练 (LoRA)"
        elif "W" in n and "norm" not in n:
            assert not p.requires_grad, f"{n} 应冻结 (底座)"
    print("[PASS] test_student_only_lora_trainable")
    return True


def test_distill_loss_components():
    """7. distill_loss 各项返回正确 (total, hidden, logit)"""
    y_s = torch.randn(2, 4, 256)
    y_t = torch.randn(2, 4, 256)
    l_s = torch.randn(2, 4, 100)
    l_t = torch.randn(2, 4, 100)
    losses = distill_loss(y_s, y_t, l_s, l_t)
    assert "total" in losses and "hidden" in losses and "logit" in losses
    # 不要求 requires_grad (因为 y_t/l_t 是 randn 不是 requires_grad)
    # 但公式应保持
    # total 应大致是 alpha*hidden + beta*logit 的加权
    expected = 0.7 * losses["hidden"] + 0.3 * losses["logit"]
    assert torch.allclose(losses["total"], expected, atol=1e-5), "total 不符合加权公式"
    print("[PASS] test_distill_loss_components")
    return True


def test_distill_loss_gradient():
    """8. distill_loss 反向时学生梯度可达"""
    y_s = torch.randn(2, 4, 256, requires_grad=True)
    y_t = torch.randn(2, 4, 256)
    l_s = torch.randn(2, 4, 100, requires_grad=True)
    l_t = torch.randn(2, 4, 100)
    losses = distill_loss(y_s, y_t, l_s, l_t)
    losses["total"].backward()
    assert y_s.grad is not None and y_s.grad.abs().sum() > 0
    assert l_s.grad is not None and l_s.grad.abs().sum() > 0
    print("[PASS] test_distill_loss_gradient")
    return True


TESTS = [
    test_mixture_aligner_output_shape,
    test_mixture_aligner_heterogeneous_dims,
    test_mixture_aligner_gradient_flow,
    test_student_model_output_shape,
    test_student_lora_param_savings,
    test_student_only_lora_trainable,
    test_distill_loss_components,
    test_distill_loss_gradient,
]


if __name__ == "__main__":
    print("=== V13.0 MixtureAligner + StudentModel + distill — 单元测试 (8 项) ===")
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