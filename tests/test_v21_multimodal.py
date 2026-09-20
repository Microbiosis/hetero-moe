"""V21.0 — MultiModalStudent (多模态 BERT+ViT) 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research.multimodal.v21_multimodal import MultiModalStudent, LoRAHiddenAdapter


def _get_text_inputs(batch_size: int, seq_len: int = 16):
    """构造 text 输入."""
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }


def _get_image_pixels(batch_size: int = 1, image_size: int = 224):
    """构造 image 输入 (fake pixels)."""
    return torch.randn(batch_size, 3, image_size, image_size)


def test_lora_hidden_adapter_shape():
    """1. LoRAHiddenAdapter 输出形状正确"""
    adapter = LoRAHiddenAdapter(hidden_size=64, rank=8)
    h = torch.randn(2, 8, 64)
    out = adapter(h)
    assert out.shape == h.shape
    print("[PASS] test_lora_hidden_adapter_shape")
    return True


def test_lora_hidden_adapter_initial_zero_delta():
    """2. LoRAHiddenAdapter 初始化: B=0 时 delta=0 (恒等)"""
    adapter = LoRAHiddenAdapter(hidden_size=64, rank=8)
    h = torch.randn(2, 8, 64)
    out = adapter(h)
    assert torch.allclose(out, h, atol=1e-5)
    print("[PASS] test_lora_hidden_adapter_initial_zero_delta")
    return True


def test_multimodal_student_init():
    """3. MultiModalStudent 初始化: BERT + ViT 都加载, LoRA + head 可训练"""
    student = MultiModalStudent(d_out=128, lora_rank=4)
    # BERT 冻结
    n_bert_trainable = sum(p.numel() for p in student.text_bert.parameters() if p.requires_grad)
    assert n_bert_trainable == 0
    # ViT 冻结
    n_vit_trainable = sum(p.numel() for p in student.image_vit.parameters() if p.requires_grad)
    assert n_vit_trainable == 0
    # 总可训练 > 0 (LoRA + head + router)
    assert student.trainable_param_count() > 0
    print(f"[PASS] test_multimodal_student_init (trainable: {student.trainable_param_count()})")
    return True


def test_multimodal_student_encode_text_only():
    """4. encode_text 单独可用 (纯文本编码)"""
    student = MultiModalStudent(d_out=64, lora_rank=4)
    text_inputs = _get_text_inputs(batch_size=2, seq_len=8)
    y = student.encode_text(text_inputs["input_ids"])
    assert y.shape == (2, 8, 64), f"shape {y.shape}"
    print("[PASS] test_multimodal_student_encode_text_only")
    return True


def test_multimodal_student_encode_image_only():
    """5. encode_image 单独可用 (纯图像编码)"""
    student = MultiModalStudent(d_out=64, lora_rank=4)
    pixels = _get_image_pixels(batch_size=2)
    y = student.encode_image(pixels)
    # ViT 输出 patch tokens (含或不含 CLS): [B, (224/16)^2 = 196, 64]
    S_expected = (224 // 16) ** 2  # 196
    assert y.shape[0] == 2 and y.shape[2] == 64
    assert S_expected - 5 <= y.shape[1] <= S_expected, \
        f"shape {y.shape}, S 期望在 {S_expected-5} ~ {S_expected}"
    print(f"[PASS] test_multimodal_student_encode_image_only (S={y.shape[1]})")
    return True


def test_multimodal_student_forward_shape():
    """6. forward 输出形状正确 (text + image 融合)"""
    student = MultiModalStudent(d_out=64, lora_rank=4)
    text_inputs = _get_text_inputs(batch_size=2, seq_len=16)
    pixels = _get_image_pixels(batch_size=2)
    y, weights = student(text_inputs, pixels)
    # y 形状 = [B, min(S_text=16, S_image=195), D]
    assert y.shape == (2, 16, 64), f"y shape {y.shape}"
    # weights 形状 = [B, S, 2] (text vs image)
    assert weights.shape == (2, 16, 2), f"weights shape {weights.shape}"
    print("[PASS] test_multimodal_student_forward_shape")
    return True


def test_multimodal_student_modality_weights_sum_to_one():
    """7. 模态路由器权重 softmax 后和为 1 (每 token 权重归一化)"""
    student = MultiModalStudent(d_out=64, lora_rank=4)
    text_inputs = _get_text_inputs(batch_size=2, seq_len=8)
    pixels = _get_image_pixels(batch_size=2)
    _, weights = student(text_inputs, pixels)
    weights_sum = weights.sum(dim=-1)  # [B, S]
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), \
        "模态权重 softmax 后应和为 1"
    print("[PASS] test_multimodal_student_modality_weights_sum_to_one")
    return True


def test_multimodal_student_gradient_flow():
    """8. MultiModalStudent 所有可训练参数梯度可达 (BERT/ViT 冻结)"""
    student = MultiModalStudent(d_out=64, lora_rank=4)
    text_inputs = _get_text_inputs(batch_size=1, seq_len=8)
    pixels = _get_image_pixels(batch_size=1)
    y, weights = student(text_inputs, pixels)
    loss = y.sum() + weights.sum()
    loss.backward()
    # text/image LoRA + head 应有梯度
    assert student.text_lora.A.grad is not None
    assert student.image_lora.A.grad is not None
    assert student.text_head.weight.grad is not None
    assert student.image_head.weight.grad is not None
    assert student.W_modality.grad is not None
    # BERT / ViT 冻结, 不应有梯度
    n_bert_grad = sum(1 for p in student.text_bert.parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0)
    n_vit_grad = sum(1 for p in student.image_vit.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_bert_grad == 0, f"BERT 有 {n_bert_grad} 个参数有梯度"
    assert n_vit_grad == 0, f"ViT 有 {n_vit_grad} 个参数有梯度"
    print("[PASS] test_multimodal_student_gradient_flow")
    return True


TESTS = [
    test_lora_hidden_adapter_shape,
    test_lora_hidden_adapter_initial_zero_delta,
    test_multimodal_student_init,
    test_multimodal_student_encode_text_only,
    test_multimodal_student_encode_image_only,
    test_multimodal_student_forward_shape,
    test_multimodal_student_modality_weights_sum_to_one,
    test_multimodal_student_gradient_flow,
]


if __name__ == "__main__":
    print("=== V21.0 MultiModalStudent (BERT + ViT) — 单元测试 (8 项) ===")
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