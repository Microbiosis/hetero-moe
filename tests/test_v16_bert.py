"""V16.0 — BERTStudent (真实 BERT + LoRA) 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research.distillation.v16_bert_student import BERTStudent, LoRAAdapter


def test_lora_adapter_shape():
    """1. LoRAAdapter 输出形状正确"""
    adapter = LoRAAdapter(hidden_size=64, rank=8)
    h = torch.randn(2, 8, 64)
    out = adapter(h)
    assert out.shape == h.shape, f"shape {out.shape}"
    print("[PASS] test_lora_adapter_shape")
    return True


def test_lora_adapter_initial_zero_delta():
    """2. LoRAAdapter 初始化: B=0 时 delta=0 (恒等映射)"""
    adapter = LoRAAdapter(hidden_size=64, rank=8)
    h = torch.randn(2, 8, 64)
    # B 初始化为 0, 所以 delta 应该 = 0
    out = adapter(h)
    assert torch.allclose(out, h, atol=1e-5), "B=0 时 LoRA 应是恒等映射"
    print("[PASS] test_lora_adapter_initial_zero_delta")
    return True


def test_bert_student_output_shape():
    """3. BERTStudent 前向输出形状正确"""
    student = BERTStudent(model_name="huawei-noah/TinyBERT_General_4L_312D",
                           d_out=256, lora_rank=8)
    input_ids = torch.randint(0, 1000, (2, 16))
    y = student(input_ids)
    assert y.shape == (2, 16, 256), f"shape {y.shape}"
    print("[PASS] test_bert_student_output_shape")
    return True


def test_bert_student_bert_frozen():
    """4. BERTStudent BERT 原有权重冻结, 仅 LoRA + head 可训练"""
    student = BERTStudent(d_out=256, lora_rank=8)
    n_bert_trainable = sum(p.numel() for p in student.bert.parameters() if p.requires_grad)
    assert n_bert_trainable == 0, f"BERT 仍有 {n_bert_trainable} 个可训练参数, 应全部冻结"
    # LoRA + head 应可训练
    trainable = student.trainable_param_count()
    assert trainable > 0, "无可训练参数"
    print(f"[PASS] test_bert_student_bert_frozen (trainable: {trainable})")
    return True


def test_bert_student_param_savings():
    """5. BERTStudent LoRA 参数远小于 BERT 总参数 (<5%)"""
    student = BERTStudent(d_out=256, lora_rank=8)
    trainable = student.trainable_param_count()
    bert_params = student.bert_param_count()
    ratio = trainable / bert_params
    assert ratio < 0.05, f"LoRA 占比过高: {ratio:.2%} ({trainable}/{bert_params})"
    print(f"[PASS] test_bert_student_param_savings (LoRA {ratio:.2%} of BERT)")
    return True


def test_bert_student_gradient_flow():
    """6. BERTStudent LoRA + head 梯度可达.

    LoRA 初始化: B=0, A 随机. 初始步 B.grad 非零但 A.grad=0
    (链式法则: dL/dA = dL/d(delta) · B^T, B=0 截断).
    这是 LoRA 标准行为: B 先"激活", A 通过 B 反向传播获得梯度.

    测试验证:
    1. 初始步 B 和 head 有梯度, 且 A.grad 恰为 0 (B=0 结构性截断)
    2. BERT 仍冻结 (无梯度)
    3. 强制把 B 改成非零后, A.grad 应非零 (证明 LoRA 数学路径正确)

    注 (损失必须非退化): 早前版本用 ``y.sum()`` 作损失, 但 forward 末端是
    ``nn.LayerNorm``, 其输出对通道求和的梯度在 gamma 均匀时恒为 0 ——
    ``sum(LN(z))`` 只依赖 bias, 与 z 无关, 实测 ``d(sum(LN(z)))/dz ≈ 1e-6``
    纯浮点噪声. 于是所有梯度断言都只剩噪声量级 (1e-5~1e-3), ``A.grad`` 是否
    恰好非零随运行浮动, 导致本测试偶发失败 (实测 10 次跑挂 5 次).
    改用 ``(y**2).mean()`` 获得真实梯度路径 (量级 1e-2~1e-1, 高出 2~3 个数量级).
    """
    torch.manual_seed(0)  # 固定输入采样与 BERT dropout, 保证可复现
    student = BERTStudent(d_out=256, lora_rank=8)
    input_ids = torch.randint(0, 1000, (1, 16))
    y = student(input_ids)
    (y ** 2).mean().backward()  # 非退化损失 (见 docstring 注)
    # LoRA B (初始为 0 但收到梯度)
    assert student.lora.B.grad is not None and student.lora.B.grad.abs().sum() > 0, \
        "LoRA B 无梯度"
    # A 初始严格为 0: B=0 使 delta≡0, A 的梯度链被结构性截断 (非浮点噪声)
    assert student.lora.A.grad is not None and student.lora.A.grad.abs().sum() == 0, \
        "B=0 时 A.grad 应为 0 (LoRA 初始化标准行为)"
    # head
    assert student.head.weight.grad is not None and student.head.weight.grad.abs().sum() > 0, \
        "head 无梯度"
    # norm (LN 参数)
    assert student.norm.weight.grad is not None and student.norm.weight.grad.abs().sum() > 0, \
        "norm 无梯度"
    # BERT 仍冻结
    n_bert_grad = sum(1 for p in student.bert.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_bert_grad == 0, f"BERT 有 {n_bert_grad} 个参数收到了梯度"

    # 强制 B 非零, 验证 A.grad 路径
    with torch.no_grad():
        student.lora.B.data = torch.randn_like(student.lora.B) * 0.1
    student.zero_grad()
    y2 = student(input_ids)
    (y2 ** 2).mean().backward()
    assert student.lora.A.grad is not None and student.lora.A.grad.abs().sum() > 0, \
        "B 非零后 LoRA A 仍未收到梯度 (LoRA 数学路径错误)"
    print("[PASS] test_bert_student_gradient_flow")
    return True


def test_bert_student_attention_mask():
    """7. BERTStudent attention_mask 正确传递"""
    student = BERTStudent(d_out=256, lora_rank=8)
    input_ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16)
    mask[0, 8:] = 0  # 第二个样本只看前 8 个 token
    y = student(input_ids, attention_mask=mask)
    assert y.shape == (2, 16, 256)
    print("[PASS] test_bert_student_attention_mask")
    return True


def test_bert_student_step_decreases_loss():
    """8. BERTStudent 单底座 + LoRA 训练可降低 loss"""
    student = BERTStudent(d_out=256, lora_rank=8)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, 1000, (1, 16))
        target = torch.zeros(1, 16, 256)
        target[:, :, :128] = 1.0
        y = student(input_ids)
        loss = F.mse_loss(y, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    assert avg_last < avg_first, f"训练无效: {avg_first:.4f} → {avg_last:.4f}"
    print(f"[PASS] test_bert_student_step_decreases_loss ({avg_first:.4f} → {avg_last:.4f})")
    return True


TESTS = [
    test_lora_adapter_shape,
    test_lora_adapter_initial_zero_delta,
    test_bert_student_output_shape,
    test_bert_student_bert_frozen,
    test_bert_student_param_savings,
    test_bert_student_gradient_flow,
    test_bert_student_attention_mask,
    test_bert_student_step_decreases_loss,
]


if __name__ == "__main__":
    print("=== V16.0 BERTStudent (真实 BERT + LoRA) — 单元测试 (8 项) ===")
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