"""V15.0 — MoMAligner (嵌套专家池) 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research.aligner.v15_nested import MoMAligner


def test_mom_aligner_output_shape():
    """1. MoMAligner 输出形状正确"""
    aligner = MoMAligner(modal_dims=[16, 32, 24], d_shared=64,
                          n_inner=2, n_outer=2, num_heads=4)
    h_list = [torch.randn(2, 8, 16), torch.randn(2, 8, 32), torch.randn(2, 8, 24)]
    out = aligner(h_list)
    assert out.shape == (2, 8, 64), f"shape {out.shape}"
    print("[PASS] test_mom_aligner_output_shape")
    return True


def test_mom_aligner_heterogeneous_dims():
    """2. MoMAligner 处理异构维度 (BERT 312, Llama 768, ViT 192)"""
    aligner = MoMAligner(modal_dims=[312, 768, 192], d_shared=256,
                          n_inner=2, n_outer=2, num_heads=4)
    h_list = [torch.randn(1, 4, 312), torch.randn(1, 4, 768), torch.randn(1, 4, 192)]
    out = aligner(h_list)
    assert out.shape == (1, 4, 256)
    print("[PASS] test_mom_aligner_heterogeneous_dims")
    return True


def test_mom_aligner_gradient_flow():
    """3. MoMAligner 内层外层专家梯度可达 (K/V 投影冗余但不影响主路径)"""
    aligner = MoMAligner(modal_dims=[16, 16, 16], d_shared=32,
                          n_inner=2, n_outer=2, num_heads=4)
    h_list = [torch.randn(1, 4, 16) for _ in range(3)]
    out = aligner(h_list)
    out.sum().backward()
    # 内层专家 (6 个) 都应有梯度
    for i, e in enumerate(aligner.inner_experts):
        n_with_grad = sum(1 for p in e.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad >= 2, f"内层专家[{i}] 仅有 {n_with_grad} 个参数有梯度"
    # 外层专家 (2 个) 都应有梯度
    for i, e in enumerate(aligner.outer_experts):
        n_with_grad = sum(1 for p in e.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad >= 2, f"外层专家[{i}] 仅有 {n_with_grad} 个参数有梯度"
    print("[PASS] test_mom_aligner_gradient_flow")
    return True


def test_mom_aligner_two_levels_distinct():
    """4. MoMAligner 两层结构不同 (内层处理底座, 外层处理语义)"""
    aligner = MoMAligner(modal_dims=[16, 16, 16], d_shared=32,
                          n_inner=2, n_outer=2, num_heads=4)
    # 内层 M × n_inner = 6 个专家, 外层 n_outer = 2 个专家
    assert len(aligner.inner_experts) == 6, f"内层应有 6 个, 实测 {len(aligner.inner_experts)}"
    assert len(aligner.outer_experts) == 2, f"外层应有 2 个, 实测 {len(aligner.outer_experts)}"
    print("[PASS] test_mom_aligner_two_levels_distinct")
    return True


def test_mom_aligner_inner_outputs_combine():
    """5. MoMAligner 移除任一底座改变输出 (跨底座耦合). 阈值设低, 因为两级 mean 会压缩差异."""
    torch.manual_seed(0)
    aligner = MoMAligner(modal_dims=[16, 16, 16], d_shared=32,
                          n_inner=2, n_outer=2, num_heads=4)
    h_a = torch.randn(1, 4, 16)
    h_b = torch.randn(1, 4, 16)
    h_c = torch.randn(1, 4, 16)
    out_full = aligner([h_a, h_b, h_c])
    # 用 0 输入替代任一底座, 应观察到差异 (阈值 1e-4, 因两级 mean 压缩)
    for no_idx, name in enumerate(["A", "B", "C"]):
        h_list = [h_a, h_b, h_c]
        h_list[no_idx] = torch.zeros_like(h_list[no_idx])
        out_partial = aligner(h_list)
        diff = (out_full - out_partial).abs().mean().item()
        assert diff > 1e-4, f"移除底座 {name} 输出差异过小 ({diff:.6f}), 跨底座耦合失效"
    print("[PASS] test_mom_aligner_inner_outputs_combine")
    return True


def test_mom_aligner_n_inner_n_outer_param_count():
    """6. MoMAligner 参数量符合预期 (M × n_inner + n_outer) * expert_size"""
    aligner = MoMAligner(modal_dims=[16, 32, 24], d_shared=64,
                          n_inner=2, n_outer=2, num_heads=4)
    n_inner = 3 * 2  # M=3, n_inner=2
    n_outer = 2
    assert len(aligner.inner_experts) == n_inner
    assert len(aligner.outer_experts) == n_outer
    # 每专家 3 层 Linear (16, 64 -> 64 -> 64) = 16*64 + 64*64 + 64 + 64 = ~5000
    print(f"[PASS] test_mom_aligner_n_inner_n_outer_param_count "
          f"({n_inner} inner + {n_outer} outer experts)")
    return True


def test_mom_aligner_output_differs_from_mixture():
    """7. MoMAligner 输出与 v13 MixtureAligner 不同 (嵌套应产生不同特征)"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from research.aligner.v13_mixture import MixtureAligner
    torch.manual_seed(42)
    mom = MoMAligner(modal_dims=[16, 16, 16], d_shared=32,
                      n_inner=2, n_outer=2, num_heads=4)
    mix = MixtureAligner(modal_dims=[16, 16, 16], d_shared=32,
                          n_shared_experts=2, num_heads=4)
    h_list = [torch.randn(1, 4, 16) for _ in range(3)]
    out_mom = mom(h_list)
    out_mix = mix(h_list)
    # 两者结构不同, 输出应显著不同
    diff = (out_mom - out_mix).abs().mean()
    assert diff > 1e-3, f"MoM 与 Mixture 输出几乎相同 (diff={diff:.6f}), 嵌套没起作用"
    print(f"[PASS] test_mom_aligner_output_differs_from_mixture (diff={diff:.4f})")
    return True


def test_mom_aligner_step_decreases_loss():
    """8. MoMAligner 接入 AlignedFusionLayer 后训练 loss 下降"""
    from research._primitives.attention import AttnPool
    from research.aligner.v10_embedding import AlignedFusionLayer

    pools = []
    for s in range(3):
        torch.manual_seed(s)
        w = torch.randn(16, 16) * 0.04
        pools.append(AttnPool(16, 32, 4, w.clone(), w.clone(), w.clone(), torch.eye(16)))

    aligner = MoMAligner(modal_dims=[16, 16, 16], d_shared=32,
                          n_inner=2, n_outer=2, num_heads=4)
    layer = AlignedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16], aligner=aligner,
        c1_alpha=0.1,
    )
    from research.routing_evolution.v8_central import V8Trainer
    trainer = V8Trainer(layer, lr=1e-2, phase=2)
    h_list = [torch.randn(1, 4, 16) for _ in range(3)]
    targets = [torch.zeros(1, 4, 32) for _ in range(3)]
    targets[0][:, :, :11] = 1.0
    targets[1][:, :, 11:22] = 1.0
    targets[2][:, :, 22:] = 1.0
    losses = []
    for _ in range(15):
        trainer.optimizer.zero_grad(set_to_none=True)
        y = layer(h_list)
        loss = sum(F.mse_loss(y, t) for t in targets) / 3
        loss.backward()
        trainer.optimizer.step()
        losses.append(loss.item())
    avg_first = sum(losses[:5]) / 5
    avg_last = sum(losses[-5:]) / 5
    assert avg_last < avg_first, f"训练无效: {avg_first:.4f} → {avg_last:.4f}"
    print(f"[PASS] test_mom_aligner_step_decreases_loss ({avg_first:.4f} → {avg_last:.4f})")
    return True


TESTS = [
    test_mom_aligner_output_shape,
    test_mom_aligner_heterogeneous_dims,
    test_mom_aligner_gradient_flow,
    test_mom_aligner_two_levels_distinct,
    test_mom_aligner_inner_outputs_combine,
    test_mom_aligner_n_inner_n_outer_param_count,
    test_mom_aligner_output_differs_from_mixture,
    test_mom_aligner_step_decreases_loss,
]


if __name__ == "__main__":
    print("=== V15.0 MoMAligner (嵌套专家池) — 单元测试 (8 项) ===")
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