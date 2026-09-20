"""V10.0 语义对齐器 — 单元测试 (8 项)。

不依赖 transformers. 用随机张量构造.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.aligner.v10_embedding import (
    SemanticAligner, CrossArchMeanAligner, CrossArchAttnAligner,
    AlignedFusionLayer,
)


def _make_pools(n=3, D_m=32, D_shared=64, H=4):
    pools = []
    for s in range(n):
        torch.manual_seed(s)
        w = torch.randn(D_m, D_m) * 0.04
        pools.append(AttnPool(D_m, D_shared, H, w.clone(), w.clone(), w.clone(), torch.eye(D_m)))
    return pools


def test_mean_aligner_output_shape():
    """1. CrossArchMeanAligner 输出形状正确"""
    aligner = CrossArchMeanAligner(modal_dims=[16, 32, 24], d_shared=64)
    h_list = [torch.randn(2, 8, 16), torch.randn(2, 8, 32), torch.randn(2, 8, 24)]
    out = aligner(h_list)
    assert out.shape == (2, 8, 64), f"shape {out.shape}"
    print("[PASS] test_mean_aligner_output_shape")
    return True


def test_attn_aligner_output_shape():
    """2. CrossArchAttnAligner 输出形状正确"""
    aligner = CrossArchAttnAligner(modal_dims=[16, 32, 24], d_shared=64, num_heads=4)
    h_list = [torch.randn(2, 8, 16), torch.randn(2, 8, 32), torch.randn(2, 8, 24)]
    out = aligner(h_list)
    assert out.shape == (2, 8, 64)
    print("[PASS] test_attn_aligner_output_shape")
    return True


def test_aligner_gradient_flow():
    """3. aligner 参数梯度可达"""
    aligner = CrossArchAttnAligner(modal_dims=[16, 32], d_shared=32, num_heads=4)
    h_list = [torch.randn(1, 4, 16), torch.randn(1, 4, 32)]
    out = aligner(h_list)
    out.sum().backward()
    # down_projections 梯度
    for proj in aligner.down_projections:
        for p in proj.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0
    # 共享 W_k / W_v 梯度
    for p in [aligner.W_k.weight, aligner.W_v.weight]:
        assert p.grad is not None and p.grad.abs().sum() > 0
    # q_projections 梯度
    for qp in aligner.q_projections:
        for p in qp.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0
    print("[PASS] test_aligner_gradient_flow")
    return True


def test_attn_aligner_cross_modal_dependence():
    """4. attn aligner 输出依赖所有底座 (移除任一底座, 输出显著变化)"""
    torch.manual_seed(0)
    aligner = CrossArchAttnAligner(modal_dims=[16, 32, 24], d_shared=64, num_heads=4)
    h_a, h_b, h_c = torch.randn(1, 4, 16), torch.randn(1, 4, 32), torch.randn(1, 4, 24)
    out_full = aligner([h_a, h_b, h_c])
    out_no_a = aligner([torch.zeros_like(h_a), h_b, h_c])
    out_no_b = aligner([h_a, torch.zeros_like(h_b), h_c])
    # 移除任一底座都应显著改变输出
    assert (out_full - out_no_a).abs().mean() > 1e-3
    assert (out_full - out_no_b).abs().mean() > 1e-3
    print("[PASS] test_attn_aligner_cross_modal_dependence")
    return True


def test_aligned_layer_forward_shape():
    """5. AlignedFusionLayer 输入 h_list, 输出 [B, S, d_shared]"""
    aligner = CrossArchMeanAligner(modal_dims=[16, 16, 16], d_shared=32)
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = AlignedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16], aligner=aligner,
    )
    h_list = [torch.randn(2, 4, 16) for _ in range(3)]
    y = layer(h_list)
    assert y.shape == (2, 4, 32), f"shape {y.shape}"
    print("[PASS] test_aligned_layer_forward_shape")
    return True


def test_aligned_layer_gate_style_central():
    """6. AlignedFusionLayer 使用 v9_gate_central.GateStyleWorkspace (修正 C)"""
    from research._primitives.central_mechanism import GateStyleWorkspace
    aligner = CrossArchAttnAligner(modal_dims=[16, 16, 16], d_shared=32, num_heads=4)
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = AlignedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16], aligner=aligner,
        c1_alpha=0.1,
    )
    # 验证 GateStyleWorkspace 配置正确
    assert isinstance(layer.cw_attn, GateStyleWorkspace), \
        f"cw_attn 应为 GateStyleWorkspace, 实测 {type(layer.cw_attn).__name__}"
    assert isinstance(layer.cw_ffn, GateStyleWorkspace)
    assert layer.cw_attn.alpha == 0.1
    assert layer.cw_ffn.alpha == 0.1
    # 前向
    h_list = [torch.randn(2, 4, 16) for _ in range(3)]
    y = layer(h_list)
    assert y.shape == (2, 4, 32)
    print("[PASS] test_aligned_layer_gate_style_central")
    return True


def test_aligner_handles_different_modal_dims():
    """7. attn aligner 处理异构维度 (BERT 312, Llama 768, ViT 192)"""
    aligner = CrossArchAttnAligner(modal_dims=[312, 768, 192], d_shared=256, num_heads=4)
    h_list = [torch.randn(1, 4, 312), torch.randn(1, 4, 768), torch.randn(1, 4, 192)]
    out = aligner(h_list)
    assert out.shape == (1, 4, 256)
    print("[PASS] test_aligner_handles_different_modal_dims")
    return True


def test_aligner_layer_trainable_params():
    """8. AlignedFusionLayer trainable params 包括 aligner + 双路由 + 中枢 + γ/β/α"""
    aligner = CrossArchMeanAligner(modal_dims=[16, 32, 24], d_shared=64)
    pools = _make_pools(n=3, D_m=32, D_shared=64)
    layer = AlignedFusionLayer(
        d_shared=64, d_ff=128, attn_pools=pools,
        modal_dims=[16, 32, 24], aligner=aligner,
    )
    n_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    # aligner 全可训练 + 双路由 + 中枢 + γ/β/α
    assert n_trainable > 1000, f"too few trainable: {n_trainable}"
    # FFN 冻结
    for p in layer.w_gates:
        assert not p.requires_grad
    print("[PASS] test_aligner_layer_trainable_params")
    return True


TESTS = [
    test_mean_aligner_output_shape,
    test_attn_aligner_output_shape,
    test_aligner_gradient_flow,
    test_attn_aligner_cross_modal_dependence,
    test_aligned_layer_forward_shape,
    test_aligned_layer_gate_style_central,
    test_aligner_handles_different_modal_dims,
    test_aligner_layer_trainable_params,
]


if __name__ == "__main__":
    print("=== V10.0 语义对齐器 — 单元测试 (8 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    import sys
    sys.exit(0 if passed == len(TESTS) else 1)