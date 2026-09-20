"""V12.0 — ComposedFusionLayer (v9 中枢 + v10 aligner 组合) 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from archive.negative_findings.v12_compose import ComposedFusionLayer


def _make_pools(n=3, D_m=32, D_shared=64, H=4):
    pools = []
    for s in range(n):
        torch.manual_seed(s)
        w = torch.randn(D_m, D_m) * 0.04
        pools.append(AttnPool(D_m, D_shared, H, w.clone(), w.clone(), w.clone(), torch.eye(D_m)))
    return pools


def test_composed_returns_aligned_fusion():
    """1. ComposedFusionLayer 返回 AlignedFusionLayer 实例"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
    )
    assert isinstance(layer, AlignedFusionLayer)
    print("[PASS] test_composed_returns_aligned_fusion")
    return True


def test_composed_uses_gate_central():
    """2. ComposedFusionLayer 默认使用 v9_gate_central.GateStyleWorkspace"""
    from research._primitives.central_mechanism import GateStyleWorkspace
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
    )
    assert isinstance(layer.cw_attn, GateStyleWorkspace), \
        f"cw_attn 应为 GateStyleWorkspace, 实测 {type(layer.cw_attn).__name__}"
    assert isinstance(layer.cw_ffn, GateStyleWorkspace)
    assert layer.cw_attn.alpha == 0.1, f"alpha 应为 0.1, 实测 {layer.cw_attn.alpha}"
    print("[PASS] test_composed_uses_gate_central")
    return True


def test_composed_uses_attn_aligner():
    """3. ComposedFusionLayer 默认使用 CrossArchAttnAligner"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
    )
    assert isinstance(layer.aligner, CrossArchAttnAligner)
    print("[PASS] test_composed_uses_attn_aligner")
    return True


def test_composed_forward_shape():
    """4. ComposedFusionLayer 前向输出形状正确"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
    )
    h_list = [torch.randn(2, 4, 16) for _ in range(3)]
    y = layer(h_list)
    assert y.shape == (2, 4, 32)
    print("[PASS] test_composed_forward_shape")
    return True


def test_composed_gradient_flow():
    """5. ComposedFusionLayer 梯度流 (aligner + 中枢 + 路由都可达)"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
    )
    h_list = [torch.randn(1, 4, 16) for _ in range(3)]
    y = layer(h_list)
    y.sum().backward()
    # aligner 参数
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in layer.aligner.parameters())
    # 路由参数
    assert layer.W_router_attn.grad is not None and layer.W_router_attn.grad.abs().sum() > 0
    assert layer.W_router_ffn.grad is not None and layer.W_router_ffn.grad.abs().sum() > 0
    # 中枢参数 (gate-style 用 W_t)
    assert layer.cw_attn.c.grad is not None
    assert layer.cw_ffn.W_t.grad is not None
    print("[PASS] test_composed_gradient_flow")
    return True


def test_composed_heterogeneous_dims():
    """6. ComposedFusionLayer 处理异构维度 (BERT 312, Llama 768, ViT 192)"""
    pools = _make_pools(n=3, D_m=192, D_shared=256)
    layer = ComposedFusionLayer(
        d_shared=256, d_ff=512, attn_pools=pools,
        modal_dims=[312, 768, 192], num_heads=4,
    )
    h_list = [torch.randn(1, 4, 312), torch.randn(1, 4, 768), torch.randn(1, 4, 192)]
    y = layer(h_list)
    assert y.shape == (1, 4, 256)
    print("[PASS] test_composed_heterogeneous_dims")
    return True


def test_composed_overrides_c1_alpha():
    """7. ComposedFusionLayer 支持 c1_alpha (gate-style 调制强度) 覆盖"""
    from research._primitives.central_mechanism import GateStyleWorkspace
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
        c1_alpha=0.3,
    )
    assert isinstance(layer.cw_attn, GateStyleWorkspace)
    assert layer.cw_attn.alpha == 0.3, f"alpha 应为 0.3, 实测 {layer.cw_attn.alpha}"
    assert layer.cw_ffn.alpha == 0.3
    print("[PASS] test_composed_overrides_c1_alpha")
    return True


def test_composed_step_decreases_loss():
    """8. ComposedFusionLayer 训练可降低 loss (V8Trainer)"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    from research.routing_evolution.v8_central import V8Trainer

    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = ComposedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools,
        modal_dims=[16, 16, 16],
    )
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
    # 后 5 步均值应小于前 5 步均值
    avg_first = sum(losses[:5]) / 5
    avg_last = sum(losses[-5:]) / 5
    assert avg_last < avg_first, f"训练无效: {avg_first:.4f} → {avg_last:.4f}"
    print(f"[PASS] test_composed_step_decreases_loss ({avg_first:.4f} → {avg_last:.4f})")
    return True


TESTS = [
    test_composed_returns_aligned_fusion,
    test_composed_uses_gate_central,
    test_composed_uses_attn_aligner,
    test_composed_forward_shape,
    test_composed_gradient_flow,
    test_composed_heterogeneous_dims,
    test_composed_overrides_c1_alpha,
    test_composed_step_decreases_loss,
]


if __name__ == "__main__":
    print("=== V12.0 ComposedFusionLayer — 单元测试 (8 项) ===")
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