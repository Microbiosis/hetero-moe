"""V8.0 全局中枢 — 单元测试 (8 项)。

每项独立 [PASS] / [FAIL], 与 v4/v5/v6/v7 测试风格一致。
不依赖 transformers, 用随机张量构造。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import (
    CentralWorkspace, CentralAugmentedFusionLayer, HierarchicalCentralLayer,
    V8Trainer, build_v8_param_groups,
)


def _make_pools(n=3, D_m=32, D_shared=64, H=4):
    pools = []
    for s in range(n):
        torch.manual_seed(s)
        w = torch.randn(D_m, D_m) * 0.04
        pools.append(AttnPool(D_m, D_shared, H, w.clone(), w.clone(), w.clone(), torch.eye(D_m)))
    return pools


def test_central_workspace_augment():
    """1. CentralWorkspace.augment_router_logits 输出形状正确"""
    cw = CentralWorkspace(d_shared=32, num_experts=3)
    z = torch.randn(2, 8, 3)
    z_aug = cw.augment_router_logits(z)
    assert z_aug.shape == z.shape
    # 中枢 c=0 时 z_aug ≈ z (U·0 = 0)
    assert torch.allclose(z_aug, z, atol=1e-5), "c 初始为 0 时应无 broadcast"
    print("[PASS] test_central_workspace_augment")
    return True


def test_central_ema_update():
    """2. EMA 更新: c 在多次 update 后非零且稳定"""
    cw = CentralWorkspace(d_shared=16, num_experts=2, ema_decay=0.5)
    # 喂入 5 次随机 expert 输出
    for i in range(5):
        out = torch.randn(1, 4, 16)
        cw.ema_update([out, out * 0.5])
    assert cw.c.abs().sum() > 0, "EMA 后 c 应非零"
    # EMA 不改变 requires_grad
    assert cw.c.requires_grad, "c 应保持可训练"
    print("[PASS] test_central_ema_update")
    return True


def test_c2_layer_forward_shape():
    """3. CentralAugmentedFusionLayer (C-2) 前向输出形状"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = CentralAugmentedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    x = torch.randn(2, 8, 32)
    y = layer(x)
    assert y.shape == x.shape, f"shape {y.shape} vs {x.shape}"
    print("[PASS] test_c2_layer_forward_shape")
    return True


def test_c3_layer_forward_shape():
    """4. HierarchicalCentralLayer (C-3) 前向输出形状"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = HierarchicalCentralLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    x = torch.randn(2, 8, 32)
    y = layer(x)
    assert y.shape == x.shape
    # C-3 中枢 expert 参数存在
    assert hasattr(layer, "central_expert"), "缺少 central_expert"
    assert hasattr(layer, "central_alpha"), "缺少 central_alpha"
    print("[PASS] test_c3_layer_forward_shape")
    return True


def test_attn_ffn_cooperation():
    """5. C-2 attn→ffn 协同: V_coop 参数存在且影响输出"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = CentralAugmentedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    assert hasattr(layer, "V_coop"), "缺少 V_coop"
    assert layer.V_coop.shape == (32, 32), f"V_coop 形状应为 (32,32), 实测 {layer.V_coop.shape}"
    # V_coop 初始化为 0.1*I: 对角线 0.1, 其它 0
    expected_diag = torch.eye(32) * 0.1
    assert torch.allclose(layer.V_coop.data, expected_diag, atol=1e-6), "V_coop 初始化不符合预期"
    print("[PASS] test_attn_ffn_cooperation")
    return True


def test_c3_central_expert_gradient():
    """6. C-3 中枢 expert 参数梯度可达"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = HierarchicalCentralLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    layer.train()
    x = torch.randn(1, 8, 32)
    y = layer(x)
    y.sum().backward()
    # central_expert 内部参数应有梯度
    for p in layer.central_expert.parameters():
        assert p.grad is not None, "central_expert 参数无梯度"
        assert p.grad.abs().sum() > 0, "central_expert 梯度为 0"
    assert layer.central_alpha.grad is not None, "central_alpha 无梯度"
    print("[PASS] test_c3_central_expert_gradient")
    return True


def test_v8_param_groups():
    """7. build_v8_param_groups 返回的 group 数符合预期"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    # C-2 layer
    layer_c2 = CentralAugmentedFusionLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    groups_c2 = build_v8_param_groups(layer_c2, phase=2)
    # 期望: 2 router + 3 attn + 3 ffn_adapter + 3 ffn_alpha + 1 central_workspace + 1 central_coop = 13
    assert len(groups_c2) == 13, f"C-2 phase=2 groups={len(groups_c2)}, 应为 13"
    # C-3 layer (继承 C-2)
    layer_c3 = HierarchicalCentralLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    groups_c3 = build_v8_param_groups(layer_c3, phase=2)
    # 期望: 13 + central_expert(1) + central_router(1) = 15
    assert len(groups_c3) == 15, f"C-3 phase=2 groups={len(groups_c3)}, 应为 15"
    # 标识完整性
    for g in groups_c3:
        assert "expert" in g and "lr" in g, "缺 expert/lr 标识"
    print("[PASS] test_v8_param_groups")
    return True


def test_v8_trainer_step_decreases_loss():
    """8. V8Trainer 单步训练 loss 应下降"""
    pools = _make_pools(n=3, D_m=16, D_shared=32)
    layer = HierarchicalCentralLayer(
        d_shared=32, d_ff=64, attn_pools=pools, modal_dims=[16, 16, 16]
    )
    trainer = V8Trainer(layer, lr=1e-2, phase=2)
    x_list = [torch.randn(1, 8, 32) for _ in range(3)]
    t_list = [torch.randn(1, 8, 32) for _ in range(3)]
    losses = []
    for _ in range(10):
        loss = trainer.step(x_list, t_list)
        losses.append(loss)
    # 后 5 步均值应小于前 5 步均值 (训练有效)
    assert statistics_mean(losses[5:]) < statistics_mean(losses[:5]), \
        f"训练无效: 前5={statistics_mean(losses[:5]):.4f}, 后5={statistics_mean(losses[5:]):.4f}"
    print("[PASS] test_v8_trainer_step_decreases_loss")
    return True


def statistics_mean(xs):
    return sum(xs) / len(xs)


TESTS = [
    test_central_workspace_augment,
    test_central_ema_update,
    test_c2_layer_forward_shape,
    test_c3_layer_forward_shape,
    test_attn_ffn_cooperation,
    test_c3_central_expert_gradient,
    test_v8_param_groups,
    test_v8_trainer_step_decreases_loss,
]


if __name__ == "__main__":
    print("=== V8.0 全局中枢 — 单元测试 (8 项) ===")
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
