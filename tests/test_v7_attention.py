"""V7.0 Attn + FFN 双路由 — 单元测试 (8 项)。

不依赖 transformers, 用随机张量构造 AttnPool。
每项测试独立 [PASS] / [FAIL] 标记, 与 v4/v5/v6 测试风格一致。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.routing_evolution.v7_attention import (
    AttnPool, CrossArchAttnFFNFusionLayer, V7Trainer, build_v7_param_groups,
    LR_ROUTER, LR_ADAPTER, LR_ALPHA,
)


def _build_layer():
    """构造一个最小 v7 layer (3 attn + 3 ffn, 同架构)"""
    D_m, D_shared = 32, 64
    H = 4
    pools = []
    for s in range(3):
        torch.manual_seed(s)
        w = torch.randn(D_m, D_m) * 0.04
        pools.append(AttnPool(D_m, D_shared, H, w.clone(), w.clone(), w.clone(), torch.eye(D_m)))
    layer = CrossArchAttnFFNFusionLayer(
        d_shared=D_shared, d_ff=4 * D_shared,
        attn_pools=pools, modal_dims=[D_m, D_m, D_m],
        top_k_attn=1, top_k_ffn=1,
    )
    return layer, D_shared, D_m


def test_attn_pool_output_shape():
    """1. AttnPool 输入 [B,S,D_shared] 输出同形状"""
    D_m, D_shared, H = 16, 32, 4
    w = torch.randn(D_m, D_m) * 0.04
    ap = AttnPool(D_m, D_shared, H, w.clone(), w.clone(), w.clone(), torch.eye(D_m))
    x = torch.randn(2, 8, D_shared)
    y = ap(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    print("[PASS] test_attn_pool_output_shape")
    return True


def test_dual_router_independent():
    """2. W_router_attn 与 W_router_ffn 梯度独立"""
    layer, D_shared, D_m = _build_layer()
    x = torch.randn(2, 8, D_shared)
    t = torch.zeros(2, 8, D_shared)
    t[:, :, :D_shared//3] = 1.0
    loss = F.mse_loss(layer(x), t)
    loss.backward()
    grad_a = layer.W_router_attn.grad
    grad_f = layer.W_router_ffn.grad
    assert grad_a is not None, "W_router_attn 没有梯度"
    assert grad_f is not None, "W_router_ffn 没有梯度"
    assert grad_a.abs().sum() > 0 and grad_f.abs().sum() > 0, \
        "至少一个 router 梯度全为 0"
    # 独立性: 两者的梯度形状相同但应明显不同 (非全等)
    assert grad_a.shape == grad_f.shape
    assert not torch.allclose(grad_a, grad_f), "两 router 梯度应独立"
    print("[PASS] test_dual_router_independent")
    return True


def test_attn_expert_gradient_flow():
    """3. attn 通路 α_attn 梯度可达"""
    layer, D_shared, D_m = _build_layer()
    x = torch.randn(1, 8, D_shared)
    t = torch.zeros(1, 8, D_shared); t[:, :, :D_shared//3] = 1.0
    y = layer(x)
    y.sum().backward()
    for m in range(layer.num_experts_attn):
        assert layer.alphas_attn[m].grad is not None, f"alphas_attn[{m}] 没有梯度"
        assert layer.attn_pools[m].W_o.grad is not None, f"attn_pools[{m}].W_o 没有梯度"
    print("[PASS] test_attn_expert_gradient_flow")
    return True


def test_ffn_expert_gradient_flow():
    """4. ffn 通路 α_ffn 梯度可达"""
    layer, D_shared, D_m = _build_layer()
    x = torch.randn(1, 8, D_shared)
    t = torch.zeros(1, 8, D_shared); t[:, :, :D_shared//3] = 1.0
    y = layer(x)
    y.sum().backward()
    for m in range(layer.num_experts_ffn):
        assert layer.alphas[m].grad is not None, f"alphas[{m}] 没有梯度"
        assert layer.gammas[m].grad is not None, f"gammas[{m}] 没有梯度"
        assert layer.betas[m].grad is not None, f"betas[{m}] 没有梯度"
    print("[PASS] test_ffn_expert_gradient_flow")
    return True


def test_per_expert_param_groups():
    """5. build_v7_param_groups 返回的 group 数 = M_attn + M_ffn*2 + (1 or 2 routers)"""
    layer, D_shared, D_m = _build_layer()
    # phase=2: 2 router + M_attn + M_ffn*2 (adapter + alpha 各一组)
    groups2 = build_v7_param_groups(layer, phase=2)
    expected2 = 2 + 3 + 3 * 2  # = 11
    assert len(groups2) == expected2, f"phase=2 groups={len(groups2)}, 应为 {expected2}"
    # phase=1: 0 router + M_attn + M_ffn*2
    groups1 = build_v7_param_groups(layer, phase=1)
    expected1 = 0 + 3 + 3 * 2  # = 9
    assert len(groups1) == expected1, f"phase=1 groups={len(groups1)}, 应为 {expected1}"
    # 每组都有 'expert' 标识
    for g in groups2:
        assert "expert" in g, "缺少 expert 标识"
        assert "lr" in g
    print("[PASS] test_per_expert_param_groups")
    return True


def test_p_m_orthogonal_init():
    """6. P_m 正交初始化: P_m @ P_m^T ≈ I"""
    layer, D_shared, D_m = _build_layer()
    for m, p in enumerate(layer.P_m):
        # P_m: D_shared × D_m, 当 D_shared > D_m 时正交初始化产生 D_m × D_m 的正交矩阵
        # 检查 P_m^T @ P_m ≈ I (D_m)
        gram = p.t() @ p
        eye = torch.eye(D_m)
        err = (gram - eye).abs().max().item()
        assert err < 1e-5, f"P_m[{m}] 不正交, max_err={err}"
        # 冻结
        assert not p.requires_grad, f"P_m[{m}] 应冻结"
    print("[PASS] test_p_m_orthogonal_init")
    return True


def test_dual_route_phase1_router_freeze():
    """7. phase=1 时两个 router 均冻结, 适配器可训"""
    layer, D_shared, D_m = _build_layer()
    trainer = V7Trainer(layer, lr=1e-2, phase=1)
    assert not layer.W_router_attn.requires_grad, "phase=1 应冻结 W_router_attn"
    assert not layer.W_router_ffn.requires_grad, "phase=1 应冻结 W_router_ffn"
    # attn W_o 与 ffn γ/β/α 应可训
    assert layer.attn_pools[0].W_o.requires_grad, "W_o 应可训"
    assert layer.gammas[0].requires_grad, "γ 应可训"
    # 切到 phase=2
    trainer.begin_phase(2)
    assert layer.W_router_attn.requires_grad, "phase=2 应解冻"
    assert layer.W_router_ffn.requires_grad, "phase=2 应解冻"
    print("[PASS] test_dual_route_phase1_router_freeze")
    return True


def test_attn_only_path():
    """8. 仅 attn 通路时输出 != 输入 (残差不为零)"""
    layer, D_shared, D_m = _build_layer()
    x = torch.randn(1, 8, D_shared)
    attn_sum, ahat = layer._attn_path(x)
    y_attn_only = x + attn_sum
    assert not torch.allclose(y_attn_only, x), "仅 attn 通路应贡献非零残差"
    # α̂ 应为 0/1 (K=1 STE)
    assert ahat.shape == (1, 8, layer.num_experts_attn)
    # K=1 时每行恰好一个 1
    ones_per_token = (ahat > 0).sum(dim=-1)
    assert (ones_per_token == 1).all(), "K=1 时每 token 应恰好选 1 个 attn 专家"
    print("[PASS] test_attn_only_path")
    return True


TESTS = [
    test_attn_pool_output_shape,
    test_dual_router_independent,
    test_attn_expert_gradient_flow,
    test_ffn_expert_gradient_flow,
    test_per_expert_param_groups,
    test_p_m_orthogonal_init,
    test_dual_route_phase1_router_freeze,
    test_attn_only_path,
]


if __name__ == "__main__":
    print("=== V7.0 Attn + FFN 双路由 — 单元测试 (8 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    sys.exit(0 if passed == len(TESTS) else 1)
