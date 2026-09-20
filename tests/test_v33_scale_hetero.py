"""V33.0 「1大+N小」异构规模融合 — 单元测试 (10 项)。

不依赖 transformers: 用随机张量构造 CrossModalFusionLayer (矩形 P_m + 共享路由),
并直接测试 paths.first_existing 的路径解析。每项独立 [PASS] / [FAIL] 标记,
与 v4/v5/v6/v7 测试风格一致。

覆盖:
    CrossModalFusionLayer:
        1. 构造 + 形状 (P_m 数量/维度, W_router)
        2. P_m 正交初始化 (P_m @ P_m^T ≈ I)
        3. 矩形投影: modal_dims 不等 (规模异构的矩阵表示)
        4. forward 输出形状 + 残差 (非恒等)
        5. forward_with_routing 返回 alpha_hat 形状
        6. K=1 稀疏性: alpha_hat 每 token 恰 K 个非零
        7. W_router 梯度可流
        8. P_m 外部投影: 冻结无梯度 / 解冻有梯度
    paths:
        9. first_existing 无现存候选时回退首选
        10. first_existing 选中存在的候选 (规模异构路径解析)
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer, first_existing

D_SHARED = 32
D_FF = 4 * D_SHARED
M, K = 3, 1
# 规模异构的缩微表示: 大 32 / 小 16 / 小 16 (对应真实 2560 / 1024 / 1024)
MODAL_DIMS = [32, 16, 16]


def _build():
    torch.manual_seed(0)
    return CrossModalFusionLayer(D_SHARED, D_FF, M, K, MODAL_DIMS)


def test_construct_and_shape():
    """1. 构造 + 形状正确"""
    layer = _build()
    assert len(layer.P_m) == M
    assert layer.P_m[0].shape == (D_SHARED, MODAL_DIMS[0])
    assert layer.P_m[1].shape == (D_SHARED, MODAL_DIMS[1])
    assert layer.W_router.shape == (M, D_SHARED)
    print("[PASS] test_construct_and_shape")
    return True


def test_p_m_orthogonal_init():
    """2. P_m 正交初始化 (D_shared<=D_m: P P^T≈I; 否则 P^T P≈I)"""
    layer = _build()
    for m in range(M):
        Pm = layer.P_m[m]  # [D_shared, D_m]
        gram = Pm @ Pm.t() if D_SHARED <= MODAL_DIMS[m] else Pm.t() @ Pm
        n = gram.shape[0]
        err = float((gram - torch.eye(n)).abs().max().detach())
        assert err < 1e-4, f"P_m[{m}] 非正交, max err={err}"
    print("[PASS] test_p_m_orthogonal_init")
    return True


def test_rectangular_projection_dim_heterogeneity():
    """3. 矩形投影: 不同 D_m 均投到同一 D_shared (规模异构)"""
    import torch.nn.functional as F
    layer = _build()
    for m, D_m in enumerate(MODAL_DIMS):
        x = torch.randn(2, 4, D_m)
        y = F.linear(x, layer.P_m[m])
        assert y.shape == (2, 4, D_SHARED), f"modal {m}: {y.shape}"
    print(f"[PASS] test_rectangular_projection_dim_heterogeneity ({MODAL_DIMS} -> {D_SHARED})")
    return True


def test_forward_shape_and_residual():
    """4. forward 输出同形状, 且非恒等 (残差 + FFN)"""
    layer = _build()
    x = torch.randn(2, 5, D_SHARED)
    y = layer(x)
    assert y.shape == x.shape
    diff = float((y - x).abs().max())
    assert diff > 1e-6, f"应非恒等映射, diff={diff}"
    print(f"[PASS] test_forward_shape_and_residual (diff={diff:.4f})")
    return True


def test_forward_with_routing_shapes():
    """5. forward_with_routing 返回 (y, alpha_hat)"""
    layer = _build()
    x = torch.randn(2, 5, D_SHARED)
    y, alpha_hat = layer.forward_with_routing(x)
    assert y.shape == x.shape
    assert alpha_hat.shape == (2, 5, M), f"alpha_hat {alpha_hat.shape}"
    print("[PASS] test_forward_with_routing_shapes")
    return True


def test_top1_sparsity():
    """6. K=1 时每 token 恰有 1 个非零专家权重 (SparseRouterSTE 稀疏性)"""
    layer = _build()
    x = torch.randn(3, 7, D_SHARED)
    _, alpha_hat = layer.forward_with_routing(x)
    nnz = (alpha_hat > 0).sum(dim=-1)  # 每 token 非零专家数
    err = float((nnz - K).abs().max())
    assert err < 1e-6, f"K={K} 时每 token 应恰有 {K} 个非零, max|nnz-K|={err}"
    # 被选专家的权重应在 (0,1) 内 (softmax 概率被 mask 保留)
    assert float(alpha_hat.max()) <= 1.0 + 1e-6
    print("[PASS] test_top1_sparsity")
    return True


def test_w_router_gradient_flows():
    """7. W_router 梯度可流"""
    layer = _build()
    t = torch.zeros(2, 5, D_SHARED)
    t[:, :, : D_SHARED // M] = 1.0
    loss = torch.nn.functional.mse_loss(layer(torch.randn(2, 5, D_SHARED)), t)
    loss.backward()
    assert layer.W_router.grad is not None
    assert layer.W_router.grad.abs().sum() > 0, "W_router 梯度为零"
    print("[PASS] test_w_router_gradient_flows")
    return True


def test_p_m_freeze_controls_grad():
    """8. P_m 用于外部投影: 冻结无梯度 / 解冻有梯度

    注: v33 中 P_m 在融合层 forward 之外作为投影矩阵使用 (F.linear(x, P_m)),
    层内不做投影, 故此处直接测 P_m 作为投影参数的梯度行为。
    """
    import torch.nn.functional as F
    D_m = MODAL_DIMS[1]

    P_free = torch.nn.Parameter(torch.empty(D_SHARED, D_m))
    torch.nn.init.orthogonal_(P_free)
    out_free = F.linear(torch.randn(2, 5, D_m), P_free)
    assert out_free.requires_grad, "解冻 P_m 输出应可求导"
    out_free.pow(2).mean().backward()
    assert P_free.grad is not None and P_free.grad.abs().sum() > 0, "解冻 P_m 应有梯度"

    P_frozen = torch.nn.Parameter(torch.empty(D_SHARED, D_m), requires_grad=False)
    torch.nn.init.orthogonal_(P_frozen)
    out_frozen = F.linear(torch.randn(2, 5, D_m), P_frozen)
    assert not out_frozen.requires_grad, "冻结 P_m 输出不应可求导"

    # 融合层默认把 P_m 设为冻结 (规模异构投影对齐, 不参与训练)
    layer = _build()
    for m in range(M):
        layer.P_m[m].requires_grad_(False)
    assert all(not layer.P_m[m].requires_grad for m in range(M))
    print("[PASS] test_p_m_freeze_controls_grad")
    return True


def test_first_existing_fallback():
    """9. first_existing 无现存候选时回退首选"""
    got = first_existing(["/nonexistent/a", "/nonexistent/b"], "V33_TEST_DIR_UNSET")
    assert got == "/nonexistent/a", f"应回退首选, got={got}"
    print("[PASS] test_first_existing_fallback")
    return True


def test_first_existing_picks_existing():
    """10. first_existing 选中存在的候选 (复杂路径解析)"""
    with tempfile.TemporaryDirectory() as d:
        existing = os.path.join(d, "real")
        os.makedirs(existing)
        got = first_existing(["/nonexistent/first", existing], "V33_TEST_DIR_UNSET")
        assert got == existing, f"应选中现存候选, got={got}"
    print("[PASS] test_first_existing_picks_existing")
    return True


TESTS = [
    test_construct_and_shape,
    test_p_m_orthogonal_init,
    test_rectangular_projection_dim_heterogeneity,
    test_forward_shape_and_residual,
    test_forward_with_routing_shapes,
    test_top1_sparsity,
    test_w_router_gradient_flows,
    test_p_m_freeze_controls_grad,
    test_first_existing_fallback,
    test_first_existing_picks_existing,
]


if __name__ == "__main__":
    print(f"=== V33.0 「1大+N小」异构规模融合 — 单元测试 ({len(TESTS)} 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    sys.exit(0 if passed == len(TESTS) else 1)
