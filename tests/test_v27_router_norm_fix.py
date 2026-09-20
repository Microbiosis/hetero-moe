"""V27.0 — router-norm / adaptive-ema held-out 修正补丁 单元测试 (12 项).

测试覆盖:
    1.  v27 公开 API import OK
    2.  apply_l2_to_c 修改 layer._l2_c_params
    3.  get_l2_to_c_loss 返回 tensor 且依赖 c 值
    4.  apply_l2_to_c 不影响 gate 模式 (无 U 但有 c)
    5.  router_dropout 挂载 _dropout 属性到 cb_attn / cb_ffn
    6.  extend_warmup_steps 修改 adaptive_warmup_steps
    7.  noise_inject_c 挂载 _c_noise_std 到 cb_attn / cb_ffn
    8.  apply_all_patches 同时应用 4 个补丁
    9.  RegularizedTheoryLayer 4 mode 都能 forward
    10. RegularizedTheoryLayer 应用 l2 后, compute_total_loss 含 L2 项
    11. l2 项对 c 梯度有贡献
    12. 端到端冒烟: 1 seed × 1 mode × 1 patch 跑通
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn.functional as F

from research.corpus_scale.v27_router_norm_fix import (
    apply_l2_to_c, get_l2_to_c_loss,
    router_dropout, extend_warmup_steps, noise_inject_c,
    apply_all_patches, DEFAULT_PATCH_CONFIGS, PATCH_NAMES,
    RegularizedTheoryLayer, compute_total_loss,
)


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _make_stub_attn_pools(d_shared=256):
    """构造 stub AttnPool 列表."""
    from research._primitives.attention import AttnPool
    pools = []
    for D_m in [312, 768, 192]:
        w_q = torch.randn(D_m, D_m) * 0.02
        w_k = torch.randn(D_m, D_m) * 0.02
        w_v = torch.randn(D_m, D_m) * 0.02
        w_o_native = torch.eye(D_m)
        ap = AttnPool(D_m=D_m, D_shared=d_shared, num_heads=4,
                       w_q=w_q, w_k=w_k, w_v=w_v, w_o_native=w_o_native)
        pools.append(ap)
    return pools


class _StubTextTok:
    pad_token_id = 0
    def __call__(self, sentences=None, **kw):
        if sentences is None:
            sentences = ["stub"]
        if not isinstance(sentences, list):
            sentences = [sentences]
        return {"input_ids": torch.randint(0, 100, (len(sentences), 16))}


class _StubVitProc:
    def __call__(self, images=None, **kw):
        return {"pixel_values": torch.randn(1, 3, 224, 224)}


class _StubModel:
    def __init__(self, D_m):
        self.D_m = D_m
    def __call__(self, input_ids_or_pixels):
        if input_ids_or_pixels.dim() == 2:
            # text input_ids → [B, S, D_m]
            return type("O", (), {"last_hidden_state": torch.randn(input_ids_or_pixels.shape[0], 16, self.D_m) * 0.1})()
        else:
            # image pixel_values → [1, 197, D_m]
            return type("O", (), {"last_hidden_state": torch.randn(1, 197, self.D_m) * 0.1})()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_v27_imports():
    """1. v27 公开 API 全部 import OK."""
    assert callable(apply_l2_to_c)
    assert callable(get_l2_to_c_loss)
    assert callable(router_dropout)
    assert callable(extend_warmup_steps)
    assert callable(noise_inject_c)
    assert callable(apply_all_patches)
    assert isinstance(DEFAULT_PATCH_CONFIGS, dict)
    assert isinstance(PATCH_NAMES, list)
    assert RegularizedTheoryLayer is not None
    assert callable(compute_total_loss)
    assert len(PATCH_NAMES) == 5
    print(f"[PASS] test_v27_imports (5 patches: {PATCH_NAMES})")
    return True


def test_apply_l2_to_c_modifies_layer():
    """2. apply_l2_to_c 修改 layer._l2_c_params 和 _l2_lambda_c."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    apply_l2_to_c(layer, lambda_c=2e-4)
    assert hasattr(layer, "_l2_c_params")
    assert len(layer._l2_c_params) == 2  # cb_attn.c + cb_ffn.c
    assert layer._l2_lambda_c == 2e-4
    # c 参数应是真实 parameter (not None)
    assert layer._l2_c_params[0] is layer.cb_attn.c
    assert layer._l2_c_params[1] is layer.cb_ffn.c
    print(f"[PASS] test_apply_l2_to_c_modifies_layer (lambda=2e-4, 2 c params)")
    return True


def test_get_l2_to_c_loss_returns_tensor():
    """3. get_l2_to_c_loss 返回 tensor 且依赖 c 值."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    apply_l2_to_c(layer, lambda_c=1e-4)
    # 初始 c=0, L2 loss = 0
    loss_zero = get_l2_to_c_loss(layer)
    assert loss_zero.item() == 0.0, f"c=0 时 L2 应=0, 实测 {loss_zero.item()}"
    # 让 c 非零
    with torch.no_grad():
        layer.cb_attn.c.data = torch.ones(256) * 0.1
        layer.cb_ffn.c.data = torch.ones(256) * 0.2
    loss_nonzero = get_l2_to_c_loss(layer)
    expected = (0.1**2 + 0.2**2) * 256
    assert abs(loss_nonzero.item() - expected) < 1e-3, \
        f"L2 应 = {expected}, 实测 {loss_nonzero.item()}"
    print(f"[PASS] test_get_l2_to_c_loss_returns_tensor (c=0→0, c=0.1/0.2→{loss_nonzero.item():.2f})")
    return True


def test_apply_l2_to_c_works_for_all_modes():
    """4. apply_l2_to_c 对所有 4 mode 都生效 (因为都有 c 参数)."""
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        pools = _make_stub_attn_pools()
        layer = RegularizedTheoryLayer(
            d_shared=256, num_experts=3,
            modal_dims=[312, 768, 192], attn_pools=pools, mode=mode,
        )
        apply_l2_to_c(layer, lambda_c=1e-4)
        assert hasattr(layer, "_l2_c_params"), f"{mode}: 应挂载 _l2_c_params"
        assert len(layer._l2_c_params) == 2
    print(f"[PASS] test_apply_l2_to_c_works_for_all_modes (4 modes)")
    return True


def test_router_dropout_attaches_dropout_layer():
    """5. router_dropout 挂载 _dropout 属性到 cb_attn / cb_ffn."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    router_dropout(layer, p=0.2)
    assert hasattr(layer.cb_attn, "_dropout")
    assert hasattr(layer.cb_ffn, "_dropout")
    assert isinstance(layer.cb_attn._dropout, torch.nn.Dropout)
    assert layer.cb_attn._dropout.p == 0.2
    assert layer._router_dropout_p == 0.2
    print(f"[PASS] test_router_dropout_attaches_dropout_layer (p=0.2)")
    return True


def test_extend_warmup_steps_modifies_adaptive_ema():
    """6. extend_warmup_steps 修改 adaptive_warmup_steps (仅 adaptive-ema)."""
    # adaptive-ema 模式
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="adaptive-ema",
    )
    # 默认 warmup=10
    assert layer.cb_attn.adaptive_warmup_steps == 10
    extend_warmup_steps(layer, new_warmup=100)
    assert layer.cb_attn.adaptive_warmup_steps == 100
    assert layer.cb_ffn.adaptive_warmup_steps == 100
    # router-norm 模式: warmup 不应生效 (mode 不匹配)
    layer2 = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    # router-norm 没有 adaptive_warmup_steps 属性, extend_warmup_steps 应跳过
    extend_warmup_steps(layer2, new_warmup=100)  # 不抛异常即 OK
    print(f"[PASS] test_extend_warmup_steps_modifies_adaptive_ema (10 → 100)")
    return True


def test_noise_inject_c_attaches_std():
    """7. noise_inject_c 挂载 _c_noise_std 到 cb_attn / cb_ffn."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    noise_inject_c(layer, std=0.02)
    assert layer.cb_attn._c_noise_std == 0.02
    assert layer.cb_ffn._c_noise_std == 0.02
    assert layer._c_noise_std == 0.02
    print(f"[PASS] test_noise_inject_c_attaches_std (std=0.02)")
    return True


def test_apply_all_patches_combines_all_4():
    """8. apply_all_patches 同时应用 4 个补丁."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="adaptive-ema",
    )
    apply_all_patches(layer, {
        "apply_l2_to_c": 1e-4,
        "router_dropout": 0.1,
        "extend_warmup_steps": 50,
        "noise_inject_c": 0.01,
    })
    # 验证 4 个补丁都生效
    assert hasattr(layer, "_l2_c_params"), "L2 未挂载"
    assert hasattr(layer.cb_attn, "_dropout"), "Dropout 未挂载"
    assert layer.cb_attn.adaptive_warmup_steps == 50, "Warmup 未修改"
    assert layer.cb_attn._c_noise_std == 0.01, "Noise std 未挂载"
    print(f"[PASS] test_apply_all_patches_combines_all_4 (4 patches 全部生效)")
    return True


def test_regularized_theory_layer_all_modes_forward():
    """9. RegularizedTheoryLayer 4 mode 都能 forward."""
    pools = _make_stub_attn_pools()
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        layer = RegularizedTheoryLayer(
            d_shared=256, num_experts=3,
            modal_dims=[312, 768, 192], attn_pools=pools, mode=mode,
        )
        h_text = torch.randn(6, 16, 312)
        h_code = torch.randn(6, 16, 768)
        h_img = torch.randn(6, 16, 192)
        y = layer([h_text, h_code, h_img])
        assert y.shape == (6, 16, 256), f"{mode}: shape 应 = (6, 16, 256), 实测 {y.shape}"
    print(f"[PASS] test_regularized_theory_layer_all_modes_forward")
    return True


def test_compute_total_loss_with_l2():
    """10. compute_total_loss 含 L2 项 (应用 apply_l2_to_c 后)."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
        patch_config={"apply_l2_to_c": 1e-4},
    )
    h_text = torch.randn(6, 16, 312)
    h_code = torch.randn(6, 16, 768)
    h_img = torch.randn(6, 16, 192)
    targets = [torch.zeros(6, 16, 256) for _ in range(3)]
    h_by_modal = [h_text, h_code, h_img]
    t_by_modal = targets
    loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=3)
    # loss 应该是 mse + lambda * l2 项
    assert loss.requires_grad, "loss 应对 c 有梯度"
    assert loss.item() > 0, "loss 应 > 0"
    print(f"[PASS] test_compute_total_loss_with_l2 (loss={loss.item():.6f})")
    return True


def test_l2_term_provides_gradient_to_c():
    """11. l2 项对 c 梯度有贡献."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
        patch_config={"apply_l2_to_c": 1e-3},
    )
    h_text = torch.randn(6, 16, 312)
    h_code = torch.randn(6, 16, 768)
    h_img = torch.randn(6, 16, 192)
    h_by_modal = [h_text, h_code, h_img]
    t_by_modal = [torch.zeros(6, 16, 256) for _ in range(3)]
    loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=3)
    loss.backward()
    # c 应有非零梯度 (来自 L2 项, 因 c 初始为 0 梯度应为 0, 让 c 非零后再测)
    with torch.no_grad():
        layer.cb_attn.c.data = torch.ones(256) * 0.5
        layer.cb_ffn.c.data = torch.ones(256) * 0.5
    # 重新计算
    opt = torch.optim.SGD(layer.parameters(), lr=0)
    opt.zero_grad()
    loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=3)
    loss.backward()
    # c 应有非零梯度
    assert layer.cb_attn.c.grad is not None
    assert layer.cb_attn.c.grad.abs().sum() > 0, \
        f"c.grad 应 > 0 (来自 L2 项), 实测 {layer.cb_attn.c.grad.abs().sum()}"
    print(f"[PASS] test_l2_term_provides_gradient_to_c (cb_attn.c.grad.abs.sum()={layer.cb_attn.c.grad.abs().sum():.4f})")
    return True


def test_end_to_end_smoke_with_patch():
    """12. 端到端冒烟: 1 seed × 1 mode × 1 patch 跑通."""
    pools = _make_stub_attn_pools()
    layer = RegularizedTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
        patch_config={"apply_l2_to_c": 1e-4, "router_dropout": 0.1,
                      "extend_warmup_steps": 50, "noise_inject_c": 0.01},
    )
    h_text = torch.randn(6, 16, 312)
    h_code = torch.randn(6, 16, 768)
    h_img = torch.randn(6, 16, 192)
    h_by_modal = [h_text, h_code, h_img]
    t_by_modal = [torch.zeros(6, 16, 256) for _ in range(3)]
    layer.train()
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()
    # 评估
    layer.eval()
    with torch.no_grad():
        y = layer(h_by_modal)
        mse = sum(F.mse_loss(y, t_by_modal[c]) for c in range(3)) / 3
    assert math.isfinite(mse.item()), f"fuse 应有限, 实测 {mse.item()}"
    print(f"[PASS] test_end_to_end_smoke_with_patch (fuse={mse.item():.4f}, 4 patches 全开)")
    return True


TESTS = [
    test_v27_imports,
    test_apply_l2_to_c_modifies_layer,
    test_get_l2_to_c_loss_returns_tensor,
    test_apply_l2_to_c_works_for_all_modes,
    test_router_dropout_attaches_dropout_layer,
    test_extend_warmup_steps_modifies_adaptive_ema,
    test_noise_inject_c_attaches_std,
    test_apply_all_patches_combines_all_4,
    test_regularized_theory_layer_all_modes_forward,
    test_compute_total_loss_with_l2,
    test_l2_term_provides_gradient_to_c,
    test_end_to_end_smoke_with_patch,
]


if __name__ == "__main__":
    print("=== V27.0 router-norm / adaptive-ema held-out 修正 — 单元测试 (12 项) ===")
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
