"""V22.0 — CentralTheory (中枢理论分析) 单元测试 (12 项).

v22.0 完整实现新增 4 项测试覆盖:
    9.  freeze_u 函数式 patch
    10. freeze_c 函数式 patch
    11. set_u_init_scale 函数式 patch
    12. diagnose_theory_layer 诊断工具
    13. AdaptiveEMACentral warmup 行为
    14. ema_enable/disable 控制
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn.functional as F

from research._primitives.central_theory import (
    CentralTheory,
    gate_style_analysis,
    RouterNormCentral,
    AdaptiveEMACentral,
    freeze_u,
    freeze_c,
    set_u_init_scale,
    diagnose_theory_layer,
)


def test_gate_style_analysis_returns_all_5_hypotheses():
    """1. gate_style_analysis 返回 5 个假设的解释"""
    analysis = gate_style_analysis()
    expected = ["H1_U_train_disturbance", "H2_EMA_too_slow", "H3_too_few_steps",
                "H4_small_student_capacity", "H5_central_position"]
    for hyp in expected:
        assert hyp in analysis, f"missing {hyp}"
        assert len(analysis[hyp]) > 50, f"{hyp} explanation too short"
    print("[PASS] test_gate_style_analysis_returns_all_5_hypotheses")
    return True


def test_central_theory_broadcast_mode():
    """2. broadcast 模式: z += U @ c"""
    model = CentralTheory(d_shared=32, num_experts=3, mode="broadcast")
    z = torch.zeros(1, 4, 3)
    z_out = model.augment_router_logits(z)
    # c=0, U@0 = 0, 所以 z_out = z
    assert torch.allclose(z_out, z, atol=1e-3)
    # 让 c 非零
    with torch.no_grad():
        model.c.data = torch.ones(32) * 0.5
    expected_bias = model.U @ model.c
    z_out = model.augment_router_logits(z)
    assert torch.allclose(z_out[0, 0, :], expected_bias, atol=1e-4)
    print("[PASS] test_central_theory_broadcast_mode")
    return True


def test_central_theory_gate_mode_temperature():
    """3. gate 模式: z 除以温度 (1+α·tanh(W_t·c))"""
    torch.manual_seed(0)
    model = CentralTheory(d_shared=32, num_experts=3, mode="gate")
    z = torch.ones(1, 4, 3) * 2.0
    # c=0, W_t=0: t_scalar = 0, temperature = 1.0, z 不变
    z_out = model.augment_router_logits(z)
    assert torch.allclose(z_out, z, atol=1e-5), "c=0, W_t=0 时温度应=1"
    # 让 W_t 和 c 都非零: temperature 应 > 1, z 应被缩小
    with torch.no_grad():
        model.c.data = torch.ones(32) * 1.0
        model.W_t.data = torch.ones(32) * 1.0
    z_out = model.augment_router_logits(z)
    # t_scalar = 32, tanh(32) ≈ 1.0, temperature = 1 + 0.1*1.0 = 1.1
    assert (z_out < z).all(), "W_t+c 非零时温度>1, z 应被缩小"
    print("[PASS] test_central_theory_gate_mode_temperature")
    return True


def test_central_theory_router_norm_mode():
    """4. router-norm 模式: LayerNorm 路由 logits"""
    model = RouterNormCentral(d_shared=32, num_experts=3)
    z = torch.randn(2, 8, 3) * 5   # 大值测试
    z_out = model.augment_router_logits(z)
    # LayerNorm 应归一化每 token 的 3 个 expert logits
    # 检查每个 token 的输出 mean≈0, std≈1
    mean_per_token = z_out.mean(dim=-1)
    std_per_token = z_out.std(dim=-1, unbiased=False)
    assert torch.allclose(mean_per_token, torch.zeros_like(mean_per_token), atol=1e-5), \
        "router-norm 应让 mean=0"
    assert torch.allclose(std_per_token, torch.ones_like(std_per_token), atol=1e-5), \
        "router-norm 应让 std=1"
    print("[PASS] test_central_theory_router_norm_mode")
    return True


def test_central_theory_adaptive_ema_initial_alpha_zero():
    """5. adaptive-ema 初始: α=0 (修复后, 与 v22.0 之前 α=0.5 不同)"""
    model = AdaptiveEMACentral(d_shared=32, num_experts=3)
    assert torch.allclose(model.adaptive_alpha, torch.tensor(0.0)), \
        f"v22.0 修复后 α 应初始化为 0, 实测 {model.adaptive_alpha.item()}"
    # 调用 ema_update (用 in-place, 不验证 c 值, 只验证不报错)
    with torch.no_grad():
        outs = [torch.randn(1, 4, 32) for _ in range(3)]
        model.ema_update(outs)
    print(f"[PASS] test_central_theory_adaptive_ema_initial_alpha_zero (α={model.adaptive_alpha.item():.3f})")
    return True


def test_central_theory_all_modes_gradient_flow():
    """6. 4 种模式都能反向传播 (用 loss 让所有模式的关键参数收到梯度)."""
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        torch.manual_seed(0)
        model = CentralTheory(d_shared=32, num_experts=3, mode=mode)
        # 让 c 非零, 确保每个模式都有可训练参数被激活
        with torch.no_grad():
            model.c.data = torch.ones(32) * 0.3
            if mode == "gate":
                model.W_t.data = torch.ones(32) * 0.3
        z = torch.randn(1, 4, 3)
        z_out = model.augment_router_logits(z)
        target = torch.zeros_like(z_out)
        loss = F.mse_loss(z_out, target)
        loss.backward()
        # c 一定有梯度 (每个 mode 都直接用 c)
        assert model.c.grad is not None and model.c.grad.abs().sum() > 0, \
            f"{mode}: c.grad 异常"
        # U: broadcast/router-norm/adaptive-ema 用 U; gate 不直接用 U (温度路径)
        if mode in ("broadcast", "router-norm", "adaptive-ema"):
            assert model.U.grad is not None and model.U.grad.abs().sum() > 0, \
                f"{mode}: U.grad 异常"
        else:  # gate
            assert model.W_t.grad is not None and model.W_t.grad.abs().sum() > 0, \
                f"{mode}: W_t.grad 异常"
    print("[PASS] test_central_theory_all_modes_gradient_flow")
    return True


def test_central_theory_different_modes_produce_different_outputs():
    """7. 4 种 mode 在 c 非零时产生不同输出"""
    z = torch.randn(2, 4, 3)
    outs = {}
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        torch.manual_seed(42)
        model = CentralTheory(d_shared=32, num_experts=3, mode=mode)
        with torch.no_grad():
            model.c.data = torch.ones(32) * 0.3
            if mode == "gate":
                model.W_t.data = torch.ones(32) * 0.3
        outs[mode] = model.augment_router_logits(z)
    # tensor 不可 hash, 用 first-token-value tuple 作 sig
    sigs = [tuple(o[0, 0, :].tolist()) for o in outs.values()]
    unique = len(set(sigs))
    assert unique >= 3, f"4 mode 应至少 3 种输出, 实测 {unique}"
    print(f"[PASS] test_central_theory_different_modes_produce_different_outputs ({unique} unique)")
    return True


def test_adaptive_ema_warmup_uses_base_decay():
    """8. adaptive-ema warmup 期间用 base_decay=0.9 (新增, v22.0 完整实现).

    修复前: α=0.5 初始化 + c.norm()=0 → decay = sigmoid(0)=0.5 (远低于 broadcast 0.9)
    修复后: α=0 初始化 → warmup 期间完全用 base_decay=0.9 (与 broadcast 一致)
    """
    torch.manual_seed(0)
    model = AdaptiveEMACentral(d_shared=32, num_experts=3, adaptive_warmup_steps=5)
    model.train()
    # warmup 期间: decay = base_decay = 0.9 (不受 adaptive_alpha 影响)
    with torch.no_grad():
        outs = [torch.randn(1, 4, 32) for _ in range(3)]
        # 第 1 步: decay 应 = 0.9 (warmup)
        c_before = model.c.data.clone()
        model.ema_update(outs)
        c_after_warmup_1 = model.c.data.clone()
        # c_after = 0.9 * c_before + 0.1 * new_c (base_decay=0.9)
        stacked = torch.stack(outs, dim=0)
        new_c_expected = stacked.mean(dim=(0, 1, 2))
        c_expected = 0.9 * c_before + 0.1 * new_c_expected
        assert torch.allclose(c_after_warmup_1, c_expected, atol=1e-5), \
            f"warmup step 应使用 base_decay=0.9"
    print(f"[PASS] test_adaptive_ema_warmup_uses_base_decay (warmup_steps={model.adaptive_warmup_steps})")
    return True


# ---------------------------------------------------------------------------
# v22.0 完整实现 — 新增 4 项测试 (覆盖 freeze_u / freeze_c / set_u / diagnose / ema 控制)
# ---------------------------------------------------------------------------


def test_freeze_u_patches_router_norm_and_broadcast():
    """9. freeze_u() 函数式 patch (修正 A 移植).

    对 RouterNormCentral 和 AdaptiveEMACentral 生效;
    对 GateStyleWorkspace 抛 TypeError (符合 v9_gate_central.freeze_broadcast_matrix 语义).
    """
    # RouterNormCentral: U 可冻结
    rn = RouterNormCentral(d_shared=32, num_experts=3)
    assert rn.U.requires_grad is True, "freeze 前 U 应可训练"
    freeze_u(rn)
    assert rn.U.requires_grad is False, "freeze 后 U 应冻结"
    # AdaptiveEMACentral: U 可冻结
    ae = AdaptiveEMACentral(d_shared=32, num_experts=3)
    freeze_u(ae)
    assert ae.U.requires_grad is False
    # GateStyleWorkspace: 没有 U 属性, 抛 TypeError
    g = CentralTheory(d_shared=32, num_experts=3, mode="gate")
    try:
        freeze_u(g)
        assert False, "gate 模式应抛 TypeError"
    except TypeError:
        pass
    print("[PASS] test_freeze_u_patches_router_norm_and_broadcast")
    return True


def test_freeze_c_patches_all_modes():
    """10. freeze_c() 对所有 4 种模式都生效."""
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        m = CentralTheory(d_shared=32, num_experts=3, mode=mode)
        assert m.c.requires_grad is True, f"{mode}: freeze 前 c 应可训练"
        freeze_c(m)
        assert m.c.requires_grad is False, f"{mode}: freeze 后 c 应冻结"
    print("[PASS] test_freeze_c_patches_all_modes")
    return True


def test_set_u_init_scale_rescales_U():
    """11. set_u_init_scale() 重新初始化 U 为更小 scale (修正 B 移植)."""
    rn = RouterNormCentral(d_shared=32, num_experts=3)
    original_std = rn.U.std().item()
    # 0.001 应该比默认的 ~0.001 还小 (默认 RouterNormCentral 已用 0.001, 但 set_u 更小)
    set_u_init_scale(rn, scale=0.0001)
    new_std = rn.U.std().item()
    assert new_std < original_std, \
        f"set_u(0.0001) 后 std 应更小: 原 {original_std:.5f} → 新 {new_std:.5f}"
    # gate 模式应抛 TypeError
    g = CentralTheory(d_shared=32, num_experts=3, mode="gate")
    try:
        set_u_init_scale(g)
        assert False, "gate 模式应抛 TypeError"
    except TypeError:
        pass
    print(f"[PASS] test_set_u_init_scale_rescales_U ({original_std:.5f} → {new_std:.5f})")
    return True


def test_diagnose_theory_layer_returns_correct_diagnosis():
    """12. diagnose_theory_layer 诊断每种模式 + 稳定性状态."""
    # broadcast + U 可训练 → H1 风险
    b = CentralTheory(d_shared=32, num_experts=3, mode="broadcast")
    d = diagnose_theory_layer(b)
    assert d["mode"] == "broadcast"
    assert "H1" in d["diagnosis"], f"broadcast + U 可训练应警告 H1: {d['diagnosis']}"
    # broadcast + U 冻结 → OK
    freeze_u(b)
    d = diagnose_theory_layer(b)
    assert "OK" in d["diagnosis"], f"freeze_u 后 broadcast 应 OK: {d['diagnosis']}"
    # gate → OK (天然无 H1)
    g = CentralTheory(d_shared=32, num_experts=3, mode="gate")
    d = diagnose_theory_layer(g)
    assert "OK" in d["diagnosis"]
    assert d["U_trainable"] == "N/A (gate 模式)"
    # adaptive-ema warmup → OK
    ae = AdaptiveEMACentral(d_shared=32, num_experts=3)
    ae.train()
    d = diagnose_theory_layer(ae)
    assert "warmup" in d["diagnosis"], f"adaptive-ema warmup 应报 warmup: {d['diagnosis']}"
    print("[PASS] test_diagnose_theory_layer_returns_correct_diagnosis")
    return True


def test_ema_enable_disable_controls_ema_update():
    """13. ema_enable/disable 控制 EMA 更新 (对齐 v8._ema_enabled 语义)."""
    rn = RouterNormCentral(d_shared=32, num_experts=3)
    rn.train()
    c_initial = rn.c.data.clone()
    outs = [torch.randn(1, 4, 32) for _ in range(3)]
    # 启用 EMA: c 应变化
    rn.enable_ema()
    rn.ema_update(outs)
    c_after_enable = rn.c.data.clone()
    assert not torch.allclose(c_after_enable, c_initial), "启用 EMA 时 c 应更新"
    # 禁用 EMA: c 不变
    rn.disable_ema()
    rn.ema_update(outs)
    c_after_disable = rn.c.data.clone()
    assert torch.allclose(c_after_disable, c_after_enable), "禁用 EMA 时 c 不应更新"
    # eval 模式: c 不变 (即使 enable)
    rn.enable_ema()
    rn.eval()
    rn.ema_update(outs)
    assert torch.allclose(rn.c.data, c_after_disable), "eval 模式 c 不应更新"
    print("[PASS] test_ema_enable_disable_controls_ema_update")
    return True


def test_adaptive_ema_step_count_increments():
    """14. AdaptiveEMACentral._step_count 在 ema_update 后正确递增."""
    ae = AdaptiveEMACentral(d_shared=32, num_experts=3)
    ae.train()
    assert ae._step_count == 0
    outs = [torch.randn(1, 4, 32) for _ in range(3)]
    for i in range(5):
        ae.ema_update(outs)
        assert ae._step_count == i + 1, f"step {i+1}: _step_count 应 = {i+1}, 实测 {ae._step_count}"
    # reset
    ae.reset_step_count()
    assert ae._step_count == 0
    print("[PASS] test_adaptive_ema_step_count_increments")
    return True


TESTS = [
    test_gate_style_analysis_returns_all_5_hypotheses,
    test_central_theory_broadcast_mode,
    test_central_theory_gate_mode_temperature,
    test_central_theory_router_norm_mode,
    test_central_theory_adaptive_ema_initial_alpha_zero,
    test_central_theory_all_modes_gradient_flow,
    test_central_theory_different_modes_produce_different_outputs,
    test_adaptive_ema_warmup_uses_base_decay,
    test_freeze_u_patches_router_norm_and_broadcast,
    test_freeze_c_patches_all_modes,
    test_set_u_init_scale_rescales_U,
    test_diagnose_theory_layer_returns_correct_diagnosis,
    test_ema_enable_disable_controls_ema_update,
    test_adaptive_ema_step_count_increments,
]


if __name__ == "__main__":
    print("=== V22.0 CentralTheory (中枢理论分析) — 单元测试 (14 项) ===")
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
