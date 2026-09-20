"""V9.0 — 中枢机制统一封装单元测试 (12 项).

覆盖:
    BroadcastWorkspace (v8.0 原版):
        1. 构造 + 形状
        2. augment 形状 + 偏置不为零
        3. EMA 更新 c
        4. 修正 A: freeze_broadcast_matrix 冻结 U
        5. 修正 B: set_broadcast_init_scale 重新初始化 U

    GateStyleWorkspace (v9.0 修正 C, 推荐):
        6. 初始状态退化为 identity
        7. W_t + c 非零时调制 logits
        8. EMA 更新 c 但 forward 不变 (W_t=0 锁温度)

    工厂 + 诊断:
        9. make_workspace 工厂两种 mode
        10. diagnose_c1_root_cause 报告 H1/H2/OK
        11. diagnose_c1_root_cause 对 GateStyleWorkspace 报告"无 U 矩阵"
        12. diagnose_c1_root_cause 对无 cw_attn 报告 "no cw_attn"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.routing_evolution.v9_gate_central import (
    BroadcastWorkspace,
    GateStyleWorkspace,
    make_workspace,
    freeze_broadcast_matrix,
    set_broadcast_init_scale,
    diagnose_c1_root_cause,
)


# ---------- BroadcastWorkspace ----------


def test_broadcast_construct_and_shape():
    """1. BroadcastWorkspace 构造 + 形状正确"""
    cw = BroadcastWorkspace(d_shared=8, num_experts=3)
    assert cw.c.shape == (8,)
    assert cw.U.shape == (3, 8)
    z = torch.randn(2, 4, 3)
    z_aug = cw.augment_router_logits(z)
    assert z_aug.shape == z.shape
    print("[PASS] test_broadcast_construct_and_shape")
    return True


def test_broadcast_bias_nonzero_after_c_set():
    """2. c 非零时, augment 应有偏置 (z_aug - z ≠ 0)"""
    cw = BroadcastWorkspace(d_shared=8, num_experts=3)
    cw.c.data = torch.ones(8) * 0.5
    z = torch.randn(2, 4, 3)
    z_aug = cw.augment_router_logits(z)
    diff = float((z_aug - z).abs().max().detach())
    assert diff > 1e-4, f"c 设置后 broadcast 应有偏置, diff={diff}"
    print(f"[PASS] test_broadcast_bias_nonzero_after_c_set (diff={diff:.4f})")
    return True


def test_broadcast_ema_updates_c():
    """3. BroadcastWorkspace EMA 更新 c 后, c norm 增加"""
    cw = BroadcastWorkspace(d_shared=8, num_experts=3)
    c_before = cw.c.norm().item()
    expert_outputs = [torch.randn(2, 4, 8) for _ in range(3)]
    cw.ema_update(expert_outputs)
    c_after = cw.c.norm().item()
    assert c_after > c_before, f"EMA 后 c norm 应增加, {c_before} -> {c_after}"
    print(f"[PASS] test_broadcast_ema_updates_c ({c_before:.4f} -> {c_after:.4f})")
    return True


def test_freeze_broadcast_matrix():
    """4. 修正 A: freeze_broadcast_matrix 冻结 BroadcastWorkspace.U"""
    cw = BroadcastWorkspace(d_shared=8, num_experts=3)
    assert cw.U.requires_grad is True, "初始 U 应可训练"
    freeze_broadcast_matrix(cw)
    assert cw.U.requires_grad is False, "freeze 后 U 应冻结"
    print("[PASS] test_freeze_broadcast_matrix")
    return True


def test_set_broadcast_init_scale():
    """5. 修正 B: set_broadcast_init_scale 重新初始化 U 为小 scale"""
    cw = BroadcastWorkspace(d_shared=8, num_experts=3)
    old_std = float(cw.U.std().detach())
    set_broadcast_init_scale(cw, scale=0.001)
    new_std = float(cw.U.std())
    assert new_std < old_std, f"scale 应减小, old={old_std}, new={new_std}"
    assert new_std < 0.01, f"scale=0.001 后 std 应 < 0.01, 实测 {new_std}"
    print(f"[PASS] test_set_broadcast_init_scale (new_std={new_std:.6f})")
    return True


# ---------- GateStyleWorkspace ----------


def test_gate_initial_identity():
    """6. GateStyleWorkspace 初始 W_t=0,c=0 时退化为 identity"""
    cw = GateStyleWorkspace(d_shared=8, num_experts=3)
    z = torch.randn(2, 4, 3)
    z_aug = cw.augment_router_logits(z)
    assert torch.allclose(z, z_aug, atol=1e-6), \
        f"初始未退化为 identity, diff={float((z_aug - z).abs().max())}"
    print("[PASS] test_gate_initial_identity")
    return True


def test_gate_w_t_and_c_modulate():
    """7. W_t + c 同时非零时调制 logits"""
    cw = GateStyleWorkspace(d_shared=8, num_experts=3)
    cw.W_t.data = torch.ones(8) * 0.5
    cw.c.data = torch.ones(8) * 0.3
    z = torch.randn(2, 4, 3)
    z_aug = cw.augment_router_logits(z)
    diff = float((z_aug - z).abs().max().detach())
    assert diff > 1e-4, f"W_t+c 设置后未调制 z, diff={diff}"
    print(f"[PASS] test_gate_w_t_and_c_modulate (diff={diff:.4f})")
    return True


def test_gate_ema_locks_temperature():
    """8. EMA 更新 c 但 W_t=0 保证温度不变, forward 不变"""
    cw = GateStyleWorkspace(d_shared=8, num_experts=3)
    z = torch.randn(2, 4, 3)
    z_before = cw.augment_router_logits(z)
    expert_outputs = [torch.randn(2, 4, 8) for _ in range(3)]
    cw.ema_update(expert_outputs)
    z_after = cw.augment_router_logits(z)
    assert torch.allclose(z_before, z_after, atol=1e-6), \
        "EMA 后 z 应不变 (W_t=0 锁定温度=1)"
    assert cw.c.norm().item() > 0.01, \
        f"EMA 后 c 应有非零 norm, 实测 {cw.c.norm().item()}"
    print(f"[PASS] test_gate_ema_locks_temperature (c_norm={cw.c.norm().item():.4f})")
    return True


# ---------- 工厂 + 诊断 ----------


def test_make_workspace_factory():
    """9. make_workspace 工厂支持两种 mode"""
    cw_gate = make_workspace(mode="gate", d_shared=8, num_experts=3)
    assert isinstance(cw_gate, GateStyleWorkspace)
    cw_bc = make_workspace(mode="broadcast", d_shared=8, num_experts=3)
    assert isinstance(cw_bc, BroadcastWorkspace)
    try:
        make_workspace(mode="invalid", d_shared=8, num_experts=3)
        assert False, "应抛 ValueError"
    except ValueError:
        pass
    print("[PASS] test_make_workspace_factory")
    return True


def test_diagnose_broadcast_h1():
    """10. diagnose_c1_root_cause: BroadcastWorkspace.U 训练时诊断 H1"""
    class _FakeLayer:
        pass
    fake = _FakeLayer()
    fake.cw_attn = BroadcastWorkspace(d_shared=8, num_experts=3)
    diag = diagnose_c1_root_cause(fake)
    assert "H1" in diag["diagnosis"], f"应诊断 H1, 实测 {diag}"
    assert diag["U_trainable"] is True
    print(f"[PASS] test_diagnose_broadcast_h1 ({diag['diagnosis']})")
    return True


def test_diagnose_gate_no_u():
    """11. diagnose_c1_root_cause: GateStyleWorkspace 报告无 U 矩阵"""
    class _FakeLayer:
        pass
    fake = _FakeLayer()
    fake.cw_attn = GateStyleWorkspace(d_shared=8, num_experts=3)
    diag = diagnose_c1_root_cause(fake)
    assert diag["has_U_attr"] is False
    assert "GateStyleWorkspace" in diag["diagnosis"]
    print(f"[PASS] test_diagnose_gate_no_u ({diag['diagnosis']})")
    return True


def test_diagnose_no_cw_attn():
    """12. diagnose_c1_root_cause: 无 cw_attn 报告 'no cw_attn'"""
    class _FakeLayer:
        pass
    diag = diagnose_c1_root_cause(_FakeLayer())
    assert diag["diagnosis"] == "no cw_attn"
    print("[PASS] test_diagnose_no_cw_attn")
    return True


TESTS = [
    # BroadcastWorkspace
    test_broadcast_construct_and_shape,
    test_broadcast_bias_nonzero_after_c_set,
    test_broadcast_ema_updates_c,
    test_freeze_broadcast_matrix,
    test_set_broadcast_init_scale,
    # GateStyleWorkspace
    test_gate_initial_identity,
    test_gate_w_t_and_c_modulate,
    test_gate_ema_locks_temperature,
    # 工厂 + 诊断
    test_make_workspace_factory,
    test_diagnose_broadcast_h1,
    test_diagnose_gate_no_u,
    test_diagnose_no_cw_attn,
]


if __name__ == "__main__":
    print("=== V9.0 中枢机制统一封装 — 单元测试 (12 项) ===")
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
    import sys
    sys.exit(0 if passed == len(TESTS) else 1)
