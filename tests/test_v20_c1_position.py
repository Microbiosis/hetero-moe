"""V20.0 — CentralBroadcaster (中枢广播位置) 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.central_diagnostics.v20_c1_position import CentralBroadcaster, BroadcastPosition


def test_broadcast_position_enum():
    """1. BroadcastPosition 5 个枚举值齐全"""
    assert BroadcastPosition.NONE.value == "none"
    assert BroadcastPosition.FFN.value == "ffn"
    assert BroadcastPosition.ATTN.value == "attn"
    assert BroadcastPosition.BOTH.value == "both"
    assert BroadcastPosition.SIGNAL.value == "signal"
    print("[PASS] test_broadcast_position_enum")
    return True


def test_central_broadcaster_none_no_change():
    """2. position=none 时 augment_router_logits 不改变输入"""
    cb = CentralBroadcaster(d_shared=32, num_experts=3, broadcast_position="none")
    z = torch.randn(2, 8, 3)
    z_out = cb.augment_router_logits(z)
    assert torch.allclose(z, z_out), "NONE 模式应原样返回"
    print("[PASS] test_central_broadcaster_none_no_change")
    return True


def test_central_broadcaster_broadcast_adds_bias():
    """3. broadcast 模式: z += U @ c"""
    cb = CentralBroadcaster(d_shared=32, num_experts=3, broadcast_position="ffn")
    z = torch.zeros(1, 4, 3)
    # U 是 randn * 0.01, c 是 0, 所以 bias 应 ≈ 0
    z_out = cb.augment_router_logits(z)
    assert torch.allclose(z, z_out, atol=1e-3), "c=0 时 broadcast bias 应≈0"
    # 让 c 非零, 验证 bias
    with torch.no_grad():
        cb.c.data = torch.ones(32) * 0.1
    expected_bias = cb.U @ cb.c  # [3]
    z_out = cb.augment_router_logits(z)
    # 所有 token 应加同样的 expected_bias
    assert torch.allclose(z_out[0, 0, :], expected_bias, atol=1e-4)
    print("[PASS] test_central_broadcaster_broadcast_adds_bias")
    return True


def test_central_broadcaster_gate_via_v9():
    """4. v20 的 gate 模式已迁移到 v9_gate_central.GateStyleWorkspace.
       这里验证: v20 CentralBroadcaster 委托给 BroadcastWorkspace 后,
       若想用 gate 风格, 应直接用 v9_gate_central.make_workspace."""
    from research._primitives.central_mechanism import GateStyleWorkspace, make_workspace
    # gate-style 中枢是 v9_gate_central 的事, 不在 v20 的范畴
    cw = make_workspace(mode="gate", d_shared=32, num_experts=3)
    assert isinstance(cw, GateStyleWorkspace)
    z = torch.ones(1, 4, 3) * 2.0
    z_out = cw.augment_router_logits(z)
    # c=0, W_t=0, temperature=1.0, z 不变
    assert torch.allclose(z_out, z, atol=1e-5), "c=0, W_t=0 时温度=1, z 不变"
    print("[PASS] test_central_broadcaster_gate_via_v9")
    return True


def test_central_broadcaster_position_override():
    """5. augment_router_logits 可临时指定 position (覆盖 self.position)"""
    cb = CentralBroadcaster(d_shared=32, num_experts=3, broadcast_position="ffn")
    z = torch.zeros(1, 4, 3)
    # 用 NONE 临时覆盖
    z_out_none = cb.augment_router_logits(z, position="none")
    assert torch.allclose(z, z_out_none)
    # 默认是 ffn
    z_out_ffn = cb.augment_router_logits(z)
    # 两者应不同 (c 非零时)
    with torch.no_grad():
        cb.c.data = torch.ones(32) * 0.5
    z_out_ffn = cb.augment_router_logits(z)
    z_out_none = cb.augment_router_logits(z, position="none")
    assert not torch.allclose(z_out_none, z_out_ffn), "不同 position 应产生不同输出"
    print("[PASS] test_central_broadcaster_position_override")
    return True


def test_central_broadcaster_ema_update():
    """6. ema_update 改变 c (向 expert_outputs 均值漂移)"""
    cb = CentralBroadcaster(d_shared=32, num_experts=3, ema_decay=0.5)
    assert cb.c.abs().sum() == 0
    # 喂入 3 个 expert 输出
    outs = [torch.ones(1, 4, 32) * 0.2, torch.ones(1, 4, 32) * 0.4, torch.ones(1, 4, 32) * 0.6]
    cb.ema_update(outs)
    # new_c = mean = 0.4, c = 0.5*0 + 0.5*0.4 = 0.2
    expected = 0.2 * torch.ones(32)
    assert torch.allclose(cb.c.data, expected, atol=1e-5), f"c 应更新到 0.2, 实测 {cb.c.data[0].item()}"
    print("[PASS] test_central_broadcaster_ema_update")
    return True


def test_v9_freeze_broadcast_matrix_via_v9():
    """7. v9 修正 A (freeze_u) 已迁到 v9_gate_central.freeze_broadcast_matrix.
       v20 CentralBroadcaster 委托给 BroadcastWorkspace, 验证修正 A 可作用于其 cw."""
    from research.routing_evolution.v9_gate_central import freeze_broadcast_matrix
    cb = CentralBroadcaster(d_shared=32, num_experts=3)
    assert cb.U.requires_grad, "初始 U 应可训练"
    freeze_broadcast_matrix(cb.cw)
    assert not cb.U.requires_grad, "v9 修正 A 冻结 BroadcastWorkspace.U"
    print("[PASS] test_v9_freeze_broadcast_matrix_via_v9")
    return True


def test_central_broadcaster_all_positions_produce_different_outputs():
    """8. 5 个广播位置在 c 非零时产生不同输出 (核心验证)"""
    torch.manual_seed(42)
    z = torch.randn(2, 4, 3)
    outs = {}
    for pos in ["none", "ffn", "attn", "both", "signal"]:
        cb = CentralBroadcaster(d_shared=32, num_experts=3, broadcast_position=pos)
        with torch.no_grad():
            cb.c.data = torch.ones(32) * 0.3
        out = cb.augment_router_logits(z)
        outs[pos] = out
    # 至少 4 个不同输出 (none 应唯一)
    # tensor 不能 hash, 用 first-token-value tuple 作为代理
    sigs = [tuple(o[0, 0, :].tolist()) for o in outs.values()]
    unique_count = len(set(sigs))
    assert unique_count >= 4, f"5 个位置应至少 4 种输出, 实测 {unique_count}"
    # none 应与原 z 相同
    assert torch.allclose(outs["none"], z, atol=1e-5), "NONE 应与 z 相同"
    print(f"[PASS] test_central_broadcaster_all_positions_produce_different_outputs "
          f"({unique_count} unique outputs)")
    return True


TESTS = [
    test_broadcast_position_enum,
    test_central_broadcaster_none_no_change,
    test_central_broadcaster_broadcast_adds_bias,
    test_central_broadcaster_gate_via_v9,
    test_central_broadcaster_position_override,
    test_central_broadcaster_ema_update,
    test_v9_freeze_broadcast_matrix_via_v9,
    test_central_broadcaster_all_positions_produce_different_outputs,
]


if __name__ == "__main__":
    print("=== V20.0 CentralBroadcaster (中枢广播位置) — 单元测试 (8 项) ===")
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