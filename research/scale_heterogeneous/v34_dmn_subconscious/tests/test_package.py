"""V34.0 DMN 潜意识创造性联想层 — 单元测试 (13 项)。

不依赖 transformers: 用随机张量构造 DMNSubconscious + FusionLayer。
每项独立 [PASS] / [FAIL] 标记, 与 v4/v5/v6/v7 测试风格一致。

覆盖:
    DMNSubconscious:
        1. 构造 + buffer 形状 (C / memory / mem_count)
        2. push_memory 写入滑动窗口 + 计数递增
        3. memory 空时 associate 返回零向量
        4. push 后 associate 返回 [D] 表征
        5. 联想表征随 memory 内容变化 (非常量)
        6. r 参与梯度: q_proj 权重收到梯度 (关键设计修正点)
        7. update_C 累积慢变状态
        8. reset 清空 C / memory / count
    FusionLayer 双管注入:
        9. route_bias_proj 零初始化 → 初始路由调制为 0
        10. 双管注入 (dmn+r) 改变输出 vs 无注入 (r 非零时)
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "hetero_moe")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research._primitives.testing import run_suite


import torch

from research.scale_heterogeneous.v34_dmn_subconscious import DMNSubconscious, FusionLayer

D_SHARED = 16
M, K = 3, 1
MODAL_DIMS = [32, 16, 16]


def test_dmn_construct():
    """1. 构造 + buffer 形状"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    assert dmn.C.shape == (D_SHARED,)
    assert dmn.memory.shape == (8, M, D_SHARED)
    assert int(dmn.mem_count) == 0
    assert dmn.route_bias_proj.out_features == M
    print("[PASS] test_dmn_construct")
    return True


def test_push_memory():
    """2. push_memory 写入 + 计数递增"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    for t in range(3):
        dmn.push_memory(torch.randn(M, D_SHARED))
    assert int(dmn.mem_count) == 3, f"mem_count={int(dmn.mem_count)}"
    assert dmn.memory[:3].abs().sum() > 0
    print("[PASS] test_push_memory")
    return True


def test_associate_zero_when_empty():
    """3. memory 空时 associate 返回零向量"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    r = dmn.associate(torch.randn(D_SHARED))
    assert r.shape == (D_SHARED,)
    assert r.abs().sum() == 0, "空 memory 应返回零向量"
    print("[PASS] test_associate_zero_when_empty")
    return True


def test_associate_shape_after_push():
    """4. push 后 associate 返回 [D]"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    dmn.push_memory(torch.randn(M, D_SHARED))
    r = dmn.associate(torch.randn(D_SHARED))
    assert r.shape == (D_SHARED,)
    assert r.abs().sum() > 0, "有 memory 时联想表征应非零"
    print("[PASS] test_associate_shape_after_push")
    return True


def test_associate_depends_on_memory():
    """5. 不同 memory 给出不同联想 (联想确实读取 memory)"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    seed = torch.randn(D_SHARED)
    dmn.push_memory(torch.randn(M, D_SHARED))
    r1 = dmn.associate(seed).detach()
    dmn.reset()
    dmn.push_memory(torch.randn(M, D_SHARED) * 5.0)
    r2 = dmn.associate(seed).detach()
    assert float((r1 - r2).abs().max()) > 1e-4, "联想应随 memory 内容变化"
    print("[PASS] test_associate_depends_on_memory")
    return True


def test_r_participates_in_gradient():
    """6. r 参与梯度: q_proj 权重收到梯度 (关键设计修正)"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    dmn.push_memory(torch.randn(M, D_SHARED))
    r = dmn.associate(torch.randn(D_SHARED))
    r.pow(2).mean().backward()
    g = dmn.q_proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "联想权重应收到梯度 (r 必须接入梯度链)"
    print("[PASS] test_r_participates_in_gradient")
    return True


def test_update_C_accumulates():
    """7. update_C 累积慢变状态 C"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    c0 = dmn.C.norm().item()
    dmn.update_C(torch.ones(D_SHARED), eta=0.1)
    c1 = dmn.C.norm().item()
    assert c1 > c0, f"C 应累积, {c0} -> {c1}"
    assert dmn.C.requires_grad is False, "C 是 buffer 不应参与优化"
    print(f"[PASS] test_update_C_accumulates ({c0:.3f} -> {c1:.3f})")
    return True


def test_reset_clears_state():
    """8. reset 清空 C / memory / count"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    dmn.push_memory(torch.randn(M, D_SHARED))
    dmn.update_C(torch.ones(D_SHARED), eta=0.1)
    dmn.reset()
    assert int(dmn.mem_count) == 0
    assert dmn.memory.abs().sum() == 0
    assert dmn.C.abs().sum() == 0
    print("[PASS] test_reset_clears_state")
    return True


def test_route_bias_zero_init():
    """9. route_bias_proj 零初始化 → 初始路由调制为 0"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    r = torch.randn(D_SHARED)
    out = dmn.route_bias_proj(r)
    assert out.abs().sum() == 0, "零初始化下初始路由调制应为 0 (完美退化)"
    print("[PASS] test_route_bias_zero_init")
    return True


def test_double_channel_changes_output():
    """10. 双管注入 (dmn+r) 改变输出 vs 无注入 (r 非零时)"""
    torch.manual_seed(0)
    layer = FusionLayer(D_SHARED, 4 * D_SHARED, M, K, MODAL_DIMS, lam=0.5, mu=0.5)
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    # 让 route_bias_proj 非零 (模拟训练后)
    with torch.no_grad():
        dmn.route_bias_proj.weight.normal_(0, 0.1)
    r = torch.randn(D_SHARED)
    x = torch.randn(2, 5, D_SHARED)
    y_plain = layer(x)
    y_inj = layer(x, dmn, r)
    diff = float((y_plain - y_inj).abs().max())
    assert diff > 1e-5, f"双管注入应改变输出, diff={diff}"
    print(f"[PASS] test_double_channel_changes_output (diff={diff:.4f})")
    return True


def test_inject_perfect_degradation_at_zero_C():
    """11. 完美退化: C=0 时 inject(r) 恒等于 r (与未接 C 的 v34 一致)"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    r = torch.randn(D_SHARED)
    out = dmn.inject(r)
    assert torch.allclose(out, r, atol=1e-6), "C 为零时不应改变 r"
    print("[PASS] test_inject_perfect_degradation_at_zero_C")
    return True


def test_inject_subtracts_accumulated_direction():
    """12. C 累积后 inject 确实抵消慢变方向 (非平凡, 且方向相关)"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    r = torch.zeros(D_SHARED)
    r[0] = 1.0
    base = r.clone()
    for _ in range(5):
        dmn.update_C(r, eta=0.1)
    out = dmn.inject(base)
    # C 沿 +e0 累积 ⇒ 输出沿 e0 的分量被压低
    assert out[0] < base[0], f"累积方向应被抵消, {base[0]:.4f} -> {out[0]:.4f}"
    # 正交分量不受影响 (Ĉ 只沿 C 方向 = e0, 故 e1 分量应原样保留)
    r2 = base.clone(); r2[1] = 1.0
    out2 = dmn.inject(r2)
    assert abs(out2[1].item() - r2[1].item()) < 1e-6, (
        f"正交分量应保持不变, 实得 {out2[1].item():.6f} vs {r2[1].item():.6f}")
    assert torch.allclose(out2[2:], base[2:], atol=1e-6), "未涉及的分量应保持不变"
    print(f"[PASS] test_inject_subtracts_accumulated_direction ({base[0]:.4f} -> {out[0]:.4f})")
    return True


def test_inject_is_pure_forward_gate():
    """13. 稳态门控是纯前向的: 梯度场与未接 C 时逐元素相同, 且 C 无梯度"""
    dmn = DMNSubconscious(D_SHARED, M, window=8)
    r0 = torch.randn(D_SHARED)
    for _ in range(3):
        dmn.update_C(r0, eta=0.1)

    # 参考: 不经 inject 的梯度场
    ref = r0.clone().requires_grad_(True)
    ref.mul(2.0).sum().backward()

    r = r0.clone().requires_grad_(True)
    dmn.inject(r).mul(2.0).sum().backward()

    assert dmn.C.grad is None, "C 是 buffer, 不应有 .grad"
    assert torch.allclose(r.grad, ref.grad, atol=1e-6), (
        "稳态只改前向数值, 梯度场应与未接 C 时完全一致")
    print("[PASS] test_inject_is_pure_forward_gate")
    return True


TESTS = [
    test_dmn_construct,
    test_push_memory,
    test_associate_zero_when_empty,
    test_associate_shape_after_push,
    test_associate_depends_on_memory,
    test_r_participates_in_gradient,
    test_update_C_accumulates,
    test_reset_clears_state,
    test_route_bias_zero_init,
    test_double_channel_changes_output,
    test_inject_perfect_degradation_at_zero_C,
    test_inject_subtracts_accumulated_direction,
    test_inject_is_pure_forward_gate,
]


TITLE = f"V34.0 DMN 潜意识联想层 — 单元测试 ({len(TESTS)} 项)"


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
