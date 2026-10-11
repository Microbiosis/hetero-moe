"""v43.0 — 中枢结构审计算子单元测试 (8 项)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_fusion")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V43.0 中枢结构审计算子 — 单元测试 (8 项)"

import torch

from research.routing_evolution.v43_central_audit import (
    broadcast_params,
    capacity_ratio,
    gate_params,
    is_identity_at_init,
    order_break_rate,
    order_preserved,
)

D, M = 256, 3


def _ws(kind="gate", num_experts=M, d_shared=D):
    from research._primitives.central_mechanism import (
        BroadcastWorkspace, GateStyleWorkspace,
    )
    if kind == "gate":
        return GateStyleWorkspace(d_shared=d_shared, num_experts=num_experts, alpha=0.1)
    return BroadcastWorkspace(d_shared=d_shared, num_experts=num_experts)


def test_param_formulas_match_instantiation():
    """1. 参数量解析式与真实实例化逐位一致"""
    b, g = _ws("broadcast"), _ws("gate")
    real_b = sum(p.numel() for p in b.parameters() if p.requires_grad)
    real_g = sum(p.numel() for p in g.parameters() if p.requires_grad)
    assert broadcast_params(D, M)["total"] == real_b, (broadcast_params(D, M), real_b)
    assert gate_params(D, M)["total"] == real_g, (gate_params(D, M), real_g)
    print(f"[PASS] test_param_formulas_match_instantiation (b={real_b}, g={real_g})")
    return True


def test_gate_uses_fewer_params_than_broadcast():
    """2. 核心事实: gate 的参数**少于** broadcast —— 容量混杂方向与担心的相反"""
    r = capacity_ratio(D, M)
    assert r < 1.0, f"gate 应当更省参数, 实际比值 {r}"
    assert gate_params(D, M)["total"] * 2 == broadcast_params(D, M)["total"]
    print(f"[PASS] test_gate_uses_fewer_params_than_broadcast (ratio={r:.2f})")
    return True


def test_gate_param_independent_of_expert_count():
    """3. gate 的参数量与专家数无关, broadcast 的随 M 线性增长"""
    for m in (2, 3, 8, 16):
        assert gate_params(D, m)["total"] == 2 * D
        assert broadcast_params(D, m)["total"] == D * (1 + m)
    print("[PASS] test_gate_param_independent_of_expert_count")
    return True


def test_perfect_degradation_at_init():
    """4. 完美退化: 初始 W_t=0,c=0 ⇒ 输出恒等于输入"""
    g = _ws("gate")
    z = torch.randn(4, 7, M)
    ok, dev = is_identity_at_init(g, z)
    assert ok, f"初始状态应恒等, 实测最大偏差 {dev:.3e}"
    print(f"[PASS] test_perfect_degradation_at_init (dev={dev:.3e})")
    return True


def test_scalar_temperature_preserves_order():
    """5. 标量温度在结构上保序"""
    torch.manual_seed(0)
    for _ in range(500):
        z = torch.randn(M)
        t = 1.0 + 0.1 * torch.tanh(torch.randn(1))
        assert order_preserved(z, t), (z.tolist(), t)
    print("[PASS] test_scalar_temperature_preserves_order")
    return True


def test_widened_temperature_breaks_order():
    """6. 展宽为逐专家温度后保序性不再成立 —— 容量与机制不可分离"""
    r = order_break_rate(n_trials=5000)
    assert r["scalar_break_rate"] == 0.0, r
    assert r["wide_break_rate"] > 0.0, f"展宽后应当出现保序破坏, 实得 {r}"
    print(f"[PASS] test_widened_temperature_breaks_order "
          f"(scalar={r['scalar_break_rate']:.3f}, wide={r['wide_break_rate']:.3f})")
    return True


def test_order_break_rate_is_seeded_and_reproducible():
    """7. 随机检验必须可复现, 且破坏率随试验数单调收敛"""
    a = order_break_rate(n_trials=2000)
    b = order_break_rate(n_trials=2000)
    assert a == b, "同样的调用必须给出同样结果"
    c = order_break_rate(n_trials=8000)
    assert c["wide_break_rate"] > 0.0 and c["scalar_break_rate"] == 0.0
    print("[PASS] test_order_break_rate_is_seeded_and_reproducible")
    return True


def test_identity_holds_after_training_c():
    """8. 完美退化依赖 W_t=0; W_t 非零后不再恒等 —— 说明该性质是结构性的而非恒真"""
    g = _ws("gate")
    with torch.no_grad():
        g.W_t.fill_(0.7)
        g.c.fill_(0.3)
    z = torch.randn(2, 3, M)
    ok, dev = is_identity_at_init(g, z)
    assert not ok and dev > 1e-3, f"W_t 非零后不应恒等, dev={dev}"
    print(f"[PASS] test_identity_holds_after_training_c (dev={dev:.4f})")
    return True


TESTS = [
    test_param_formulas_match_instantiation,
    test_gate_uses_fewer_params_than_broadcast,
    test_gate_param_independent_of_expert_count,
    test_perfect_degradation_at_init,
    test_scalar_temperature_preserves_order,
    test_widened_temperature_breaks_order,
    test_order_break_rate_is_seeded_and_reproducible,
    test_identity_holds_after_training_c,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
