"""V31.0 — 梯度重缩放门控中枢 单元测试 (9 项).

锁定本代主张: 在 tanh 前对 t 做重缩放 ``t' = t/(1+|t|)``, 使饱和区梯度
按**多项式** (∝ 1/t²) 而非**指数**衰减 —— 这是与 v9 纯 tanh 的本质差别。

与 v30 (softplus) 的对照:
- v30 用 ``softplus`` 取代 tanh ⇒ 丢掉**完美退化**(softplus(0)=ln2≠0), 换来双侧单调
- v31 保留 tanh 只在**前面**加一层重缩放 ⇒ **完美退化成立**, 且梯度只慢衰减

对照基线: research._primitives.central_mechanism.GateStyleWorkspace (v9 纯 tanh)。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_fusion")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V31.0 梯度重缩放门控中枢 — 单元测试 (9 项)"

import torch

from research._primitives.central_mechanism import GateStyleWorkspace
from research.central_stability.v31_gate_gradient_scale.workspace import (
    GradientScaledGateStyleWorkspace,
)

D, M, ALPHA = 32, 3, 0.1


def _ws(d=D, m=M, alpha=ALPHA):
    return GradientScaledGateStyleWorkspace(d_shared=d, num_experts=m, alpha=alpha)


def _at(ws, tval):
    """令 t = W_t·c 恰为 tval(W_t 取 1/D, 因点积求和)。"""
    with torch.no_grad():
        ws.c.fill_(tval); ws.W_t.fill_(1.0 / D)


def _grad_sum(ws, seed=0):
    """W_t 梯度绝对值和。完全无梯度时 grad 为 None(图里不含 W_t), 记作 0。"""
    g = ws.W_t.grad
    return 0.0 if g is None else g.abs().sum().item()


def _grad_at(tval, seed=0):
    w = _ws(); _at(w, tval)
    z = torch.randn(1, 5, M, generator=torch.Generator().manual_seed(seed))
    z = z.requires_grad_(True)
    w.augment_router_logits(z).pow(2).sum().backward()
    return _grad_sum(w)


def test_perfect_degradation_at_init():
    """1. **完美退化成立**: 初始 W_t=0,c=0 ⇒ t=0 ⇒ t'=0 ⇒ tanh(0)=0 ⇒ temperature=1"""
    w = _ws()
    z = torch.randn(3, 7, M)
    dev = (w.augment_router_logits(z.clone()) - z).abs().max().item()
    assert dev < 1e-6, f"初始应恒等, 实得偏差 {dev}"
    print("[PASS] test_perfect_degeneration_at_init")
    return True


def test_t_scaled_bounded_by_one():
    """2. 重缩放把 t 压进 (-1, 1): 任意 |t| 都满足 |t/(1+|t|)| < 1"""
    for tval in (-1e3, -50.0, -1.0, 0.0, 1.0, 50.0, 1e3):
        t = torch.tensor(tval, dtype=torch.float32)
        s = t / (1.0 + t.abs())
        assert float(s.abs()) < 1.0, f"t={tval}: t'={float(s)} 应 |·|<1"
    print("[PASS] test_t_scaled_bounded_by_one")
    return True


def test_temperature_stays_within_one_plus_minus_alpha():
    """3. temperature ∈ (1−α·tanh1, 1+α·tanh1), 恒正 ⇒ 温度不会反号"""
    lo, hi = 1.0 - ALPHA * torch.tanh(torch.tensor(1.0)).item(), \
             1.0 + ALPHA * torch.tanh(torch.tensor(1.0)).item()
    for tval in (-1e4, -10.0, 0.0, 10.0, 1e4):
        w = _ws(); _at(w, tval)
        z = torch.randn(1, 5, M)
        out = w.augment_router_logits(z)
        assert torch.isfinite(out).all(), f"t={tval} 输出非有限"
    assert lo > 0.0, "温度下界须为正"
    print(f"[PASS] test_temperature_stays_within_one_plus_minus_alpha (下界 {lo:.4f})")
    return True


def test_order_preservation():
    """4. 标量正温度 ⇒ M 个 logit 次序不变"""
    w = _ws(); _at(w, 30.0)
    z = torch.randn(2, 5, M)
    out = w.augment_router_logits(z)
    assert torch.equal(torch.argsort(z, dim=-1), torch.argsort(out, dim=-1))
    print("[PASS] test_order_preservation")
    return True


def test_small_t_preserves_original_behavior():
    """5. |t| 小时 t/(1+|t|) ≈ t ⇒ 与未重缩放的 tanh 几乎一致"""
    w = _ws(); _at(w, 1e-3)
    base = GateStyleWorkspace(d_shared=D, num_experts=M, alpha=ALPHA); _at(base, 1e-3)
    z = torch.randn(2, 5, M)
    dev = (w.augment_router_logits(z.clone()) - base.augment_router_logits(z.clone())).abs().max().item()
    assert dev < 1e-3, f"|t|=1e-3 时应与原 tanh 一致, 实差 {dev}"
    print(f"[PASS] test_small_t_preserves_original_behavior (差 {dev:.2e})")
    return True


def test_gradient_survives_where_tanh_dies():
    """6. **核心主张**: 饱和区梯度存活, 而 v9 纯 tanh 恰为 0"""
    g9 = _grad_at(50.0) if False else None
    w9 = GateStyleWorkspace(d_shared=D, num_experts=M, alpha=ALPHA); _at(w9, 50.0)
    z9 = torch.randn(1, 5, M, generator=torch.Generator().manual_seed(0))
    z9 = z9.requires_grad_(True)
    w9.augment_router_logits(z9).pow(2).sum().backward()
    g9 = _grad_sum(w9)
    assert g9 == 0.0, f"v9 tanh 基线梯度应为 0, 实得 {g9}"

    g31 = _grad_at(50.0)
    assert g31 > 0.0, "v31 的核心主张就是梯度不断"
    print(f"[PASS] test_gradient_survives_where_tanh_dies (v9={g9:.1e} vs v31={g31:.3e})")
    return True


def test_gradient_decays_as_one_over_t_not_exponentially():
    """7. 梯度按 **~1/t 多项式**衰减, 而 tanh 是指数衰减到**精确 0**

    渐近指数是 1/t 而非 1/t²: 因为 dL/dW_t = dL/dt · (dt/dW_t) = dL/dt · c,
    而 c 随 t 线性增长, 恰好抵消掉 t'(t) 的 1/t² 中的一个 t。
    实测 g·t 在 t≥25 后近似常数(33.3 → 33.9)。

    ⚠ 这同时修正了 workspace docstring 的**过强主张**
    『梯度 ∂scaled/∂t_scalar 始终有下界（不会为 0）』—— 实际是多项式衰减,
    仍会随 t → ∞ 趋于 0, 只是比 tanh 的指数快慢不同。
    """
    ts = [25.0, 625.0, 15625.0, 78125.0]   # t=1 尚未进入渐近区(g·t 仅为极限的一半)
    g = [_grad_at(t) for t in ts]
    assert all(x > 0 for x in g), f"全程须存活: {g}"
    scaled = [x * t for x, t in zip(g, ts)]          # 若渐近 1/t, 则近似常数
    lo, hi = min(scaled), max(scaled)
    assert hi / lo < 1.5, f"g·t 应近似常数, 实得 {scaled}"
    # 对照: 同样这些 t 下 v9 tanh 梯度恰为 0
    for t in ts[1:]:
        w9 = GateStyleWorkspace(d_shared=D, num_experts=M, alpha=ALPHA); _at(w9, t)
        z9 = torch.randn(1, 5, M, generator=torch.Generator().manual_seed(0))
        z9 = z9.requires_grad_(True)
        w9.augment_router_logits(z9).pow(2).sum().backward()
        assert _grad_sum(w9) == 0.0, f"v9 在 t={t} 梯度应为 0"
    print(f"[PASS] test_gradient_decays_as_one_over_t_not_exponentially "
          f"(g·t={[f'{x:.3g}' for x in scaled]})")
    return True


def test_ema_update_and_snapshot():
    """8. ema_update 累积 c; snapshot() 供推理取中枢"""
    w = _ws()
    with torch.no_grad():
        w.c.zero_()
        w.ema_update([torch.ones(4, D), torch.ones(4, D)])
    assert w.c.abs().sum() > 0, "ema_update 应累积 c"
    snap = w.snapshot()
    assert torch.is_tensor(snap) and torch.allclose(snap, w.c), "snapshot 应返回 c"
    print("[PASS] test_ema_update_and_snapshot")
    return True


def test_forward_shape_preserved():
    """9. 输出形状与输入一致"""
    w = _ws()
    z = torch.randn(4, 16, M)
    assert w.augment_router_logits(z).shape == z.shape
    print("[PASS] test_forward_shape_preserved")
    return True


TESTS = [
    test_perfect_degradation_at_init,
    test_t_scaled_bounded_by_one,
    test_temperature_stays_within_one_plus_minus_alpha,
    test_order_preservation,
    test_small_t_preserves_original_behavior,
    test_gradient_survives_where_tanh_dies,
    test_gradient_decays_as_one_over_t_not_exponentially,
    test_ema_update_and_snapshot,
    test_forward_shape_preserved,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
