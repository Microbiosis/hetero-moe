"""V56.0 维度作为容量分配 — 单元测试 (8 项)。不加载模型。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
from research._primitives.testing import run_suite

TITLE = "V56.0 维度作为容量分配 — 单元测试 (8 项)"

import torch
import torch.nn as nn
from research.scale_heterogeneous.v56_capacity_allocation.audit import (
    eff_rank_pr, eff_rank_entropy, rank_report,
    make_head, build_input, train_head, hidden_act_rank, block_ablate, n_params,
    WIDTHS, REPRS, HEADS, PREDICTIONS,
)


def test_eff_rank_on_low_rank():
    """低秩数据的有效秩必须 ≈ 真实秩。"""
    torch.manual_seed(0)
    Z = torch.randn(512, 2560)
    low = Z @ torch.randn(2560, 8) @ torch.randn(8, 2560)
    assert abs(eff_rank_pr(low) - 8.0) < 1.0, eff_rank_pr(low)
    assert eff_rank_pr(Z) > 100, eff_rank_pr(Z)
    print(f"[PASS] 秩-8 数据 eff_rank={eff_rank_pr(low):.2f}; 满秩 randn={eff_rank_pr(Z):.0f}")
    return True


def test_rank_report_fields():
    r = rank_report("x", torch.randn(64, 128))
    assert set(r) == {"name", "ambient_dim", "eff_rank_pr", "eff_rank_entropy", "occupied_frac"}
    assert 0 < r["occupied_frac"] <= 1
    print(f"[PASS] rank_report 字段齐全, occupied_frac={r['occupied_frac']:.3f}")
    return True


def test_linear_head_equals_additive_family():
    """★ concat + 线性 == add 族 (数学等价), 所以 linear 头解不开 XOR。"""
    torch.manual_seed(0)
    n = 512
    a = (torch.rand(n) > 0.5).long(); b = (torch.rand(n) > 0.5).long()
    y = a ^ b
    h = torch.zeros(n, 64); h[torch.arange(n), a] = 1.0   # h 只带 a
    c = torch.zeros(n, 64); c[torch.arange(n), b] = 1.0   # c 只带 b
    x = torch.cat([h, c], 1)
    tr, te = torch.arange(384), torch.arange(384, 512)
    _, acc_lin = train_head(x, y, tr, te, head_mode="linear", steps=600, seed=0)
    assert acc_lin < 0.75, acc_lin
    print(f"[PASS] linear 头在严格 XOR 上退到 {acc_lin:.3f} (≈chance)")
    return True


def test_mlp_solves_xor_at_tiny_width():
    """★ P2 的核心: 宽度 2 就足够解 XOR (维度不是约束)。"""
    torch.manual_seed(0)
    n = 512
    a = (torch.rand(n) > 0.5).long(); b = (torch.rand(n) > 0.5).long()
    y = a ^ b
    h = torch.zeros(n, 64); h[torch.arange(n), a] = 1.0
    c = torch.zeros(n, 64); c[torch.arange(n), b] = 1.0
    x = torch.cat([h, c], 1)
    tr, te = torch.arange(384), torch.arange(384, 512)
    for w in (2, 8):
        _, acc = train_head(x, y, tr, te, hidden=w, head_mode="mlp", steps=600, seed=0)
        assert acc > 0.95, (w, acc)
    print("[PASS] MLP 宽度 2 与 8 都解出 XOR (acc>0.95) → 宽度不是约束")
    return True


def test_heads_are_not_linear():
    """linear 头无隐藏层; mlp 头必须含一层非线性 (GELU)。"""
    lin = make_head("linear", 128)
    mlp = make_head("mlp", 128, hidden=16)
    assert isinstance(lin, nn.Linear)
    assert any(isinstance(m, nn.GELU) for m in mlp.modules())
    print(f"[PASS] linear={type(lin).__name__}, mlp 含 GELU; 参数 {n_params(lin)} vs {n_params(mlp)}")
    return True


def test_build_input_shapes():
    """proj 让入口与容器等宽; native 保留原生宽度; 块边界正确。"""
    torch.manual_seed(0)
    h = torch.randn(32, 2560); c = torch.randn(32, 512)
    P = nn.Linear(512, 2560)
    xp, dh_p, de_p = build_input(h, c, "proj", P)
    xn, dh_n, de_n = build_input(h, c, "native", P)
    assert xp.shape == (32, 2560 + 2560) and (dh_p, de_p) == (2560, 2560)
    assert xn.shape == (32, 2560 + 512) and (dh_n, de_n) == (2560, 512)
    print(f"[PASS] proj→{xp.shape[1]} native→{xn.shape[1]}, 块边界正确")
    return True


def test_block_ablate_zeroes_each_side():
    """按块消融: 抹掉任一侧都必须掉准确率 (XOR 两侧都必需)。"""
    torch.manual_seed(0)
    n = 512
    a = (torch.rand(n) > 0.5).long(); b = (torch.rand(n) > 0.5).long()
    y = a ^ b
    h = torch.zeros(n, 64); h[torch.arange(n), a] = 1.0
    c = torch.zeros(n, 64); c[torch.arange(n), b] = 1.0
    x = torch.cat([h, c], 1)
    tr, te = torch.arange(384), torch.arange(384, 512)
    head, acc = train_head(x, y, tr, te, hidden=32, head_mode="mlp", steps=600, seed=0)
    bl = block_ablate(head, x, y, te, 64, 64)
    assert bl["drop_hbase"] > 0.2 and bl["drop_entry"] > 0.2, bl
    print(f"[PASS] 消融: drop_hbase={bl['drop_hbase']:.2f} drop_entry={bl['drop_entry']:.2f}")
    return True


def test_predictions_and_grid_declared():
    assert set(PREDICTIONS) == {"P1", "P2", "P3", "P4"}
    assert len(WIDTHS) == 10 and WIDTHS[0] == 4 and WIDTHS[-1] == 2048
    assert set(REPRS) == {"proj", "native"} and set(HEADS) == {"linear", "mlp"}
    print(f"[PASS] 四条判决 + 网格声明齐全 (10 宽度 4..2048)")
    return True


TESTS = [
    test_eff_rank_on_low_rank,
    test_rank_report_fields,
    test_linear_head_equals_additive_family,
    test_mlp_solves_xor_at_tiny_width,
    test_heads_are_not_linear,
    test_build_input_shapes,
    test_block_ablate_zeroes_each_side,
    test_predictions_and_grid_declared,
]

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
