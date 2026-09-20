"""V39.0 — γ2 加性 vs 交互融合框架 单元测试 (10 项)。

不依赖 transformers / tabldm: 用构造性 XOR 数据测试融合头的**表达能力**差异
(这是 γ2 相对 γ 的核心修正), 以及双数据集加载器与注册表。

覆盖:
    heads (表达力 — γ2 核心):
        1. make_head: add 返回 Linear / concat 返回 MLP(Sequential)
        2. make_head: 未知 mode 抛 ValueError
        3. fuse: add 保持维度, concat 拼接维度
        4. fuse: 未知 mode 抛 ValueError
        5. **加性头无法解 XOR** (构造性: 训练后 acc 显著低于 1.0)
        6. **交互头可解 XOR** (可达 acc ≥ 0.95)
        7. 两模式参数形状正确 (concat 输入维 = 2*d_big)
    data / registry:
        8. DATASETS 注册表含 adult + shoppers
        9. shoppers loader: 形状/二值标签/文本含访客类型
        10. adult loader: XOR 标签二值 + 平衡
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from research.scale_heterogeneous.v39_g2_interaction import (
    make_head, fuse, logits_of, DATASETS,
    adult_loader, shoppers_loader,
)


def test_make_head_types():
    """1. make_head: add → Linear; concat → Sequential(MLP)."""
    h_add = make_head("add", 64, 2)
    h_cat = make_head("concat", 64, 2)
    assert isinstance(h_add, nn.Linear), f"add 应为 Linear, got {type(h_add)}"
    assert isinstance(h_cat, nn.Sequential), f"concat 应为 MLP, got {type(h_cat)}"
    print("[PASS] test_make_head_types")
    return True


def test_make_head_invalid_mode():
    """2. make_head: 未知 mode 抛 ValueError."""
    try:
        make_head("bogus", 64, 2)
    except ValueError:
        print("[PASS] test_make_head_invalid_mode")
        return True
    raise AssertionError("未知 mode 应抛 ValueError")


def test_fuse_dimensions():
    """3. fuse: add 保持 D_BIG; concat 拼接为 2*D_BIG."""
    hb = torch.randn(4, 64)
    c = torch.randn(4, 64)
    assert fuse(hb, c, "add").shape == (4, 64)
    assert fuse(hb, c, "concat").shape == (4, 128)
    print("[PASS] test_fuse_dimensions")
    return True


def test_fuse_invalid_mode():
    """4. fuse: 未知 mode 抛 ValueError."""
    try:
        fuse(torch.randn(2, 8), torch.randn(2, 8), "nope")
    except ValueError:
        print("[PASS] test_fuse_invalid_mode")
        return True
    raise AssertionError("未知 mode 应抛 ValueError")


def _xor_data(n=256, d=32, seed=0):
    """构造 XOR 任务: a (文本侧) 与 b (表格侧) 各为二元条件, y = a XOR b.

    两侧各自编码在不相交的维度子空间 (模拟 h_base / c_entry 的独立来源),
    使加性头无法通过重排特征"绕开"交互需求。
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, 2, (n,), generator=g)
    b = torch.randint(0, 2, (n,), generator=g)
    y = (a != b).long()
    h_base = torch.zeros(n, d)
    c_side = torch.zeros(n, d)
    idx = torch.arange(n)
    h_base[idx, a * 8] = 1.0        # 文本侧: 前 16 维承载 a
    c_side[idx, 16 + b * 8] = 1.0   # 表格侧: 后 16 维承载 b
    return h_base, c_side, y


def _train_xor(mode, steps=700, seed=0):
    torch.manual_seed(seed)
    h_base, c_side, y = _xor_data(seed=seed)
    d = h_base.shape[1]
    P = nn.Linear(d, d)
    head = make_head(mode, d, 2)
    opt = torch.optim.Adam(list(P.parameters()) + list(head.parameters()), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(
            logits_of(head, h_base, P(c_side), mode), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (logits_of(head, h_base, P(c_side), mode).argmax(1) == y).float().mean().item()
    return acc


def test_additive_head_cannot_solve_xor():
    """5. 加性头无法解 XOR (表达力上限) — γ 的"三臂打平"根因."""
    acc = _train_xor("add")
    assert acc < 0.9, f"加性头本应无法解 XOR (退化), 实测 acc={acc:.3f}"
    print(f"[PASS] test_additive_head_cannot_solve_xor (acc={acc:.3f} < 0.9, 退化)")
    return True


def test_interaction_head_solves_xor():
    """6. 交互头可解 XOR — γ2 修正的立论基础."""
    acc = _train_xor("concat")
    assert acc >= 0.95, f"交互头应可解 XOR, 实测 acc={acc:.3f}"
    print(f"[PASS] test_interaction_head_solves_xor (acc={acc:.3f} >= 0.95)")
    return True


def test_head_param_shapes():
    """7. 两模式参数形状正确 (concat 的输入维 = 2*d_big)."""
    d, n_cls = 64, 2
    h_add = make_head("add", d, n_cls)
    assert h_add.weight.shape == (n_cls, d)
    h_cat = make_head("concat", d, n_cls)
    assert h_cat[0].weight.shape == (128, d * 2), f"{h_cat[0].weight.shape}"
    assert h_cat[2].weight.shape == (n_cls, 128)
    print("[PASS] test_head_param_shapes")
    return True


def test_datasets_registry():
    """8. DATASETS 注册表含 adult + shoppers, 且为 (callable, str)."""
    assert set(DATASETS) == {"adult", "shoppers"}, f"{set(DATASETS)}"
    for name, (fn, label) in DATASETS.items():
        assert callable(fn), f"{name} 加载器不可调用"
        assert isinstance(label, str) and label
    print("[PASS] test_datasets_registry")
    return True


def test_shoppers_loader():
    """9. shoppers loader: 形状 [n,10]/[n], 二值标签, 文本含访客类型."""
    texts, X, y = shoppers_loader.load_shoppers_crossmodal(64, 42)
    assert X.shape == (64, 10), f"{X.shape}"
    assert y.shape == (64,) and set(y.unique().tolist()) <= {0, 1}
    assert all("visitor" in t for t in texts), "文本应含访客类型 (文本侧条件)"
    print("[PASS] test_shoppers_loader")
    return True


def test_adult_loader():
    """10. adult loader: 形状 [n,6]/[n], 二值标签, 正负平衡."""
    texts, X, y = adult_loader.load_adult_crossmodal(64, 42, decouple=True)
    assert X.shape == (64, 6), f"{X.shape}"
    assert set(y.unique().tolist()) <= {0, 1}
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    assert n_pos == n_neg == 32, f"应各 32, got pos={n_pos} neg={n_neg}"
    assert all("working as" in t for t in texts), "文本应含职业 (文本侧条件)"
    print("[PASS] test_adult_loader")
    return True


TESTS = [
    test_make_head_types,
    test_make_head_invalid_mode,
    test_fuse_dimensions,
    test_fuse_invalid_mode,
    test_additive_head_cannot_solve_xor,
    test_interaction_head_solves_xor,
    test_head_param_shapes,
    test_datasets_registry,
    test_shoppers_loader,
    test_adult_loader,
]


if __name__ == "__main__":
    print(f"=== V39.0 γ2 加性 vs 交互融合 — 单元测试 ({len(TESTS)} 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    sys.exit(0 if passed == len(TESTS) else 1)
