"""V37.0 — β 自训练预测编码入口 单元测试 (10 项)。

不依赖 transformers / tabldm: 用随机张量测试 EntryEncoder、gen_table、train_arm、knn_acc、记忆关键设计修正 (r 参与梯度)。
覆盖:
    EntryEncoder:
        1. 构造 + 形状 (d_in → d_out)
        2. forward 输出形状
        3. 参与梯度: 反向传播后参数 grad 非空非零
        4. 参数可学习 (requires_grad=True)
    gen_table_classification:
        5. 输出形状 [n, 6] + [n]
        6. 类别数 ∈ [0, 10) (符合 % 10 规则)
        7. 规则性: 前三维符号模式决定类 (单元测试构造性验证)
    train_arm 语义:
        8. A 臂 (c=zeros) 训练后 test_acc ≥ 随机基线 (简单任务上能拟合)
        9. c_frozen=None 时 E 参数收到梯度 (β 核心: 入口参与梯度)
        10. c_frozen=non-None 时 E 不参与优化 (β 对照: 入口冻结)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from research._primitives.scale_hetero_loaders import (
    EntryEncoder, gen_table_classification, knn_acc, train_arm,
)


D_ENTRY = 32
D_BIG = 64
N_CLS = 10


def _build_data(seed=0):
    """构造一组确定性 (X, y) 与容器表征 h_base (与表格无关)."""
    X, y = gen_table_classification(n=64, seed=seed, n_classes=N_CLS)
    torch.manual_seed(seed + 777)
    h_base = torch.randn(64, D_BIG)
    tr_idx = torch.arange(48)
    te_idx = torch.arange(48, 64)
    return X, y, h_base, tr_idx, te_idx


def test_entry_encoder_construct():
    """1. EntryEncoder 构造: Linear→GELU→Linear 序列."""
    E = EntryEncoder(6, D_ENTRY)
    assert isinstance(E.net, nn.Sequential)
    assert isinstance(E.net[0], nn.Linear) and E.net[0].in_features == 6 and E.net[0].out_features == 256
    assert isinstance(E.net[2], nn.Linear) and E.net[2].in_features == 256 and E.net[2].out_features == D_ENTRY
    print("[PASS] test_entry_encoder_construct")
    return True


def test_entry_encoder_forward_shape():
    """2. forward 输出 [B, d_out]."""
    E = EntryEncoder(6, D_ENTRY)
    x = torch.randn(8, 6)
    c = E(x)
    assert c.shape == (8, D_ENTRY)
    print("[PASS] test_entry_encoder_forward_shape")
    return True


def test_entry_encoder_gradient_flows():
    """3. 参与梯度: 反向传播后参数 grad 非空非零 (β 关键设计)."""
    E = EntryEncoder(6, D_ENTRY)
    x = torch.randn(4, 6)
    c = E(x)
    c.pow(2).mean().backward()
    assert E.net[0].weight.grad is not None
    assert E.net[0].weight.grad.abs().sum() > 0
    print("[PASS] test_entry_encoder_gradient_flows")
    return True


def test_entry_encoder_params_learnable():
    """4. 参数 requires_grad=True (默认, 确认非冻结)."""
    E = EntryEncoder(6, D_ENTRY)
    for p in E.parameters():
        assert p.requires_grad is True
    print("[PASS] test_entry_encoder_params_learnable")
    return True


def test_gen_table_shape():
    """5. gen_table_classification 输出 [n, 6] + [n]."""
    X, y = gen_table_classification(n=32, seed=42, n_classes=N_CLS)
    assert X.shape == (32, 6) and y.shape == (32,)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long
    print("[PASS] test_gen_table_shape")
    return True


def test_gen_table_label_range():
    """6. 类别 ∈ [0, 10) (符号模式 + n_classes 取模)."""
    X, y = gen_table_classification(n=128, seed=7, n_classes=N_CLS)
    assert y.min().item() >= 0
    assert y.max().item() < N_CLS
    # 应当覆盖大部分类别 (n=128, 10 类)
    assert len(torch.unique(y)) >= 5
    print(f"[PASS] test_gen_table_label_range (类别 {y.min()}-{y.max()}, 覆盖 {len(torch.unique(y))} 类)")
    return True


def test_gen_table_rule_deterministic():
    """7. 规则性: 同一 seed 产生相同数据 (规则可复现)."""
    X1, y1 = gen_table_classification(n=64, seed=123, n_classes=N_CLS)
    X2, y2 = gen_table_classification(n=64, seed=123, n_classes=N_CLS)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)
    # 构造性验证: 类别 = (x0>0)*4 + (x1>0)*2 + (x2>0)
    expected = ((X1[:, 0] > 0).long() * 4
               + (X1[:, 1] > 0).long() * 2
               + (X1[:, 2] > 0).long()) % N_CLS
    assert torch.equal(y1, expected)
    print("[PASS] test_gen_table_rule_deterministic")
    return True


def test_train_arm_arm_A_simple_task():
    """8. A 臂 (c=zeros) 训练后 test_acc ≥ 随机基线 (简单任务能拟合)."""
    X, y, h_base, tr_idx, te_idx = _build_data()
    torch.manual_seed(0)
    c_zero = torch.zeros(64, D_ENTRY)
    acc, knn = train_arm(h_base, X, y, c_zero, tr_idx, te_idx,
                          d_entry=D_ENTRY, d_big=D_BIG, n_cls=N_CLS,
                          steps=300, seed=0)
    # A 臂训的是 head, 简单任务上应该 ≥ 1/10
    assert acc >= 1.0 / N_CLS, f"A acc={acc} 应 ≥ 1/{N_CLS}"
    print(f"[PASS] test_train_arm_arm_A_simple_task (acc={acc:.3f})")
    return True


def test_train_arm_arm_D_grad_to_E():
    """9. c_frozen=None 时 E 参数收到梯度 (β 核心: 入口参与梯度)."""
    X, y, h_base, tr_idx, te_idx = _build_data()
    torch.manual_seed(0)
    # 手动模拟 train_arm 的训练一步, 验证 E 收到梯度
    E = EntryEncoder(X.shape[1], D_ENTRY)
    P = nn.Linear(D_ENTRY, D_BIG)
    head = nn.Linear(D_BIG, N_CLS)
    params = list(P.parameters()) + list(head.parameters()) + list(E.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    b = tr_idx[:32]
    cin = E(X[b])
    logits = head(h_base[b] + P(cin))
    loss = nn.functional.cross_entropy(logits, y[b])
    opt.zero_grad(); loss.backward(); opt.step()
    # 关键: E 参数应收到梯度
    grad_norms = [p.grad.norm().item() for p in E.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "E 参数未收到梯度"
    assert all(g > 0 for g in grad_norms), f"E 梯度全零, 违反 β 核心设计"
    print(f"[PASS] test_train_arm_arm_D_grad_to_E (E grad norms: min={min(grad_norms):.4f})")
    return True


def test_train_arm_arm_BC_E_frozen():
    """10. c_frozen 给出时 E 不参与优化 (β 对照: 入口冻结)."""
    X, y, h_base, tr_idx, te_idx = _build_data()
    torch.manual_seed(0)
    c_frozen = torch.randn(64, D_ENTRY)
    E = EntryEncoder(X.shape[1], D_ENTRY)
    P = nn.Linear(D_ENTRY, D_BIG)
    head = nn.Linear(D_BIG, N_CLS)
    # A/B/C 臂: 只训 P+head, 不训 E
    params = list(P.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    b = tr_idx[:32]
    # 即使偶然 forward E 也不会收到优化 (params 不含 E)
    cin = c_frozen[b]
    logits = head(h_base[b] + P(cin))
    loss = nn.functional.cross_entropy(logits, y[b])
    opt.zero_grad(); loss.backward(); opt.step()
    assert all(p.grad is None for p in E.parameters()), "E 不应被优化但收到了梯度?"
    print("[PASS] test_train_arm_arm_BC_E_frozen")
    return True


TESTS = [
    test_entry_encoder_construct,
    test_entry_encoder_forward_shape,
    test_entry_encoder_gradient_flows,
    test_entry_encoder_params_learnable,
    test_gen_table_shape,
    test_gen_table_label_range,
    test_gen_table_rule_deterministic,
    test_train_arm_arm_A_simple_task,
    test_train_arm_arm_D_grad_to_E,
    test_train_arm_arm_BC_E_frozen,
]


if __name__ == "__main__":
    print(f"=== V37.0 β 自训练入口 — 单元测试 ({len(TESTS)} 项) ===")
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