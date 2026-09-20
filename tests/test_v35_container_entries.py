"""V35.0-α 大模型容器 + 异质入口注入 — 单元测试 (10 项)。

不依赖 transformers / tabldm: 用 dummy 容器模型测试注入前向语义,
用随机张量测试投影工具与路径解析。每项独立 [PASS] / [FAIL] 标记。

覆盖:
    inject:
        1. ProjHead 输出形状
        2. ProjHead 权重正交初始化
        3. make_P 冻结 + 正交
    paths:
        4. 路径常量可解析 (str)
        5. first_existing 语义 (回退 / 命现存)
    entries.big_forward (dummy 容器):
        6. 无 prefix 时输出 == embed_tokens(input_ids)
        7. 有 prefix 时输出 == embed + prior (broadcast mean)
        8. prefix 的 prior = mean over dim=1 (异质入口 [3,D] → [1,D])
        9. 输出 dtype 为 float32 (容器 forward 后 .float())
        10. prefix=None 与 prefix=零 不一致语义被正确区分 (零注入 == 无注入的数值)
"""
import sys, os, tempfile
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from research.scale_heterogeneous.v35_container_entries import (
    ProjHead, make_P, big_forward, first_existing,
    MSA_DIR, EMB_DIR, RER_DIR, TABLDM_CKPT,
)


class _DummyContainer(nn.Module):
    """最小 dummy 容器: 只实现 big_forward 需要的 embed_tokens + forward。"""
    def __init__(self, vocab=20, D=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, D)
        self.D = D
        self.last_embed = None

    def forward(self, input_ids=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        self.last_embed = x.detach().clone()
        return SimpleNamespace(last_hidden_state=x)


def test_projhead_shape():
    """1. ProjHead 输出形状 D_in → D_out"""
    h = ProjHead(16, 5)
    x = torch.randn(3, 16)
    y = h(x)
    assert y.shape == (3, 5), f"{y.shape}"
    print("[PASS] test_projhead_shape")
    return True


def test_projhead_orthogonal_init():
    """2. ProjHead 权重正交初始化 (D_out <= D_in 时 W W^T ≈ I)"""
    h = ProjHead(16, 5)
    gram = h.fc.weight @ h.fc.weight.t()
    err = float((gram - torch.eye(5)).abs().max())
    assert err < 1e-4, f"非正交, max|W W^T - I|={err}"
    assert float(h.fc.bias.abs().sum()) == 0, "bias 应初始化为 0"
    print("[PASS] test_projhead_orthogonal_init")
    return True


def test_make_P_frozen_orthogonal():
    """3. make_P 冻结 + 正交"""
    P = make_P(24, 8)
    assert P.requires_grad is False, "make_P 应冻结"
    gram = P @ P.t()
    err = float((gram - torch.eye(8)).abs().max())
    assert err < 1e-4, f"非正交, err={err}"
    print("[PASS] test_make_P_frozen_orthogonal")
    return True


def test_paths_resolvable():
    """4. 路径常量可解析 (str)"""
    for name, v in [("MSA_DIR", MSA_DIR), ("EMB_DIR", EMB_DIR),
                    ("RER_DIR", RER_DIR), ("TABLDM_CKPT", TABLDM_CKPT)]:
        assert isinstance(v, str) and len(v) > 0, f"{name}={v!r}"
    print("[PASS] test_paths_resolvable")
    return True


def test_first_existing_semantics():
    """5. first_existing 回退 / 命现存"""
    assert first_existing(["/nope/a", "/nope/b"], "V35_X_UNSET") == "/nope/a"
    with tempfile.TemporaryDirectory() as d:
        assert first_existing(["/nope/a", d], "V35_X_UNSET") == d
    print("[PASS] test_first_existing_semantics")
    return True


def test_big_forward_no_prefix():
    """6. 无 prefix 时输出 == embed_tokens(input_ids)"""
    torch.manual_seed(0)
    m = _DummyContainer(vocab=20, D=8)
    ids = torch.randint(0, 20, (2, 4))
    out = big_forward(m, ids, prefix=None)
    ref = m.embed_tokens(ids).float()
    assert out.shape == (2, 4, 8)
    assert torch.allclose(out, ref, atol=1e-6), "无 prefix 应为纯 embed"
    print("[PASS] test_big_forward_no_prefix")
    return True


def test_big_forward_with_prefix():
    """7. 有 prefix 时输出 == embed + prior (broadcast mean)"""
    torch.manual_seed(0)
    m = _DummyContainer(vocab=20, D=8)
    ids = torch.randint(0, 20, (2, 4))
    prefix = torch.randn(2, 3, 8)
    out = big_forward(m, ids, prefix=prefix)
    prior = prefix.mean(dim=1, keepdim=True)  # [2,1,8]
    ref = (m.embed_tokens(ids) + prior).float()
    assert torch.allclose(out, ref, atol=1e-6), "prefix 应作为 prior 加到 embed"
    print("[PASS] test_big_forward_with_prefix")
    return True


def test_prefix_prior_heterogeneous_entries():
    """8. 异质入口 [3, D] 经 mean(dim=1) 融为 [1, D] prior"""
    torch.manual_seed(0)
    m = _DummyContainer(vocab=20, D=8)
    ids = torch.randint(0, 20, (1, 4))
    # 模拟 3 个异质入口的 prefix (TabLDM/embedding/reranker)
    prefix = torch.tensor([[[1.0] * 8], [[2.0] * 8], [[3.0] * 8]])  # [1,3,8]
    out = big_forward(m, ids, prefix=prefix)
    embed = m.embed_tokens(ids).float()
    # prior 应为 (1+2+3)/3 = 2.0
    expected_prior = 2.0
    got_prior = float((out - embed).mean())
    assert abs(got_prior - expected_prior) < 1e-4, f"prior={got_prior} 期望 {expected_prior}"
    print("[PASS] test_prefix_prior_heterogeneous_entries")
    return True


def test_big_forward_returns_float32():
    """9. big_forward 输出 float32 (容器 bf16 后 .float())"""
    m = _DummyContainer(vocab=20, D=8).to(torch.bfloat16)
    ids = torch.randint(0, 20, (2, 4))
    out = big_forward(m, ids)
    assert out.dtype == torch.float32, f"dtype={out.dtype}"
    print("[PASS] test_big_forward_returns_float32")
    return True


def test_zero_prefix_equals_no_prefix():
    """10. 零 prefix 数值上等于无 prefix (prior 加法不影响时)"""
    torch.manual_seed(0)
    m1 = _DummyContainer(vocab=20, D=8)
    m2 = _DummyContainer(vocab=20, D=8)
    m2.load_state_dict(m1.state_dict())
    ids = torch.randint(0, 20, (2, 4))
    out_none = big_forward(m1, ids, prefix=None)
    out_zero = big_forward(m2, ids, prefix=torch.zeros(2, 3, 8))
    assert torch.allclose(out_none, out_zero, atol=1e-6), "零 prefix 应等价无 prefix"
    print("[PASS] test_zero_prefix_equals_no_prefix")
    return True


TESTS = [
    test_projhead_shape,
    test_projhead_orthogonal_init,
    test_make_P_frozen_orthogonal,
    test_paths_resolvable,
    test_first_existing_semantics,
    test_big_forward_no_prefix,
    test_big_forward_with_prefix,
    test_prefix_prior_heterogeneous_entries,
    test_big_forward_returns_float32,
    test_zero_prefix_equals_no_prefix,
]


if __name__ == "__main__":
    print(f"=== V35.0-α 大模型容器+异质入口注入 — 单元测试 ({len(TESTS)} 项) ===")
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
    sys.exit(0 if passed == len(TESTS) else 1)
