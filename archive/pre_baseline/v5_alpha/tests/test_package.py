"""V5.0-α — 序列级路由 最小自包含测试 (5 项).

pre-baseline 归档包, 本轮**不补 held-out**(其 SPEC 明示"结论不应对外引用"),
补的是最小存在性与行为检查 —— 此前该包 artifacts.tests / forwarder_tests 均为 null,
是全仓唯一连"包能自己 import、类能不能跑"都没验过的包。

注意: ``SeqSparseRouterSTE`` 是 ``torch.autograd.Function``, 必须用 ``.apply(alpha, k)``
调用, 不能实例化 —— 上一版测试误按普通 nn.Module 调用, 触发 torch 2.x 的
"Legacy autograd function with non-static forward method" 报错。那是我的测试写错,
不是类坏了。
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

TITLE = "V5.0-α 序列级路由 — 最小自包含测试 (5 项)"

import torch

from archive.pre_baseline.v5_alpha import SeqSparseRouterSTE

M, K = 4, 1


def test_package_importable():
    """1. 包可独立 import (自包含的最低要求)"""
    assert SeqSparseRouterSTE is not None
    print("[PASS] test_package_importable")
    return True


def test_shape_preserved():
    """2. .apply 后保持 [B, M] 形状 (调用方再广播到 [B,S,M])"""
    alpha = torch.softmax(torch.randn(2, M), dim=-1)
    out = SeqSparseRouterSTE.apply(alpha, K)
    assert out.shape == alpha.shape == (2, M), out.shape
    print(f"[PASS] test_shape_preserved ({tuple(out.shape)})")
    return True


def test_forward_keeps_soft_weights_on_topk():
    """3. 前向保留 Top-K 位置的**软权重**(V1 修正: 非 binary), 其余置零"""
    alpha = torch.softmax(torch.randn(3, M), dim=-1)
    out = SeqSparseRouterSTE.apply(alpha, K)
    for b in range(3):
        nz = out[b] != 0
        assert int(nz.sum()) == K, f"应有且仅有 {K} 个非零, 实得 {int(nz.sum())}"
        assert torch.allclose(out[b][nz], alpha[b][nz]), "Top-K 位置应保留软权重原值"
    print("[PASS] test_forward_keeps_soft_weights_on_topk")
    return True


def test_backward_passes_gradient_only_through_topk():
    """4. 反向 STE: 梯度只透传到 Top-K 位置, 其余为 0"""
    alpha = torch.softmax(torch.randn(2, M), dim=-1)
    a = alpha.clone().requires_grad_(True)
    SeqSparseRouterSTE.apply(a, K).sum().backward()
    out = SeqSparseRouterSTE.apply(alpha, K).detach()
    assert a.grad is not None
    for b in range(2):
        nz = out[b] != 0
        assert (a.grad[b][nz] != 0).all(), "Top-K 位置应有梯度"
        assert (a.grad[b][~nz] == 0).all(), "非 Top-K 位置梯度必须为 0"
    print("[PASS] test_backward_passes_gradient_only_through_topk")
    return True


def test_broadcast_to_sequence_dim():
    """5. 序列级用法: 先把 [B,S,M] 池化到 [B,M], 路由后再广播回 [B,S,M]"""
    B, S = 2, 5
    a = torch.softmax(torch.randn(B, S, M), dim=-1).requires_grad_(True)
    a_hat_seq = SeqSparseRouterSTE.apply(a.mean(dim=1), K)          # [B, M] 序列级路由
    a_hat = a_hat_seq.unsqueeze(1).expand(-1, S, -1)               # [B, S, M]
    a_hat.pow(2).sum().backward()
    assert a_hat_seq.shape == (B, M) and a_hat.shape == (B, S, M)
    assert a.grad is not None and a.grad.abs().sum() > 0, "端到端梯度应非零"
    print(f"[PASS] test_broadcast_to_sequence_dim ({tuple(a_hat_seq.shape)} -> {tuple(a_hat.shape)})")
    return True


TESTS = [test_package_importable, test_shape_preserved,
         test_forward_keeps_soft_weights_on_topk,
         test_backward_passes_gradient_only_through_topk,
         test_broadcast_to_sequence_dim]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
