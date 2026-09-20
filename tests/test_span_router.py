"""§4.3 span 级稀疏路由 STE 测试 (v5.1)。

覆盖:
  1. build_span_id: 边界 mask → span_id 正确性
  2. forward 同 span 共享路由: 同 span 内所有 token 路由完全相同
  3. forward Top-K per span: 每 span 独立选 Top-K (非 per-token)
  4. backward STE 隔离: 非 Top-K span 位置梯度为 0
  5. backward 同 span 共享梯度: 同 span token 梯度完全相同 (均分)
  6. 跨 span 不泄漏: span A 的梯度不影响 span B 的 STE 决策
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from archive.pre_baseline.v5_alpha.span_router import build_span_id, SpanSparseRouterSTE


def test_build_span_id():
    """build_span_id: 边界 mask → span_id 正确性 + 首 token 强制边界。"""
    # 3 个 span: [0,1] [2,3,4] [5..11]
    bm = torch.zeros(2, 12, dtype=torch.bool)
    bm[0, [0, 2, 5]] = True
    span_id, num_spans = build_span_id(bm)
    assert num_spans == 3, f"应 3 个 span, 实际 {num_spans}"
    assert span_id[0].tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    # 首 token 强制边界
    bm2 = torch.zeros(1, 5, dtype=torch.bool)
    bm2[0, [2]] = True  # 未设首 token
    sid2, ns2 = build_span_id(bm2)
    assert sid2[0, 0] == 0, "首 token 应强制为 span 0"
    assert ns2 == 2


def test_same_span_same_routing():
    """同 span 内所有 token 路由权重完全相同 (v5.1 §4.3 步骤 4 广播)。"""
    torch.manual_seed(0)
    B, S, M, K = 2, 8, 4, 2
    z = torch.randn(B, S, M)
    bm = torch.zeros(B, S, dtype=torch.bool)
    bm[:, 0] = True; bm[:, 4] = True  # 两 span: [0..3], [4..7]
    span_id, num_spans = build_span_id(bm)
    ah = SpanSparseRouterSTE.apply(z, span_id, num_spans, K)
    # 同 span 内 token 路由应完全相同
    assert torch.allclose(ah[0, 0], ah[0, 1]), "同 span token 0,1 路由应相同"
    assert torch.allclose(ah[0, 2], ah[0, 3]), "同 span token 2,3 路由应相同"
    assert torch.allclose(ah[0, 4], ah[0, 7]), "同 span token 4..7 路由应相同"
    # 不同 span 路由应不同 (随机 z 下几乎必然)
    assert not torch.allclose(ah[0, 0], ah[0, 4], atol=1e-6), "不同 span 路由应不同"


def test_topk_per_span_not_per_token():
    """每 span 独立 Top-K, 而非全局 per-token (v5.1 §4.3 步骤 3)。"""
    torch.manual_seed(1)
    B, S, M, K = 1, 6, 4, 1
    z = torch.zeros(B, S, M)
    # span 0 = [0,1]: 偏好 expert 0
    z[0, 0:2, 0] = 10.0
    # span 1 = [2,3,4,5]: 偏好 expert 2
    z[0, 2:6, 2] = 10.0
    bm = torch.zeros(B, S, dtype=torch.bool)
    bm[0, [0, 2]] = True  # span0=[0,1], span1=[2,3,4,5]
    span_id, num_spans = build_span_id(bm)
    ah = SpanSparseRouterSTE.apply(z, span_id, num_spans, K)
    # span 0 应选 expert 0
    assert ah[0, 0, 0] > 0.5 and ah[0, 0, 1] == 0
    # span 1 应选 expert 2
    assert ah[0, 2, 2] > 0.5 and ah[0, 2, 0] == 0


def test_backward_ste_isolates_non_topk():
    """反向 STE: 非 Top-K span 位置梯度为 0 (三粒度 STE 不变量)。"""
    torch.manual_seed(2)
    B, S, M, K = 2, 8, 4, 1
    z = torch.randn(B, S, M, requires_grad=True)
    bm = torch.zeros(B, S, dtype=torch.bool)
    bm[:, 0] = True; bm[:, 4] = True
    span_id, num_spans = build_span_id(bm)
    ah = SpanSparseRouterSTE.apply(z, span_id, num_spans, K)
    ah.sum().backward()
    # 找每个 span 的 Top-K 专家
    z_span_manual = z.detach().mean(dim=1, keepdim=True).expand_as(z)  # 近似
    # 直接检查: 被选中的 span 专家梯度非零, 未被选中为零
    # 手动计算 span 级 softmax 的 Top-K
    z_s = torch.zeros(B, num_spans, M)
    for b in range(B):
        for j in range(num_spans):
            toks = (span_id[b] == j).nonzero(as_tuple=True)[0]
            z_s[b, j] = z.detach()[b, toks].mean(dim=0)
    alpha_s = z_s.softmax(-1)
    _, topk_idx = alpha_s.topk(K, dim=-1)
    for b in range(B):
        for j in range(num_spans):
            topk_expert = topk_idx[b, j].item()
            for m in range(M):
                if m != topk_expert:
                    # 该 span 的非 Top-K 专家, 梯度应为 0
                    toks = (span_id[b] == j).nonzero(as_tuple=True)[0]
                    grad_at_span = z.grad[b, toks, m]
                    assert grad_at_span.abs().sum() < 1e-6, \
                        f"span {j} 非 Top-K 专家 {m} 梯度应为 0, 实际 {grad_at_span.abs().sum()}"


def test_backward_same_span_shared_gradient():
    """反向: 同 span 内 token 梯度完全相同 (均分 span 级梯度)。"""
    torch.manual_seed(3)
    B, S, M, K = 1, 6, 3, 1
    z = torch.randn(B, S, M, requires_grad=True)
    bm = torch.zeros(B, S, dtype=torch.bool)
    bm[0, [0, 3]] = True  # span0=[0,1,2], span1=[3,4,5]
    span_id, num_spans = build_span_id(bm)
    ah = SpanSparseRouterSTE.apply(z, span_id, num_spans, K)
    ah.sum().backward()
    # span 0 内 3 个 token 梯度应完全相同 (各得 span_grad / 3)
    for m in range(M):
        g0 = z.grad[0, 0, m]
        for s in [1, 2]:
            assert torch.allclose(z.grad[0, s, m], g0, atol=1e-6), \
                f"span0 token {s} 与 token 0 梯度应相同 (均分), 实际 {z.grad[0,s,m]} vs {g0}"
        g3 = z.grad[0, 3, m]
        for s in [4, 5]:
            assert torch.allclose(z.grad[0, s, m], g3, atol=1e-6), \
                f"span1 token {s} 与 token 3 梯度应相同"


def test_backward_no_cross_span_leakage():
    """跨 span 不泄漏: span A 的梯度不应影响 span B 的 STE 决策。"""
    torch.manual_seed(4)
    B, S, M, K = 1, 6, 3, 1
    z = torch.randn(B, S, M, requires_grad=True)
    bm = torch.zeros(B, S, dtype=torch.bool)
    bm[0, [0, 3]] = True
    span_id, num_spans = build_span_id(bm)
    ah = SpanSparseRouterSTE.apply(z, span_id, num_spans, K)
    # 只对 span 0 的 token 求 loss (不触及 span 1)
    loss = ah[0, :3].sum()  # 仅 span 0
    loss.backward()
    # span 1 的梯度应为 0 (未被损失触及, 且 STE 隔离)
    for s in [3, 4, 5]:
        for m in range(M):
            assert z.grad[0, s, m] == 0.0, \
                f"span1 token {s} 梯度应为 0 (未触及), 实际 {z.grad[0,s,m]}"
    # span 0 应有非零梯度
    assert z.grad[0, :3].abs().sum() > 0, "span 0 应有梯度"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[PASS] {name}")
    print("\nSpanSparseRouterSTE 测试全部通过.")
