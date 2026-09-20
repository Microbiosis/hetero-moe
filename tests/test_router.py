"""§2.3 / §5 — 稀疏路由与容量路由验证。

覆盖: SparseRouterSTE 前向掩码 + 反向掩码、capacity_route 容量上限、
溢出降级、重归一化、全溢出跳过、C 越大溢出越少 (§8.3)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hetero_fusion.core.router import SparseRouterSTE, capacity_route


def test_sparse_router_forward_mask():
    """前向: 非 Top-K 位置置 0 (§2.3)。float32 精度下用容差比较。"""
    alpha = torch.tensor([[0.1, 0.6, 0.2, 0.1]])
    out = SparseRouterSTE.apply(alpha, 2)
    # top2 = idx 1, 2
    assert abs(out[0, 1].item() - 0.6) < 1e-6
    assert abs(out[0, 2].item() - 0.2) < 1e-6
    assert abs(out[0, 0].item()) < 1e-6 and abs(out[0, 3].item()) < 1e-6


def test_sparse_router_backward_mask():
    """反向: 仅 Top-K 位置透传梯度, 防止梯度泄漏 (§2.3)。"""
    alpha = torch.tensor([[0.4, 0.3, 0.2, 0.1]], requires_grad=True)
    out = SparseRouterSTE.apply(alpha, 2)
    out.sum().backward()
    g = alpha.grad
    # top2 idx = 0, 1
    assert g[0, 0] == 1.0 and g[0, 1] == 1.0
    assert g[0, 2] == 0.0 and g[0, 3] == 0.0


def test_capacity_renorm_sums_to_one():
    """§5.2: 实际激活集合重归一化, 权重和为 1 (未跳过 token)。"""
    torch.manual_seed(0)
    B, S, M, K = 2, 8, 4, 2
    alpha = torch.softmax(torch.randn(B, S, M), dim=-1)
    alpha_t, valid, stats = capacity_route(alpha, K, capacity_factor=1.25)
    # 每个 token 的有效权重和应为 1 (除非全溢出跳过)
    sums = alpha_t.sum(dim=-1)
    skipped = sums == 0
    non_skipped_sums = sums[~skipped]
    assert torch.allclose(non_skipped_sums, torch.ones_like(non_skipped_sums), atol=1e-5)
    # valid 位置应与 alpha_t 非零一致
    assert (alpha_t > 0).equal(valid)


def test_capacity_higher_C_less_overflow():
    """§8.3: C=1.5 溢出率应 <= C=1.0。"""
    torch.manual_seed(1)
    B, S, M, K = 2, 32, 4, 2
    # 构造不均衡路由 (部分专家过载)
    z = torch.randn(B, S, M)
    z[..., 0] += 2.0  # expert 0 偏好
    alpha = torch.softmax(z, dim=-1)
    _, _, s10 = capacity_route(alpha, K, 1.0)
    _, _, s15 = capacity_route(alpha, K, 1.5)
    of10 = sum(s10["per_expert_overflow"])
    of15 = sum(s15["per_expert_overflow"])
    assert of15 <= of10, f"C=1.5 溢出 {of15} 应 <= C=1.0 溢出 {of10}"


def test_capacity_skip_when_all_overflow():
    """§5.2: 所有候选均溢出 → Valid=∅ → token 跳过 (y_t=x_t)。"""
    # 极端: 所有 token 只偏好同一专家, K=1, C=1 → 仅 1 个 token 被接纳, 其余跳过
    B, S, M, K = 1, 6, 3, 1
    alpha = torch.zeros(B, S, M)
    alpha[..., 0] = 1.0  # 全部只选 expert 0
    alpha_t, valid, stats = capacity_route(alpha, K, capacity_factor=1.0)
    # cap = max(1, 1*6/3) = 2; expert0 接纳 2, 其余 4 个 token 全跳过
    assert stats["skip_tokens"] == 4, f"期望 4 个 token 跳过, 实际 {stats['skip_tokens']}"
    skipped = alpha_t.sum(dim=-1) == 0
    assert skipped.sum().item() == 4


def test_capacity_per_expert_stats_consistent():
    """请求 + 溢出 = 接纳; 接纳 <= capacity。"""
    torch.manual_seed(2)
    B, S, M, K = 2, 16, 4, 2
    alpha = torch.softmax(torch.randn(B, S, M), dim=-1)
    _, valid, stats = capacity_route(alpha, K, 1.25)
    cap = stats["capacity_per_expert"]
    for m in range(M):
        accepted = valid[..., m].sum().item()
        req = stats["per_expert_requested"][m]
        ovf = stats["per_expert_overflow"][m]
        assert accepted == req - ovf, f"expert {m}: 接纳 {accepted} != 请求 {req} - 溢出 {ovf}"
        assert accepted <= cap


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\nAll router tests passed.")
