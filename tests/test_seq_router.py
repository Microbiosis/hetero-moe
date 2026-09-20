"""v5.0-α.1 §2.2 核心算子回归测试 (固化用户已验证的 V1+V2+STE+sum-pool)。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from archive.pre_baseline.v5_alpha.seq_router import SeqSparseRouterSTE


def _leaf_alpha(*shape, seed=0):
    """构造可作为叶节点接收梯度的 softmax 输出。"""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(*shape, generator=g)
    return torch.softmax(logits, dim=-1).detach().requires_grad_(True)


def test_v1_soft_weights_not_binary():
    """V1: forward 返回软权重 (softmax 概率), 非 binary {0,1}。"""
    alpha = _leaf_alpha(2, 4, seed=0)
    out = SeqSparseRouterSTE.apply(alpha, 2)  # [B, M]
    assert (out == 1.0).sum() == 0, "存在二值 1.0, V1 未修正"
    assert (out == 0.0).sum() == 4, f"非 TopK 应为 0, 实际零值数 {(out==0).sum()}"
    nz = out[out > 0]
    assert torch.allclose(nz, alpha[out > 0]), "TopK 位置应保留 softmax 概率"


def test_v2_backward_shape_no_mean_error():
    """V2: backward 收到 [B,M] (非 [B,S,M]), 不再 mean(dim=1)。"""
    alpha = _leaf_alpha(2, 4, seed=1)
    out = SeqSparseRouterSTE.apply(alpha, 2)  # [B, M]
    S = 8
    out_b = out.unsqueeze(1).expand(-1, S, -1)  # [B, S, M]
    (out_b * 1.0).sum().backward()
    assert alpha.grad is not None
    assert alpha.grad.shape == (2, 4), f"grad shape 应 [B,M]=[2,4], 实际 {alpha.grad.shape}"


def test_v2_sum_pool_not_mean():
    """V2: expand 反向是 sum 非 mean。S=8, 权重 1 → grad=8.0 (sum) 而非 1.0 (mean)。"""
    alpha = _leaf_alpha(1, 4, seed=2)
    out = SeqSparseRouterSTE.apply(alpha, 4)  # K=4: 全 TopK, mask 全 1
    S = 8
    out_b = out.unsqueeze(1).expand(-1, S, -1)  # [1, S, 4]
    (out_b * 1.0).sum().backward()
    # alpha.grad[b,m] = sum_s(1 * mask[b,m]) = 8.0 (mask 全 1)
    assert torch.allclose(alpha.grad, torch.full_like(alpha.grad, 8.0)), \
        f"expand 反向应为 sum (grad=8.0), 实际 {alpha.grad}"


def test_ste_mask_isolates_non_topk():
    """STE: 非 TopK 位置 alpha_seq 梯度精确为 0。"""
    alpha = _leaf_alpha(2, 4, seed=3)
    out = SeqSparseRouterSTE.apply(alpha, 2)
    out_b = out.unsqueeze(1).expand(-1, 8, -1)
    out_b.sum().backward()
    topk_mask = (out > 0)
    assert (alpha.grad[~topk_mask].abs().sum() == 0), "非 TopK 位置 STE 隔离失效"
    assert (alpha.grad[topk_mask].abs().sum() > 0), "TopK 位置梯度缺失"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[PASS] {name}")
    print("\nSeqSparseRouterSTE 核心算子测试通过 (V1+V2+STE).")
