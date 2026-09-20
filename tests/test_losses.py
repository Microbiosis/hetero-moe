"""§3 — 损失函数验证 (含手算基准)。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from hetero_fusion.core.losses import lm_loss, balance_loss, z_loss, total_loss


def test_lm_loss_correctness():
    """§3.1 交叉熵手算: logits=[2,1,0], target=0 → -log(softmax(2,1,0)[0])。"""
    logits = torch.tensor([[[2.0, 1.0, 0.0]]])  # [1,1,3]
    targets = torch.tensor([[0]])
    loss = lm_loss(logits, targets)
    sm0 = math.exp(2) / (math.exp(2) + math.exp(1) + math.exp(0))
    expected = -math.log(sm0)
    assert abs(loss.item() - expected) < 1e-5, f"{loss.item()} vs {expected}"


def test_balance_loss_handcomputed():
    """§3.2 手算: B=S=1, M=2, K=1, alpha=[0.7,0.3], topk→expert0。
    f=[0.7,0], P=[1,0], L = M·Σ f·P = 2·(0.7·1+0·0)=1.4。"""
    alpha = torch.tensor([[[0.7, 0.3]]])  # [1,1,2]
    alpha_hat = torch.tensor([[[0.7, 0.0]]])  # 稀疏化后
    loss = balance_loss(alpha, alpha_hat, top_k=1)
    assert abs(loss.item() - 1.4) < 1e-5, f"{loss.item()} vs 1.4"


def test_balance_loss_balanced_lt_imbalanced():
    """§3.2 性质: 均衡路由的 L_balance 严格小于不均衡路由 (负载均衡目标)。

    用 SparseRouterSTE 从 alpha 派生 alpha_hat, 保证二者 Top-K 选择一致,
    避免 torch.topk 在并列值上选末尾索引导致的错位。
    """
    from hetero_fusion.core.router import SparseRouterSTE
    M, K = 4, 2
    B, S = 4, 8

    # 均衡: 每 token 偏好不同的专家对, 使 P_m 趋于均匀
    z_bal = torch.randn(B, S, M)
    # 强制每 token 的 top2 轮转覆盖所有专家
    for i in range(B * S):
        b, s = i // S, i % S
        z_bal[b, s, (2 * s) % M] += 3.0
        z_bal[b, s, (2 * s + 1) % M] += 3.0
    alpha_bal = torch.softmax(z_bal, dim=-1)
    ah_bal = SparseRouterSTE.apply(alpha_bal, K)
    l_bal = balance_loss(alpha_bal, ah_bal, K)

    # 不均衡: 所有 token 只偏好 expert 0
    z_imb = torch.zeros(B, S, M)
    z_imb[..., 0] = 5.0
    alpha_imb = torch.softmax(z_imb, dim=-1)
    ah_imb = SparseRouterSTE.apply(alpha_imb, K)
    l_imb = balance_loss(alpha_imb, ah_imb, K)

    assert l_bal < l_imb, (
        f"均衡 {l_bal:.4f} 应小于不均衡 {l_imb:.4f}"
    )


def test_z_loss_correctness():
    """§3.3: z=[0,0] → logsumexp=log(2), L=log²(2)。"""
    z = torch.zeros(1, 1, 2)
    loss = z_loss(z)
    expected = math.log(2) ** 2
    assert abs(loss.item() - expected) < 1e-5, f"{loss.item()} vs {expected}"


def test_z_loss_positive_and_penalizes_large():
    """大 logits → 更大 z-loss (防止路由置信度爆炸)。"""
    z_small = torch.zeros(1, 1, 4)
    z_large = torch.full((1, 1, 4), 5.0)
    assert z_loss(z_large) > z_loss(z_small)


def test_total_loss_aggregation():
    """§3: total = lm + λ1·balance + λ2·z, 各分量可追溯。"""
    logits = torch.randn(2, 8, 16)
    targets = torch.randint(0, 16, (2, 8))
    alpha = torch.softmax(torch.randn(2, 8, 4), dim=-1)
    alpha_hat = torch.zeros(2, 8, 4)
    alpha_hat[..., :2] = alpha[..., :2]
    z = torch.randn(2, 8, 4)
    out = total_loss(logits, targets, alpha, alpha_hat, z, top_k=2,
                     lam_balance=0.01, lam_z=0.001)
    expected = out["lm"] + 0.01 * out["balance"] + 0.001 * out["z"]
    assert abs(out["total"].item() - expected.item()) < 1e-4
    assert all(k in out for k in ("total", "lm", "balance", "z"))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\nAll loss tests passed.")
