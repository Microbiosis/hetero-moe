"""§7.2 mask 隔离验收: 非 TopK 专家的 γ_m/β_m/α_m 梯度为 0, TopK 非零。

推理链 (可追溯): out = Σ α̂_m·α_m·F_m; ∂out/∂γ_m = α̂_m·α_m·∂F_m/∂γ_m。
非 TopK 时 α̂_m=0 → 该项梯度链=0 → γ_m/β_m/α_m 梯度=0。
W_router 仍经 softmax Jacobian 交叉项接收梯度 (预期, 非bug)。

设计: B=4 (InfoNCE 非平凡, B=2 时 loss≡0 无梯度); M=8, K=1
      → 4 序列各选 1 专家, 至多 4 个 TopK, 至少 4 个对所有序列非 TopK (隔离, 可测)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
from archive.pre_baseline.v5_alpha.infonce import infonce_loss


def test_7_2_non_topk_adapter_grads_zero():
    torch.manual_seed(42)
    D, D_FF, M, K = 32, 64, 8, 1  # M=8,K=1,B=4 → ≥4 隔离专家保证
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity="sequence")
    realistic_ffn_init(layer, D, D_FF)

    B, S = 4, 8
    x = torch.randn(B, S, D).detach()  # 随机 x, 无结构化 (§7.2 测梯度流非收敛)

    y, z, alpha_hat, alpha_hat_seq = layer(x)
    e = y.mean(dim=1)
    loss = infonce_loss(e)
    assert loss.item() > 1e-3, f"InfoNCE 应非平凡 (>0), 实际 {loss.item()} (B=2 会平凡为 0)"
    loss.backward()

    isolated, topk_with_grad = 0, 0
    for m in range(M):
        ahs_m = alpha_hat_seq[:, m]  # [B]
        is_isolated = (ahs_m.abs().sum() == 0)
        if is_isolated:
            isolated += 1
            for name, param in (("γ", layer.gammas[m]), ("β", layer.betas[m]),
                                ("α", layer.alphas[m])):
                g = param.grad
                assert g is None or g.abs().sum() == 0, \
                    f"非 TopK 专家 {m} 的 {name} 梯度应为 0, 实际 {g}"
        else:
            g = layer.gammas[m].grad
            assert g is not None and g.abs().sum() > 0, \
                f"TopK 专家 {m} 的 γ 梯度应非零"
            topk_with_grad += 1

    # 至少 4 个隔离 (M=8, B=4, K=1 保证) + 至少 1 个 TopK 有梯度
    assert isolated >= 4, f"应至少 4 个隔离专家 (M=8,B·K=4), 实际 {isolated}"
    assert topk_with_grad >= 1, "应至少 1 个 TopK 专家梯度非零"
    # W_router 经 softmax Jacobian 交叉项接收梯度 (预期, 非bug)
    assert layer.W_router.grad is not None and layer.W_router.grad.abs().sum() > 0, \
        "W_router 应经 softmax 交叉项接收梯度"
    print(f"  隔离专家 {isolated}/{M}, TopK 有梯度 {topk_with_grad}, "
          f"W_router grad {layer.W_router.grad.abs().sum().item():.2e}")


if __name__ == "__main__":
    test_7_2_non_topk_adapter_grads_zero()
    print("[PASS] §7.2 非 TopK γ/β/α 梯度=0, TopK 非零, W_router 经 softmax 交叉项有梯度")
