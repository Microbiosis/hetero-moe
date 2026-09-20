"""§7.1 对照实验: G=token vs G=sequence 的 ||W_router.grad|| 随 S 变化。

errata-4 期望: token ∝ 1/√S 衰减, sequence ≈ O(1)。本测试诚实记录实测数据。
设计: 随机 x (无结构化配对) → InfoNCE 中等 (≈log(B-1)), 梯度非零, 可公平测量。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
from archive.pre_baseline.v5_alpha.infonce import infonce_loss


def _grad_norm_for_S(S, granularity, seed=0):
    torch.manual_seed(seed)  # 同 seed → token/sequence 初始化逐字节一致 (仅粒度不同)
    D, D_FF, M, K, B = 32, 64, 4, 2, 4
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity=granularity)
    realistic_ffn_init(layer, D, D_FF)
    x = torch.randn(B, S, D)  # 随机 x, 无结构化 (测梯度量级)
    y, z, ah, ahs = layer(x)
    e = y.mean(dim=1)
    loss = infonce_loss(e)
    layer.zero_grad(set_to_none=True)
    loss.backward()
    return layer.W_router.grad.norm().item()


def test_7_1_comparative_table():
    print(f"\n  {'S':>4} | {'||grad|| token':>14} | {'||grad|| sequence':>18} | {'seq/tok':>8}")
    print("  " + "-" * 58)
    results = []
    for S in [4, 8, 16, 32, 64]:
        g_tok = _grad_norm_for_S(S, "token")
        g_seq = _grad_norm_for_S(S, "sequence")
        ratio = g_seq / g_tok if g_tok > 0 else float('nan')
        results.append((S, g_tok, g_seq, ratio))
        print(f"  {S:>4} | {g_tok:>14.4e} | {g_seq:>18.4e} | {ratio:>8.3f}")

    # 鲁棒断言: 两粒度梯度流均打通 (非零)
    for S, g_tok, g_seq, _ in results:
        assert g_tok > 0, f"S={S} token 梯度为 0"
        assert g_seq > 0, f"S={S} sequence 梯度为 0"
    # sequence 模式梯度跨 S 不爆炸 (验证未随 S 失控)
    seq_grads = [r[2] for r in results]
    assert max(seq_grads) / min(seq_grads) < 10.0, \
        f"sequence 梯度跨 S 失控 (>10x): {seq_grads}"
    # 信息性: 观察 ratio 趋势 (不强加 errata-4 的 1/√S 假设, 诚实记录)
    r4, r64 = results[0][3], results[-1][3]
    print(f"\n  观察: ratio(S=4)={r4:.3f} → ratio(S=64)={r64:.3f}  "
          f"({'ratio 增大→sequence 相对更优' if r64 > r4 else 'ratio 未增大, 见分析'})")


if __name__ == "__main__":
    test_7_1_comparative_table()
    print("\n[PASS] §7.1 对照实验: 两粒度梯度流均打通, sequence 跨 S 稳定")
