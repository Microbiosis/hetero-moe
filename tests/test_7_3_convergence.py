"""§7.3 InfoNCE 收敛验收 (小规模, CPU 可行)。

2-layer EmbeddingFusionLayer (G=sequence), 结构化配对 (共享基底, 可学习放大),
非饱和温度 t=0.5 (避免低温度下 positives 过近导致 loss≡0 无梯度)。
期望: InfoNCE Loss 稳定下降。真实 0.5B 级收敛留给 A100 (§8.1 级)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
from archive.pre_baseline.v5_alpha.infonce import infonce_loss
from archive.pre_baseline.v5_alpha.seq_losses import balance_loss_seq, z_loss_seq


def test_7_3_infonce_decreases():
    torch.manual_seed(7)
    D, D_FF, M, K, N_LAYERS = 64, 128, 4, 2, 2
    B, S = 8, 16
    TEMP = 0.5  # 非饱和温度 (t=0.05 会使结构化 positives 过近→loss≡0)

    layers = nn.ModuleList([
        EmbeddingFusionLayer(D, D_FF, M, K, granularity="sequence") for _ in range(N_LAYERS)
    ])
    for l in layers:
        realistic_ffn_init(l, D, D_FF)

    # 结构化配对: 每 pair 共享基底 + 独立噪声 (有可学习放大的相关性, 非过近)
    base = torch.randn(B // 2, S, D)
    noise = 0.4 * torch.randn(B, S, D)
    x = torch.empty(B, S, D)
    x[0::2] = base + noise[0::2]
    x[1::2] = base + noise[1::2]
    x = x.detach()

    opt = torch.optim.AdamW([p for l in layers for p in l.parameters() if p.requires_grad],
                            lr=1e-2)

    losses = []
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        y = x
        last_z, last_ahs = None, None
        for layer in layers:
            y, z, ah, ahs = layer(y)
            last_z, last_ahs = z, ahs
        e = y.mean(dim=1)
        l_infonce = infonce_loss(e, temperature=TEMP)
        l_bal = balance_loss_seq(torch.softmax(last_z, dim=-1), last_ahs, K)
        l_z = z_loss_seq(last_z)
        loss = l_infonce + 0.01 * l_bal + 0.001 * l_z
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 6 == 0 or step == 29:
            print(f"  step {step:2d}  total={loss.item():.4f}  infonce={l_infonce.item():.4f}  "
                  f"balance={l_bal.item():.4f}  z={l_z.item():.4f}")

    print(f"\n  InfoNCE 起始: {losses[0]:.4f}")
    print(f"  InfoNCE 终值: {losses[-1]:.4f}")
    print(f"  降幅: {(losses[0]-losses[-1])/max(losses[0],1e-9)*100:.1f}%")
    assert losses[-1] < losses[0], f"InfoNCE 未下降: {losses[0]:.4f}→{losses[-1]:.4f}"
    assert all(not (math.isnan(l) or math.isinf(l)) for l in losses), "出现 NaN/inf, SwiGLU 震荡"


if __name__ == "__main__":
    test_7_3_infonce_decreases()
    print("\n[PASS] §7.3 InfoNCE 稳定下降, 无 NaN/inf 震荡")
