"""v5.1 §7.1 新 metric 验证 (适配版, CPU 可行)。

适配 (基于 §7.1 证伪 + 设计审查):
  A. RDC → 输入扰动 per-token 选择翻转率 (架构无 dropout, 改输入扰动; 不改架构)
  B. L_balance 收敛 → InfoNCE 收敛对比 (balance 在 λ1=0.01 下上升, 不可达 50%)

验证假设: sequence routing 修复 token 级欠定 (S 决策 vs 1 信号)。
  - 若 sequence 收敛更快/更稳 → 修复有效
  - 若两者相当 → 小规模下欠定未显现 (scaling matter, 诚实记录)
  - 若 token 更优 → 假设进一步证伪
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
from archive.pre_baseline.v5_alpha.infonce import infonce_loss
from archive.pre_baseline.v5_alpha.seq_losses import z_loss_seq


D, D_FF, M, K, N_LAYERS = 64, 128, 4, 2, 2
B, S = 8, 16
TEMP = 0.5
STEPS = 30


def build_model(granularity, seed=7):
    torch.manual_seed(seed)  # 同 seed → 两粒度初始化逐字节一致 (仅粒度不同)
    layers = nn.ModuleList([
        EmbeddingFusionLayer(D, D_FF, M, K, granularity=granularity) for _ in range(N_LAYERS)
    ])
    for l in layers:
        realistic_ffn_init(l, D, D_FF)
    return layers


def make_data(seed=7):
    torch.manual_seed(seed)
    base = torch.randn(B // 2, S, D)
    noise = 0.4 * torch.randn(B, S, D)
    x = torch.empty(B, S, D)
    x[0::2] = base + noise[0::2]
    x[1::2] = base + noise[1::2]
    return x.detach()


def train(layers, x):
    opt = torch.optim.AdamW([p for l in layers for p in l.parameters() if p.requires_grad], lr=1e-2)
    history = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y = x
        last_z = None
        for layer in layers:
            y, z, ah, ahs = layer(y)
            last_z = z
        e = y.mean(dim=1)
        l_inf = infonce_loss(e, temperature=TEMP)
        l_z = z_loss_seq(last_z)
        loss = l_inf + 0.001 * l_z
        loss.backward()
        opt.step()
        history.append(l_inf.item())
    return history


def compute_flip_rate(layers, x, num_runs=10, eps=0.05):
    """输入扰动下 per-token 路由选择翻转率 (适配 RDC, 架构无 dropout)。

    sequence: ah 为 [B,S,M] 广播 → 翻转率 = 序列级翻转率 (pool 抑制扰动)
    token:    ah 为 [B,S,M] 独立 → 翻转率 = per-token 翻转率 (欠定→游走)
    """
    for l in layers:
        l.eval()
    sels = []
    with torch.no_grad():
        for _ in range(num_runs):
            x_p = x + eps * torch.randn_like(x)
            y = x_p
            for layer in layers:
                y, z, ah, ahs = layer(y)
            sels.append((ah > 0).float())  # [B, S, M]
    flip = 0.0
    cnt = 0
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            flip += (sels[i] != sels[j]).float().mean().item()
            cnt += 1
    return flip / cnt


def main():
    x = make_data()

    print("=" * 64)
    print("v5.1 §7.1 新 metric 验证 (适配版: 翻转率 + InfoNCE 收敛)")
    print("=" * 64)

    results = {}
    for gran in ("token", "sequence"):
        layers = build_model(gran)
        hist = train(layers, x)
        flip = compute_flip_rate(layers, x)
        # 收敛步数: InfoNCE 降至初始 70% 所需步数
        thresh = 0.70 * hist[0]
        n70 = next((s for s, v in enumerate(hist) if v <= thresh), STEPS)
        results[gran] = {
            "final_infonce": hist[-1],
            "init_infonce": hist[0],
            "steps_to_70pct": n70,
            "flip_rate": flip,
            "history": hist,
        }
        print(f"\n--- {gran} ---")
        print(f"  InfoNCE: {hist[0]:.4f} → {hist[-1]:.4f}  (降幅 {(hist[0]-hist[-1])/hist[0]*100:.1f}%)")
        print(f"  收敛步数 (至初始 70%): {n70}/{STEPS}")
        print(f"  路由翻转率 (输入扰动 ε=0.05): {flip:.4f}")

    print("\n" + "=" * 64)
    print("对照分析")
    print("=" * 64)
    t, s = results["token"], results["sequence"]
    print(f"  {'metric':<28} | {'token':>10} | {'sequence':>10} | {'seq/tok':>8} | {'判定':>10}")
    print("  " + "-" * 80)
    # B': InfoNCE 收敛 (sequence 应更快/更低)
    ratio_final = s["final_infonce"] / t["final_infonce"]
    print(f"  {'最终 InfoNCE':<28} | {t['final_infonce']:>10.4f} | {s['final_infonce']:>10.4f} | "
          f"{ratio_final:>8.3f} | {'seq更优' if ratio_final < 1.0 else 'token更优':>10}")
    ratio_steps = s["steps_to_70pct"] / max(t["steps_to_70pct"], 1)
    print(f"  {'收敛步数(至70%)':<28} | {t['steps_to_70pct']:>10} | {s['steps_to_70pct']:>10} | "
          f"{ratio_steps:>8.3f} | {'seq更快' if s['steps_to_70pct'] < t['steps_to_70pct'] else 'token更快':>10}")
    # A: 翻转率 (sequence 应更低=更稳)
    ratio_flip = s["flip_rate"] / t["flip_rate"] if t["flip_rate"] > 0 else float('nan')
    print(f"  {'路由翻转率(欠定→高)':<28} | {t['flip_rate']:>10.4f} | {s['flip_rate']:>10.4f} | "
          f"{ratio_flip:>8.3f} | {'seq更稳' if s['flip_rate'] < t['flip_rate'] else 'token更稳':>10}")

    print("\n判定门 (v5.1 §7.1):")
    a_pass = s["flip_rate"] < t["flip_rate"]
    b_pass = s["final_infonce"] < t["final_infonce"] or s["steps_to_70pct"] < t["steps_to_70pct"]
    print(f"  Metric A (翻转率 seq<tok): {'PASS' if a_pass else 'FAIL'}")
    print(f"  Metric B' (InfoNCE 收敛 seq 更优): {'PASS' if b_pass else 'FAIL'}")
    if a_pass and b_pass:
        print("  → §7.1 PASS: sequence routing 验证为有效修复 (欠定被缓解)")
    elif a_pass or b_pass:
        print("  → §7.1 PARTIAL: 部分支持, 需更大规模验证")
    else:
        print("  → §7.1 FAIL: 小规模下 sequence routing 未显现优势 (scaling 可能关键)")


if __name__ == "__main__":
    main()
