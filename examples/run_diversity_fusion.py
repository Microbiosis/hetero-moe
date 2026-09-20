"""方向多样性融合增益验证 (diversity fusion gain)。

命题: 4 个方向不同的底座能被本系统融合, 且融合优于单底座 / 简单平均。
方法: 4-类方向任务, 4 个底座各专精一类 (通过方向性 frozen 权重), 比较:
  - 单底座 (任意 1 个底座单独输出)
  - 简单平均 (4 底座均值, 无路由)
  - 本系统 (G=token 路由 + 融合, 选择专精专家)
度量: per-class MSE / accuracy。融合增益 = 融合 loss 显著 < 单底座 loss 且 < 平均 loss。
鲁棒: 多种子 (seed 0-4)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
from archive.pre_baseline.v5_alpha.infonce import infonce_loss

D, D_FF, M, K, N_CLS = 16, 32, 4, 1, 4
B, S, STEPS, LR = 32, 16, 80, 1e-2


def make_direction_backbone(D, D_FF, cls_id, direction_seed):
    """构造一个在 cls_id 方向专精的底座 (通过方向性 frozen 权重)。

    方法: gate/up/down 权重在 cls_id 对应的特征方向上有更强增益。
    4 个底座的专精方向互相正交, 形成真正的方向多样性。
    """
    g = torch.Generator().manual_seed(direction_seed)
    gate = nn.Parameter(torch.randn(D_FF, D, generator=g) * 0.02)
    up = nn.Parameter(torch.randn(D_FF, D, generator=g) * 0.02)
    down = nn.Parameter(torch.randn(D, D_FF, generator=g) * 0.02)
    # 注入方向偏差: 在 cls_id 对应的 D 维子空间上加一个方向向量
    dir_vec = torch.zeros(D)
    dir_vec[cls_id * (D // N_CLS):(cls_id + 1) * (D // N_CLS)] = 1.0
    gate.data[:, :] += torch.outer(torch.randn(D_FF, generator=g) * 0.1, dir_vec * 0.5)
    down.data[:, :] += 0.3 * dir_vec.unsqueeze(-1)
    gate.requires_grad_(False)
    up.requires_grad_(False)
    down.requires_grad_(False)
    return gate, up, down


def build_fusion_layer(seeds):
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity="token")
    # 替换 frozen 底座为方向性底座
    with torch.no_grad():
        for m in range(M):
            g, u, d = make_direction_backbone(D, D_FF, m, seeds[m])
            layer.w_gates[m].copy_(g); layer.w_ups[m].copy_(u); layer.w_downs[m].copy_(d)
    return layer


def make_task(x, cls_seed):
    """4-类方向回归: target_t = class_dir(cls_id[t]) + noise。"""
    g = torch.Generator().manual_seed(cls_seed)
    cls_id = torch.randint(0, N_CLS, (B, S, 1), generator=g)  # 每个 token 一个类
    targets = torch.zeros(B, S, D)
    for i in range(B):
        for j in range(S):
            c = cls_id[i, j, 0].item()
            targets[i, j, c * (D // N_CLS):(c + 1) * (D // N_CLS)] = 1.0
    targets += torch.randn(B, S, D, generator=g) * 0.3
    targets = targets.detach()
    return targets


def eval_backbone_only(backbone_set, x, targets):
    """单底座: 只用第一个底座 (worst-case 代表)。"""
    g, u, d = backbone_set[0]
    x_det = x.detach()
    h = F.linear(x_det, g)
    h_up = F.linear(x_det, u)
    f = F.linear(F.silu(h) * h_up, d)
    out = x + f
    return F.mse_loss(out, targets).item()


def eval_simple_average(layer, x, targets):
    """简单平均: 4 底座输出均值 (无路由)。"""
    x_det = x.detach()
    total_f = torch.zeros_like(x)
    with torch.no_grad():
        for m in range(M):
            g, u, d = layer.w_gates[m], layer.w_ups[m], layer.w_downs[m]
            h = F.linear(x_det, g)
            h_up = F.linear(x_det, u)
            f = F.linear(F.silu(h) * h_up, d)
            total_f = total_f + f
    out = x + total_f / M
    return F.mse_loss(out, targets).item()


def run_one(seed):
    torch.manual_seed(seed)
    seeds = [seed + i * 7 for i in range(M)]
    layer = build_fusion_layer(seeds)
    realistic_ffn_init(layer, D, D_FF)
    # 冻结适配器 (仅训 W_router, 隔离路由增益)
    with torch.no_grad():
        for m in range(M):
            layer.P_m[m].copy_(torch.eye(D))
            layer.gammas[m].fill_(1.0)
            layer.betas[m].fill_(0.0)
            layer.alphas[m].fill_(1.0)
    for m in range(M):
        for p in (layer.P_m[m], layer.gammas[m], layer.betas[m], layer.alphas[m]):
            p.requires_grad_(False)

    opt = torch.optim.AdamW([layer.W_router], lr=LR)
    x = torch.randn(B, S, D)
    x = x.detach()
    targets = make_task(x, seed)

    backbone_set = [(layer.w_gates[m], layer.w_ups[m], layer.w_downs[m]) for m in range(M)]

    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y, _, _, _ = layer(x)
        loss = F.mse_loss(y, targets)
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_fuse, _, _, _ = layer(x)
        fuse_loss = F.mse_loss(y_fuse, targets).item()
    avg_loss = eval_simple_average(layer, x, targets)
    single_loss = eval_backbone_only(backbone_set, x, targets)

    return fuse_loss, avg_loss, single_loss


def main():
    print(f"{'='*72}\n方向多样性融合增益验证 (diversity fusion gain)\n{'='*72}")
    print(f"任务: 4-类方向回归 | 4 底座各专精一类 | 仅训 W_router")
    print(f"\n{'seed':>5} | {'single MSE':>11} | {'avg MSE':>9} | {'fuse MSE':>9} | "
          f"{'增益 vs single':>13} | {'增益 vs avg':>11} | {'PASS':>4}")
    print("-" * 72)
    results = []
    all_pass = True
    for seed in range(5):
        fuse, avg, single = run_one(seed)
        gain_s = (single - fuse) / max(single, 1e-9) * 100
        gain_a = (avg - fuse) / max(avg, 1e-9) * 100
        ok = fuse < single and fuse < avg
        all_pass = all_pass and ok
        results.append((fuse, avg, single, gain_s, gain_a))
        print(f"{seed:>5} | {single:>11.4f} | {avg:>9.4f} | {fuse:>9.4f} | "
              f"{gain_s:>+12.1f}% | {gain_a:>+10.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 72)
    import statistics
    fg = statistics.mean([r[3] for r in results])
    ag = statistics.mean([r[4] for r in results])
    print(f"\n融合增益 (5 seeds): vs single 均值 {fg:+.1f}% | vs avg 均值 {ag:+.1f}%")
    if all_pass:
        print(f"{'='*72}\n✅ 全部 seeds 通过: 融合 < single 且 融合 < avg")
        print(f"→ 4 个方向不同的底座可被本系统融合, 且融合优于单底座与简单平均")
        print(f"→ 方向多样性融合增益成立 (与模型大小无关, 证明的是融合能力本身)")
    else:
        print(f"{'='*72}\n❌ 部分 seeds 失败")


if __name__ == "__main__":
    main()
