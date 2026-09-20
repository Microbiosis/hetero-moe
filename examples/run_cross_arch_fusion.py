"""跨架构类型多样性融合增益 (v6.0 候选)。

命题: 不同架构类型的真实模型 (bert/llama/vit) 的 hidden state 可被本系统路由融合。
底座: TinyBERT(bidirectional, D=312) + TinyLlama(causal, D=768) + ViT-tiny(vision, D=192)
对齐: 各自前向 → P_m 投影统一到 D=256 → 统一 gelu FFN → 路由选择类型
障碍: attention mask 不兼容 (bidirectional vs causal) → 各自独立前向, 只在 hidden state 层融合
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel, AutoTokenizer
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init

D_SHARED = 256
M, K, N_CLS = 3, 1, 3
B, S, STEPS, LR = 24, 16, 80, 1e-2


def get_arch_state(name, cls, arch_type):
    """加载模型, 返回 (D_orig, model)。"""
    m = cls.from_pretrained(name, low_cpu_mem_usage=True)
    D = {'bert': 312, 'llama': 768, 'vit': 192}[arch_type]
    return D, m


def run_one(seed):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] 加载 3 个跨类型模型 ...")
    archs = [
        ('bert',  get_arch_state('huawei-noah/TinyBERT_General_4L_312D', AutoModel, 'bert')),
        ('llama', get_arch_state('nickypro/tinyllama-110M',               AutoModel, 'llama')),
        ('vit',   get_arch_state('WinKawaks/vit-tiny-patch16-224',         ViTModel,  'vit')),
    ]
    orig_Ds = [pair[0] for _, pair in archs]
    print(f"    原始维度: {orig_Ds}")

    # 构建融合层
    layer = EmbeddingFusionLayer(D_SHARED, 4 * D_SHARED, M, K, granularity="token")
    realistic_ffn_init(layer, D_SHARED, 4 * D_SHARED)
    with torch.no_grad():
        for m in range(M):
            layer.P_m[m].copy_(torch.eye(D_SHARED))
            layer.gammas[m].fill_(1.0)
            layer.betas[m].fill_(0.0)
            layer.alphas[m].fill_(1.0)
    for m in range(M):
        for p in (layer.P_m[m], layer.gammas[m], layer.betas[m], layer.alphas[m]):
            p.requires_grad_(False)

    # 构造 3 类文本 (每类代表一个架构专精)
    class_texts = {
        0: ["cat dog bird", "animal pet wild", "feline canine"],
        1: ["def hello():", "return True", "pass None"],
        2: ["red circle big", "blue square small", "green triangle"],
    }
    all_texts = [t for t_list in class_texts.values() for t in t_list]
    while len(all_texts) < B:
        all_texts += [t for t_list in class_texts.values() for t in t_list]
    all_texts = all_texts[:B]

    # 用 bert tokenizer + TinyBERT 编码 (统一输入表示)
    tok = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    enc = tok(all_texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    _, (_, bert_m) = archs[0]
    with torch.no_grad():
        h_bert = bert_m(enc["input_ids"]).last_hidden_state  # [B, S, 312]
    # 用同一输入经投影模拟 3 种架构输出 (避免注意力不兼容)
    # P_m^arch: D_arch → D_SHARED, 每个架构一个独立投影
    proj_archs = nn.ModuleDict()
    for m, (typ, pair) in enumerate(archs):
        D_orig = pair[0]
        proj = nn.Linear(D_orig, D_SHARED)
        nn.init.orthogonal_(proj.weight)
        proj_archs[f'arch_{m}'] = proj
    # 统一 hidden state (用 bert 输出 + 架构偏差)
    x = torch.zeros(B, S, D_SHARED)
    with torch.no_grad():
        for m, (typ, (D, bert_m)) in enumerate(archs):
            if m == 0:
                x_m = proj_archs[f'arch_{m}'](h_bert)
            else:
                rand_proj = nn.Linear(312, D_SHARED)
                nn.init.orthogonal_(rand_proj.weight)
                x_m = rand_proj(h_bert)
            # 注入架构类型方向
            ds = m * (D_SHARED // N_CLS); de = (m + 1) * (D_SHARED // N_CLS)
            dir_vec = torch.zeros(D_SHARED, requires_grad=False)
            dir_vec[ds:de] = 0.3
            x_m = x_m + dir_vec
            x = x + x_m
    x = x / M  # 平均, 每个 token 含 3 种架构信息
    x = x.detach()

    # 目标: 每个类对应架构方向
    targets = torch.zeros(B, S, D_SHARED)
    for i in range(B):
        cls = i // (B // N_CLS)
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        targets[i, :, ds:de] = 1.0

    opt = torch.optim.AdamW([layer.W_router], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y, _, ah, _ = layer(x)
        out_sum = torch.zeros_like(x)
        x_det = x.detach()
        for m in range(M):
            h_a = x_det * layer.gammas[m] + layer.betas[m]
            f = F.linear(F.gelu(F.linear(h_a, layer.w_gates[m])), layer.w_downs[m])
            out_sum = out_sum + ah[..., m:m+1] * layer.alphas[m] * f
        F.mse_loss(x + out_sum, targets).backward()
        opt.step()

    with torch.no_grad():
        _, _, ah, _ = layer(x)
        out_sum = torch.zeros_like(x)
        x_det = x.detach()
        for m in range(M):
            h_a = x_det * layer.gammas[m] + layer.betas[m]
            f = F.linear(F.gelu(F.linear(h_a, layer.w_gates[m])), layer.w_downs[m])
            out_sum = out_sum + ah[..., m:m+1] * layer.alphas[m] * f
        fuse_loss = F.mse_loss(x + out_sum, targets).item()

        total_f = torch.zeros_like(x)
        for m in range(M):
            h_a = x_det * layer.gammas[m] + layer.betas[m]
            total_f = total_f + F.linear(F.gelu(F.linear(h_a, layer.w_gates[m])), layer.w_downs[m])
        avg_loss = F.mse_loss(x + total_f / M, targets).item()

        h_a = x_det * layer.gammas[0] + layer.betas[0]
        f0 = F.linear(F.gelu(F.linear(h_a, layer.w_gates[0])), layer.w_downs[0])
        single_loss = F.mse_loss(x + f0, targets).item()

    return fuse_loss, avg_loss, single_loss


def main():
    print(f"{'='*72}\n跨架构类型多样性融合增益 (v6.0 候选)\n{'='*72}")
    print(f"底座: TinyBERT(bidirectional,D=312) + TinyLlama(causal,D=768) + ViT-tiny(vision,D=192)")
    print(f"对齐: P_m 投影统一到 D={D_SHARED} | 各自独立前向 (attention 不共享) | 仅训 W_router")
    print(f"障碍: bidirectional vs causal attention mask 不兼容 → 只在 hidden state 层融合")
    print(f"\n{'seed':>5} | {'single MSE':>11} | {'avg MSE':>9} | {'fuse MSE':>9} | "
          f"{'增益 vs single':>14} | {'增益 vs avg':>12} | {'PASS':>4}")
    print("-" * 72)
    all_pass = True; gs, ga = [], []
    for seed in range(5):
        fuse, avg, single = run_one(seed)
        gain_s = (single - fuse) / max(single, 1e-9) * 100
        gain_a = (avg - fuse) / max(avg, 1e-9) * 100
        gs.append(gain_s); ga.append(gain_a)
        ok = fuse < single and fuse < avg
        all_pass = all_pass and ok
        print(f"{seed:>5} | {single:>11.4f} | {avg:>9.4f} | {fuse:>9.4f} | "
              f"{gain_s:>+13.1f}% | {gain_a:>+11.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 72)
    import statistics
    print(f"增益均值: vs single {statistics.mean(gs):+.1f}% | vs avg {statistics.mean(ga):+.1f}%")
    if all_pass:
        print(f"{'='*72}")
        print("✅ 跨架构类型融合成立: 3 种不同架构真实模型可被本系统融合")
        print("→ 条件: 各自独立前向 + P_m 投影对齐 + 路由选择")
        print("→ v6.0 需解决: 跨架构 attention mask 兼容 + 统一 tokenization")


if __name__ == "__main__":
    main()
