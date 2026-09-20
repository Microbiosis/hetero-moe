"""方向多样性融合增益 (真实 TinyBERT 底座)。

命题: 4 个方向不同的真实模型可被本系统融合, 且优于单底座 / 简单平均。
底座: huawei-noah/TinyBERT_General_4L_312D (4L, D=312, D_FF=1200, ~11M)。
结构: TinyBERT 标准 BERT FFN: f(x) = output.dense(gelu(intermediate.dense(x)))
方向: 4 份最后一层 FFN, 各注入一个方向偏差 (模拟不同数据微调)。
任务: 4-类文本方向回归。
度量: per-class MSE, 比较 单底座 / 简单平均 / 融合。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init

D = 312
D_FF = 1200
M, K, N_CLS = 4, 1, 4
B, S, STEPS, LR = 32, 16, 60, 1e-2


def bert_ffn(x, w_inter, w_output):
    """TinyBERT 标准 FFN: output.dense(gelu(intermediate.dense(x)))"""
    h = F.linear(x, w_inter)
    return F.linear(F.gelu(h), w_output)


def make_direction_ffn(base_inter, base_output, cls_id):
    """从 TinyBERT FFN 权重注入方向偏差, 模拟不同数据微调。"""
    inter = base_inter.clone()
    output = base_output.clone()
    ds = cls_id * (D // N_CLS); de = (cls_id + 1) * (D // N_CLS)
    inter[:, ds:de] *= 1.5  # 放大该方向激活
    output[ds:de, :] *= 1.3  # 放大该方向输出
    return nn.Parameter(inter), nn.Parameter(output)


def load_tinybert():
    name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def build_fusion_layer(model):
    last = model.encoder.layer[-1]
    base_inter = last.intermediate.dense.weight  # [D_FF, D] = [1200, 312]
    base_output = last.output.dense.weight       # [D, D_FF] = [312, 1200]
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity="token")
    with torch.no_grad():
        for m in range(M):
            w_int, w_out = make_direction_ffn(base_inter, base_output, m)
            layer.w_gates[m].copy_(w_int)    # 复用为 gate (=intermediate)
            layer.w_ups[m].copy_(w_int)      # 复用为 up (=intermediate)
            layer.w_downs[m].copy_(w_out)    # output dense
    return layer


def run_one(seed):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] 加载 TinyBERT ...")
    model, tokenizer = load_tinybert()
    layer = build_fusion_layer(model)
    realistic_ffn_init(layer, D, D_FF)
    # 冻结适配器, 仅训 W_router
    with torch.no_grad():
        for m in range(M):
            layer.P_m[m].copy_(torch.eye(D))
            layer.gammas[m].fill_(1.0)
            layer.betas[m].fill_(0.0)
            layer.alphas[m].fill_(1.0)
    for m in range(M):
        for p in (layer.P_m[m], layer.gammas[m], layer.betas[m], layer.alphas[m]):
            p.requires_grad_(False)

    # 用 TinyBERT 编码 4 类文本
    class_texts = [
        ["cat dog bird", "animal pet wild", "feline canine fowl"],
        ["car truck bike", "vehicle motor engine", "transport drive"],
        ["happy sad angry", "emotion feel mood", "joy grief rage"],
        ["one two three", "number count math", "digit sum calc"],
    ]
    all_texts = [t for cls in class_texts for t in cls]
    while len(all_texts) < B:
        all_texts += [t for cls in class_texts for t in cls]
    all_texts = all_texts[:B]
    enc = tokenizer(all_texts, padding="max_length", truncation=True,
                    max_length=S, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    x = h.detach()

    # 目标: 每个类对应方向子空间 = 1
    targets = torch.zeros(B, S, D)
    for i in range(B):
        cls = i // (B // N_CLS)
        ds = cls * (D // N_CLS); de = (cls + 1) * (D // N_CLS)
        targets[i, :, ds:de] = 1.0

    # --- 训练 W_router (用正确 BERT FFN gelu, 非 SwiGLU) ---
    opt = torch.optim.AdamW([layer.W_router], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y, z, ah, ahs = layer(x)
        # 用正确 BERT FFN 重算输出 (替换层内 SwiGLU)
        out_sum = torch.zeros_like(x)
        x_det = x.detach()
        for m in range(M):
            h_aligned = x_det * layer.gammas[m] + layer.betas[m]
            f_m = bert_ffn(h_aligned, layer.w_gates[m], layer.w_downs[m])
            out_sum = out_sum + ah[..., m:m+1] * layer.alphas[m] * f_m
        y = x + out_sum
        loss = F.mse_loss(y, targets)
        loss.backward()
        opt.step()

    # 评估 (正确 BERT FFN)
    with torch.no_grad():
        y, _, ah, _ = layer(x)
        out_sum = torch.zeros_like(x)
        x_det = x.detach()
        for m in range(M):
            h_aligned = x_det * layer.gammas[m] + layer.betas[m]
            f_m = bert_ffn(h_aligned, layer.w_gates[m], layer.w_downs[m])
            out_sum = out_sum + ah[..., m:m+1] * layer.alphas[m] * f_m
        fuse_loss = F.mse_loss(x + out_sum, targets).item()

        # 简单平均
        total_f = torch.zeros_like(x)
        for m in range(M):
            total_f = total_f + bert_ffn(x_det, layer.w_gates[m], layer.w_downs[m])
        avg_loss = F.mse_loss(x + total_f / M, targets).item()

        # 单底座
        f0 = bert_ffn(x_det, layer.w_gates[0], layer.w_downs[0])
        single_loss = F.mse_loss(x + f0, targets).item()

    return fuse_loss, avg_loss, single_loss


def main():
    print(f"{'='*76}\n方向多样性融合增益 (真实 TinyBERT 底座)\n{'='*76}")
    print(f"底座: huawei-noah/TinyBERT_General_4L_312D (4L, D={D}, D_FF={D_FF})")
    print(f"任务: 4-类文本方向回归 | 4 底座各专精一类 | 仅训 W_router")
    print(f"\n{'seed':>5} | {'single MSE':>11} | {'avg MSE':>9} | {'fuse MSE':>9} | "
          f"{'增益 vs single':>14} | {'增益 vs avg':>12} | {'PASS':>4}")
    print("-" * 76)
    all_pass = True
    gains_s, gains_a = [], []
    for seed in range(5):
        fuse, avg, single = run_one(seed)
        gain_s = (single - fuse) / max(single, 1e-9) * 100
        gain_a = (avg - fuse) / max(avg, 1e-9) * 100
        gains_s.append(gain_s); gains_a.append(gain_a)
        ok = fuse < single and fuse < avg
        all_pass = all_pass and ok
        print(f"{seed:>5} | {single:>11.4f} | {avg:>9.4f} | {fuse:>9.4f} | "
              f"{gain_s:>+13.1f}% | {gain_a:>+11.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 76)
    import statistics
    print(f"\n融合增益 (5 seeds): vs single 均值 {statistics.mean(gains_s):+.1f}% | "
          f"vs avg 均值 {statistics.mean(gains_a):+.1f}%")
    if all_pass:
        print(f"{'='*76}")
        print("✅ 全部 seeds 通过: 融合 < single 且 融合 < avg")
        print("→ 4 个方向不同的真实 TinyBERT 底座可被本系统融合")
    else:
        print("❌ 部分 seeds 失败")


if __name__ == "__main__":
    main()
