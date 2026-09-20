"""V7.0 跨架构族 Attn + FFN 双路由 — 跨架构基准 (v7-A)。

复用 v6 的 HF 加载器与编码路径, 在原生 attention 层抽取 Q/K/V/O 矩阵,
构建 AttnPool, 接入 v7.0 双路由。

v7-A 相对 v6 的关键差异:
    - 新增 attn 通路 (AttnPool × 3)
    - 双路由独立选择 (W_router_attn, W_router_ffn)
    - per-expert param group (V7Trainer)
    - 评估块新增 attn_only 第 4 组对照

用法: bash run.sh v7a
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, ViTModel, AutoTokenizer, ViTImageProcessor

from research.routing_evolution.v7_attention import (
    AttnPool, CrossArchAttnFFNFusionLayer, V7Trainer,
)

D_SHARED = 256
N_CLS = 3
B, S, STEPS, LR = 18, 16, 100, 1e-2
SEEDS = 5


# ---- 抽取底座原生 attn Q/K/V/O ----

def extract_attn_bert(model) -> tuple:
    """TinyBERT / Bert-like: encoder.layer[0].attention.self.{query,key,value}"""
    attn = model.encoder.layer[0].attention.self
    return (
        attn.query.weight.detach().clone().float(),   # [D, D]
        attn.key.weight.detach().clone().float(),
        attn.value.weight.detach().clone().float(),
        # TinyBERT 没有显式 o_proj, 用单位矩阵代替 (P_m 投影会吸收这部分)
        torch.eye(attn.query.weight.shape[0]),
    ), 12  # num_heads


def extract_attn_llama(model) -> tuple:
    """TinyLlama / Llama-like: layers[0].self_attn.{q_proj,k_proj,v_proj,o_proj}"""
    sa = model.layers[0].self_attn
    return (
        sa.q_proj.weight.detach().clone().float(),
        sa.k_proj.weight.detach().clone().float(),
        sa.v_proj.weight.detach().clone().float(),
        sa.o_proj.weight.detach().clone().float(),
    ), 12


def extract_attn_vit(model) -> tuple:
    """ViT: layers[0].attention.{q_proj,k_proj,v_proj,o_proj}"""
    attn = model.layers[0].attention
    return (
        attn.q_proj.weight.detach().clone().float(),
        attn.k_proj.weight.detach().clone().float(),
        attn.v_proj.weight.detach().clone().float(),
        attn.o_proj.weight.detach().clone().float(),
    ), 3


def load_encoders_and_attn():
    print("  加载 TinyBERT (text, attn 4 heads -> 12 heads)...")
    bert_tok = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    bert = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    bert_w, bert_heads = extract_attn_bert(bert)

    print("  加载 TinyLlama (code, attn 12 heads)...")
    llama_tok = AutoTokenizer.from_pretrained('nickypro/tinyllama-110M')
    llama_tok.pad_token = llama_tok.eos_token
    llama = AutoModel.from_pretrained('nickypro/tinyllama-110M')
    llama_w, llama_heads = extract_attn_llama(llama)

    print("  加载 ViT-tiny (image, attn 3 heads)...")
    vit_proc = ViTImageProcessor.from_pretrained('WinKawaks/vit-tiny-patch16-224')
    vit = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
    vit_w, vit_heads = extract_attn_vit(vit)

    return (
        (bert_tok, bert, 312, bert_w, bert_heads),
        (llama_tok, llama, 768, llama_w, llama_heads),
        (vit_proc, vit, 192, vit_w, vit_heads),
    )


# ---- 编码 (复用 v6) ----

def encode_text(texts, tok, model):
    """texts: list[str] → hidden [len(texts), S, D] (float32)"""
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach().float()  # [B_text, S, 312]


def encode_code(code_texts, tok, model):
    enc = tok(code_texts, padding="max_length", truncation=True, max_length=S,
              return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach().float()  # [B_code, S, 768]


def encode_image(img_array, processor, model):
    img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
    pix = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(pix["pixel_values"])
    h = out.last_hidden_state[:, 1:, :][:, :S, :]  # [1, S, 192]
    return h.detach().float()  # [1, S, 192]


def build_layer(encoders):
    (_, _, D_bert, w_bert, h_bert), \
    (_, _, D_llama, w_llama, h_llama), \
    (_, _, D_vit, w_vit, h_vit) = encoders
    attn_pools = [
        AttnPool(D_bert,   D_SHARED, h_bert,   *w_bert),
        AttnPool(D_llama,  D_SHARED, h_llama,  *w_llama),
        AttnPool(D_vit,    D_SHARED, h_vit,    *w_vit),
    ]
    return CrossArchAttnFFNFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools,
        modal_dims=[D_bert, D_llama, D_vit],
        top_k_attn=1, top_k_ffn=1,
    )


def run_one(seed, encoders):
    torch.manual_seed(seed)
    (bert_tok, bert, D_bert, _, _), \
    (llama_tok, llama, D_llama, _, _), \
    (vit_proc, vit, D_vit, _, _) = encoders
    print(f"  [seed {seed}] 编码 + 构建 v7 双路由层 + 训练...")

    # 1. 编码每模态 (注意: encode_* 已不再 squeeze 0)
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)         # [3, S, 312]
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)       # [3, S, 768]
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)            # [1, S, 192]

    # 2. 构建 v7 双路由层
    layer = build_layer(encoders)

    # 3. 每模态 6 个样本, 共 B=18; 每样本单独 [S, D_m] (后续 unsqueeze 0 加 batch)
    modal_seqs = []
    modal_indices = []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]                    # [S, D_m]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            perturbed = src + 0.1 * torch.randn(src.shape, generator=g)
            modal_seqs.append(perturbed)
            modal_indices.append(cls)

    # 4. 目标: 每模态一个 class direction
    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED)
        t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    assert len(targets) == B

    # 5. 投影到 D_shared
    x_shared_list = []
    for h, cls in zip(modal_seqs, modal_indices):
        x = h.unsqueeze(0)  # [1, S, D_m]
        x_shared = F.linear(x, layer.P_m[cls])  # [1, S, D_SHARED]
        x_shared_list.append(x_shared)

    # 6. 训练
    trainer = V7Trainer(layer, lr=LR, phase=2)
    for step in range(STEPS):
        trainer.step(x_shared_list, [t.unsqueeze(0) for t in targets])

    # 7. 评估 (4 组)
    with torch.no_grad():
        fuse_loss, attn_only_loss, ffn_only_loss, avg_loss = 0.0, 0.0, 0.0, 0.0
        for i, (x_shared, t) in enumerate(zip(x_shared_list, targets)):
            t_b = t.unsqueeze(0)
            # 7.1 fuse (attn + ffn)
            y_fuse = layer(x_shared)
            fuse_loss += F.mse_loss(y_fuse, t_b).item()
            # 7.2 attn_only (强制 ffn 通路不贡献)
            attn_sum, _ = layer._attn_path(x_shared)
            y_attn = x_shared + attn_sum
            attn_only_loss += F.mse_loss(y_attn, t_b).item()
            # 7.3 ffn_only (强制 attn 通路不贡献)
            ffn_sum, _ = layer._ffn_path(x_shared)
            y_ffn = x_shared + ffn_sum
            ffn_only_loss += F.mse_loss(y_ffn, t_b).item()
            # 7.4 avg (所有 ffn 专家等权, 无路由 STE)
            x_det = x_shared.detach()
            total_f = torch.zeros_like(x_shared)
            for m in range(layer.num_experts_ffn):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                gate = F.linear(h_a, layer.w_gates[m])
                up   = F.linear(h_a, layer.w_ups[m])
                f    = F.linear(F.silu(gate) * up, layer.w_downs[m])
                total_f = total_f + f
            y_avg = x_shared + total_f / layer.num_experts_ffn
            avg_loss += F.mse_loss(y_avg, t_b).item()
        fuse_loss /= B; attn_only_loss /= B; ffn_only_loss /= B; avg_loss /= B

    return fuse_loss, attn_only_loss, ffn_only_loss, avg_loss


def main():
    print("=" * 78)
    print("V7.0 跨架构族 Attn + FFN 双路由 — v7-A 跨架构基准")
    print("=" * 78)
    print("Text→TinyBERT(D=312) + Code→TinyLlama(D=768) + Image→ViT-tiny(D=192)")
    print("每模态原生 Q/K/V/O + P_m 矩形投影 → 共享 D_shared=256")
    print("双路由: W_router_attn + W_router_ffn, 每 token 独立选 1 attn + 1 ffn")
    print("per-expert param group (V7Trainer)")
    print()
    encoders = load_encoders_and_attn()

    print(f"\n{'seed':>5} | {'fuse MSE':>9} | {'attn_only':>10} | {'ffn_only':>9} | "
          f"{'avg MSE':>9} | {'vs avg':>8} | {'PASS':>4}")
    print("-" * 78)
    all_pass = True
    gs = []
    ao_sum = fo_sum = 0.0
    for seed in range(SEEDS):
        fuse, attn_only, ffn_only, avg = run_one(seed, encoders)
        ao_sum += attn_only
        fo_sum += ffn_only
        gain = (avg - fuse) / max(avg, 1e-9) * 100
        gs.append(gain)
        ok = fuse < avg
        all_pass = all_pass and ok
        print(f"{seed:>5} | {fuse:>9.4f} | {attn_only:>10.4f} | {ffn_only:>9.4f} | "
              f"{avg:>9.4f} | {gain:>+7.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 78)
    print(f"\n增益均值 vs avg: {statistics.mean(gs):+.1f}%  "
          f"(范围 [{min(gs):+.1f}%, {max(gs):+.1f}%])")
    print(f"attn_only 均值: {ao_sum/SEEDS:.4f} | ffn_only 均值: {fo_sum/SEEDS:.4f}")
    print("=" * 78)
    if all_pass:
        print("✅ V7.0 双路由成立: Attn + FFN 独立选择, 融合 MSE < avg 基线")
        print("→ 关键: 每模态原生 attention + 矩形 P_m + per-expert param group")
    else:
        print("❌ 部分 seeds 失败 — 需分析 attn/ffn 通路贡献")


if __name__ == "__main__":
    main()
