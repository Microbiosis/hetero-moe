"""V8.0 全局中枢三路线 — 跨架构基准 (5 seeds)。

四种配置同时跑, 直接对比中枢贡献:
    baseline (无中枢):          v7 CrossArchAttnFFNFusionLayer
    C-1 (中枢 token):           CentralAugmentedFusionLayer + cw_attn/cw_ffn (但关 V_coop)
    C-2 (中枢 + attn→ffn 协同): CentralAugmentedFusionLayer 完整 (cw + V_coop)
    C-3 (层级中枢):             HierarchicalCentralLayer (cw + V_coop + central_expert)

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可

每个 seed 输出: 4 行的 MSE, 算相对 baseline 的增益.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from experiment_env import log_experiment_env

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, ViTModel, AutoTokenizer, ViTImageProcessor

from research._primitives.attention import AttnPool
from research.routing_evolution.v7_attention import CrossArchAttnFFNFusionLayer, V7Trainer
from research.routing_evolution.v8_central import (
    CentralAugmentedFusionLayer, HierarchicalCentralLayer, V8Trainer,
)

D_SHARED = 256
N_CLS = 3
B, S, STEPS, LR = 18, 16, 100, 1e-2
SEEDS = 5


# ---- 复用 v7-A 的 attention 抽取函数 ----
def extract_attn_bert(model):
    attn = model.encoder.layer[0].attention.self
    return (
        attn.query.weight.detach().clone().float(),
        attn.key.weight.detach().clone().float(),
        attn.value.weight.detach().clone().float(),
        torch.eye(attn.query.weight.shape[0]),
    ), 12


def extract_attn_llama(model):
    sa = model.layers[0].self_attn
    return (
        sa.q_proj.weight.detach().clone().float(),
        sa.k_proj.weight.detach().clone().float(),
        sa.v_proj.weight.detach().clone().float(),
        sa.o_proj.weight.detach().clone().float(),
    ), 12


def extract_attn_vit(model):
    attn = model.layers[0].attention
    return (
        attn.q_proj.weight.detach().clone().float(),
        attn.k_proj.weight.detach().clone().float(),
        attn.v_proj.weight.detach().clone().float(),
        attn.o_proj.weight.detach().clone().float(),
    ), 3


def load_encoders():
    print("  加载 HF 模型...")
    bert_tok = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    bert = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    w_bert, _ = extract_attn_bert(bert)

    llama_tok = AutoTokenizer.from_pretrained('nickypro/tinyllama-110M')
    llama_tok.pad_token = llama_tok.eos_token
    llama = AutoModel.from_pretrained('nickypro/tinyllama-110M')
    w_llama, _ = extract_attn_llama(llama)

    vit_proc = ViTImageProcessor.from_pretrained('WinKawaks/vit-tiny-patch16-224')
    vit = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
    w_vit, _ = extract_attn_vit(vit)

    return (
        (bert_tok, bert, 312, w_bert),
        (llama_tok, llama, 768, w_llama),
        (vit_proc, vit, 192, w_vit),
    )


def encode_text(texts, tok, model):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach().float()


def encode_code(code_texts, tok, model):
    enc = tok(code_texts, padding="max_length", truncation=True, max_length=S,
              return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach().float()


def encode_image(img_array, processor, model):
    img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
    pix = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(pix["pixel_values"])
    h = out.last_hidden_state[:, 1:, :][:, :S, :]
    return h.detach().float()


def make_pools(encoders):
    (_, _, D_bert, w_bert), (_, _, D_llama, w_llama), (_, _, D_vit, w_vit) = encoders
    return [
        AttnPool(D_bert,   D_SHARED, 12, *w_bert),
        AttnPool(D_llama,  D_SHARED, 12, *w_llama),
        AttnPool(D_vit,    D_SHARED, 3,  *w_vit),
    ]


def make_layer(mode, attn_pools, encoders):
    """mode ∈ {baseline, c1, c2, c3}"""
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    if mode == "baseline":
        return CrossArchAttnFFNFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
        )
    elif mode == "c1":
        # 只用 C-1: 中枢广播, 但关 V_coop
        layer = CentralAugmentedFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
        )
        layer.V_coop.data.zero_()   # 关协同
        return layer
    elif mode == "c2":
        return CentralAugmentedFusionLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
        )
    elif mode == "c3":
        return HierarchicalCentralLayer(
            d_shared=D_SHARED, d_ff=4 * D_SHARED,
            attn_pools=attn_pools, modal_dims=modal_dims,
        )
    else:
        raise ValueError(f"unknown mode {mode}")


def make_trainer(layer):
    """按 layer 类型选 trainer"""
    if isinstance(layer, CrossArchAttnFFNFusionLayer):
        return V7Trainer(layer, lr=LR, phase=2)
    else:
        return V8Trainer(layer, lr=LR, phase=2)


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] mode={mode} ...")

    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)

    layer = make_layer(mode, attn_pools, encoders)

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            perturbed = src + 0.1 * torch.randn(src.shape, generator=g)
            modal_seqs.append(perturbed)
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)

    # 投影到 D_shared
    x_shared_list = []
    for h, cls in zip(modal_seqs, modal_indices):
        x = h.unsqueeze(0)
        x_shared = F.linear(x, layer.P_m[cls])
        x_shared_list.append(x_shared)

    trainer = make_trainer(layer)
    for step in range(STEPS):
        trainer.step(x_shared_list, [t.unsqueeze(0) for t in targets])

    # 评估
    with torch.no_grad():
        fuse_loss, avg_loss = 0.0, 0.0
        for x_shared, t in zip(x_shared_list, targets):
            t_b = t.unsqueeze(0)
            y = layer(x_shared)
            fuse_loss += F.mse_loss(y, t_b).item()
            # avg 基线: 各 ffn expert 等权
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
        fuse_loss /= B; avg_loss /= B
    return fuse_loss, avg_loss


def main():
    _env = log_experiment_env("run_v8_full")
    print(f"  时间:       {_env['timestamp']}")
    print(f"  Python:     {_env['python_version'].split()[0]}")
    print(f"  PyTorch:    {_env['pytorch_version']}")
    print(f"  CUDA:       {_env['cuda_available']} ({_env['cuda_version']})")
    print(f"  设备:       {_env['device']}")
    print(f"  CPU 核心:   {_env['cpu_count']}")
    print("=" * 78)
    print()

    print("=" * 78)
    print("V8.0 全局中枢三路线 — 跨架构基准")
    print("=" * 78)
    print("baseline (无中枢) / C-1 (中枢 token) / C-2 (+ attn→ffn 协同) / C-3 (+ 层级中枢)")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    modes = ["baseline", "c1", "c2", "c3"]
    # 结果矩阵: results[mode] = list of (fuse, avg) per seed
    results = {m: [] for m in modes}

    print(f"\n每个 mode × {SEEDS} seeds")
    print("-" * 78)
    for mode in modes:
        for seed in range(SEEDS):
            fuse, avg = run_seed(mode, seed, encoders, attn_pools)
            results[mode].append((fuse, avg))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>10} | {'fuse MSE':>9} | {'avg MSE':>9} | {'vs baseline':>13} | {'vs avg':>8} | {'PASS':>4}")
    print("-" * 78)
    base_fuse_mean = statistics.mean([f for f, _ in results["baseline"]])
    for mode in modes:
        fuse_mean = statistics.mean([f for f, _ in results[mode]])
        avg_mean  = statistics.mean([a for _, a in results[mode]])
        gain_v_base = (base_fuse_mean - fuse_mean) / max(base_fuse_mean, 1e-9) * 100
        gain_v_avg  = (avg_mean - fuse_mean) / max(avg_mean, 1e-9) * 100
        ok = fuse_mean < avg_mean
        print(f"{mode:>10} | {fuse_mean:>9.4f} | {avg_mean:>9.4f} | {gain_v_base:>+12.1f}% | "
              f"{gain_v_avg:>+7.1f}% | {'✅' if ok else '❌':>4}")
    print("=" * 78)
    print()
    print("判定:")
    print(f"  C-1 中枢贡献: {base_fuse_mean - statistics.mean([f for f, _ in results['c1']]):.4f} MSE 降低 vs baseline")
    print(f"  C-2 协同贡献: {statistics.mean([f for f, _ in results['c1']]) - statistics.mean([f for f, _ in results['c2']]):.4f}")
    print(f"  C-3 层级贡献: {statistics.mean([f for f, _ in results['c2']]) - statistics.mean([f for f, _ in results['c3']]):.4f}")
    print()
    print("✅ V8.0 验证完成 — 详见 docs/V8_0_SPEC.md")


if __name__ == "__main__":
    main()
