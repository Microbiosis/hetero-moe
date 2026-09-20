"""V6.0 — 跨架构族异构模态融合 (Cross-architecture Heterogeneous Modal Fusion)。

核心命题: 不同架构类型的模型 (encoder / decoder / vision) 可在各自独立前向后,
在隐藏态层通过 P_m 投影对齐, 被共享路由融合, 且优于单模态 / 简单平均。

v6.0 架构:
  Text  → TinyBERT (bidirectional, D=312) → h_text  → P_text → h_text'
  Code  → TinyLlama (causal, D=768)      → h_code  → P_code → h_code'
  Image → ViT-tiny (vision, D=192)        → h_img   → P_img  → h_img'
                                                                       ↓
                                                           [B, S, D_shared]
                                                                       ↓
                                                  Shared Fusion Router (top-k STE)
                                                                       ↓
                                                           Shared FFN + Output Head

关键设计决策 (v6.0 vs v5.1):
  1. 每模态用原生架构独立前向 (无共享 attention)
  2. 每模态独立 tokenizer
  3. P_m 投影对齐 D_m → D_shared (矩形投影, 非方阵)
  4. 共享路由在 D_shared 空间选择
  5. 多模态 loss: 每模态一个 class direction, 路由选对应模态

解决 v5.1 跨架构测试的负结果 (+4%):
  上次用统一 BERT 编码模拟跨类型 → 抹平了类型差异 → 路由价值趋零
  本次用每模态原生编码 → 保留真正的架构差异 → 路由可识别模态类型
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModel, ViTModel, AutoTokenizer, ViTImageProcessor
from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.router import SparseRouterSTE

D_SHARED = 256
M, K, N_CLS = 3, 1, 3  # 3 模态, 每模态 1 类方向
B, S, STEPS, LR = 18, 16, 100, 1e-2


def load_encoders():
    """加载 3 个跨架构编码器 + 各自 tokenizer/processor。"""
    print("  加载 TinyBERT (text)...")
    bert_tok = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    bert = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

    print("  加载 TinyLlama (code)...")
    llama_tok = AutoTokenizer.from_pretrained('nickypro/tinyllama-110M')
    llama_tok.pad_token = llama_tok.eos_token
    llama = AutoModel.from_pretrained('nickypro/tinyllama-110M')

    print("  加载 ViT-tiny (image)...")
    vit_proc = ViTImageProcessor.from_pretrained('WinKawaks/vit-tiny-patch16-224')
    vit = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')

    return (bert_tok, bert, 312), (llama_tok, llama, 768), (vit_proc, vit, 192)


def encode_text(texts, tok, model):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach().squeeze(0)  # [S, 312]


def encode_code(code_texts, tok, model):
    enc = tok(code_texts, padding="max_length", truncation=True, max_length=S,
              return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach().squeeze(0)  # [S, 768]


def encode_image(img_array, processor, model):
    img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
    pix = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(pix["pixel_values"])
    h = out.last_hidden_state[0, 1:, :].unsqueeze(0)  # [1, N, D]
    h = h[:, :S, :]
    return h.detach().squeeze(0)  # [S, 192]


class CrossModalFusionLayer(nn.Module):
    """V6.0 跨模态融合层。

    每模态独立 P_m: D_m → D_shared (矩形投影)。
    共享 W_router 在 D_shared 空间做 token 级路由。
    3 个专家 = 3 个模态方向, γ/β/α 可训练对齐参数。
    """

    def __init__(self, d_shared: int, d_ff: int, num_experts: int, top_k: int,
                 modal_dims: list, quant_bits: int = 4, quant_group_size: int = 128):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.modal_dims = modal_dims

        # ---- 每模态矩形投影 P_m: D_m → D_shared (正交初始化) ----
        self.P_m = nn.ParameterList([nn.Parameter(torch.empty(d_shared, D_m)) for D_m in modal_dims])
        for m in range(len(modal_dims)):
            nn.init.orthogonal_(self.P_m[m])

        # ---- 每专家 SwiGLU FFN ----
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02) for _ in range(num_experts)])
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02) for _ in range(num_experts)])
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_shared, d_ff) * 0.02) for _ in range(num_experts)])
        for pl in (self.w_gates, self.w_ups, self.w_downs):
            for p in pl:
                p.requires_grad = False

        # ---- 共享对齐参数 γ, β, α (每专家, 可训练) ----
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_shared)) for _ in range(num_experts)])
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_shared)) for _ in range(num_experts)])
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])

        # ---- 共享路由 W_router: D_shared → M ----
        self.W_router = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)

    def _swiglu(self, x, w_gate, w_up, w_down):
        gate = F.linear(x, w_gate)
        up = F.linear(x, w_up)
        return F.linear(F.silu(gate) * up, w_down)

    def _compute_output(self, x_shared):
        """计算融合输出 (可复用于训练+评估)"""
        x_det = x_shared.detach()
        z = F.linear(x_det, self.W_router)  # [B, S, M]
        alpha = F.softmax(z, dim=-1)
        alpha_hat = SparseRouterSTE.apply(alpha, self.top_k)

        out_sum = torch.zeros_like(x_shared)
        for m in range(self.num_experts):
            h_aligned = x_det * self.gammas[m] + self.betas[m]
            f_m = self._swiglu(h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_m_q = FakeQuantSTE.apply(f_m, self.quant_bits, self.quant_group_size)
            out_sum = out_sum + alpha_hat[..., m:m+1] * self.alphas[m] * f_m_q
        return x_shared + out_sum, alpha_hat

    def forward(self, x_shared: torch.Tensor):
        """前向: x_shared [B, S, D_shared] → y [B, S, D_shared]"""
        y, _ = self._compute_output(x_shared)
        return y

    def forward_with_routing(self, x_shared):
        """返回 (y, alpha_hat) 用于分析"""
        return self._compute_output(x_shared)


def run_one(seed, encoders):
    torch.manual_seed(seed)
    (bert_tok, bert, D_bert), (llama_tok, llama, D_llama), (vit_proc, vit, D_vit) = encoders
    print(f"  [seed {seed}] 编码 3 模态 + 训练...")

    # 1. 编码每模态的输入 (各自原生架构)
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)  # [S, 312]

    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)  # [S, 768]

    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)  # [S, 192]

    # 2. 构建跨模态融合层
    modal_dims = [D_bert, D_llama, D_vit]
    layer = CrossModalFusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims)

    # 冻结 P_m (只训 W_router + γ/β/α)
    for m in range(M):
        layer.P_m[m].requires_grad_(False)

    # 3. 构造多模态 batch: 每模态 6 个样本, 共 B=18
    modal_seqs = []
    modal_indices = []
    for cls, (h, D_m) in enumerate([(h_text, D_bert), (h_code, D_llama), (h_img, D_vit)]):
        for i in range(6):
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            perturbed = h + 0.1 * torch.randn(S, D_m, generator=g)
            modal_seqs.append(perturbed)
            modal_indices.append(cls)

    # 4. 构建目标: 每模态一个 class direction
    targets = torch.zeros(B, S, D_SHARED)
    for i in range(B):
        cls = i // 6
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        targets[i, :, ds:de] = 1.0

    # 5. 训练: W_router + γ/β/α
    trainable_params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    opt = torch.optim.AdamW(trainable_params, lr=LR)

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        losses = []
        for i, (xi, mod_idx) in enumerate(zip(modal_seqs, modal_indices)):
            xi = xi.unsqueeze(0)  # [1, S, D_m]
            xi_shared = F.linear(xi, layer.P_m[mod_idx])  # [1, S, D_SHARED]
            y = layer(xi_shared)  # 融合输出 [1, S, D_SHARED]
            loss = F.mse_loss(y, targets[i].unsqueeze(0))
            losses.append(loss)
        total = sum(losses) / len(losses)
        total.backward()
        opt.step()

    # 6. 评估: 融合 / 单模态 / 平均
    with torch.no_grad():
        fuse_loss = 0.0
        single_loss = 0.0
        avg_loss = 0.0

        for i, (xi, mod_idx) in enumerate(zip(modal_seqs, modal_indices)):
            xi = xi.unsqueeze(0)
            xi_shared = F.linear(xi, layer.P_m[mod_idx])  # [1, S, D_SHARED]
            x_det = xi_shared.detach()

            # --- 融合: 用 layer.forward() ---
            y_fuse = layer(xi_shared)
            fuse_loss += F.mse_loss(y_fuse, targets[i].unsqueeze(0)).item()

            # --- 单模态: 只用专家 0 的 FFN, 无路由 ---
            h_a0 = x_det * layer.gammas[0] + layer.betas[0]
            f0 = layer._swiglu(h_a0, layer.w_gates[0], layer.w_ups[0], layer.w_downs[0])
            y_single = xi_shared + layer.alphas[0] * f0
            single_loss += F.mse_loss(y_single, targets[i].unsqueeze(0)).item()

            # --- 平均: 所有专家等权 (无路由, 无 STE) ---
            total_f = torch.zeros_like(xi_shared)
            for m in range(M):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
                total_f = total_f + f
            y_avg = xi_shared + total_f / M
            avg_loss += F.mse_loss(y_avg, targets[i].unsqueeze(0)).item()

        N = B
        fuse_loss /= N; single_loss /= N; avg_loss /= N

    return fuse_loss, avg_loss, single_loss


def main():
    print(f"{'='*78}")
    print("V6.0 跨架构族异构模态融合 (Cross-architecture Heterogeneous Modal Fusion)")
    print(f"{'='*78}")
    print("Text→TinyBERT(bidirectional D=312) + Code→TinyLlama(causal D=768)")
    print("        + Image→ViT-tiny(vision D=192)")
    print("每模态独立前向 + 独立 tokenizer + 矩形 P_m 投影对齐 → 共享路由")
    print(f"投影: D_text={312} → D_shared={D_SHARED}, D_code={768} → D_shared={D_SHARED},")
    print(f"       D_img={192} → D_shared={D_SHARED}")
    print()
    encoders = load_encoders()

    print(f"\n{'seed':>5} | {'single MSE':>11} | {'avg MSE':>9} | {'fuse MSE':>9} | "
          f"{'增益 vs single':>14} | {'增益 vs avg':>12} | {'PASS':>4}")
    print("-" * 78)
    all_pass = True; gs, ga = [], []
    for seed in range(5):
        fuse, avg, single = run_one(seed, encoders)
        gain_s = (single - fuse) / max(single, 1e-9) * 100
        gain_a = (avg - fuse) / max(avg, 1e-9) * 100
        gs.append(gain_s); ga.append(gain_a)
        ok = fuse < single and fuse < avg
        all_pass = all_pass and ok
        print(f"{seed:>5} | {single:>11.4f} | {avg:>9.4f} | {fuse:>9.4f} | "
              f"{gain_s:>+13.1f}% | {gain_a:>+11.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 78)
    import statistics
    print(f"\n增益均值: vs single {statistics.mean(gs):+.1f}% | vs avg {statistics.mean(ga):+.1f}%")
    print(f"增益范围: vs single [{min(gs):+.1f}%, {max(gs):+.1f}%] | vs avg [{min(ga):+.1f}%, {max(ga):+.1f}%]")
    print(f"{'='*78}")
    if all_pass:
        print("✅ V6.0 跨架构融合成立: 3 种不同架构真实模型可被本系统融合")
        print("→ 关键: 每模态独立前向 + 独立 tokenizer + 矩形 P_m 投影对齐")
        print(f"→ 对比 v5.1 跨架构测试 (+4%): 本次用原生编码替代统一编码, 保留类型差异")
    else:
        print("❌ 部分 seeds 失败 — 需分析原因")


if __name__ == "__main__":
    main()