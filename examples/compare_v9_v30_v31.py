"""v9 vs v30 vs v31 对比实验。

对比三种 gate-style 变体在相同条件下的表现：
- v9: 原版 tanh (已知 grad_norm_central = 0)
- v30: softplus 替代 tanh (方向 A)
- v31: gradient-scale 替代 tanh (方向 B)

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 5 seeds (0-4)
- 复现: 直接运行此脚本即可
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

from research._primitives.attention import AttnPool
from research.aligner.v10_embedding import CrossArchAttnAligner
from research.central_stability.v22_central_theory import CentralTheory, freeze_u, set_u_init_scale
from research.routing_evolution.v9_gate_central.workspace import GateStyleWorkspace
from research.central_stability.v30_gate_softplus.workspace import SoftplusGateStyleWorkspace
from research.central_stability.v31_gate_gradient_scale.workspace import GradientScaledGateStyleWorkspace
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)

# ---- 实验环境 ----
_ENV = log_experiment_env("v9_v30_v31_compare")

# ---- 学习率隔离常量 ----
LR_ROUTER = 1e-4
LR_ADAPTER = 1e-2
LR_ALPHA = 1e-3
LR_CENTRAL = 5e-3
LR_COO = 5e-3
GRAD_CLIP_NORM = 1.0


class TheoryLayerWithMode(nn.Module):
    """支持三种 gate-style 变体的 TheoryLayer。"""

    def __init__(self, d_shared, num_experts, modal_dims, attn_pools, mode):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = mode

        # aligner (冻结)
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=d_shared, num_heads=4)
        for p in aligner.parameters():
            p.requires_grad_(False)
        self.aligner = aligner

        # attn 路由
        self.attn_pools = nn.ModuleList(attn_pools)
        self.W_router_attn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.alphas_attn = nn.ParameterList([nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])

        # ffn 路由
        self.W_router_ffn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.alphas_ffn = nn.ParameterList([nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])

        d_ff = 4 * d_shared
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(d_ff, d_shared) * 0.02, requires_grad=False) for _ in range(num_experts)])
        self.w_ups = nn.ParameterList([nn.Parameter(torch.randn(d_ff, d_shared) * 0.02, requires_grad=False) for _ in range(num_experts)])
        self.w_downs = nn.ParameterList([nn.Parameter(torch.randn(d_shared, d_ff) * 0.02, requires_grad=False) for _ in range(num_experts)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(d_shared)) for _ in range(num_experts)])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(d_shared)) for _ in range(num_experts)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)])

        # V_coop (C-2)
        self.V_coop = nn.Parameter(torch.eye(d_shared) * 0.1)

        # 中央中枢 - 根据 mode 选择不同实现
        if mode == "v9":
            self.cb_attn = GateStyleWorkspace(d_shared, num_experts, alpha=0.1)
            self.cb_ffn = GateStyleWorkspace(d_shared, num_experts, alpha=0.1)
        elif mode == "v30":
            self.cb_attn = SoftplusGateStyleWorkspace(d_shared, num_experts, alpha=0.1)
            self.cb_ffn = SoftplusGateStyleWorkspace(d_shared, num_experts, alpha=0.1)
        elif mode == "v31":
            self.cb_attn = GradientScaledGateStyleWorkspace(d_shared, num_experts, alpha=0.1)
            self.cb_ffn = GradientScaledGateStyleWorkspace(d_shared, num_experts, alpha=0.1)
        else:
            raise ValueError(f"unknown mode {mode}")

        self._ema_enabled = True

    def disable_ema(self):
        self._ema_enabled = False

    def enable_ema(self):
        self._ema_enabled = True

    def _attn_path(self, x_shared):
        x_det = x_shared.detach()
        z_a = F.linear(x_det, self.W_router_attn)
        z_a = self.cb_attn.augment_router_logits(z_a)
        alpha_a = F.softmax(z_a, dim=-1)
        from hetero_fusion.core.router import SparseRouterSTE
        ahat_a = SparseRouterSTE.apply(alpha_a, 1)
        attn_sum = torch.zeros_like(x_shared)
        for m in range(self.num_experts):
            a_out = self.attn_pools[m](x_shared)
            attn_sum = attn_sum + ahat_a[..., m:m+1] * self.alphas_attn[m] * a_out.detach()
        return attn_sum, ahat_a

    def _ffn_path(self, x_shared, attn_sum):
        x_det = x_shared.detach()
        coop_ctx = F.linear(attn_sum.detach(), self.V_coop)
        z_f = F.linear(x_det + coop_ctx, self.W_router_ffn)
        z_f = self.cb_ffn.augment_router_logits(z_f)
        alpha_f = F.softmax(z_f, dim=-1)
        from hetero_fusion.core.router import SparseRouterSTE
        ahat_f = SparseRouterSTE.apply(alpha_f, 1)
        ffn_sum = torch.zeros_like(x_shared)
        for m in range(self.num_experts):
            h_a = x_det * self.gammas[m] + self.betas[m]
            f_m = F.linear(F.silu(F.linear(h_a, self.w_gates[m])) * F.linear(h_a, self.w_ups[m]), self.w_downs[m])
            from hetero_fusion.core.quant import FakeQuantSTE
            f_q = FakeQuantSTE.apply(f_m, 4, 128)
            ffn_sum = ffn_sum + ahat_f[..., m:m+1] * self.alphas_ffn[m] * f_q.detach()
        return ffn_sum, ahat_f

    def forward(self, h_list):
        x_shared = self.aligner(h_list)
        attn_sum, _ = self._attn_path(x_shared)
        ffn_sum, _ = self._ffn_path(x_shared, attn_sum)
        if self._ema_enabled and self.training:
            self.cb_attn.ema_update([x_shared])
            self.cb_ffn.ema_update([x_shared])
        return x_shared + attn_sum + ffn_sum


def build_param_groups(layer):
    groups = []
    groups.append({"params": [layer.W_router_attn], "lr": LR_ROUTER, "expert": "router_attn"})
    for m, ap in enumerate(layer.attn_pools):
        groups.append({"params": [ap.W_o], "lr": LR_ADAPTER, "expert": f"attn_{m}"})
    groups.append({"params": [layer.W_router_ffn], "lr": LR_ROUTER, "expert": "router_ffn"})
    for m in range(layer.num_experts):
        groups.append({"params": [layer.gammas[m], layer.betas[m]], "lr": LR_ADAPTER, "expert": f"ffn_{m}_adapter"})
        groups.append({"params": [layer.alphas[m]], "lr": LR_ALPHA, "expert": f"ffn_{m}_alpha"})
    central_params = []
    for cb in (layer.cb_attn, layer.cb_ffn):
        central_params.append(cb.c)
        if hasattr(cb, "W_t"):
            central_params.append(cb.W_t)
        if hasattr(cb, "U"):
            central_params.append(cb.U)
    groups.append({"params": central_params, "lr": LR_CENTRAL, "expert": "central_workspace"})
    groups.append({"params": [layer.V_coop], "lr": LR_COO, "expert": "central_coop"})
    return groups


def prepare_data(seed, n_text=18, n_code=6, n_image=6):
    """准备数据，所有模态共享相同序列数。"""
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = load_encoders()

    text_templates = [f"text sample {i}" for i in range(max(n_text, 6))]
    code_templates = [f"code sample {i}" for i in range(max(n_code, 6))]
    img_array = np.random.randint(0, 256, (224, 224, 3))

    h_text = encode_text(text_templates[:n_text], bert_tok, bert)
    h_code = encode_text(code_templates[:n_code], llama_tok, llama)
    h_img = encode_image(img_array, vit_proc, vit)

    n_min = min(h_text.shape[0], h_code.shape[0], h_img.shape[0])

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text[:n_min], h_code[:n_min], h_img[:n_min]]):
        for i in range(n_min):
            src = h_block[i]
            g = torch.Generator().manual_seed(seed * 10 + cls * n_min + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED)
        t[:, ds:de] = 1.0
        for _ in range(n_min * 3):
            targets.append(t)
    return modal_seqs, modal_indices, targets


def run_seed(mode, seed, attn_pools, n_text=18, n_code=6, n_image=6, steps=STEPS):
    """单 seed 训练 + 评估。"""
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = prepare_data(seed, n_text, n_code, n_image)

    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = load_encoders()
    layer = TheoryLayerWithMode(
        d_shared=D_SHARED, num_experts=3,
        modal_dims=[D_bert, D_llama, D_vit],
        attn_pools=attn_pools, mode=mode,
    )

    # 按模态分组数据
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)

    layer.train()
    opt = torch.optim.AdamW(build_param_groups(layer), lr=LR)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        h_list = h_by_modal
        y = layer(h_list)
        total_loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), GRAD_CLIP_NORM)
        opt.step()

    # 评估
    layer.eval()
    with torch.no_grad():
        h_list = h_by_modal
        y = layer(h_list)
        fuse_loss = sum(F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
                        for c in range(N_CLS)
                        for s in range(len(t_by_modal[c]))) / sum(len(t_by_modal[c]) for c in range(N_CLS))
    return fuse_loss


def main():
    print("=" * 78)
    print("v9 vs v30 vs v31 对比实验")
    print("=" * 78)
    print("3 变体 × 5 seeds = 15 run")
    print()

    attn_pools = make_pools(load_encoders())
    modes = ["v9", "v30", "v31"]
    results = {m: [] for m in modes}

    for mode in modes:
        for seed in range(5):
            try:
                fuse = run_seed(mode, seed, attn_pools, steps=200)
                results[mode].append(fuse)
                print(f"  [{mode:>6} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>6} seed {seed}] FAILED: {type(e).__name__}: {e}")
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>6} | {'fuse MSE':>9} | {'std':>6}")
    print("-" * 78)
    for m in modes:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>6} | {'N/A':>9} | {'N/A':>6}")
            continue
        m_mean = statistics.mean(vals)
        m_std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{m:>6} | {m_mean:>9.4f} | {m_std:>6.4f}")
    print("=" * 78)

    # 两两对比
    print("\n对比:")
    for m1 in modes:
        for m2 in modes:
            if m1 >= m2:
                continue
            vals1 = [x for x in results[m1] if not np.isnan(x)]
            vals2 = [x for x in results[m2] if not np.isnan(x)]
            if vals1 and vals2:
                mean1 = statistics.mean(vals1)
                mean2 = statistics.mean(vals2)
                gain = (mean1 - mean2) / max(mean1, 1e-9) * 100
                print(f"  {m1} vs {m2}: {gain:>+6.1f}% ({mean1:.4f} vs {mean2:.4f})")


if __name__ == "__main__":
    main()
