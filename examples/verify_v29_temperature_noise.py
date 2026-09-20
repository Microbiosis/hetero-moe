"""验证 v29 gate 退步的温度噪声假设。

实验 1：对比小语料 vs 大语料下 gate-style 的温度波动
实验 2：验证 v22 adaptive-ema 的 MSE 值
实验 3：对比 gate vs adaptive-ema 在不同语料规模下的表现

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 3-5 seeds
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
from research._primitives.central_theory import (
    CentralTheory,
    RouterNormCentral,
    AdaptiveEMACentral,
    freeze_u,
    set_u_init_scale,
)
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)

# ---- 学习率隔离常量 ----
LR_ROUTER = 1e-4
LR_ADAPTER = 1e-2
LR_ALPHA = 1e-3
LR_CENTRAL = 5e-3
LR_COO = 5e-3
GRAD_CLIP_NORM = 1.0
DEFAULT_FREEZE_U_FOR_ROUTER_NORM = True


class TheoryLayerWithTempLog(nn.Module):
    """带温度日志的 TheoryLayer，用于记录每个 step 的 tau 值。"""

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

        # 中央中枢
        self.cb_attn = CentralTheory(d_shared, num_experts, mode=mode)
        self.cb_ffn = CentralTheory(d_shared, num_experts, mode=mode)

        if mode == "router-norm" and DEFAULT_FREEZE_U_FOR_ROUTER_NORM:
            freeze_u(self.cb_attn)
            freeze_u(self.cb_ffn)

        if mode == "adaptive-ema":
            set_u_init_scale(self.cb_attn, scale=0.001)
            set_u_init_scale(self.cb_ffn, scale=0.001)

        self._ema_enabled = True
        self.temp_log = []  # 记录每个 step 的 tau 值

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

        # 记录 gate-style 的温度值
        if self.mode == "gate" and self.training:
            with torch.no_grad():
                for cb in (self.cb_attn, self.cb_ffn):
                    if hasattr(cb, 'W_t') and hasattr(cb, 'c'):
                        t_scalar = torch.dot(cb.W_t, cb.c)
                        tau = 1.0 + 0.1 * torch.tanh(t_scalar)
                        self.temp_log.append(tau.item())

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
        if hasattr(cb, "U"):
            central_params.append(cb.U)
        if hasattr(cb, "W_t"):
            central_params.append(cb.W_t)
    groups.append({"params": central_params, "lr": LR_CENTRAL, "expert": "central_workspace"})
    groups.append({"params": [layer.V_coop], "lr": LR_COO, "expert": "central_coop"})
    return groups


def prepare_data(seed, n_text=18, n_code=6, n_image=6):
    """准备数据，可调整样本量。所有模态共享相同序列数，避免 aligner stack 报错。"""
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
    """单 seed 训练 + 评估，返回 (fuse_mse, temp_log)。"""
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = prepare_data(seed, n_text, n_code, n_image)

    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = load_encoders()
    layer = TheoryLayerWithTempLog(
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
    temp_log = layer.temp_log
    with torch.no_grad():
        h_list = h_by_modal
        y = layer(h_list)
        fuse_loss = 0.0
        total_samples = sum(len(t_by_modal[c]) for c in range(N_CLS))
        for c in range(N_CLS):
            for s in range(len(t_by_modal[c])):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= total_samples
    return fuse_loss, temp_log


def experiment_1_temperature_variance():
    """实验 1：对比小语料 vs 大语料下 gate-style 的温度波动。"""
    print("=" * 78)
    print("实验 1: 温度波动对比 (小语料 vs 大语料)")
    print("=" * 78)

    attn_pools = make_pools(load_encoders())
    modes = ["gate"]
    corpus_sizes = [
        ("small", 18, 6, 6),
        ("medium", 32, 12, 12),
        ("large", 64, 24, 24),
    ]

    results = {}
    for corpus_name, n_text, n_code, n_image in corpus_sizes:
        print(f"\n语料规模: {corpus_name} (text={n_text}, code={n_code}, image={n_image})")
        for mode in modes:
            temps_all = []
            fuse_all = []
            for seed in range(3):  # 3 seeds for speed
                fuse, temp_log = run_seed(mode, seed, attn_pools, n_text, n_code, n_image, steps=200)
                temps_all.extend(temp_log)
                fuse_all.append(fuse)
                print(f"  [seed {seed}] fuse={fuse:.4f}, temp_samples={len(temp_log)}")

            if temps_all:
                temp_mean = statistics.mean(temps_all)
                temp_std = statistics.stdev(temps_all) if len(temps_all) > 1 else 0.0
                temp_min = min(temps_all)
                temp_max = max(temps_all)
                print(f"  温度统计: mean={temp_mean:.4f}, std={temp_std:.4f}, min={temp_min:.4f}, max={temp_max:.4f}")
                print(f"  温度范围: [{temp_min:.4f}, {temp_max:.4f}] (理论范围 [0.9, 1.1])")
            else:
                print("  警告: 未记录到温度数据")

            fuse_mean = statistics.mean(fuse_all)
            print(f"  fuse MSE: {fuse_mean:.4f} (±{statistics.stdev(fuse_all) if len(fuse_all) > 1 else 0.0:.4f})")

    print("\n" + "=" * 78)


def experiment_2_v22_adaptive_ema():
    """实验 2：验证 v22 adaptive-ema 的 MSE 值。"""
    print("=" * 78)
    print("实验 2: v22 四种中枢模式对比")
    print("=" * 78)
    print("4 变体 × 5 seeds = 20 run")

    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {m: [] for m in ["broadcast", "gate", "router-norm", "adaptive-ema"]}

    for mode in results.keys():
        for seed in range(5):
            try:
                fuse, _ = run_seed(mode, seed, attn_pools, steps=200)
                results[mode].append(fuse)
                print(f"  [{mode:>14} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>14} seed {seed}] FAILED: {e}")
                results[mode].append(float("nan"))

    print("\n" + "=" * 78)
    print(f"{'mode':>14} | {'fuse MSE':>9} | {'vs broadcast':>14}")
    print("-" * 78)
    broadcast_vals = [m for m in results["broadcast"] if not np.isnan(m)]
    if not broadcast_vals:
        print("全部失败, 无法汇总")
        return
    broadcast_mean = statistics.mean(broadcast_vals)
    for m in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>14} | {'N/A':>9} | {'N/A':>14}")
            continue
        m_mean = statistics.mean(vals)
        m_std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gain = (broadcast_mean - m_mean) / max(broadcast_mean, 1e-9) * 100
        print(f"{m:>14} | {m_mean:>9.4f} (±{m_std:.3f}) | {gain:>+13.1f}%")
    print("=" * 78)


def experiment_3_corpus_size_sweep():
    """实验 3：对比 gate vs adaptive-ema 在不同语料规模下的表现。"""
    print("=" * 78)
    print("实验 3: 语料规模对比 (gate vs adaptive-ema)")
    print("=" * 78)

    attn_pools = make_pools(load_encoders())
    modes = ["gate", "adaptive-ema"]
    corpus_sizes = [
        ("tiny", 8, 4, 4),
        ("small", 18, 6, 6),
        ("medium", 32, 12, 12),
        ("large", 64, 24, 24),
    ]

    for corpus_name, n_text, n_code, n_image in corpus_sizes:
        print(f"\n语料规模: {corpus_name} (text={n_text}, code={n_code}, image={n_image})")
        for mode in modes:
            fuse_all = []
            for seed in range(3):
                fuse, _ = run_seed(mode, seed, attn_pools, n_text, n_code, n_image, steps=200)
                fuse_all.append(fuse)
            fuse_mean = statistics.mean(fuse_all)
            fuse_std = statistics.stdev(fuse_all) if len(fuse_all) > 1 else 0.0
            print(f"  {mode:>14}: {fuse_mean:>8.4f} (±{fuse_std:.3f})")

    print("\n" + "=" * 78)


def main():
    print("v29 验证实验: 温度噪声假设与中枢模式对比")
    print("=" * 78)

    experiment_1_temperature_variance()
    experiment_2_v22_adaptive_ema()
    experiment_3_corpus_size_sweep()


if __name__ == "__main__":
    main()
