"""v29 gate 退步根因诊断实验。

对比 small/large 语料下 gate-style 的训练动态，定位退步根因。

诊断维度：
1. 中枢向量 c 的统计特性（范数、熵、分布）
2. 温度 τ 的训练动态（是否饱和、波动）
3. 路由 logits 的统计特性（均值、标准差、熵）
4. 梯度流（central 参数梯度范数）
5. 训练集 vs held-out 差距（过拟合指标）

实验环境:
- 硬件: CPU only (无 GPU)
- 软件: Python 3.x, PyTorch 2.0+, 小规模张量 D≤128
- 随机种子: 3 seeds
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


class DiagnosticTheoryLayer(nn.Module):
    """带诊断日志的 TheoryLayer。"""

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

        if mode == "router-norm":
            freeze_u(self.cb_attn)
            freeze_u(self.cb_ffn)

        if mode == "adaptive-ema":
            set_u_init_scale(self.cb_attn, scale=0.001)
            set_u_init_scale(self.cb_ffn, scale=0.001)

        self._ema_enabled = True

        # 诊断日志
        self.diag_log = {
            "c_norm_attn": [],
            "c_norm_ffn": [],
            "tau_attn": [],
            "tau_ffn": [],
            "router_logits_mean": [],
            "router_logits_std": [],
            "router_logits_entropy": [],
            "grad_norm_central": [],
            "grad_norm_router": [],
            "train_loss": [],
            "eval_loss": [],
        }

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
        return attn_sum, ahat_a, z_a

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
        return ffn_sum, ahat_f, z_f

    def forward(self, h_list):
        x_shared = self.aligner(h_list)
        attn_sum, ahat_a, z_a = self._attn_path(x_shared)
        ffn_sum, ahat_f, z_f = self._ffn_path(x_shared, attn_sum)

        # 诊断日志
        if self.training:
            with torch.no_grad():
                # 中枢向量 c 的统计
                if hasattr(self.cb_attn, 'c'):
                    self.diag_log["c_norm_attn"].append(self.cb_attn.c.norm().item())
                if hasattr(self.cb_ffn, 'c'):
                    self.diag_log["c_norm_ffn"].append(self.cb_ffn.c.norm().item())

                # 温度 τ
                if hasattr(self.cb_attn, 'W_t') and hasattr(self.cb_attn, 'c'):
                    t_scalar = torch.dot(self.cb_attn.W_t, self.cb_attn.c)
                    tau = 1.0 + 0.1 * torch.tanh(t_scalar)
                    self.diag_log["tau_attn"].append(tau.item())
                if hasattr(self.cb_ffn, 'W_t') and hasattr(self.cb_ffn, 'c'):
                    t_scalar = torch.dot(self.cb_ffn.W_t, self.cb_ffn.c)
                    tau = 1.0 + 0.1 * torch.tanh(t_scalar)
                    self.diag_log["tau_ffn"].append(tau.item())

                # 路由 logits 统计
                logits = torch.cat([z_a, z_f], dim=-1)
                self.diag_log["router_logits_mean"].append(logits.mean().item())
                self.diag_log["router_logits_std"].append(logits.std().item())
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                self.diag_log["router_logits_entropy"].append(entropy.item())

        if self._ema_enabled and self.training:
            self.cb_attn.ema_update([x_shared])
            self.cb_ffn.ema_update([x_shared])
        return x_shared + attn_sum + ffn_sum

    def log_gradients(self):
        """记录梯度范数（应在 backward 后调用）。"""
        with torch.no_grad():
            central_params = []
            router_params = []
            for cb in (self.cb_attn, self.cb_ffn):
                if hasattr(cb, 'c') and cb.c.grad is not None:
                    central_params.append(cb.c.grad.norm().item())
                if hasattr(cb, 'W_t') and cb.W_t.grad is not None:
                    central_params.append(cb.W_t.grad.norm().item())
                if hasattr(cb, 'U') and cb.U.grad is not None:
                    central_params.append(cb.U.grad.norm().item())
            central_params.extend([p.grad.norm().item() for p in [self.V_coop] if p.grad is not None])
            router_params.extend([p.grad.norm().item() for p in [self.W_router_attn, self.W_router_ffn] if p.grad is not None])

            self.diag_log["grad_norm_central"].append(statistics.mean(central_params) if central_params else 0.0)
            self.diag_log["grad_norm_router"].append(statistics.mean(router_params) if router_params else 0.0)


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
    """单 seed 训练 + 评估，返回诊断日志。"""
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = prepare_data(seed, n_text, n_code, n_image)

    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = load_encoders()
    layer = DiagnosticTheoryLayer(
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
        layer.log_gradients()
        opt.step()

        # 每 50 step 记录训练/验证损失
        if (step + 1) % 50 == 0 or step == 0:
            layer.eval()
            with torch.no_grad():
                y_eval = layer(h_list)
                train_loss = sum(F.mse_loss(y, t_by_modal[c]).item() for c in range(N_CLS)) / N_CLS
                # 用独立噪声构造 held-out 数据
                h_list_heldout = []
                for c in range(N_CLS):
                    noise = torch.randn_like(h_by_modal[c]) * 0.05
                    h_list_heldout.append(h_by_modal[c] + noise)
                y_heldout = layer(h_list_heldout)
                # 与目标比较（目标已在 shared space）
                eval_loss = F.mse_loss(y_heldout, t_by_modal[0]).item()
                layer.diag_log["train_loss"].append((step + 1, train_loss))
                layer.diag_log["eval_loss"].append((step + 1, eval_loss))
            layer.train()

    # 最终评估
    layer.eval()
    with torch.no_grad():
        h_list = h_by_modal
        y = layer(h_list)
        fuse_loss = sum(F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
                        for c in range(N_CLS)
                        for s in range(len(t_by_modal[c]))) / sum(len(t_by_modal[c]) for c in range(N_CLS))
    return fuse_loss, layer.diag_log


def print_summary(logs, label):
    """打印诊断摘要。"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for key in ["c_norm_attn", "c_norm_ffn", "tau_attn", "tau_ffn",
                "router_logits_mean", "router_logits_std", "router_logits_entropy",
                "grad_norm_central", "grad_norm_router"]:
        vals = logs[key]
        if not vals:
            continue
        print(f"  {key:>22}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals):.4f}, "
              f"min={min(vals):.4f}, max={max(vals):.4f}")

    # 训练/验证差距
    if logs["train_loss"] and logs["eval_loss"]:
        final_train = logs["train_loss"][-1][1]
        final_eval = logs["eval_loss"][-1][1]
        gap = final_eval - final_train
        print(f"  {'train_loss':>22}: {final_train:.4f}")
        print(f"  {'eval_loss':>22}: {final_eval:.4f}")
        print(f"  {'gap (eval-train)':>22}: {gap:+.4f}")


def main():
    print("=" * 78)
    print("v29 gate 退步根因诊断")
    print("=" * 78)

    attn_pools = make_pools(load_encoders())
    modes = ["gate"]
    corpus_sizes = [
        ("small", 18, 6, 6),
        ("large", 64, 24, 24),
    ]

    all_logs = {}
    for corpus_name, n_text, n_code, n_image in corpus_sizes:
        print(f"\n语料规模: {corpus_name} (text={n_text}, code={n_code}, image={n_image})")
        for mode in modes:
            logs_list = []
            fuse_list = []
            for seed in range(3):
                fuse, logs = run_seed(mode, seed, attn_pools, n_text, n_code, n_image, steps=200)
                logs_list.append(logs)
                fuse_list.append(fuse)
                print(f"  [seed {seed}] fuse={fuse:.4f}")

            # 合并日志（取平均）
            merged = {}
            for key in logs_list[0].keys():
                all_vals = []
                for log in logs_list:
                    all_vals.extend(log[key])
                if all_vals and isinstance(all_vals[0], (int, float)):
                    merged[key] = all_vals
                else:
                    merged[key] = logs_list[-1][key]  # 用最后一个 seed 的
            all_logs[f"{corpus_name}_{mode}"] = merged
            print(f"  fuse MSE: {statistics.mean(fuse_list):.4f} (±{statistics.stdev(fuse_list):.4f})")

    # 打印对比
    for corpus_name, n_text, n_code, n_image in corpus_sizes:
        key = f"{corpus_name}_gate"
        if key in all_logs:
            print_summary(all_logs[key], f"{corpus_name} - gate")

    print("\n" + "=" * 78)
    print("诊断完成")
    print("=" * 78)


if __name__ == "__main__":
    main()
