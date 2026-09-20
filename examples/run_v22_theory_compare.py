"""V22.0 — 中枢机制理论分析端到端 (完整实现).

4 变体 × 5 seeds = 20 run:
    broadcast     — v8.0 默认 (z += U @ c), 对照基线
    gate          — v9.0 修正 C (z / temperature), 当前最优
    router-norm   — v22 新 (LayerNorm(z + U @ c))
    adaptive-ema  — v22 新 (broadcast + decay = sigmoid(α·||c||), α 初始 0 + warmup)

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
from research._primitives.central_theory import (
    CentralTheory,
    RouterNormCentral,
    AdaptiveEMACentral,
    freeze_u,
    set_u_init_scale,
)
from examples.run_v8_full import (
        C. **_ema_enabled 控制**: 训练时 EMA, eval 时不更新
        D. **梯度裁剪 (clip_grad_norm_=1.0)**: 防止 U 训练扰动
        E. **RouterNormCentral 默认 freeze_u (修正 A)**: 避免 H1
        F. **AdaptiveEMACentral warmup=10 步**: 前 10 步用 base_decay=0.9, 之后才用自适应

    预期: 4 模式 fuse MSE 都应收敛到 ~5-10 区间, 与 v8/v9 同量级.

核心判断: RouterNorm / AdaptiveEMA 是否能达到 gate-style 同等或更好的效果?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ---- 学习率隔离常量 (与 v8_trainer.LR_* 完全一致) ----
LR_ROUTER = 1e-4
LR_ADAPTER = 1e-2
LR_ALPHA = 1e-3
LR_CENTRAL = 5e-3
LR_COO = 5e-3

# ---- 稳定性 patch: 梯度裁剪阈值 (新增, 之前未用) ----
GRAD_CLIP_NORM = 1.0

# ---- 稳定性 patch: RouterNorm 默认 freeze_u (新增) ----
DEFAULT_FREEZE_U_FOR_ROUTER_NORM = True


class TheoryLayer(nn.Module):
    """v22.0 完整版融合层.

    与原 v22 TheoryLayer 的差异:
        1. 引入 V_coop 协同矩阵 (C-2, 对齐 v8 CentralAugmentedFusionLayer)
        2. aligner 参数全部冻结 (q_projections / W_k / W_v / down / up / out)
        3. attn_pools W_o 显式可训练 (其它内部已 register_buffer 冻结)
        4. _ema_enabled 字段 (eval 时关闭)
        5. 可选 freeze_u (router-norm 默认开)

    Args:
        mode:           中枢模式 (4 选 1)
        stability_patches: dict, 可覆盖默认稳定性 patch
            - freeze_u_router_norm: bool, router-norm 是否默认 freeze_u
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        modal_dims: List[int],
        attn_pools: List[AttnPool],
        mode: str,
        stability_patches: Dict = None,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = mode
        if stability_patches is None:
            stability_patches = {}
        self.freeze_u_router_norm = stability_patches.get(
            "freeze_u_router_norm", DEFAULT_FREEZE_U_FOR_ROUTER_NORM
        )

        # ---- aligner (冻结, 与 v10 CrossArchAttnAligner 协议一致) ----
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=d_shared, num_heads=4)
        for p in aligner.parameters():
            p.requires_grad_(False)
        self.aligner = aligner

        # ---- attn 路由 (与 v8 完全一致) ----
        self.attn_pools = nn.ModuleList(attn_pools)
        self.W_router_attn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.alphas_attn = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)]
        )

        # ---- ffn 路由 (与 v8 完全一致) ----
        self.W_router_ffn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.alphas_ffn = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)]
        )
        # FFN 冻结权重
        d_ff = 4 * d_shared
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02, requires_grad=False) for _ in range(num_experts)]
        )
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_shared) * 0.02, requires_grad=False) for _ in range(num_experts)]
        )
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_shared, d_ff) * 0.02, requires_grad=False) for _ in range(num_experts)]
        )
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_shared)) for _ in range(num_experts)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_shared)) for _ in range(num_experts)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)]
        )

        # ---- V_coop (C-2 协同矩阵, 对齐 v8) ----
        self.V_coop = nn.Parameter(torch.eye(d_shared) * 0.1)

        # ---- 中央中枢 (4 模式) ----
        self.cb_attn = CentralTheory(d_shared, num_experts, mode=mode)
        self.cb_ffn = CentralTheory(d_shared, num_experts, mode=mode)

        # ---- 稳定性 patch: RouterNorm 默认 freeze_u ----
        if mode == "router-norm" and self.freeze_u_router_norm:
            freeze_u(self.cb_attn)
            freeze_u(self.cb_ffn)

        # ---- 稳定性 patch: AdaptiveEMA 修正 B 移植 (U init scale 更小) ----
        if mode == "adaptive-ema":
            set_u_init_scale(self.cb_attn, scale=0.001)
            set_u_init_scale(self.cb_ffn, scale=0.001)

        # EMA 控制 (对齐 v8._ema_enabled)
        self._ema_enabled = True

    def disable_ema(self):
        self._ema_enabled = False

    def enable_ema(self):
        self._ema_enabled = True

    def _attn_path(self, x_shared: torch.Tensor):
        """attn 通路: 返回 (attn_sum, alpha_hat_attn, attn_expert_outs)"""
        x_det = x_shared.detach()
        z_a = F.linear(x_det, self.W_router_attn)
        z_a = self.cb_attn.augment_router_logits(z_a)
        alpha_a = F.softmax(z_a, dim=-1)
        from hetero_fusion.core.router import SparseRouterSTE
        ahat_a = SparseRouterSTE.apply(alpha_a, 1)
        attn_sum = torch.zeros_like(x_shared)
        attn_outs = []
        for m in range(self.num_experts):
            a_out = self.attn_pools[m](x_shared)
            attn_outs.append(a_out.detach())
            attn_sum = attn_sum + ahat_a[..., m:m+1] * self.alphas_attn[m] * a_out
        return attn_sum, ahat_a, attn_outs

    def _ffn_path(self, x_shared: torch.Tensor, attn_sum: torch.Tensor):
        """ffn 通路 (含 V_coop C-2 协同): 返回 (ffn_sum, alpha_hat_ffn, ffn_expert_outs)"""
        x_det = x_shared.detach()
        # C-2: ffn 路由 logits 受 attn 输出影响
        coop_ctx = F.linear(attn_sum.detach(), self.V_coop)
        z_f = F.linear(x_det + coop_ctx, self.W_router_ffn)
        z_f = self.cb_ffn.augment_router_logits(z_f)
        alpha_f = F.softmax(z_f, dim=-1)
        from hetero_fusion.core.router import SparseRouterSTE
        ahat_f = SparseRouterSTE.apply(alpha_f, 1)
        ffn_sum = torch.zeros_like(x_shared)
        ffn_outs = []
        for m in range(self.num_experts):
            h_a = x_det * self.gammas[m] + self.betas[m]
            f_m = F.linear(F.silu(F.linear(h_a, self.w_gates[m])) *
                           F.linear(h_a, self.w_ups[m]), self.w_downs[m])
            from hetero_fusion.core.quant import FakeQuantSTE
            f_q = FakeQuantSTE.apply(f_m, 4, 128)
            ffn_outs.append(f_q.detach())
            ffn_sum = ffn_sum + ahat_f[..., m:m+1] * self.alphas_ffn[m] * f_q
        return ffn_sum, ahat_f, ffn_outs

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        """前向 (与 v8.0 CentralAugmentedFusionLayer.forward 一致).

        Args:
            h_list: list of [S, D_m], 每个模态一个 [S, D_m] 张量 (无 batch dim)
        Returns:
            [S, D_shared]
        """
        x_shared = self.aligner(h_list)
        attn_sum, _, attn_outs = self._attn_path(x_shared)
        ffn_sum, _, ffn_outs = self._ffn_path(x_shared, attn_sum)
        # EMA 中枢更新 (仅训练时)
        if self._ema_enabled and self.training:
            self.cb_attn.ema_update(attn_outs)
            self.cb_ffn.ema_update(attn_outs)
        return x_shared + attn_sum + ffn_sum


def build_theory_param_groups(layer: TheoryLayer) -> List[Dict]:
    """构造 v22 TheoryLayer 的 per-expert param group (与 v8 完全一致).

    包含:
        - W_router_attn / W_router_ffn (lr=LR_ROUTER)
        - 每 attn 专家 W_o (lr=LR_ADAPTER)
        - 每 ffn 专家 γ/β (lr=LR_ADAPTER), α (lr=LR_ALPHA)
        - C-1 中枢: cb_attn.c, cb_ffn.c, U / W_t (lr=LR_CENTRAL)
        - C-2 协同: V_coop (lr=LR_COO)
    """
    groups: List[Dict] = []
    # attn router
    groups.append({"params": [layer.W_router_attn], "lr": LR_ROUTER, "expert": "router_attn"})
    # 每 attn 专家 (W_o)
    for m, ap in enumerate(layer.attn_pools):
        groups.append({"params": [ap.W_o], "lr": LR_ADAPTER, "expert": f"attn_{m}"})
    # ffn router
    groups.append({"params": [layer.W_router_ffn], "lr": LR_ROUTER, "expert": "router_ffn"})
    # 每 ffn 专家
    for m in range(layer.num_experts):
        groups.append({
            "params": [layer.gammas[m], layer.betas[m]],
            "lr": LR_ADAPTER, "expert": f"ffn_{m}_adapter",
        })
        groups.append({
            "params": [layer.alphas[m]],
            "lr": LR_ALPHA, "expert": f"ffn_{m}_alpha",
        })
    # C-1 中枢 (cb_attn, cb_ffn)
    central_params = []
    for cb in (layer.cb_attn, layer.cb_ffn):
        central_params.append(cb.c)
        if hasattr(cb, "U"):
            central_params.append(cb.U)
        if hasattr(cb, "W_t"):
            central_params.append(cb.W_t)
    groups.append({"params": central_params, "lr": LR_CENTRAL, "expert": "central_workspace"})
    # C-2 协同矩阵 V_coop
    groups.append({"params": [layer.V_coop], "lr": LR_COO, "expert": "central_coop"})
    return groups


def prepare_data(seed, encoders):
    """复用原版 prepare_data: 18 个样本 × 3 个模态."""
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)
    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)
    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    return modal_seqs, modal_indices, targets


def run_seed(mode: str, seed: int, encoders, attn_pools):
    """单 seed 训练 + 评估.

    关键改进:
        - per-expert param group (build_theory_param_groups)
        - 梯度裁剪 (clip_grad_norm_=GRAD_CLIP_NORM)
        - 训练时 layer.train(), 评估时 layer.eval() (EMA 自动停)
        - 修复: layer 期望 h_list=[h_A, h_B, h_C], 不能传单个 h_by_modal[c]
    """
    torch.manual_seed(seed)
    print(f"  [seed {seed}] mode={mode} ...")
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)

    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    layer = TheoryLayer(
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

    # per-expert param group 优化器 (与 v8 完全一致)
    layer.train()
    opt = torch.optim.AdamW(build_theory_param_groups(layer), lr=LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        # 关键修复: layer 期望 h_list=[h_A, h_B, h_C], 不是单个张量
        h_list = h_by_modal  # 3 元素 list, 每元素 [6, S, D_m]
        y = layer(h_list)    # [6, S, D_shared]
        # 损失: 所有 3 模态 target 的均值
        total_loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        total_loss.backward()
        # 稳定性 patch: 梯度裁剪
        torch.nn.utils.clip_grad_norm_(layer.parameters(), GRAD_CLIP_NORM)
        opt.step()

    # 评估: eval 模式 (EMA 自动停), layer 输入整个 h_list
    layer.eval()
    with torch.no_grad():
        h_list = h_by_modal
        y = layer(h_list)  # [6, S, D_shared]
        fuse_loss = 0.0
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


MODES = ["broadcast", "gate", "router-norm", "adaptive-ema"]


def main():
    print("=" * 78)
    print("V22.0 — 中枢机制理论分析端到端 (完整实现)")
    print("=" * 78)
    print("4 变体 × 5 seeds = 20 run")
    print("稳定性 patch: V_coop + per-expert param group + 梯度裁剪 + RouterNorm freeze_u")
    print("核心判断: RouterNorm / AdaptiveEMA 是否能达到 gate-style 同等效果?")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {m: [] for m in MODES}

    print(f"\n每个 mode × {SEEDS} seeds")
    for mode in MODES:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(mode, seed, encoders, attn_pools)
                results[mode].append(fuse)
                print(f"  [{mode:>14} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>14} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>14} | {'fuse MSE':>9} | {'vs broadcast':>14}")
    print("-" * 78)
    broadcast_vals = [m for m in results["broadcast"] if not np.isnan(m)]
    if not broadcast_vals:
        print("全部失败, 无法汇总")
        return
    broadcast_mean = statistics.mean(broadcast_vals)
    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>14} | {'N/A':>9} | {'N/A':>14}")
            continue
        m_mean = statistics.mean(vals)
        m_std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gain = (broadcast_mean - m_mean) / max(broadcast_mean, 1e-9) * 100
        print(f"{m:>14} | {m_mean:>9.4f} (±{m_std:.3f}) | {gain:>+13.1f}%")
    print("=" * 78)
    print()
    f_b = broadcast_mean
    f_g = statistics.mean([m for m in results["gate"] if not np.isnan(m)])
    f_r = statistics.mean([m for m in results["router-norm"] if not np.isnan(m)])
    f_a = statistics.mean([m for m in results["adaptive-ema"] if not np.isnan(m)])
    print(f"  broadcast (v8.0 默认):    {f_b:.4f}")
    print(f"  gate (v9.0 修正 C):       {f_g:.4f}  (vs broadcast: {(f_b - f_g) / f_b * 100:+.1f}%)")
    print(f"  router-norm (v22 新):     {f_r:.4f}  (vs broadcast: {(f_b - f_r) / f_b * 100:+.1f}%)")
    print(f"  adaptive-ema (v22 新):   {f_a:.4f}  (vs broadcast: {(f_b - f_a) / f_b * 100:+.1f}%)")
    print()
    best = min(f_g, f_r, f_a)
    if best == f_g:
        print("✅ gate-style 仍是最佳, v22 新方案未超越")
    elif best == f_r:
        print(f"✅ router-norm 超越 gate-style, 新结构层修复!")
    else:
        print(f"✅ adaptive-ema 超越 gate-style, 新结构层修复!")


if __name__ == "__main__":
    main()
