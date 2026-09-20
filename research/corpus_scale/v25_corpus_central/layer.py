"""v25.0 — v22 CentralTheory + v18 MiniCorpus 跨版本集成 (首个跨版本包).

设计:
    - 复用 v22_central_theory.CentralTheory (不重写)
    - 复用 v18_real_corpus.MiniCorpus (不重写)
    - RealCorpusTheoryLayer 继承 v22 TheoryLayer 的全部结构与稳定性 patch
    - 仅替换数据准备函数 (text 模态用真实 Wikipedia 句子)

可追溯性:
    RealCorpusTheoryLayer    -> 委托 v22 TheoryLayer, text 模态用 MiniCorpus
    prepare_real_corpus_data -> 真实 Wikipedia 句子 + 合成 code/image (诚实标记)
    build_v25_param_groups   -> 与 v22 build_theory_param_groups 完全一致 (per-expert 隔离)

公平性声明:
    - 仅 text 模态用 MiniCorpus.train_sentences (真实 Wikipedia, 16 句)
    - code/image 模态保留 v22 合成扰动 (无真实语料库)
    - 评估: 训练集 + held-out 双评估 (held-out 用 MiniCorpus.held_out 6 句)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 委托 v22 + v18
from research._primitives.attention import AttnPool
from research.aligner.v10_embedding import CrossArchAttnAligner
from research._primitives.central_theory import (
    CentralTheory,
    freeze_u,
    set_u_init_scale,
)
from research._primitives.real_corpus import MiniCorpus

# v22 TheoryLayer 的常量 (与 run_v22_theory_compare.py 完全一致)
LR_ROUTER = 1e-4
LR_ADAPTER = 1e-2
LR_ALPHA = 1e-3
LR_CENTRAL = 5e-3
LR_COO = 5e-3
GRAD_CLIP_NORM = 1.0
DEFAULT_FREEZE_U_FOR_ROUTER_NORM = True


class RealCorpusTheoryLayer(nn.Module):
    """v25.0 — RealCorpus + CentralTheory 融合层.

    与 v22 TheoryLayer 的差异:
        - 结构完全相同 (V_coop + per-expert param group + 6 项 patch)
        - 数据准备侧由 prepare_real_corpus_data 替换 (text 模态用 MiniCorpus)
        - 评估时区分训练集与 held-out (v18 风格)
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

        # ---- aligner (冻结) ----
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=d_shared, num_heads=4)
        for p in aligner.parameters():
            p.requires_grad_(False)
        self.aligner = aligner

        # ---- attn 路由 ----
        self.attn_pools = nn.ModuleList(attn_pools)
        self.W_router_attn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.alphas_attn = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)]
        )

        # ---- ffn 路由 ----
        self.W_router_ffn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
        self.alphas_ffn = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)]
        )
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

        # ---- V_coop (C-2) ----
        self.V_coop = nn.Parameter(torch.eye(d_shared) * 0.1)

        # ---- 中枢 (4 模式, 委托 v22) ----
        self.cb_attn = CentralTheory(d_shared, num_experts, mode=mode)
        self.cb_ffn = CentralTheory(d_shared, num_experts, mode=mode)

        if mode == "router-norm" and self.freeze_u_router_norm:
            freeze_u(self.cb_attn)
            freeze_u(self.cb_ffn)
        if mode == "adaptive-ema":
            set_u_init_scale(self.cb_attn, scale=0.001)
            set_u_init_scale(self.cb_ffn, scale=0.001)

        self._ema_enabled = True

    def disable_ema(self):
        self._ema_enabled = False

    def enable_ema(self):
        self._ema_enabled = True

    def _attn_path(self, x_shared: torch.Tensor):
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
        x_det = x_shared.detach()
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
            ffn_sum = ffn_sum + ahat_f[..., m:m+1] * self.alphas[m] * f_q
        return ffn_sum, ahat_f, ffn_outs

    def forward(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        """前向 (与 v22.0 完整实现完全一致).

        Args:
            h_list: list of [B, S, D_m], 每模态一个 batched 张量 (3 模态)
        Returns:
            [B, S, D_shared]
        """
        x_shared = self.aligner(h_list)
        attn_sum, _, attn_outs = self._attn_path(x_shared)
        ffn_sum, _, ffn_outs = self._ffn_path(x_shared, attn_sum)
        if self._ema_enabled and self.training:
            self.cb_attn.ema_update(attn_outs)
            self.cb_ffn.ema_update(attn_outs)
        return x_shared + attn_sum + ffn_sum


def build_v25_param_groups(layer: RealCorpusTheoryLayer) -> List[Dict]:
    """v25 per-expert param group (与 v22 build_theory_param_groups 完全一致)."""
    groups: List[Dict] = []
    groups.append({"params": [layer.W_router_attn], "lr": LR_ROUTER, "expert": "router_attn"})
    for m, ap in enumerate(layer.attn_pools):
        groups.append({"params": [ap.W_o], "lr": LR_ADAPTER, "expert": f"attn_{m}"})
    groups.append({"params": [layer.W_router_ffn], "lr": LR_ROUTER, "expert": "router_ffn"})
    for m in range(layer.num_experts):
        groups.append({
            "params": [layer.gammas[m], layer.betas[m]],
            "lr": LR_ADAPTER, "expert": f"ffn_{m}_adapter",
        })
        groups.append({
            "params": [layer.alphas[m]],
            "lr": LR_ALPHA, "expert": f"ffn_{m}_alpha",
        })
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


def prepare_real_corpus_data(
    seed: int,
    encoders,
    corpus: MiniCorpus,
    n_per_class: int = 6,
    S: int = 16,
    D_SHARED: int = 256,
    N_CLS: int = 3,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    """v25 数据准备: text 模态用真实 Wikipedia, code/image 保留合成扰动.

    与 v22 prepare_data 的差异:
        - text 模态: 从 corpus.train_sentences 中按 seed 抽取 n_per_class 句, encode_text
        - code/image: 与 v22 完全一致 (合成)

    Args:
        seed:           训练 seed (用于 wikipedia 采样)
        encoders:       (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _)
        corpus:         v18 MiniCorpus 实例
        n_per_class:    每类采样数 (默认 6, 与 v22 一致)
        S:              序列长度
        D_SHARED:       共享维度
        N_CLS:          类别数 (3)

    Returns:
        (modal_seqs, modal_indices, targets):
            modal_seqs:  list of [S, D_m] 共 N_CLS * n_per_class 个
            modal_indices: list of int, 长度同上
            targets:    list of [S, D_SHARED] 同上
    """
    from examples.run_v8_full import encode_text, encode_code, encode_image

    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # ---- text 模态: 真实 Wikipedia (n_per_class 句) ----
    g = torch.Generator().manual_seed(seed)
    n_train = len(corpus.train_sentences)
    # 至少需要 n_per_class 个 (含重复也可, MiniCorpus 内部支持)
    idx = torch.randint(0, n_train, (n_per_class,), generator=g).tolist()
    texts = [corpus.train_sentences[i] for i in idx]
    h_text = encode_text(texts, bert_tok, bert)  # [n_per_class, S, D_bert]

    # ---- code 模态: 合成 (v22 风格, 保留以维持 N_CLS 一致) ----
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)  # [3, S, D_llama]

    # ---- image 模态: 合成 ----
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)  # [1, S, D_vit]

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(n_per_class):
            src = h_block[i % n]
            g2 = torch.Generator().manual_seed(seed * 10 + cls * n_per_class + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g2))
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(n_per_class):
            targets.append(t)

    return modal_seqs, modal_indices, targets


def prepare_held_out_data(
    encoders,
    corpus: MiniCorpus,
    S: int = 16,
    D_SHARED: int = 256,
    N_CLS: int = 3,
    n_per_class: int = 6,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    """v25 held-out 数据: 全用 corpus.held_out (6 句) 重复到 6 per class.

    Args:
        encoders, corpus, S, D_SHARED, N_CLS, n_per_class: 同 prepare_real_corpus_data

    Returns:
        (modal_seqs, modal_indices, targets), 格式与 prepare_real_corpus_data 一致
    """
    from examples.run_v8_full import encode_text, encode_code, encode_image

    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # text 模态: 全 held-out (6 句) 经 encode_text → [6, S, D_bert]
    h_text = encode_text(corpus.held_out, bert_tok, bert)
    # code/image: 与 prepare_real_corpus_data 保持一致 (合成)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(n_per_class):
            src = h_block[i % n]
            modal_seqs.append(src)
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(n_per_class):
            targets.append(t)
    return modal_seqs, modal_indices, targets
