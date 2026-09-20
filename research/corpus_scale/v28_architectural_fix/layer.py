"""v28.0 — ArchitecturalFixLayer 集成层 (委托 v25 RealCorpusTheoryLayer + v28 变体).

设计:
    - 不重写 v25 RealCorpusTheoryLayer 的结构
    - 在 layer 构造时, 根据 mode 字符串选择:
        - v22 4 个原始 mode (broadcast / gate / router-norm / adaptive-ema) → RealCorpusTheoryLayer
        - v28 7 个新 mode → 用变体替换 cb_attn / cb_ffn 中的 CentralTheory 实例
    - 提供 unified interface: 11 modes 都能跑通同一段训练代码
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.corpus_scale.v25_corpus_central.layer import RealCorpusTheoryLayer

from .router_norm_variants import (
    ROUTER_NORM_VARIANT_MODES,
    ROUTER_NORM_VARIANT_CLASSES,
    make_router_norm_variant,
)
from .adaptive_ema_variants import (
    ADAPTIVE_EMA_VARIANT_MODES,
    ADAPTIVE_EMA_VARIANT_CLASSES,
    make_adaptive_ema_variant,
)


# v28 所有 mode (含 v22 原始 + v28 变体)
V22_MODES = ["broadcast", "gate", "router-norm", "adaptive-ema"]
V28_VARIANT_MODES = ROUTER_NORM_VARIANT_MODES + ADAPTIVE_EMA_VARIANT_MODES
ALL_MODES = V22_MODES + V28_VARIANT_MODES


class ArchitecturalFixLayer(RealCorpusTheoryLayer):
    """v28 增强层: 支持 11 个中枢 mode (v22 原始 4 个 + v28 变体 7 个).

    与 v25 RealCorpusTheoryLayer 的差异:
        - mode 参数接受 v28 变体字符串 (router-norm-affine-false 等)
        - 构造时, 用 v28 变体替换 cb_attn / cb_ffn (而非 v22 RouterNormCentral/AdaptiveEMACentral)

    对 v22 原始 mode, 行为完全等同 RealCorpusTheoryLayer.
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        modal_dims: List[int],
        attn_pools,
        mode: str,
        stability_patches: dict = None,
    ):
        # v28 变体 mode: 不用父类构造, 直接接管 (避免父类创建错误的 cb_attn/cb_ffn)
        if mode in V28_VARIANT_MODES:
            nn.Module.__init__(self)
            self.d_shared = d_shared
            self.num_experts = num_experts
            self.mode = mode
            if stability_patches is None:
                stability_patches = {}
            self.freeze_u_router_norm = stability_patches.get(
                "freeze_u_router_norm", True
            )

            # aligner (冻结, 与 v22 一致)
            from research.aligner.v10_embedding import CrossArchAttnAligner
            aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=d_shared, num_heads=4)
            for p in aligner.parameters():
                p.requires_grad_(False)
            self.aligner = aligner

            # attn 路由
            self.attn_pools = nn.ModuleList(attn_pools)
            self.W_router_attn = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
            self.alphas_attn = nn.ParameterList(
                [nn.Parameter(torch.tensor([0.05])) for _ in range(num_experts)]
            )

            # ffn 路由 (与 v22 一致)
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

            # V_coop
            self.V_coop = nn.Parameter(torch.eye(d_shared) * 0.1)

            # === 关键差异: 用 v28 变体替换 cb_attn / cb_ffn ===
            self.cb_attn = make_router_norm_variant(mode, d_shared, num_experts) if mode in ROUTER_NORM_VARIANT_MODES \
                else make_adaptive_ema_variant(mode, d_shared, num_experts)
            self.cb_ffn = make_router_norm_variant(mode, d_shared, num_experts) if mode in ROUTER_NORM_VARIANT_MODES \
                else make_adaptive_ema_variant(mode, d_shared, num_experts)

            # 注: RouterNormPure 内部已创建 frozen U 占位, 让 v25 build_v25_param_groups 不报 None.

            # EMA 控制
            self._ema_enabled = True
        else:
            # v22 原始 mode: 委托父类
            super().__init__(
                d_shared=d_shared,
                num_experts=num_experts,
                modal_dims=modal_dims,
                attn_pools=attn_pools,
                mode=mode,
                stability_patches=stability_patches,
            )


def group_by_modal(modal_seqs, modal_indices, targets, N_CLS):
    """工具函数: 按模态分组数据 (与 v26 一致)."""
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    return h_by_modal, t_by_modal
