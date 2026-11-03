"""MoA 异构 proposer 融合层 —— 替代 v62 同构 SwiGLU 专家。

动机（v62 诊断结论）：
  1. v62 的 4 个专家全是 SwiGLU FFN（同构），架构异构只在输入模态；
  2. 专家是 pointwise FFN，不建模 token 关系 → 探针证明融合空间只有
     线性+微弱局部结构（mlp=linear，无可学非线性）；
  3. 一次性前馈，无迭代精炼；
  4. 路由正反馈塌缩（路线1/8 踩坑）。

本模块用 MoA 库的 HeterogeneousProposerOp（Attn/SSM/Conv/SwiGLU 四异构分支）
+ CrossProposalAggregatorOp（sparse_router，内置 Load-Balancing + Z-Loss）
替换融合层，接口与 FusionOverProjSpace.forward_with_routing 兼容。

关键升级：proposer 在融合前引入序列建模，直接增加融合空间的关系表达维度。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from moa_operators import (
    MoAOperatorConfig, IterativeMoAOperatorBlock,
    NumpyMoAOperatorKernels,
)


class MoAFusionOverProjSpace(nn.Module):
    """MoA 异构 proposer 融合层，接口兼容 v62 FusionOverProjSpace。

    输入 canonical [B,S,d_space]，输出 [B,S,d_space] + alpha 占位。
    aux_loss（Load-Balancing + Z-Loss）需由训练脚本加入总 loss。
    """

    def __init__(self, d_space: int = 1024, num_proposers: int = 4,
                 n_heads: int = 8, top_k: int = 2,
                 aggregator_mode: str = 'sparse_router',
                 num_refinement_steps: int = 1,
                 enable_halting: bool = False):
        super().__init__()
        self.d_space = d_space
        self.num_proposers = num_proposers
        self.num_experts = num_proposers      # 兼容 train_v2 的专家计数
        self.top_k = top_k
        self.cfg = MoAOperatorConfig(
            d_model=d_space, n_heads=n_heads, num_proposers=num_proposers,
            top_k_proposers=top_k, proposer_mode='heterogeneous',
            aggregator_mode=aggregator_mode,
            num_refinement_steps=num_refinement_steps,
            enable_adaptive_halting=enable_halting,
            aux_loss_weight=0.01, z_loss_weight=0.001,
        )
        self.block = IterativeMoAOperatorBlock(self.cfg)
        self.last_aux_loss = 0.0
        self.last_importance = None

    def forward(self, x: torch.Tensor):
        """x: [B,S,d_space] → (y, aux_loss, metrics)。"""
        y, aux_loss, metrics = self.block(x)
        self.last_aux_loss = aux_loss
        self.last_importance = metrics.get('proposal_importance')
        return y, aux_loss, metrics

    def forward_with_routing(self, x: torch.Tensor):
        """兼容 v62 接口：返回 (y, alpha)。

        alpha 用 proposal_importance 扩展为 [B,S,num_proposers] 占位，
        让 train_v2 的负载均衡统计有值可算（真实均衡由 aux_loss 保证）。
        """
        y, aux_loss, metrics = self.block(x)
        self.last_aux_loss = aux_loss
        imp = metrics.get('proposal_importance')          # [num_proposers]
        B, S, _ = x.shape
        if imp is None:
            alpha = torch.ones(B, S, self.num_proposers, device=x.device) / self.num_proposers
        else:
            alpha = imp.view(1, 1, -1).expand(B, S, -1).to(x.device)
        return y, alpha

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return ('MoAFusionOverProjSpace\n'
                '  d_space=%d, proposers=%d (Attn/SSM/Conv/SwiGLU 异构)\n'
                '  aggregator=%s top_k=%d, refinement_steps=%d\n'
                '  参数=%d（v62 同构MoE为 %d）'
                % (self.d_space, self.num_proposers, self.cfg.aggregator_mode,
                   self.top_k, self.cfg.num_refinement_steps, n,
                   4 * 3 * 4 * self.d_space * self.d_space))
