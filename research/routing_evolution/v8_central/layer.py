"""v8.0 — C-3 层级中枢 (Hierarchical Central Layer)。

设计:
    在 CentralAugmentedFusionLayer (C-2) 基础上, 加一个"中枢 expert":
    - 中枢 expert 是一个全连接 + LayerNorm 模块 (可训练)
    - 它不参与普通 expert 池, 但接收所有 attn+ffn 输出后做"元认知聚合"
    - 聚合结果作为一个额外的"专家输出"广播回主输出

类比:
    网状激活系统 (RAS) 调节全脑警觉度.
    前额叶皮层 (PFC) 接收全脑信息做元认知决策.

实现:
    central_expert: Linear(D_shared, D_shared) + LayerNorm
    central_attn_weight: Scalar (中枢 attn 输出权重)
    central_ffn_weight: Scalar (中枢 ffn 输出权重)

forward:
    attn_sum, ffn_sum = 各 expert 池输出 (C-2)
    central_out = LN(central_expert(mean([attn_sum, ffn_sum])))
    y = x + attn_sum + ffn_sum + central_out  (中枢作为额外"准 expert")
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research._primitives.attention import AttnPool

from .central import CentralWorkspace
from .fusion import CentralAugmentedFusionLayer


class HierarchicalCentralLayer(CentralAugmentedFusionLayer):
    """v8.0 C-3: 层级中枢 = C-2 + 中枢 expert.

    在 C-2 基础上多一个 central_expert:
        输入:  mean(attn_sum, ffn_sum)  (元认知聚合)
        输出:  Linear + LayerNorm + 残差
    权重:  self.central_alpha (可学习)
    """

    def __init__(self, *args, central_alpha_init: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        # 中枢 expert: 全连接 + LayerNorm
        self.central_expert = nn.Sequential(
            nn.Linear(self.d_shared, self.d_shared),
            nn.LayerNorm(self.d_shared),
            nn.GELU(),
            nn.Linear(self.d_shared, self.d_shared),
        )
        # 中枢输出权重 (可学习, 初始小值)
        self.central_alpha = nn.Parameter(torch.tensor(central_alpha_init))
        # 中枢路由 (一个可学习的"是否启用中枢"门)
        self.W_router_central = nn.Parameter(torch.randn(1, self.d_shared) * 0.01)
        self.U_central = nn.Parameter(torch.zeros(1))  # 中枢自身的"元认知广播"

    def forward(self, x_shared: torch.Tensor) -> torch.Tensor:
        attn_sum, _, attn_outs = self._attn_path(x_shared)
        ffn_sum, _, ffn_outs = self._ffn_path(x_shared, attn_sum)

        # C-3 层级中枢: 聚合 → 元认知 → 广播
        meta_input = (attn_sum + ffn_sum) / 2
        central_out = self.central_expert(meta_input)
        # 中枢路由 logits (单 expert, 决定是否启用中枢)
        z_c = F.linear(meta_input.detach(), self.W_router_central)
        alpha_c = torch.sigmoid(z_c + self.U_central)  # [B, S, 1]
        central_contrib = alpha_c * self.central_alpha * central_out

        # EMA 中枢更新 (C-1)
        if self._ema_enabled and self.training:
            self.cw_attn.ema_update(attn_outs)
            self.cw_ffn.ema_update(ffn_outs)
            # 中枢 c 的 EMA: 用中枢 expert 自身输出
            self.cw_attn.ema_update([central_out.detach()])

        return x_shared + attn_sum + ffn_sum + central_contrib

    def forward_with_routing(self, x_shared: torch.Tensor):
        attn_sum, ahat_a, _ = self._attn_path(x_shared)
        ffn_sum, ahat_f, _ = self._ffn_path(x_shared, attn_sum)
        meta_input = (attn_sum + ffn_sum) / 2
        central_out = self.central_expert(meta_input)
        z_c = F.linear(meta_input.detach(), self.W_router_central)
        alpha_c = torch.sigmoid(z_c + self.U_central)
        central_contrib = alpha_c * self.central_alpha * central_out
        y = x_shared + attn_sum + ffn_sum + central_contrib
        return y, ahat_a, ahat_f
