"""v7.0 — per-expert 优化器训练协议。

设计:
    - 单一 AdamW + 多个 param group (每 expert 一组), 异构学习率
    - 不使用 List[AdamW] (实现复杂且对单卡 CPU 无收益)
    - phase 协议:
        phase=1: 冻结两个 router (W_router_attn, W_router_ffn), 仅训适配器
        phase=2: 解冻 router, 全量训练
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import CrossArchAttnFFNFusionLayer

# ---- 学习率隔离常量 (与 v4 §4.2/§4.3 保持一致语义) ----
LR_ROUTER = 1e-4     # W_router_attn / W_router_ffn
LR_ADAPTER = 1e-2    # attn W_o / ffn γ, β
LR_ALPHA = 1e-3      # α_attn / α_ffn


def build_v7_param_groups(
    layer: CrossArchAttnFFNFusionLayer,
    phase: int,
) -> List[Dict]:
    """构造优化器参数组: 每 expert 一个 group。

    group 索引:
        0:                            W_router_attn (若 phase=2)
        1..M_attn:                    每 attn 专家的 W_o
        M_attn+1:                     W_router_ffn (若 phase=2)
        M_attn+2..M_attn+1+M_ffn:     每 ffn 专家的 γ/β/α

    Returns:
        list[dict], 每 dict 含 {"params": [...], "lr": float, "expert": str}
    """
    groups: List[Dict] = []

    # attn router
    if phase == 2:
        groups.append({
            "params": [layer.W_router_attn],
            "lr": LR_ROUTER,
            "expert": "router_attn",
        })

    # 每 attn 专家 (W_o) 一个 group
    for m, ap in enumerate(layer.attn_pools):
        groups.append({
            "params": [ap.W_o],
            "lr": LR_ADAPTER,
            "expert": f"attn_{m}",
        })

    # ffn router
    if phase == 2:
        groups.append({
            "params": [layer.W_router_ffn],
            "lr": LR_ROUTER,
            "expert": "router_ffn",
        })

    # 每 ffn 专家 (γ/β/α) 一个 group — α 用 LR_ALPHA, γ/β 用 LR_ADAPTER
    for m in range(layer.num_experts_ffn):
        groups.append({
            "params": [layer.gammas[m], layer.betas[m]],
            "lr": LR_ADAPTER,
            "expert": f"ffn_{m}_adapter",
        })
        groups.append({
            "params": [layer.alphas[m]],
            "lr": LR_ALPHA,
            "expert": f"ffn_{m}_alpha",
        })

    return groups


class V7Trainer:
    """per-expert 优化器训练器 (单一 AdamW + 多 param group)。"""

    def __init__(
        self,
        layer: CrossArchAttnFFNFusionLayer,
        lr: float = 1e-2,
        phase: int = 2,
    ):
        self.layer = layer
        self.lr = lr
        self.phase = phase
        self._set_router_grad(phase)
        self.optimizer = torch.optim.AdamW(
            build_v7_param_groups(layer, phase), lr=lr
        )

    def _set_router_grad(self, phase: int) -> None:
        if phase == 1:
            self.layer.W_router_attn.requires_grad_(False)
            self.layer.W_router_ffn.requires_grad_(False)
        else:
            self.layer.W_router_attn.requires_grad_(True)
            self.layer.W_router_ffn.requires_grad_(True)

    def begin_phase(self, phase: int) -> None:
        self.phase = phase
        self._set_router_grad(phase)
        self.optimizer = torch.optim.AdamW(
            build_v7_param_groups(self.layer, phase), lr=self.lr
        )

    def step(
        self,
        x_shared_list: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> float:
        """单步训练: x_shared_list[i] 与 targets[i] 对齐 (per-sample MSE 累加后均值)。"""
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for x, t in zip(x_shared_list, targets):
            y = self.layer(x)
            losses.append(F.mse_loss(y, t))
        total = sum(losses) / len(losses)
        total.backward()
        self.optimizer.step()
        return float(total.item())

    def param_group_count(self) -> int:
        """返回 param group 数 (测试用)。"""
        return len(self.optimizer.param_groups)
