"""v8.0 — C-1 全局中枢 (Central Workspace, broadcast 模式, 最小实现).

设计:
    中枢 token c ∈ R^{D_shared} 是全局可学习向量, 代表"全脑概要".
    路由 logits 受 c 广播影响:  z += U @ c
    每个 step 后用 expert 输出的均值 EMA 更新 c.

类比:
    丘脑 (thalamus) 把全脑概要广播到各皮层.
    Global Workspace Theory: 意识来自信息跨脑区广播.

本文件是 v8.0 的最小实现, 仅支持 broadcast 模式.
v9 的三种修正 (freeze_u / init_scale / gate-style) 已迁到 v9_gate_central:
    修正 A (freeze_u):     v9_gate_central.freeze_broadcast_matrix(cw)
    修正 B (init_scale):   v9_gate_central.set_broadcast_init_scale(cw, 0.001)
    修正 C (gate-style):   v9_gate_central.GateStyleWorkspace (推荐, 当前最优)

v10/v12/v15 的 fusion.py 内部已统一改为调用 v9_gate_central.GateStyleWorkspace.

API:
    cw = CentralWorkspace(d_shared=256, num_experts=3)
    z_aug = cw.augment_router_logits(z_router)        # [B, S, M]
    cw.ema_update(expert_outputs)                     # 在训练 step 后调用
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CentralWorkspace(nn.Module):
    """全局中枢 token + 广播矩阵 + EMA 更新 (broadcast 模式).

    最小实现: 只支持 v8.0 broadcast (z += U @ c).
    不支持 freeze_u / mode 参数, 这些是 v9 修正 A/C 的实现, 见 v9_gate_central.

    Args:
        d_shared:    共享维度 (与 v7 的 D_shared 一致)
        num_experts: 每个路由池的专家数 (v7 中 M_attn = M_ffn = 3)
        ema_decay:   EMA 衰减系数 (默认 0.9)
        init_scale:  U 初始化 scale (默认 0.01)

    警告:
        v8 broadcast 模式被 V8.1 根因分析确认会破坏 v7 路由 (H1 假设验证).
        新代码请用 v9_gate_central.GateStyleWorkspace.
    """

    def __init__(
        self, d_shared: int, num_experts: int, ema_decay: float = 0.9,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.ema_decay = ema_decay

        # 中枢 token: 全局概要 (初始为 0, 确保开始训练时中枢不干扰路由)
        self.c = nn.Parameter(torch.zeros(d_shared))
        # 广播矩阵 U: num_experts × d_shared, 把中枢投影到路由 logits 空间
        self.U = nn.Parameter(torch.randn(num_experts, d_shared) * init_scale)

    def augment_router_logits(self, z_router: torch.Tensor) -> torch.Tensor:
        """z_router: [B, S, M] → broadcast 增强后的 logits.

        Returns:
            z_router + U @ c   (v8.0 行为, 加偏置)
        """
        bias = self.U @ self.c  # [M]
        return z_router + bias

    @torch.no_grad()
    def ema_update(self, expert_outputs: List[torch.Tensor]) -> None:
        """EMA 更新中枢 c.

        Args:
            expert_outputs: 每个 expert 的输出 [B, S, D_shared] 的 list
                           (取所有 expert 的均值作为新中枢)
        """
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)  # [E, B, S, D]
        new_c = stacked.mean(dim=(0, 1, 2))           # [D]
        self.c.data.mul_(self.ema_decay).add_(new_c, alpha=1.0 - self.ema_decay)

    def snapshot(self) -> torch.Tensor:
        """推理时取中枢快照 (避免 in-place 修改干扰其他样本)"""
        return self.c.detach().clone()
