"""§4 / v5.0 — 训练协议：冷启动与渐进解锁。

v5.0 修复:
    1. Phase 1 z_loss 梯度污染修复: Phase 1 冻结 W_router 时同时强制 _cur_lam_z = 0.0，
       防止深层 (l >= 2) 的 z_l = y_{l-1} W_router^T 把前层适配器拉向冻结随机矩阵的零空间。
    2. 支持 v5.0 异构专家投影 (in_proj/out_proj)、低秩向量门控 (gate_u/gate_v) 与可选
       LoRA/解冻参数分组。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.fusion import HeteroFusionLayer
from ..core.losses import balance_loss, lm_loss, z_loss


# ---- §4.2 / §4.3 学习率隔离常量 ----
LR_ROUTER = 1e-4     # §4.3: W_router
LR_ADAPTER = 1e-2    # §4.2/§4.3: γ_m, β_m, 异构投影适配器
LR_ALPHA = 1e-3      # §4.2/§4.3: α_m, 低秩向量门控, LoRA/解冻参数


def _dedup_trainable(params: Sequence[nn.Parameter]) -> List[nn.Parameter]:
    seen = set()
    out: List[nn.Parameter] = []
    for p in params:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def build_param_groups(layers: Sequence[HeteroFusionLayer], phase: int) -> List[Dict]:
    """构造优化器参数组 (§4.2 / §4.3 / v5.0)。

    phase=1: 不含 W_router (§4.2 冻结)；适配器用 1e-2，门控/LoRA 用 1e-3。
    phase=2: 含 W_router 1e-4，适配器用 1e-2，门控/LoRA 用 1e-3。
    """
    adapters = _dedup_trainable([p for l in layers for p in l.adapter_parameters()])
    gates_lora = _dedup_trainable([p for l in layers for p in l.gate_and_lora_parameters()])
    groups: List[Dict] = []
    if phase == 2:
        routers = _dedup_trainable([l.W_router for l in layers])
        if routers:
            groups.append({"params": routers, "lr": LR_ROUTER})
    if adapters:
        groups.append({"params": adapters, "lr": LR_ADAPTER})
    if gates_lora:
        groups.append({"params": gates_lora, "lr": LR_ALPHA})
    return groups


class HeteroTrainer:
    """两阶段训练器 (§4 / v5.0)。"""

    def __init__(
        self,
        layers: Sequence[HeteroFusionLayer],
        lm_head: Optional[Callable] = None,
        lam_balance: float = 0.01,
        lam_z: float = 0.001,
        top_k: Optional[int] = None,
        objective: str = "lm",
    ):
        """Args:
            objective: 'lm' (§3.1 交叉熵, 需 lm_head) 或 'mse' (隐藏态回归)。
        """
        self.layers = list(layers)
        self.lm_head = lm_head
        self.lam_balance = float(lam_balance)
        self.lam_z = float(lam_z)
        self.top_k = top_k if top_k is not None else self.layers[0].top_k
        self.objective = objective
        self.phase: Optional[int] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._cur_lam1 = 0.0
        self._cur_lam_z = 0.0

    def begin_phase(self, phase: int) -> None:
        """切换阶段: 设置 requires_grad + 辅助损失门控 + 重建优化器 (§4.2/§4.3)。"""
        assert phase in (1, 2), "phase 必须为 1 或 2"
        self.phase = phase
        if phase == 1:
            # §4.2: 冻结 W_router，同时关闭 balance_loss 与 z_loss（防止梯度污染前层适配器）
            for l in self.layers:
                l.W_router.requires_grad_(False)
            self._cur_lam1 = 0.0
            self._cur_lam_z = 0.0
        else:
            # §4.3: 解冻 W_router 并恢复辅助损失
            for l in self.layers:
                l.W_router.requires_grad_(True)
            self._cur_lam1 = self.lam_balance
            self._cur_lam_z = self.lam_z
        self.optimizer = torch.optim.AdamW(build_param_groups(self.layers, phase))

    def forward(self, x: torch.Tensor):
        """层间传递 (第一层输出作为第二层输入)。"""
        y = x
        layer_outs = []
        for layer in self.layers:
            y, z, alpha_hat = layer(y, phase1=(self.phase == 1))
            alpha = F.softmax(z, dim=-1)
            layer_outs.append((z, alpha, alpha_hat))
        return y, layer_outs

    def compute_loss(
        self,
        y: torch.Tensor,
        targets: torch.Tensor,
        layer_outs,
    ) -> Dict[str, torch.Tensor]:
        """§3 总损失。Phase 1 下 balance 与 z_loss 权重均为 0（但仍记录诊断值）。"""
        if self.objective == "mse":
            l_primary = F.mse_loss(y, targets)
        else:
            logits = self.lm_head(y) if self.lm_head is not None else y
            l_primary = lm_loss(logits, targets)

        z_cat = torch.stack([zo[0] for zo in layer_outs], dim=0)
        alpha_cat = torch.stack([zo[1] for zo in layer_outs], dim=0)
        ahat_cat = torch.stack([zo[2] for zo in layer_outs], dim=0)

        l_bal = sum(balance_loss(alpha_cat[i], ahat_cat[i], self.top_k) for i in range(len(layer_outs)))
        l_z = sum(z_loss(z_cat[i]) for i in range(len(layer_outs)))
        total = l_primary + self._cur_lam1 * l_bal + self._cur_lam_z * l_z
        return {"total": total, "lm": l_primary, "balance": l_bal, "z": l_z}

    def step(self, x: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """单步训练: forward → loss → backward → optimizer.step。"""
        assert self.optimizer is not None, "需先 begin_phase()"
        self.optimizer.zero_grad(set_to_none=True)
        y, layer_outs = self.forward(x)
        losses = self.compute_loss(y, targets, layer_outs)
        losses["total"].backward()
        self.optimizer.step()
        return {k: float(v.item()) if torch.is_tensor(v) else v for k, v in losses.items()}
