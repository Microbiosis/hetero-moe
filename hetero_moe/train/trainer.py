"""§4 / v5.1 — 训练协议：冷启动、渐进解锁与跨阶段 Adam 动量状态无损保留。

v5.0 / v5.1 修复:
    1. Phase 1 z_loss 梯度污染修复: Phase 1 冻结 W_router 时同时强制 _cur_lam_z = 0.0，
       防止深层 (l >= 2) 的 z_l = y_{l-1} W_router^T 把前层适配器拉向冻结随机矩阵的零空间。
    2. 支持 v5.0/v5.1 异构专家投影 (in_proj/out_proj)、低秩向量门控 (gate_u/gate_v) 与内部子层 LoRA 分组。
    3. 跨阶段 Adam 动量保留 (v5.1):
       `begin_phase(2)` 从 Phase 1 切换到 Phase 2 时不再销毁重建 `AdamW`（避免清空 Phase 1 积累的
       `exp_avg` / `exp_avg_sq` 一阶二阶动量导致冷启动震荡），而是通过 `optimizer.add_param_group`
       增量注入新解冻的 `W_router` 参数组。
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
LR_ALPHA = 1e-3      # §4.2/§4.3: α_m, 低秩向量门控, 内部子层 LoRA/解冻参数


def _dedup_trainable(params: Sequence[nn.Parameter]) -> List[nn.Parameter]:
    seen = set()
    out: List[nn.Parameter] = []
    for p in params:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def build_param_groups(layers: Sequence[HeteroFusionLayer], phase: int) -> List[Dict]:
    """构造优化器参数组 (§4.2 / §4.3 / v5.1)。

    phase=1: 不含 W_router (§4.2 冻结)；适配器用 1e-2，门控/LoRA 用 1e-3。
    phase=2: 含 W_router 1e-4，适配器用 1e-2，门控/LoRA 用 1e-3。
    """
    adapters = _dedup_trainable([p for l in layers for p in l.adapter_parameters()])
    gates_lora = _dedup_trainable([p for l in layers for p in l.gate_and_lora_parameters()])
    groups: List[Dict] = []
    if phase == 2:
        routers = _dedup_trainable([l.W_router for l in layers])
        if routers:
            groups.append({"params": routers, "lr": LR_ROUTER, "group_role": "router"})
    if adapters:
        groups.append({"params": adapters, "lr": LR_ADAPTER, "group_role": "adapter"})
    if gates_lora:
        groups.append({"params": gates_lora, "lr": LR_ALPHA, "group_role": "gate_lora"})
    return groups


class HeteroTrainer:
    """两阶段训练器 (§4 / v5.1)。"""

    def __init__(
        self,
        layers: Sequence[HeteroFusionLayer],
        lm_head: Optional[Callable] = None,
        lam_balance: float = 0.01,
        lam_z: float = 0.001,
        top_k: Optional[int] = None,
        objective: str = "lm",
    ):
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

    def begin_phase(self, phase: int, preserve_optimizer_state: bool = True) -> None:
        """切换阶段: 设置 requires_grad + 辅助损失门控 + 增量更新或重建优化器 (§4.2/§4.3/v5.1)。"""
        assert phase in (1, 2), "phase 必须为 1 或 2"
        self.phase = phase
        if phase == 1:
            for l in self.layers:
                l.W_router.requires_grad_(False)
            self._cur_lam1 = 0.0
            self._cur_lam_z = 0.0
        else:
            for l in self.layers:
                l.W_router.requires_grad_(True)
            self._cur_lam1 = self.lam_balance
            self._cur_lam_z = self.lam_z

        if self.optimizer is None or not preserve_optimizer_state:
            self.optimizer = torch.optim.AdamW(build_param_groups(self.layers, phase))
            return

        # v5.1: 保留已有 Adam 动量状态 (exp_avg / exp_avg_sq)，增量同步 W_router 参数组
        existing_ids = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        if phase == 2:
            routers = _dedup_trainable([l.W_router for l in self.layers])
            new_routers = [p for p in routers if id(p) not in existing_ids]
            if new_routers:
                self.optimizer.add_param_group({"params": new_routers, "lr": LR_ROUTER, "group_role": "router"})
            for g in self.optimizer.param_groups:
                if g.get("group_role") == "router":
                    g["lr"] = LR_ROUTER
        else:
            for g in self.optimizer.param_groups:
                if g.get("group_role") == "router":
                    g["lr"] = 0.0

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
