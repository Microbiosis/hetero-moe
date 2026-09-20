"""§4 — 训练协议：冷启动与渐进解锁。

实现 §4.1 参数初始化设定、§4.2 Phase 1 预热、§4.3 Phase 2 解锁，
以及学习率隔离策略（Router 1e-4 / 适配器 1e-2 / α_m 1e-3）。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.fusion import HeteroFusionLayer
from ..core.losses import total_loss


# ---- §4.2 / §4.3 学习率隔离常量 ----
LR_ROUTER = 1e-4     # §4.3: W_router
LR_ADAPTER = 1e-2    # §4.2/§4.3: γ_m, β_m (100x 补偿学习率)
LR_ALPHA = 1e-3      # §4.2/§4.3: α_m


def build_param_groups(layers: Sequence[HeteroFusionLayer], phase: int) -> List[Dict]:
    """构造优化器参数组 (§4.2 / §4.3)。

    phase=1: 不含 W_router (§4.2 冻结)；γ/β 用 1e-2，α_m 用 1e-3。
    phase=2: 含 W_router 1e-4，γ/β 用 1e-2，α_m 用 1e-3。
    """
    gammas_betas = [p for l in layers for p in (*l.gammas, *l.betas)]
    alphas = [p for l in layers for p in l.alphas]
    if phase == 1:
        return [
            {"params": gammas_betas, "lr": LR_ADAPTER},
            {"params": alphas, "lr": LR_ALPHA},
        ]
    routers = [l.W_router for l in layers]
    return [
        {"params": routers, "lr": LR_ROUTER},
        {"params": gammas_betas, "lr": LR_ADAPTER},
        {"params": alphas, "lr": LR_ALPHA},
    ]


class HeteroTrainer:
    """两阶段训练器 (§4)。

    使用方式:
        trainer = HeteroTrainer(layers, lm_head=frozen_head)
        trainer.begin_phase(1)            # §4.2 预热 (前 10% steps)
        for batch in phase1_loader: trainer.step(x, targets)
        trainer.begin_phase(2)            # §4.3 解锁 (后 90% steps)
        for batch in phase2_loader: trainer.step(x, targets)
    """

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
            objective: 'lm' (§3.1 交叉熵, 需 lm_head) 或 'mse'
                       (§7 原型梯度流验证风格, targets 为 [B,S,D] float)。
        """
        self.layers = list(layers)
        self.lm_head = lm_head
        self.lam_balance = lam_balance
        self.lam_z = lam_z
        self.top_k = top_k if top_k is not None else self.layers[0].top_k
        self.objective = objective
        self.phase: Optional[int] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._cur_lam1 = 0.0

    def begin_phase(self, phase: int) -> None:
        """切换阶段: 设置 requires_grad + 重建优化器 (§4.2/§4.3)。"""
        assert phase in (1, 2), "phase 必须为 1 或 2"
        self.phase = phase
        if phase == 1:
            # §4.2: 冻结 W_router
            for l in self.layers:
                l.W_router.requires_grad_(False)
            self._cur_lam1 = 0.0  # §4.2 强制 λ1=0
        else:
            # §4.3: 解冻 W_router
            for l in self.layers:
                l.W_router.requires_grad_(True)
            self._cur_lam1 = self.lam_balance  # §4.3 λ1>0
        # 重建优化器 (phase 切换后状态隔离, 避免动量污染)
        self.optimizer = torch.optim.AdamW(build_param_groups(self.layers, phase))

    def forward(self, x: torch.Tensor):
        """层间传递 (§7 注: 第一层输出作为第二层输入, 真正深层而非单层并行)。"""
        y = x
        layer_outs = []  # 每层 (z, alpha, alpha_hat)
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
        """§3 总损失。L_balance / L_z 在所有融合层上求和 (每层独立贡献)。

        objective='lm'  (§3.1): targets 为 token id, 需 lm_head。
        objective='mse' (§7 原型风格): targets 为 [B,S,D] float, 直接对隐藏态回归,
                              用于梯度流验证 (不需要 lm_head)。
        """
        from ..core.losses import balance_loss, z_loss, lm_loss

        if self.objective == "mse":
            l_primary = F.mse_loss(y, targets)
        else:
            logits = self.lm_head(y) if self.lm_head is not None else y
            l_primary = lm_loss(logits, targets)
        # 汇总各层 z / alpha / alpha_hat
        z_cat = torch.stack([zo[0] for zo in layer_outs], dim=0)  # [L, B, S, M]
        alpha_cat = torch.stack([zo[1] for zo in layer_outs], dim=0)
        ahat_cat = torch.stack([zo[2] for zo in layer_outs], dim=0)
        # 逐层计算并求和
        l_bal = sum(balance_loss(alpha_cat[i], ahat_cat[i], self.top_k) for i in range(len(layer_outs)))
        l_z = sum(z_loss(z_cat[i]) for i in range(len(layer_outs)))
        total = l_primary + self._cur_lam1 * l_bal + self.lam_z * l_z
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
