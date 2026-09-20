"""v8.0 — per-expert + 中枢 param group 训练器。

相对 V7Trainer 的差异:
    - 新增 CentralWorkspace 的参数 (c, U)
    - 新增 V_coop (C-2 协同矩阵)
    - 新增 HierarchicalCentralLayer 的 central_expert 参数 + central_alpha
    - 中枢相关的 param group 用单独学习率 (LR_CENTRAL = 5e-3, 中等)
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .fusion import CentralAugmentedFusionLayer
from .layer import HierarchicalCentralLayer

LR_ROUTER = 1e-4
LR_ADAPTER = 1e-2
LR_ALPHA = 1e-3
LR_CENTRAL = 5e-3       # 中枢参数 (c, U, V_coop) — 比路由快, 比适配器慢
LR_CENTRAL_EXPERT = 1e-3  # 中枢 expert (层级中枢)


def build_v8_param_groups(
    layer: CentralAugmentedFusionLayer,
    phase: int = 2,
) -> List[Dict]:
    """构造 v8 优化器参数组。

    包含:
        - W_router_attn / W_router_ffn (phase=2)
        - 每 attn 专家的 W_o
        - 每 ffn 专家的 γ/β/α
        - C-1 中枢: cw_attn.c, cw_attn.U, cw_ffn.c, cw_ffn.U
        - C-2 协同: V_coop
        - C-3 层级中枢 (仅 HierarchicalCentralLayer):
            central_expert 参数 + central_alpha + W_router_central + U_central
    """
    groups: List[Dict] = []
    # attn router
    if phase == 2:
        groups.append({"params": [layer.W_router_attn], "lr": LR_ROUTER, "expert": "router_attn"})
    # 每 attn 专家 (W_o)
    for m, ap in enumerate(layer.attn_pools):
        groups.append({"params": [ap.W_o], "lr": LR_ADAPTER, "expert": f"attn_{m}"})
    # ffn router
    if phase == 2:
        groups.append({"params": [layer.W_router_ffn], "lr": LR_ROUTER, "expert": "router_ffn"})
    # 每 ffn 专家
    for m in range(layer.num_experts_ffn):
        groups.append({
            "params": [layer.gammas[m], layer.betas[m]],
            "lr": LR_ADAPTER, "expert": f"ffn_{m}_adapter",
        })
        groups.append({
            "params": [layer.alphas[m]],
            "lr": LR_ALPHA, "expert": f"ffn_{m}_alpha",
        })
    # C-1 中枢 (cw_attn, cw_ffn): c 必训; U (BroadcastWorkspace) 或 W_t (GateStyleWorkspace) 可选
    central_params = [layer.cw_attn.c, layer.cw_ffn.c]
    for cw in (layer.cw_attn, layer.cw_ffn):
        if hasattr(cw, "U"):
            central_params.append(cw.U)
        if hasattr(cw, "W_t"):
            central_params.append(cw.W_t)
    groups.append({
        "params": central_params,
        "lr": LR_CENTRAL, "expert": "central_workspace",
    })
    # C-2 协同矩阵 V_coop
    groups.append({
        "params": [layer.V_coop],
        "lr": LR_CENTRAL, "expert": "central_coop",
    })
    # C-3 层级中枢 (仅 HierarchicalCentralLayer)
    if isinstance(layer, HierarchicalCentralLayer):
        central_expert_params = list(layer.central_expert.parameters())
        groups.append({
            "params": central_expert_params,
            "lr": LR_CENTRAL_EXPERT, "expert": "central_expert",
        })
        groups.append({
            "params": [layer.central_alpha, layer.W_router_central, layer.U_central],
            "lr": LR_CENTRAL, "expert": "central_router",
        })
    return groups


class V8Trainer:
    """per-expert + 中枢训练器。

    用法与 V7Trainer 一致, step() 返回 per-batch MSE 标量.
    EMA 中枢更新在 layer.forward() 内部完成 (无需 trainer 干预).
    """

    def __init__(
        self,
        layer: CentralAugmentedFusionLayer,
        lr: float = 1e-2,
        phase: int = 2,
    ):
        self.layer = layer
        self.lr = lr
        self.phase = phase
        self._set_router_grad(phase)
        self.optimizer = torch.optim.AdamW(
            build_v8_param_groups(layer, phase), lr=lr
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
            build_v8_param_groups(self.layer, phase), lr=self.lr
        )

    def step(self, x_shared_list: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
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
        return len(self.optimizer.param_groups)
