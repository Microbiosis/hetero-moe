"""v27.0 — RegularizedTheoryLayer 集成层 (委托 v25 RealCorpusTheoryLayer + v27 patches).

设计:
    - 不重写 v25 RealCorpusTheoryLayer 的结构
    - 在 layer 构造后, 应用 v27 patches.py 中的补丁
    - 提供 forward_with_loss: 自动把 L2 正则项加到 MSE loss 中
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.corpus_scale.v25_corpus_central.layer import RealCorpusTheoryLayer

from .patches import (
    apply_l2_to_c,
    get_l2_to_c_loss,
    apply_all_patches,
    DEFAULT_PATCH_CONFIGS,
)


class RegularizedTheoryLayer(RealCorpusTheoryLayer):
    """v27 增强层: 在 v22 RealCorpusTheoryLayer 上应用 v27 4 个 held-out 正则化补丁.

    与 v25/v26 RealCorpusTheoryLayer 的差异:
        - 构造后自动应用 patch_config 中的补丁 (apply_l2_to_c / router_dropout / 等)
        - 提供 compute_total_loss: 自动把 L2 正则项加到 MSE loss 中
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        modal_dims: List[int],
        attn_pools,
        mode: str,
        stability_patches: Dict = None,
        patch_config: Dict = None,
    ):
        """构造 layer, 可选地应用 v27 patches.

        Args:
            stability_patches: 与 v22/v25 一致的稳定性 patch dict (freeze_u_router_norm 等)
            patch_config: v27 patch config dict, 包含要应用的 4 个补丁
        """
        super().__init__(
            d_shared=d_shared,
            num_experts=num_experts,
            modal_dims=modal_dims,
            attn_pools=attn_pools,
            mode=mode,
            stability_patches=stability_patches,
        )
        # 默认 patch_config 为空 (即 baseline, 等同 v26)
        if patch_config is None:
            patch_config = {}
        # 应用 v27 补丁
        apply_all_patches(self, patch_config)
        # 标记是否应用了补丁 (供端到端脚本验证)
        self._patch_config = patch_config


def compute_total_loss(layer: RegularizedTheoryLayer, h_by_modal, t_by_modal,
                       N_CLS: int = 3) -> torch.Tensor:
    """计算训练 loss: MSE + L2 正则.

    与 v22/v25/v26 的区别: 自动加 L2 正则 (如果 patch_config 包含 apply_l2_to_c).

    用法:
        y = layer(h_by_modal)
        loss = compute_total_loss(layer, h_by_modal, t_by_modal, N_CLS=3)
        loss.backward()
    """
    y = layer(h_by_modal)
    mse = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
    # 加 L2 正则 (如果有)
    if hasattr(layer, "_l2_c_params"):
        l2_term = get_l2_to_c_loss(layer)
        lambda_c = layer._l2_lambda_c
        total = mse + lambda_c * l2_term
    else:
        total = mse
    return total
