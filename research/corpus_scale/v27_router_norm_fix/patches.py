"""v27.0 — router-norm / adaptive-ema held-out 修正补丁.

设计:
    v26 在 fully-real-corpus 上发现 router-norm 和 adaptive-ema 在 held-out 上退步:
        - training set: gate (+11.1%) > adaptive-ema (+7.9%) > router-norm (+3.8%) > broadcast
        - held-out:    gate (+2.3%) > broadcast > router-norm (-5.1%) ≈ adaptive-ema (-6.6%)

    v22 的现有 patch (freeze_u / warmup / set_u_init_scale / clip_grad) 无法解决这个问题.
    v27 提出 4 个新 patch, 分别从梯度和 EMA 两个通道入手.

4 个补丁:
    Patch A: apply_l2_to_c     - 在 c 参数上加 L2 正则 (weight decay 风格)
    Patch B: router_dropout    - 在 router logits 上加 Dropout
    Patch C: extend_warmup_steps - 把 AdaptiveEMA 的 warmup 从 10 → 100
    Patch D: noise_inject_c    - 训练 forward 时给 c 加 Gaussian 噪声

可追溯性:
    apply_l2_to_c         -> 给 c 加 ||c||^2 项 (供训练循环 loss += lambda * term)
    router_dropout        -> 在 CentralTheory.augment_router_logits 内置 nn.Dropout
    extend_warmup_steps   -> 直接修改 central.adaptive_warmup_steps 属性
    noise_inject_c        -> 在 CentralTheory.augment_router_logits 中给 c 加噪声
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from research.central_stability.v22_central_theory import CentralTheory


# ---------------------------------------------------------------------------
# Patch A: apply_l2_to_c
# ---------------------------------------------------------------------------


def apply_l2_to_c(layer: nn.Module, lambda_c: float = 1e-4) -> None:
    """Patch A: 在 c 参数上加 L2 正则 (weight decay 风格).

    解决: c 梯度过大导致记忆训练分布.

    用法 (训练循环):
        loss = sum(F.mse_loss(...)) / N + get_l2_to_c_loss(layer) * lambda_c
        loss.backward()

    参数:
        layer: 包含 cb_attn / cb_ffn 的 layer (v22/v25/v26 的 TheoryLayer)
        lambda_c: L2 系数 (默认 1e-4, 适配 LR_CENTRAL=5e-3)
    """
    layer._l2_lambda_c = lambda_c
    # 预先计算 cb_attn.c / cb_ffn.c 的 list, 加速 loss 计算
    layer._l2_c_params = [layer.cb_attn.c, layer.cb_ffn.c]


def get_l2_to_c_loss(layer: nn.Module) -> torch.Tensor:
    """计算 c 的 L2 正则项: ||cb_attn.c||^2 + ||cb_ffn.c||^2.

    在训练循环中:
        total_loss = mse_loss + get_l2_to_c_loss(layer) * layer._l2_lambda_c
    """
    if not hasattr(layer, "_l2_c_params"):
        return torch.tensor(0.0)
    loss = torch.tensor(0.0, device=layer._l2_c_params[0].device if len(layer._l2_c_params) > 0 else "cpu")
    for c in layer._l2_c_params:
        loss = loss + (c * c).sum()
    return loss


# ---------------------------------------------------------------------------
# Patch B: router_dropout
# ---------------------------------------------------------------------------


def router_dropout(layer: nn.Module, p: float = 0.1) -> None:
    """Patch B: 在 router logits 上加 Dropout (训练时随机遮蔽专家).

    解决: 路由过拟合特定专家组合.

    实现: 在 cb_attn / cb_ffn 各自挂一个 nn.Dropout(p),
    并 monkey-patch augment_router_logits 让它在 softmax 前先 dropout.

    注: 训练时 dropout 生效, eval 时自动关闭.
    """
    layer._router_dropout_p = p
    for cb_name in ("cb_attn", "cb_ffn"):
        cb = getattr(layer, cb_name)
        cb._dropout = nn.Dropout(p=p)


def _augment_with_dropout(self, z: torch.Tensor) -> torch.Tensor:
    """CentralTheory.augment_router_logits 的 dropout 增强版本 (monkey-patch 用)."""
    out = _orig_augment_router_logits(self, z)
    if hasattr(self, "_dropout"):
        out = self._dropout(out)
    return out


_orig_augment_router_logits = CentralTheory.augment_router_logits


# ---------------------------------------------------------------------------
# Patch C: extend_warmup_steps
# ---------------------------------------------------------------------------


def extend_warmup_steps(layer: nn.Module, new_warmup: int = 100) -> None:
    """Patch C: 把 AdaptiveEMA 的 warmup 从 10 → 100 (或任意更大值).

    解决: 训练前期 c 不稳定导致过早自适应.

    注: 仅影响 mode='adaptive-ema'. 其他模式不受影响.
    """
    for cb_name in ("cb_attn", "cb_ffn"):
        cb = getattr(layer, cb_name)
        if hasattr(cb, "adaptive_warmup_steps"):
            cb.adaptive_warmup_steps = new_warmup
            cb.reset_step_count()  # 重置计数, 让 warmup 从新值重新开始


# ---------------------------------------------------------------------------
# Patch D: noise_inject_c
# ---------------------------------------------------------------------------


def noise_inject_c(layer: nn.Module, std: float = 0.01) -> None:
    """Patch D: 训练 forward 时给 c 加 Gaussian 噪声 (仅 forward, EMA 不变).

    解决: c 在训练集上收敛到特定点, 破坏泛化.

    实现: 给 cb_attn / cb_ffn 各挂一个 _noise_std 属性,
    monkey-patch augment_router_logits 在用 c 之前加噪声.
    """
    layer._c_noise_std = std
    for cb_name in ("cb_attn", "cb_ffn"):
        cb = getattr(layer, cb_name)
        cb._c_noise_std = std


# ---------------------------------------------------------------------------
# 工厂: apply_all_patches
# ---------------------------------------------------------------------------


def apply_all_patches(layer: nn.Module, patch_config: Dict[str, float]) -> None:
    """一次性应用多个 patch.

    Args:
        layer: v22/v25/v26 TheoryLayer
        patch_config: dict, 可包含以下 key:
            - "apply_l2_to_c": float, lambda_c 值 (e.g. 1e-4)
            - "router_dropout": float, dropout p 值 (e.g. 0.1)
            - "extend_warmup_steps": int, 新 warmup 值 (e.g. 100)
            - "noise_inject_c": float, 噪声 std (e.g. 0.01)
    """
    if "apply_l2_to_c" in patch_config:
        apply_l2_to_c(layer, lambda_c=patch_config["apply_l2_to_c"])
    if "router_dropout" in patch_config:
        router_dropout(layer, p=patch_config["router_dropout"])
    if "extend_warmup_steps" in patch_config:
        extend_warmup_steps(layer, new_warmup=patch_config["extend_warmup_steps"])
    if "noise_inject_c" in patch_config:
        noise_inject_c(layer, std=patch_config["noise_inject_c"])


# 默认 patch 配置 (供 examples/run_v27_*.py 使用)
DEFAULT_PATCH_CONFIGS = {
    "none":  {},  # baseline: 无 patch (与 v26 完全一致)
    "l2":    {"apply_l2_to_c": 1e-4},
    "dropout": {"router_dropout": 0.1},
    "warmup":  {"extend_warmup_steps": 100},
    "noise":   {"noise_inject_c": 0.01},
}


# Patch 名列表 (供端到端遍历)
PATCH_NAMES = ["none", "l2", "dropout", "warmup", "noise"]
