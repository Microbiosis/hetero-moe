"""v28.0 — router-norm 架构变体 (4 个).

v22 router-norm 数学形态: z_out = LayerNorm_{affine=True}(z + U @ c)
v27 证明: 这是架构性问题, LayerNorm 破坏绝对量级信息, 导致 held-out 退步.

v28 提出 4 个架构变体, 修改 LayerNorm/U 的形态:
    Variant 1: RouterNormAffineFalse  - LayerNorm affine=False, 移除可学习 scale/bias
    Variant 2: RouterNormZeroU        - 初始化 U=0, 让 c 学习 U 的生长
    Variant 3: RouterNormRMS          - 用 RMSNorm 替换 LayerNorm, 不强制 mean=0
    Variant 4: RouterNormPure         - 完全删除 U, 只对 z 做 RMSNorm

可追溯性:
    RouterNormCentral            -> v22 RouterNormCentral (held-out 退化基线)
    RouterNormAffineFalse        -> 变体 1
    RouterNormZeroU              -> 变体 2
    RouterNormRMS                -> 变体 3
    RouterNormPure               -> 变体 4

注意: 4 个变体共享相同的 EMA 更新逻辑 (与 v22 一致), 只改 augment_router_logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.central_stability.v22_central_theory import CentralTheory


class _RouterNormVariantBase(CentralTheory):
    """router-norm 变体基类.

    所有变体共享与 v22 RouterNormCentral 一致的:
        - 参数初始化 (c=zeros, U=randn*init_scale)
        - EMA 更新逻辑 (CentralTheory.ema_update)
        - 训练/eval 切换 (CentralTheory._ema_enabled)

    变体只改 augment_router_logits 的数学形态.
    """

    def __init__(self, d_shared: int, num_experts: int, init_scale: float = 0.001):
        # 注意: 不能直接调用 CentralTheory.__init__, 因为它会按 mode 分支创建不同子模块.
        # 这里手动创建变体所需的 c / U (如有) / 归一化层.
        nn.Module.__init__(self)
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = "router-norm-variant"
        self.ema_decay = 0.9
        self._step_count = 0
        self._ema_enabled = True
        self.c = nn.Parameter(torch.zeros(d_shared))
        self.U = nn.Parameter(torch.randn(num_experts, d_shared) * init_scale)
        # 变体 1/2/3 仍需 LN/RMS; 变体 4 不创建归一化层.
        self._init_normalizer()


# ---------------------------------------------------------------------------
# Variant 1: RouterNormAffineFalse (LayerNorm affine=False)
# ---------------------------------------------------------------------------


class RouterNormAffineFalse(_RouterNormVariantBase):
    """变体 1: LayerNorm affine=False, 移除 6 个可学习 scale/bias 参数.

    解决: LayerNorm 的可学习 scale/bias 拟合训练集分布, 在 held-out 上失配.
    """

    def _init_normalizer(self):
        self.router_norm = nn.LayerNorm(self.num_experts, elementwise_affine=False)
        # 标记: 变体 1 不冻结 U (默认行为), 让 U 仍可学习但 LN 不学
        self._init_scale = 0.001

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        z_with_bias = z + self.U @ self.c
        return self.router_norm(z_with_bias)


# ---------------------------------------------------------------------------
# Variant 2: RouterNormZeroU (init U=0)
# ---------------------------------------------------------------------------


class RouterNormZeroU(_RouterNormVariantBase):
    """变体 2: 初始化 U=0 (非 random), 让 U 通过训练学习生长.

    解决: 初始 U 是小随机噪声, 与 c=0 复合产生随机 bias, 影响早期训练稳定性.
    初始 U=0 让 bias 起点干净, LayerNorm(z) 在 c=0 时等价于"z 的归一化"。
    """

    def _init_normalizer(self):
        self.router_norm = nn.LayerNorm(self.num_experts, elementwise_affine=True)

    def __init__(self, d_shared: int, num_experts: int, init_scale: float = 0.0):
        # 重写 __init__ 用 U=zeros
        nn.Module.__init__(self)
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = "router-norm-variant"
        self.ema_decay = 0.9
        self._step_count = 0
        self._ema_enabled = True
        self.c = nn.Parameter(torch.zeros(d_shared))
        self.U = nn.Parameter(torch.zeros(num_experts, d_shared))
        self._init_scale = 0.0
        self.router_norm = nn.LayerNorm(self.num_experts, elementwise_affine=True)

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        z_with_bias = z + self.U @ self.c
        return self.router_norm(z_with_bias)


# ---------------------------------------------------------------------------
# Variant 3: RouterNormRMS (用 RMSNorm 替换 LayerNorm)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """RMSNorm: 只归一化均方根, 不强制 mean=0.

    forward(x) = x / sqrt(mean(x^2) + eps) * weight
    其中 weight 是可学习 (默认 init=1, 与 LayerNorm affine 类似).
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.elementwise_affine and self.weight is not None:
            x_normed = x_normed * self.weight
        return x_normed


class RouterNormRMS(_RouterNormVariantBase):
    """变体 3: 用 RMSNorm 替换 LayerNorm, 不强制 mean=0.

    解决: LayerNorm 强制 mean=0/std=1, 破坏 router 的绝对量级信息.
    RMSNorm 只归一化均方根, 保留方向信息.
    """

    def _init_normalizer(self):
        self.router_norm = _RMSNorm(self.num_experts, elementwise_affine=True)
        self._init_scale = 0.001

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        z_with_bias = z + self.U @ self.c
        return self.router_norm(z_with_bias)


# ---------------------------------------------------------------------------
# Variant 4: RouterNormPure (完全删除 U, 只对 z 做 RMSNorm)
# ---------------------------------------------------------------------------


class RouterNormPure(nn.Module):
    """变体 4: 完全删除 U (实际), 只对 z 做 RMSNorm.

    解决: U 是无界 bias 来源, 移除它; 只用 RMSNorm 稳定 router 输出.

    注: 此变体不创建 U 参数参与 augment, 但保留 frozen U 占位让 v25 build_v25_param_groups 兼容.
    c 仍存在 (接口一致) 但不影响 augment 输出.
    """

    def __init__(self, d_shared: int, num_experts: int):
        nn.Module.__init__(self)
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = "router-norm-variant"
        self.ema_decay = 0.9
        self._step_count = 0
        self._ema_enabled = True
        # c 仍存在 (保持接口一致), 但不影响 augment 输出
        self.c = nn.Parameter(torch.zeros(d_shared))
        # 占位 U: 不参与 augment, frozen. 让 v25 build_v25_param_groups 不报 None.
        self.U = nn.Parameter(torch.zeros(num_experts, d_shared), requires_grad=False)
        self.router_norm = _RMSNorm(self.num_experts, elementwise_affine=False)

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        # 纯 RMSNorm, 无 bias
        return self.router_norm(z)

    @torch.no_grad()
    def ema_update(self, expert_outputs, base_decay: float = 0.9):
        """c 仍按 v22 规则更新 (但 c 不影响 augment 输出, 仅为接口一致)."""
        if not self._ema_enabled or not self.training:
            return
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))
        self.c.data.mul_(base_decay).add_(new_c, alpha=1.0 - base_decay)
        self._step_count += 1

    def disable_ema(self):
        self._ema_enabled = False

    def enable_ema(self):
        self._ema_enabled = True

    def reset_step_count(self):
        self._step_count = 0


# 4 个变体的 mode 字符串 (供端到端脚本统一遍历)
ROUTER_NORM_VARIANT_MODES = [
    "router-norm-affine-false",
    "router-norm-zero-u",
    "router-norm-rms",
    "router-norm-pure",
]

# 变体名称到类的映射
ROUTER_NORM_VARIANT_CLASSES = {
    "router-norm-affine-false": RouterNormAffineFalse,
    "router-norm-zero-u": RouterNormZeroU,
    "router-norm-rms": RouterNormRMS,
    "router-norm-pure": RouterNormPure,
}


def make_router_norm_variant(mode: str, d_shared: int, num_experts: int):
    """工厂: 根据 mode 字符串创建对应变体."""
    if mode not in ROUTER_NORM_VARIANT_CLASSES:
        raise ValueError(f"unknown router-norm variant mode {mode!r}")
    return ROUTER_NORM_VARIANT_CLASSES[mode](d_shared, num_experts)
