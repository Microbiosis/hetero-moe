"""§2.2 / §6.2 — INT4 量化感知算子。

规范 §6.2 原文："真实实现需按 group_size=128 分块计算 scale，原型中 per-row
简化仅供验证梯度流。" 本模块实现 group_size 分块的真实版本；当
group_size >= 末维 D 时自然退化为 per-row（与规范 §7 原型逐字节一致），
从而保证向后兼容与可追溯。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class FakeQuantSTE(torch.autograd.Function):
    """INT4 伪量化，反向传播使用 Straight-Through Estimator (STE)。

    数学定义 (§2.2):
        F_m^{noisy} = FakeQuant_STE(F_m^{exact}, bits=4, group_size=128)

    反向传播必须使用 STE，防止 torch.round 丢失梯度 (§6.2)。
    由于本算子是 torch.autograd.Function，forward 内部的 round / reshape /
    pad 均不进入自动微分图，梯度完全由 backward 定义的恒等映射决定。

    Args:
        x:        任意形状张量，量化沿最后一维进行。
        bits:      量化位宽，默认 4 (§2.2)。
        group_size: 分块大小，默认 128 (§6.2)。None 表示 per-row（原型行为）。
                    当 group_size >= x.shape[-1] 时等价于 per-row。
    """

    @staticmethod
    def forward(ctx, x, bits: int = 4, group_size: int = 128):
        qmax = 2 ** (bits - 1) - 1  # 对称量化，正区间上界

        # ---- per-row 路径（规范 §7 原型行为，group_size=None 或 >= D 时触发）----
        D = x.shape[-1]
        if group_size is None or group_size >= D:
            scale = x.abs().amax(dim=-1, keepdim=True) / qmax
            scale = torch.clamp(scale, min=1e-5)
            return torch.round(x / scale) * scale

        # ---- group-wise 路径（§6.2 真实实现）----
        # 末维补齐到 group_size 整数倍，避免分块越界。
        pad = (group_size - D % group_size) % group_size
        if pad > 0:
            x_pad = F.pad(x, (0, pad))
        else:
            x_pad = x
        Dp = x_pad.shape[-1]
        n_groups = Dp // group_size

        # [..., n_groups, group_size] —— 每组独立计算 scale
        x_g = x_pad.reshape(*x.shape[:-1], n_groups, group_size)
        scale = x_g.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-5)
        x_q_g = torch.round(x_g / scale) * scale
        x_q = x_q_g.reshape(*x_pad.shape[:-1], Dp)
        if pad > 0:
            x_q = x_q[..., :D]
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE：梯度直通；bits / group_size 为 Python 标量，返回 None。
        return grad_output, None, None


def fake_quant(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """函数式封装，便于在无 requires_grad 上下文中调用。"""
    return FakeQuantSTE.apply(x, bits, group_size)


def quantization_error(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> float:
    """量化相对误差（L2），用于 §8.2 PPL 对齐验证的诊断指标。

    返回 ||x - Q(x)||_2 / ||x||_2。误差越小，量化对分布扰动越小。
    """
    with torch.no_grad():
        x_q = fake_quant(x, bits, group_size)
        num = (x - x_q).pow(2).sum().sqrt().item()
        den = x.pow(2).sum().sqrt().item() + 1e-12
        return num / den
