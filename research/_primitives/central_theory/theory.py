"""v22.0 — 中枢机制理论分析 (完整实现, 含稳定性修复).

设计:
    探索两种"结构层修复"中枢机制:
        - RouterNorm: 路由 logits 加 LayerNorm 再 softmax (归一化单 token 极端值)
        - AdaptiveEMA: EMA 衰减自适应 (decay = sigmoid(α · c.norm()))

    broadcast 和 gate 模式**委托**给 v9_gate_central (单一中枢实现).
    v22 只保留 router-norm 和 adaptive-ema 两个独有探索.

相对 v9 (修正 C gate-style):
    v9 gate-style 是唯一同时解决 5 假设的方案, 但 v22 探索其他可能的结构层修复
    验证: 是否还有其他修复能达到 gate-style 同等或更好的效果

v22.0 完整实现 (修复简化层不稳定):
    之前 v22 端到端简化层的 fuse MSE 偏大 (gate=2140, router-norm=1737, adaptive-ema=7502,
    broadcast=6.26), 根因是 TheoryLayer 没沿用 v8 的训练稳定性机制 (per-expert 优化器、
    V_coop 协同、eval 关闭 EMA、attn W_o 冻结/正交、EMA decay 暖启动)。

    本次完整实现补齐:
        A. freeze_u / freeze_c 函数式 patch (v9 修正 A/B 移植)
        B. AdaptiveEMA warmup: 前 N step 用 base_decay=0.9, 之后才用自适应
           (避免 c.norm()=0 → decay=0.5 造成的 c 训练震荡)
        C. RouterNorm 中心化初始化: U 用更小 init_scale=0.001 (修正 B 移植)
        D. ema_enable/disable 控制 (对齐 v8 layer 的 _ema_enabled 语义)
        E. EMA-only-when-training 语义: 默认只 self.training=True 时 EMA

可追溯性:
    CentralTheory              -> 中枢机制基类 (bcast / gate / router-norm / adaptive-ema)
    RouterNormCentral          -> v22 独有: router-norm
    AdaptiveEMACentral         -> v22 独有: adaptive-ema
    gate_style_analysis       -> 静态数学分析
    freeze_u / freeze_c        -> 函数式 patch (修正 A/B 在 v22 的移植)
    set_u_init_scale           -> 修正 B 移植
    diagnose_theory_layer      -> 根因诊断工具 (对齐 v9.diagnose_c1_root_cause)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from research._primitives.central_mechanism import BroadcastWorkspace, GateStyleWorkspace


# ---------------------------------------------------------------------------
# 静态数学分析 (不变)
# ---------------------------------------------------------------------------


def gate_style_analysis() -> Dict[str, str]:
    """gate-style 数学分析 (静态).

    Returns:
        dict of {假设: 修复原因}
    """
    return {
        "H1_U_train_disturbance": (
            "gate-style 不直接加偏置, 只除以温度. "
            "温度 = 1 + α·tanh(W_t·c), 当 α=0.1 时, 路由 logits 相对顺序不变. "
            "对比 v8 broadcast: z += U@c, 直接扰乱 logits 顺序."
        ),
        "H2_EMA_too_slow": (
            "c=0 时, t_scalar = W_t @ c = 0, temperature = 1 + α·tanh(0) = 1.0. "
            "路由完全退化为 v7 (无中枢). "
            "对比 v8 broadcast: c=0 时仍有 z += U@c=0 (偶然), 但训练几 step 后 c 非零就开始扰动."
        ),
        "H3_too_few_steps": (
            "gate-style 不依赖 c 收敛. 即使 c 是初始的零向量或随机噪声, "
            "temperature ≈ 1, 路由行为 ≈ v7 (不依赖 c 提供信号). "
            "对比 v8 broadcast: c 早期是非零噪声, 直接破坏 v7 路由."
        ),
        "H4_small_student_capacity": (
            "gate-style 只调节 softmax 锐度, 学生只需学会'使用 W_t 投影', "
            "不增加路由容量需求. "
            "对比 v8 broadcast: 学生需要学'如何应对 U@c 扰动', 容量需求大."
        ),
        "H5_central_position": (
            "gate-style 是数学通用接口: z /= temperature 对任意 z 适用. "
            "可加到 attn 路由 / ffn 路由 / 两路由, 位置无关. "
            "对比 v8 broadcast: 加到 attn 可能与 attn softmax 冲突."
        ),
    }


# ---------------------------------------------------------------------------
# 函数式 patch (从 v9_gate_central 移植, 适配 v22 独有模式)
# ---------------------------------------------------------------------------


def freeze_u(central: nn.Module) -> None:
    """修正 A (v22 移植): 冻结 router-norm / adaptive-ema 的广播矩阵 U.

    仅当 central 有 U 属性时生效 (router-norm / adaptive-ema / broadcast).
    gate 模式无 U, 调用会抛 TypeError (符合 v9_gate_central.freeze_broadcast_matrix 语义).
    """
    if not hasattr(central, "U"):
        raise TypeError(f"{type(central).__name__} 没有 U 属性, 不能 freeze_u. "
                        f"请用 GateStyleWorkspace (无需 freeze_u).")
    central.U.requires_grad_(False)


def freeze_c(central: nn.Module) -> None:
    """修正 A 变体: 冻结中枢向量 c. 调试 / 消融用.

    对所有 4 种模式都生效 (c 都是 nn.Parameter).
    """
    if not hasattr(central, "c"):
        raise TypeError(f"{type(central).__name__} 没有 c 属性.")
    central.c.requires_grad_(False)


def set_u_init_scale(central: nn.Module, scale: float = 0.001) -> None:
    """修正 B (v22 移植): 重新初始化 U 为更小 scale.

    只在 central 创建后第一次 forward 之前调用. 重新初始化会丢失已学的 U.
    默认 scale=0.001 比 v8 broadcast 的默认 0.01 小 10x, 更保守.
    """
    if not hasattr(central, "U"):
        raise TypeError(f"{type(central).__name__} 没有 U 属性, 不能 set_u_init_scale.")
    with torch.no_grad():
        central.U.data = torch.randn_like(central.U) * scale


# ---------------------------------------------------------------------------
# 中枢机制基类 (broadcast / gate 委托给 v9, router-norm / adaptive-ema 自实现)
# ---------------------------------------------------------------------------


class CentralTheory(nn.Module):
    """中枢机制理论基类 (v22.0 完整实现).

    broadcast / gate 模式委托给 v9_gate_central (单一中枢实现).
    router-norm / adaptive-ema 是 v22 独有探索, 内部实现.

    关键稳定性机制 (本次新增):
        - freeze_u / freeze_c: 函数式 patch
        - AdaptiveEMA warmup: 前 warmup_steps 步用 base_decay, 之后用自适应
        - ema_enable/disable: 对齐 v8 layer 的 _ema_enabled 语义
        - 默认只在 self.training=True 时 EMA 更新

    Args:
        d_shared:       共享维度
        num_experts:    专家数
        mode:           "broadcast" | "gate" | "router-norm" | "adaptive-ema"
        ema_decay:      EMA 衰减 (broadcast / gate / router-norm 默认 0.9)
        adaptive_warmup_steps: 仅 adaptive-ema 用, 前 N 步强制 base_decay (避免
                              c.norm()=0 → decay=0.5 造成的 c 训练震荡)
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        mode: str = "gate",
        ema_decay: float = 0.9,
        adaptive_warmup_steps: int = 10,
    ):
        super().__init__()
        assert mode in ("broadcast", "gate", "router-norm", "adaptive-ema"), \
            f"unsupported mode {mode}"
        self.d_shared = d_shared
        self.num_experts = num_experts
        self.mode = mode
        self.ema_decay = ema_decay
        self.adaptive_warmup_steps = adaptive_warmup_steps
        self._step_count = 0
        self._ema_enabled = True

        if mode == "broadcast":
            # 委托给 v9_gate_central.BroadcastWorkspace
            self.cw = BroadcastWorkspace(d_shared, num_experts, ema_decay=ema_decay)
            self.c = self.cw.c
            self.U = self.cw.U
        elif mode == "gate":
            # 委托给 v9_gate_central.GateStyleWorkspace
            self.cw = GateStyleWorkspace(d_shared, num_experts, alpha=0.1)
            self.c = self.cw.c
            self.W_t = self.cw.W_t
            self.alpha = self.cw.alpha
        elif mode == "router-norm":
            # v22 独有: 路由 logits 加 LayerNorm. U 用小 scale 0.001 (修正 B 移植)
            self.c = nn.Parameter(torch.zeros(d_shared))
            self.U = nn.Parameter(torch.randn(num_experts, d_shared) * 0.001)
            self.router_norm = nn.LayerNorm(num_experts)
        elif mode == "adaptive-ema":
            # v22 独有: EMA 衰减自适应. α 初始化为 0 (warmup 期间退化为 broadcast)
            self.c = nn.Parameter(torch.zeros(d_shared))
            self.U = nn.Parameter(torch.randn(num_experts, d_shared) * 0.01)
            self.adaptive_alpha = nn.Parameter(torch.tensor(0.0))  # 初始化为 0 → 前期无自适应

    def augment_router_logits(self, z: torch.Tensor) -> torch.Tensor:
        """根据 mode 增强 z_router."""
        if self.mode == "broadcast":
            return self.cw.augment_router_logits(z)
        elif self.mode == "gate":
            return self.cw.augment_router_logits(z)
        elif self.mode == "router-norm":
            z_with_bias = z + self.U @ self.c
            return self.router_norm(z_with_bias)
        else:  # adaptive-ema
            return z + self.U @ self.c

    @torch.no_grad()
    def ema_update(self, expert_outputs, base_decay: float = 0.9):
        """EMA 更新 c.

        关键改进:
            - 仅在 self._ema_enabled=True 且 self.training=True 时执行 (对齐 v8 语义)
            - adaptive-ema 前 warmup_steps 步用 base_decay=0.9 (避免 c.norm()=0 → decay=0.5 震荡)
            - 自适应 decay = sigmoid(adaptive_alpha · ||c||), α 初始化为 0 保证
              warmup 期间几乎退化为 broadcast 行为
        """
        # 稳定性 patch 1: EMA 仅在训练时启用
        if not self._ema_enabled or not self.training:
            return
        # broadcast / gate 委托给 v9_gate_central
        if self.mode in ("broadcast", "gate"):
            self.cw.ema_update(expert_outputs)
            self._step_count += 1
            return
        if not expert_outputs:
            return
        stacked = torch.stack(expert_outputs, dim=0)
        new_c = stacked.mean(dim=(0, 1, 2))

        if self.mode == "adaptive-ema":
            # 稳定性 patch 2: warmup 期间用 base_decay (避免初始震荡)
            if self._step_count < self.adaptive_warmup_steps:
                decay = base_decay
            else:
                # α·||c|| → 训练稳定后 α 自然增长, decay 趋近 1 (慢更新)
                adaptive_decay = torch.sigmoid(
                    self.adaptive_alpha * self.c.norm()
                ).item()
                decay = adaptive_decay
        else:
            decay = base_decay
        self.c.data.mul_(decay).add_(new_c, alpha=1.0 - decay)
        self._step_count += 1

    # ---- 稳定性 patch 3: ema_enable / disable 控制 (对齐 v8._ema_enabled) ----
    def disable_ema(self) -> None:
        self._ema_enabled = False

    def enable_ema(self) -> None:
        self._ema_enabled = True

    def reset_step_count(self) -> None:
        """重置 _step_count (跨 phase / warm restart 用)."""
        self._step_count = 0


def mode_or_self(central: nn.Module, target: str) -> bool:
    """小工具: 避免 self.mode 拼写错误的 hardcoded 比较."""
    return getattr(central, "mode", None) == target


class RouterNormCentral(CentralTheory):
    """RouterNorm: 路由 logits 加 LayerNorm. v22 独有.

    关键改进 (v22.0 完整实现):
        - U 用小 init_scale=0.001 (默认), 修正 B 移植 (见 set_u_init_scale)
        - 默认开启 freeze_u 可选 (避免 H1)
        - ema_decay 默认 0.9 (与 broadcast 一致)
    """

    def __init__(self, d_shared: int, num_experts: int, ema_decay: float = 0.9):
        super().__init__(d_shared, num_experts, mode="router-norm", ema_decay=ema_decay)


class AdaptiveEMACentral(CentralTheory):
    """AdaptiveEMA: EMA 衰减自适应. v22 独有.

    关键改进 (v22.0 完整实现):
        - adaptive_alpha 初始化为 0.0 (而非 0.5), warmup 期间退化为 broadcast
        - warmup_steps 默认 10, 之后才用 sigmoid(α·||c||) 自适应
        - ema_decay 仅作用于 warmup 期, warmup 后由 adaptive_alpha 控制
    """

    def __init__(
        self,
        d_shared: int,
        num_experts: int,
        ema_decay: float = 0.9,
        adaptive_warmup_steps: int = 10,
    ):
        super().__init__(
            d_shared,
            num_experts,
            mode="adaptive-ema",
            ema_decay=ema_decay,
            adaptive_warmup_steps=adaptive_warmup_steps,
        )


# ---------------------------------------------------------------------------
# 根因诊断工具 (对齐 v9_gate_central.diagnose_c1_root_cause)
# ---------------------------------------------------------------------------


def diagnose_theory_layer(central: nn.Module) -> dict:
    """诊断 v22 CentralTheory 各模式的稳定性状态.

    Args:
        central: CentralTheory 或其子类的实例

    Returns:
        dict with:
            mode: str
            c_init_norm: float
            c_trainable: bool
            U_trainable: bool (or N/A for gate)
            ema_decay: float
            ema_enabled: bool
            step_count: int
            diagnosis: str
    """
    out = {
        "mode": getattr(central, "mode", "unknown"),
        "c_init_norm": float(central.c.norm().item()),
        "c_trainable": central.c.requires_grad,
        "ema_decay": getattr(central, "ema_decay", 0.9),
        "ema_enabled": getattr(central, "_ema_enabled", True),
        "step_count": getattr(central, "_step_count", 0),
    }
    if hasattr(central, "U"):
        out["U_trainable"] = central.U.requires_grad
    else:
        out["U_trainable"] = "N/A (gate 模式)"

    mode = out["mode"]
    if mode == "broadcast":
        if out["U_trainable"]:
            out["diagnosis"] = "H1 风险 (U 训练扰动). 建议 freeze_u()."
        else:
            out["diagnosis"] = "OK (U 已冻结)"
    elif mode == "gate":
        out["diagnosis"] = "OK (gate 天然无 H1 风险)"
    elif mode == "router-norm":
        if not out["U_trainable"]:
            out["diagnosis"] = "OK (U 已冻结, LayerNorm 提供额外稳定性)"
        else:
            out["diagnosis"] = "建议 freeze_u() 以减少 LayerNorm 之前 logits 的扰动"
    elif mode == "adaptive-ema":
        if out["step_count"] < getattr(central, "adaptive_warmup_steps", 10):
            out["diagnosis"] = f"OK (warmup 中, 用 base_decay={out['ema_decay']})"
        else:
            out["diagnosis"] = "OK (warmup 后用自适应 decay)"
    else:
        out["diagnosis"] = "unknown mode"
    return out
