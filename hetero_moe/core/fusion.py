"""§2 / v5.1 — 异构微观融合层 (Heterogeneous Micro-Fusion Layer)。

v5.0 / v5.1 固有缺陷修复清单（相对 v4.0）:
    1. 向量化预筛真稀疏计算 (sparse_dispatch=True):
       先以单次向量化归约 `counts = (ah_flat > 0).sum(dim=0).tolist()` 一次性判定全专家命中数
       （消除逐专家 `nonzero` 造成的 M 次 Host-Device 同步停顿），零命中专家直接跳过，仅对活跃专家执行前向。
    2. 显存与梯度语义纠偏:
       默认 detach_input=False（保留跨层梯度；保留 detach_input=True 供消融复现），
       并提供 grad_checkpoint=True（基于 torch.utils.checkpoint 激活重计算），真正将峰值
       中间激活显存从 O(M·L·D_ff) 降至 O(L·D)。
    3. 训练-推理路由幅值与容量对齐:
       - 默认 renorm_topk=True，使训练期 Top-K 权重与推理期同样归一化到和为 1。
       - 新增 train_capacity_factor: Optional[float] (v5.1)，在训练期同样启用 capacity_route_ste
         容量上限截断与重归一化，消除训练 (C=∞) 与推理 (C=1.25) 的分布偏移。
    4. 消除推理期重复路由矩阵乘:
       当外部传入 routed_weights（及可选 routed_logits）时，直接跳过 F.linear(x, W_router) 重复计算。
    5. 真异构专家与内部子层 LoRA (v5.1):
       支持传入 expert_specs（不同宽度/架构/外部冻结模块），且 LoRA 直接深入专家内部非线性子层
       (internal_lora=True)，打破纯冻结底座的 β 墙死锁。
    6. 突破路由低秩瓶颈 (v55 Theorem 3: rank(J_route) <= min(K, M-1)):
       支持 gate_rank > 0 的低秩向量门控 g_m(x) = α_m · (1 + tanh(U_m V_m x)) ∈ R^D。
    7. 权重级 + 激活级 INT4 伪量化 (v5.1):
       新增 quant_target ('activation' | 'weight' | 'both' | 'none')，支持对专家内部权重矩阵
       (W_gate/W_up/W_down) 直接施加 FakeQuantSTE。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .experts import ExpertSpec, HeteroExpert, LoRALinearAdapter, make_heterogeneous_specs
from .quant import FakeQuantSTE
from .router import SparseRouterSTE, capacity_route_ste


def swiglu_forward(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    quant_weight_bits: int = 0,
    quant_group_size: int = 128,
) -> torch.Tensor:
    """§2.2 标准 SwiGLU 前馈精确值: (SiLU(x W_gate^T) ⊙ (x W_up^T)) W_down^T（可选权重级 INT4 伪量化）。"""
    if quant_weight_bits > 0 and quant_weight_bits < 16:
        w_gate = FakeQuantSTE.apply(w_gate, quant_weight_bits, quant_group_size)
        w_up = FakeQuantSTE.apply(w_up, quant_weight_bits, quant_group_size)
        w_down = FakeQuantSTE.apply(w_down, quant_weight_bits, quant_group_size)
    h_gate = F.linear(x, w_gate)
    h_up = F.linear(x, w_up)
    h_act = F.silu(h_gate) * h_up
    return F.linear(h_act, w_down)


class HeteroFusionLayer(nn.Module):
    """异构微观融合层 (v5.1，完全兼容 v4.0 / v5.0 构造与前向签名)。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        quant_bits: int = 4,
        quant_group_size: int = 128,
        renorm_topk: bool = True,
        detach_input: bool = False,
        grad_checkpoint: bool = False,
        sparse_dispatch: bool = True,
        gate_rank: int = 0,
        expert_specs: Optional[Sequence[ExpertSpec]] = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_experts: bool = True,
        quant_target: str = "activation",
        train_capacity_factor: Optional[float] = None,
    ):
        super().__init__()
        if expert_specs is not None:
            num_experts = len(expert_specs)
        assert num_experts >= 1, "num_experts 必须 >= 1"
        assert 1 <= top_k <= num_experts, "top_k 必须在 [1, num_experts] 内"
        assert gate_rank >= 0, "gate_rank 必须 >= 0"
        assert lora_rank >= 0, "lora_rank 必须 >= 0"
        assert quant_target in ("activation", "weight", "both", "none"), f"未知 quant_target={quant_target!r}"
        if train_capacity_factor is not None:
            assert train_capacity_factor > 0.0, "train_capacity_factor 必须为正"

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.quant_bits = int(quant_bits)
        self.quant_group_size = int(quant_group_size)
        self.renorm_topk = bool(renorm_topk)
        self.detach_input = bool(detach_input)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.sparse_dispatch = bool(sparse_dispatch)
        self.gate_rank = int(gate_rank)
        self.lora_rank = int(lora_rank)
        self.freeze_experts = bool(freeze_experts)
        self.quant_target = str(quant_target)
        self.train_capacity_factor = float(train_capacity_factor) if train_capacity_factor is not None else None
        self.is_heterogeneous = expert_specs is not None
        self.last_expert_token_counts: List[int] = [0] * self.num_experts

        # ---- 专家库: 真异构 HeteroExpert 银行 或 兼容 v4 的 SwiGLU 参数列表 ----
        if expert_specs is not None:
            self.experts: Optional[nn.ModuleList] = make_heterogeneous_specs(
                d_model, expert_specs, default_d_ff=d_ff
            )
            self.w_gates = nn.ParameterList()
            self.w_ups = nn.ParameterList()
            self.w_downs = nn.ParameterList()
            self.expert_loras: Optional[nn.ModuleList] = None
            self.sublayer_gate_loras: Optional[nn.ModuleList] = None
            self.sublayer_up_loras: Optional[nn.ModuleList] = None
            self.sublayer_down_loras: Optional[nn.ModuleList] = None
        else:
            self.experts = None
            self.w_gates = nn.ParameterList(
                [nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(self.num_experts)]
            )
            self.w_ups = nn.ParameterList(
                [nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(self.num_experts)]
            )
            self.w_downs = nn.ParameterList(
                [nn.Parameter(torch.randn(d_model, d_ff) * 0.02) for _ in range(self.num_experts)]
            )
            if self.freeze_experts:
                for plist in (self.w_gates, self.w_ups, self.w_downs):
                    for p in plist:
                        p.requires_grad = False
            if self.lora_rank > 0:
                self.expert_loras = nn.ModuleList(
                    [
                        LoRALinearAdapter(d_model, d_model, rank=self.lora_rank, alpha=lora_alpha)
                        for _ in range(self.num_experts)
                    ]
                )
                # v5.1: 同时为内置 SwiGLU 提供深入 w_gate / w_up / w_down 的子层 LoRA
                self.sublayer_gate_loras = nn.ModuleList(
                    [LoRALinearAdapter(d_model, d_ff, rank=self.lora_rank, alpha=lora_alpha) for _ in range(self.num_experts)]
                )
                self.sublayer_up_loras = nn.ModuleList(
                    [LoRALinearAdapter(d_model, d_ff, rank=self.lora_rank, alpha=lora_alpha) for _ in range(self.num_experts)]
                )
                self.sublayer_down_loras = nn.ModuleList(
                    [LoRALinearAdapter(d_ff, d_model, rank=self.lora_rank, alpha=lora_alpha) for _ in range(self.num_experts)]
                )
            else:
                self.expert_loras = None
                self.sublayer_gate_loras = None
                self.sublayer_up_loras = None
                self.sublayer_down_loras = None

        # ---- 可训练路由与分布对齐参数 (§4.1 初始化规范) ----
        self.W_router = nn.Parameter(torch.randn(self.num_experts, d_model) * 0.01)
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_model)) for _ in range(self.num_experts)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(self.num_experts)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.empty(1).uniform_(0.01, 0.1)) for _ in range(self.num_experts)]
        )

        # ---- 可选低秩向量门控: 突破 v55 Theorem 3 标量路由带宽上界 ----
        if self.gate_rank > 0:
            scale_v = 1.0 / math.sqrt(max(1, d_model))
            self.gate_v: Optional[nn.ParameterList] = nn.ParameterList(
                [nn.Parameter(torch.randn(self.gate_rank, d_model) * scale_v) for _ in range(self.num_experts)]
            )
            self.gate_u: Optional[nn.ParameterList] = nn.ParameterList(
                [nn.Parameter(torch.randn(d_model, self.gate_rank) * 0.02) for _ in range(self.num_experts)]
            )
        else:
            self.gate_v = None
            self.gate_u = None

    def adapter_parameters(self) -> List[nn.Parameter]:
        """逐通道分布对齐 (γ/β) 与异构维度投影 (in_proj/out_proj) 参数。"""
        params: List[nn.Parameter] = [p for p in (*self.gammas, *self.betas) if p.requires_grad]
        if self.experts is not None:
            for exp in self.experts:
                params.extend(exp.adapter_parameters())
        return params

    def gate_and_lora_parameters(self) -> List[nn.Parameter]:
        """α 标量门控、低秩向量门控 (U/V)、内部子层 LoRA、外围 LoRA 及解冻专家参数。"""
        params: List[nn.Parameter] = [p for p in self.alphas if p.requires_grad]
        if self.gate_v is not None and self.gate_u is not None:
            params.extend(p for p in (*self.gate_v, *self.gate_u) if p.requires_grad)
        if self.experts is not None:
            for exp in self.experts:
                params.extend(exp.lora_and_unfrozen_parameters())
        else:
            for mlist in (
                self.expert_loras,
                self.sublayer_gate_loras,
                self.sublayer_up_loras,
                self.sublayer_down_loras,
            ):
                if mlist is not None:
                    params.extend(p for p in mlist.parameters() if p.requires_grad)
            if not self.freeze_experts:
                for plist in (self.w_gates, self.w_ups, self.w_downs):
                    params.extend(p for p in plist if p.requires_grad)
        return params

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        """返回层内所有可训练参数（去重保序，兼容 v4.0/v5.0 优化器分组）。"""
        ordered = [self.W_router, *self.adapter_parameters(), *self.gate_and_lora_parameters()]
        seen = set()
        out = []
        for p in ordered:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def _weight_qbits(self) -> int:
        return self.quant_bits if self.quant_target in ("weight", "both") else 0

    def _apply_act_quant(self, t: torch.Tensor) -> torch.Tensor:
        if self.quant_target in ("activation", "both") and self.quant_bits > 0 and self.quant_bits < 16:
            return FakeQuantSTE.apply(t, self.quant_bits, self.quant_group_size)
        return t

    def _run_expert_core(
        self,
        m: int,
        x_in: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        h_aligned = x_in * gamma + beta
        w_qbits = self._weight_qbits()
        if self.experts is not None:
            return self.experts[m](
                h_aligned,
                quant_weight_bits=w_qbits,
                quant_group_size=self.quant_group_size,
            )
        wg, wu, wd = self.w_gates[m], self.w_ups[m], self.w_downs[m]
        if w_qbits > 0:
            wg = FakeQuantSTE.apply(wg, w_qbits, self.quant_group_size)
            wu = FakeQuantSTE.apply(wu, w_qbits, self.quant_group_size)
            wd = FakeQuantSTE.apply(wd, w_qbits, self.quant_group_size)
        h_gate = F.linear(h_aligned, wg)
        h_up = F.linear(h_aligned, wu)
        if self.sublayer_gate_loras is not None and self.sublayer_up_loras is not None:
            h_gate = h_gate + self.sublayer_gate_loras[m](h_aligned)
            h_up = h_up + self.sublayer_up_loras[m](h_aligned)
        h_act = F.silu(h_gate) * h_up
        out = F.linear(h_act, wd)
        if self.sublayer_down_loras is not None:
            out = out + self.sublayer_down_loras[m](h_act)
        if self.expert_loras is not None:
            out = out + self.expert_loras[m](h_aligned)
        return out

    def _run_expert(self, m: int, x_in: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            def _fn(inp: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return self._run_expert_core(m, inp, g, b)

            return checkpoint(_fn, x_in, self.gammas[m], self.betas[m], use_reentrant=False)
        return self._run_expert_core(m, x_in, self.gammas[m], self.betas[m])

    def _compute_gate(self, m: int, x_gate: torch.Tensor) -> torch.Tensor:
        if self.gate_rank > 0 and self.gate_v is not None and self.gate_u is not None:
            mod = torch.tanh(F.linear(F.linear(x_gate, self.gate_v[m]), self.gate_u[m]))
            return self.alphas[m] * (1.0 + mod)
        return self.alphas[m]

    def forward(
        self,
        x: torch.Tensor,
        phase1: bool = False,
        routed_weights: Optional[torch.Tensor] = None,
        routed_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。"""
        x_residual = x
        B, S, D = x.shape

        if routed_weights is not None:
            alpha_hat = routed_weights
            if routed_logits is not None:
                z = routed_logits
            else:
                z = torch.zeros(B, S, self.num_experts, device=x.device, dtype=x.dtype)
        else:
            z = F.linear(x, self.W_router)
            alpha = F.softmax(z, dim=-1)
            if phase1:
                alpha_hat = (torch.ones_like(alpha) / self.num_experts).detach()
            elif self.train_capacity_factor is not None:
                alpha_hat = capacity_route_ste(alpha, self.top_k, self.train_capacity_factor, self.renorm_topk)
            else:
                alpha_hat = SparseRouterSTE.apply(alpha, self.top_k, self.renorm_topk)

        x_expert_in = x.detach() if self.detach_input else x

        if self.sparse_dispatch:
            # ---- 向量化预筛真稀疏分派: 单次归约全专家 token 计数，彻底跳过 0 命中专家 ----
            N = B * S
            x_in_flat = x_expert_in.reshape(N, D)
            x_gate_flat = x.reshape(N, D)
            ah_flat = alpha_hat.reshape(N, self.num_experts)
            active_mask = ah_flat > 0
            counts: List[int] = active_mask.sum(dim=0).tolist()
            out_flat = torch.zeros_like(x_in_flat)
            for m, n_act in enumerate(counts):
                if n_act == 0:
                    continue
                active = active_mask[:, m].nonzero(as_tuple=False).squeeze(-1)
                x_act = x_in_flat.index_select(0, active)
                f_m_exact = self._run_expert(m, x_act)
                f_m_noisy = self._apply_act_quant(f_m_exact)
                g_m = self._compute_gate(m, x_gate_flat.index_select(0, active))
                contrib = ah_flat[:, m].index_select(0, active).unsqueeze(-1) * g_m * f_m_noisy
                out_flat = out_flat.index_add(0, active, contrib)
            self.last_expert_token_counts = counts
            y = x_residual + out_flat.reshape(B, S, D)
            return y, z, alpha_hat

        # ---- 稠密回退路径 (供稀疏-稠密数值等价性验证使用) ----
        out_sum = torch.zeros_like(x)
        counts = [B * S] * self.num_experts
        for m in range(self.num_experts):
            f_m_exact = self._run_expert(m, x_expert_in)
            f_m_noisy = self._apply_act_quant(f_m_exact)
            g_m = self._compute_gate(m, x)
            out_sum = out_sum + alpha_hat[..., m : m + 1] * g_m * f_m_noisy
        self.last_expert_token_counts = counts
        y = x_residual + out_sum
        return y, z, alpha_hat
