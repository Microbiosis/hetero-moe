"""§2 / v5.0 — 异构微观融合层 (Heterogeneous Micro-Fusion Layer)。

v5.0 固有缺陷修复清单（相对 v4.0）:
    1. 真稀疏计算 (sparse_dispatch=True): 仅对被路由激活的 token 子集执行对应专家前向，
       零命中专家直接跳过，避免对全部 M 专家×全序列做稠密计算；记录 last_expert_token_counts。
    2. 显存与梯度语义纠偏:
       - v4.0 声称 x.detach() 可使底座 FFN 中间激活不入图、显存降至 O(L·D)，该陈述在
         gamma/beta 可训练时在 PyTorch autograd 机制下不成立（FFN 激活仍需保存以反传至 γ/β），
         且人为切断了跨层专家路径梯度。
       - v5.0 默认 detach_input=False（保留跨层梯度；保留 detach_input=True 供消融复现），
         并提供 grad_checkpoint=True（基于 torch.utils.checkpoint 激活重计算），真正将峰值
         中间激活显存从 O(M·L·D_ff) 降至 O(L·D)。
    3. 训练-推理路由幅值对齐: 默认 renorm_topk=True，使训练期 Top-K 权重与推理期
       capacity_route 同样归一化到和为 1，消除推理 2x-4x 幅值跳变。
    4. 消除推理期重复路由矩阵乘: 当外部传入 routed_weights（及可选 routed_logits）时，
       直接跳过 F.linear(x, W_router) 与 softmax 重复计算。
    5. 真异构专家与可选 LoRA/解冻: 支持传入 expert_specs（不同宽度/架构/外部冻结模块），
       以及 lora_rank>0 / freeze_experts=False，打破纯冻结底座的 β 墙死锁。
    6. 突破路由低秩瓶颈 (v55 Theorem 3: rank(J_route) <= min(K, M-1)):
       支持 gate_rank > 0 的低秩向量门控 g_m(x) = α_m · (1 + tanh(U_m V_m x)) ∈ R^D，
       将跨专家调制带宽从标量 1–2 维提升至 O(top_k · gate_rank) 维。
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .experts import ExpertSpec, HeteroExpert, LoRALinearAdapter, make_heterogeneous_specs
from .quant import FakeQuantSTE
from .router import SparseRouterSTE


def swiglu_forward(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """§2.2 标准 SwiGLU 前馈精确值: (SiLU(x W_gate^T) ⊙ (x W_up^T)) W_down^T。"""
    h_gate = F.linear(x, w_gate)
    h_up = F.linear(x, w_up)
    h_act = F.silu(h_gate) * h_up
    return F.linear(h_act, w_down)


class HeteroFusionLayer(nn.Module):
    """异构微观融合层 (v5.0，完全兼容 v4.0 构造与前向签名)。"""

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
    ):
        super().__init__()
        if expert_specs is not None:
            num_experts = len(expert_specs)
        assert num_experts >= 1, "num_experts 必须 >= 1"
        assert 1 <= top_k <= num_experts, "top_k 必须在 [1, num_experts] 内"
        assert gate_rank >= 0, "gate_rank 必须 >= 0"
        assert lora_rank >= 0, "lora_rank 必须 >= 0"

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
            else:
                self.expert_loras = None

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
        """α 标量门控、低秩向量门控 (U/V)、LoRA 及解冻专家参数。"""
        params: List[nn.Parameter] = [p for p in self.alphas if p.requires_grad]
        if self.gate_v is not None and self.gate_u is not None:
            params.extend(p for p in (*self.gate_v, *self.gate_u) if p.requires_grad)
        if self.experts is not None:
            for exp in self.experts:
                params.extend(exp.lora_and_unfrozen_parameters())
        else:
            if self.expert_loras is not None:
                params.extend(p for p in self.expert_loras.parameters() if p.requires_grad)
            if not self.freeze_experts:
                for plist in (self.w_gates, self.w_ups, self.w_downs):
                    params.extend(p for p in plist if p.requires_grad)
        return params

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        """返回层内所有可训练参数（去重保序，兼容 v4.0 优化器分组）。"""
        ordered = [self.W_router, *self.adapter_parameters(), *self.gate_and_lora_parameters()]
        seen = set()
        out = []
        for p in ordered:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def _run_expert_core(
        self,
        m: int,
        x_in: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        h_aligned = x_in * gamma + beta
        if self.experts is not None:
            return self.experts[m](h_aligned)
        out = swiglu_forward(h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m])
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
        """前向传播。

        Args:
            x:              [B, S, D] 主干残差流。
            phase1:         True 则强制均匀路由 (§4.2)，W_router 不收梯度。
            routed_weights: [B, S, M] 预计算路由权重（推理路径 §5）。提供时跳过内部路由计算。
            routed_logits:  [B, S, M] 可选预计算路由 logits（与 routed_weights 配合消除重复矩阵乘）。

        Returns:
            y:          [B, S, D] 残差融合输出。
            z:          [B, S, M] 路由 logits。
            alpha_hat:  [B, S, M] 实际使用的路由权重。
        """
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
            else:
                alpha_hat = SparseRouterSTE.apply(alpha, self.top_k, self.renorm_topk)

        x_expert_in = x.detach() if self.detach_input else x

        if self.sparse_dispatch:
            # ---- 真稀疏分派: 仅计算每个专家命中的活跃 token ----
            N = B * S
            x_in_flat = x_expert_in.reshape(N, D)
            x_gate_flat = x.reshape(N, D)
            ah_flat = alpha_hat.reshape(N, self.num_experts)
            out_flat = torch.zeros_like(x_in_flat)
            counts: List[int] = []
            for m in range(self.num_experts):
                w_m = ah_flat[:, m]
                active = torch.nonzero(w_m > 0, as_tuple=False).squeeze(-1)
                n_act = int(active.numel())
                counts.append(n_act)
                if n_act == 0:
                    continue
                x_act = x_in_flat.index_select(0, active)
                f_m_exact = self._run_expert(m, x_act)
                f_m_noisy = FakeQuantSTE.apply(f_m_exact, self.quant_bits, self.quant_group_size)
                g_m = self._compute_gate(m, x_gate_flat.index_select(0, active))
                contrib = w_m.index_select(0, active).unsqueeze(-1) * g_m * f_m_noisy
                out_flat = out_flat.index_add(0, active, contrib)
            self.last_expert_token_counts = counts
            y = x_residual + out_flat.reshape(B, S, D)
            return y, z, alpha_hat

        # ---- 稠密回退路径 (供稀疏-稠密数值等价性验证使用) ----
        out_sum = torch.zeros_like(x)
        counts = [B * S] * self.num_experts
        for m in range(self.num_experts):
            f_m_exact = self._run_expert(m, x_expert_in)
            f_m_noisy = FakeQuantSTE.apply(f_m_exact, self.quant_bits, self.quant_group_size)
            g_m = self._compute_gate(m, x)
            out_sum = out_sum + alpha_hat[..., m : m + 1] * g_m * f_m_noisy
        self.last_expert_token_counts = counts
        y = x_residual + out_sum
        return y, z, alpha_hat
