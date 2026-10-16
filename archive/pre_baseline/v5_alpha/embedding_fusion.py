"""§2 异构融合层 (v5.0-α.1, G=sequence) + §4 模态投影 P_m。

支持 granularity='sequence' (v5.0-α) 与 'token' (v4.0 baseline) 双模式,
供 §7.1 对照实验使用。其余结构 (P_m, γ, β, α, SwiGLU, INT4, 残差) 共享。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_moe.core.quant import FakeQuantSTE
from hetero_moe.core.fusion import swiglu_forward
from hetero_moe.core.router import SparseRouterSTE  # v4.0 token 级
from archive.pre_baseline.v5_alpha.span_router import SpanSparseRouterSTE  # v5.1 span 级
from .seq_router import SeqSparseRouterSTE


def realistic_ffn_init(layer, d_model: int, d_ff: int) -> None:
    """将 §7 占位的 0.02-std FFN 权重重置为真实 Transformer 量级 (1/√D, 1/√D_ff)。

    §1.1 底座为预训练模型; §7 的 0.02 仅原型占位。真实量级方能使适配器训练
    产生可见的 InfoNCE 下降与 §8.1 梯度验证。
    """
    sg = 1.0 / (d_model ** 0.5)
    sd = 1.0 / (d_ff ** 0.5)
    with torch.no_grad():
        for p in (*layer.w_gates, *layer.w_ups):
            p.mul_(sg / 0.02)
        for p in layer.w_downs:
            p.mul_(sd / 0.02)


class EmbeddingFusionLayer(nn.Module):
    """异构融合层 (支持 token / sequence 双粒度)。

    §4.1 对齐 (alpha 阶段 D_m = D_shared):
        h_aligned = P_m(stop_grad(x)) ⊙ γ_m + β_m
        P_m ∈ R^{D×D} (方阵, identity-ish 初始化; errata-3 标注 O(D²) 参数成本)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        granularity: str = "sequence",
        quant_bits: int = 4,
        quant_group_size: int = 128,
        detach_router_input: bool = False,
    ):
        super().__init__()
        assert granularity in ("token", "sequence", "span")
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.granularity = granularity
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.detach_router_input = detach_router_input

        # ---- 冻结底座 FFN (§1.3 异构性仅存于 FFN) ----
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)]
        )
        self.w_ups = nn.ParameterList(
            [nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)]
        )
        self.w_downs = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model, d_ff) * 0.02) for _ in range(num_experts)]
        )
        for pl in (self.w_gates, self.w_ups, self.w_downs):
            for p in pl:
                p.requires_grad = False

        # ---- 可训练参数 (§4.1 + §4 P_m) ----
        self.W_router = nn.Parameter(torch.randn(num_experts, d_model) * 0.01)
        # P_m: D×D 方阵, identity + 小噪声初始化 (近 pass-through, 学习偏差)
        self.P_m = nn.ParameterList(
            [nn.Parameter(torch.eye(d_model) + torch.randn(d_model, d_model) * 0.01)
             for _ in range(num_experts)]
        )
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(d_model)) for _ in range(num_experts)]
        )
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(num_experts)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.empty(1).uniform_(0.01, 0.1)) for _ in range(num_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        span_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """前向。

        Args:
            x:       [B, S, D] 输入
            span_id: [B, S] long, 仅 granularity="span" 时需要 (每 token 的 span 索引)

        Returns:
            y:             [B, S, D] 残差融合输出
            z:             [B, M] (seq) / [B, S, M] (token) / [B, S, M] (span, 同 span 共享) 路由 logits
            alpha_hat:     [B, S, M] 实际路由权重 (广播后)
            alpha_hat_other: 粒度级路由权重 [B, M] (seq) / [B, num_spans, M] (span) / None (token)
        """
        B, S, D = x.shape
        x_residual = x
        alpha_hat_other: Optional[torch.Tensor] = None

        if self.granularity == "sequence":
            x_seq = x.mean(dim=1)  # [B, D]
            if self.detach_router_input:
                x_seq = x_seq.detach()
            z = F.linear(x_seq, self.W_router)  # [B, M]
            alpha_seq = F.softmax(z, dim=-1)
            alpha_hat_other = SeqSparseRouterSTE.apply(alpha_seq, self.top_k)  # [B, M]
            alpha_hat = alpha_hat_other.unsqueeze(1).expand(-1, S, -1)
        elif self.granularity == "span":
            assert span_id is not None, "granularity='span' 必须提供 span_id [B, S]"
            x_r = x if not self.detach_router_input else x.detach()
            z = F.linear(x_r, self.W_router)  # [B, S, M]
            num_spans = int(span_id.max().item()) + 1
            # v5.1 §4.3: z per-token → span 内 mean-pool → softmax → Top-K STE → 广播
            alpha_hat = SpanSparseRouterSTE.apply(z, span_id, num_spans, self.top_k)  # [B, S, M]
            # alpha_hat_other: span 级权重 [B, num_spans, M] (供 span 级损失)
            z_span = torch.zeros(B, num_spans, D, device=x.device)
            ones = torch.ones(S, device=x.device)
            cnt = torch.zeros(B, num_spans, device=x.device)
            for b in range(B):
                sid = span_id[b]
                ssum = torch.zeros(num_spans, D, device=x.device)
                c = torch.zeros(num_spans, device=x.device)
                c.scatter_add_(0, sid, ones)
                ssum.scatter_add_(0, sid.unsqueeze(-1).expand(-1, D), x_r[b])
                z_span[b] = ssum / c.clamp(min=1).unsqueeze(-1)
            z_span = F.linear(z_span, self.W_router)  # [B, num_spans, M]
            alpha_span = z_span.softmax(-1)
            _, idx = alpha_span.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(alpha_span).scatter_(-1, idx, 1.0)
            alpha_hat_other = alpha_span * mask  # [B, num_spans, M]
        else:  # token (v4.0 baseline)
            x_r = x if not self.detach_router_input else x.detach()
            z = F.linear(x_r, self.W_router)  # [B, S, M]
            alpha = F.softmax(z, dim=-1)
            alpha_hat = SparseRouterSTE.apply(alpha, self.top_k)  # [B, S, M]

        # ---- 逐专家融合 (§4.1 对齐 → §2.2 SwiGLU → INT4 → §2.4 残差) ----
        out_sum = torch.zeros_like(x)
        x_det = x.detach()
        for m in range(self.num_experts):
            h_aligned = F.linear(x_det, self.P_m[m]) * self.gammas[m] + self.betas[m]
            f_m = swiglu_forward(h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_m_noisy = FakeQuantSTE.apply(f_m, self.quant_bits, self.quant_group_size)
            out_sum = out_sum + alpha_hat[..., m : m + 1] * self.alphas[m] * f_m_noisy

        y = x_residual + out_sum
        return y, z, alpha_hat, alpha_hat_other
