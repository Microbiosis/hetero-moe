"""§6 span 级 rerank 融合层 (cross-encoder, v5.1 §4.3 粒度匹配)。

架构: cross-encoder 风格 — 每 (query, doc) 对拼接为独立 span:
    [q_tok...] [d_tok...]
每个 span 路由到专精专家 (span 级 Top-K, SpanSparseRouterSTE),
分类头输出 relevance 分数。粒度匹配 (v5.1 P): 决策单元=pair (span 级)
与 pair 级 list-wise InfoNCE 损失粒度一致。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_fusion.core.quant import FakeQuantSTE
from hetero_fusion.core.fusion import swiglu_forward
from .span_router import SpanSparseRouterSTE


class RerankFusionLayer(nn.Module):
    """span 级 rerank 融合层 (cross-encoder, G=span)。

    Args:
        x:         [B, S, D] 拼接 token 表示
        span_id:   [B, S] long, 每 token 所属 pair-span 索引
        num_spans: int, batch 内 pair 数

    Returns:
        scores:    [B, num_spans] 每 pair relevance 分数
        z_tok:     [B, S, M] per-token 路由 logits (供 L_z)
        alpha_span:[B, num_spans, M] 每 span 路由概率 (供 L_balance)
        ahat_span: [B, num_spans, M] 每 span 稀疏路由权重
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int,
                 quant_bits: int = 4, quant_group_size: int = 128):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size

        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)])
        self.w_ups = nn.ParameterList([nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)])
        self.w_downs = nn.ParameterList([nn.Parameter(torch.randn(d_model, d_ff) * 0.02) for _ in range(num_experts)])
        for pl in (self.w_gates, self.w_ups, self.w_downs):
            for p in pl:
                p.requires_grad = False

        self.W_router = nn.Parameter(torch.randn(num_experts, d_model) * 0.01)
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(d_model)) for _ in range(num_experts)])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(num_experts)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.empty(1).uniform_(0.01, 0.1)) for _ in range(num_experts)])
        self.cls_head = nn.Parameter(torch.randn(d_model, 1) * 0.01)

    def _pool_to_span(self, t: torch.Tensor, span_id: torch.Tensor, num_spans: int):
        """[B, S, D'] → [B, num_spans, D'] mean-pool per span."""
        B, S, Dp = t.shape
        dev = t.device
        out = torch.zeros(B, num_spans, Dp, device=dev, dtype=t.dtype)
        cnt = torch.zeros(B, num_spans, device=dev, dtype=t.dtype)
        ones = torch.ones(S, device=dev, dtype=t.dtype)
        for b in range(B):
            sid = span_id[b]
            cc = torch.zeros(num_spans, device=dev, dtype=t.dtype)
            ss = torch.zeros(num_spans, Dp, device=dev, dtype=t.dtype)
            cc.scatter_add_(0, sid, ones)
            ss.scatter_add_(0, sid.unsqueeze(-1).expand(-1, Dp), t[b])
            out[b] = ss / cc.clamp(min=1).unsqueeze(-1)
        return out

    def forward(self, x, span_id, num_spans):
        B, S, D = x.shape
        # per-token logits
        z_tok = F.linear(x, self.W_router)  # [B, S, M]
        # span 级 Top-K STE → 广播回 token 级 ahat [B, S, M]
        ahat_tok = SpanSparseRouterSTE.apply(z_tok, span_id, num_spans, self.top_k)
        # 路由概率 (未稀疏化, 供 L_balance 的 P_m)
        z_span = self._pool_to_span(z_tok, span_id, num_spans)  # [B, num_spans, M]
        alpha_span = z_span.softmax(-1)

        # ---- 逐专家融合 (per token, ahat 同 span 共享) ----
        out_sum = torch.zeros_like(x)
        x_det = x.detach()
        for m in range(self.num_experts):
            h = x_det * self.gammas[m] + self.betas[m]
            f = swiglu_forward(h, self.w_gates[m], self.w_ups[m], self.w_downs[m])
            f_noisy = FakeQuantSTE.apply(f, self.quant_bits, self.quant_group_size)
            out_sum = out_sum + ahat_tok[..., m : m + 1] * self.alphas[m] * f_noisy
        y_hidden = x + out_sum

        # ---- pool 到 per-span → 分类头 → relevance 分数 ----
        pooled = self._pool_to_span(y_hidden, span_id, num_spans)  # [B, num_spans, D]
        scores = (pooled @ self.cls_head).squeeze(-1)  # [B, num_spans]
        ahat_span = self._pool_to_span(ahat_tok, span_id, num_spans)  # [B, num_spans, M]
        return scores, z_tok, alpha_span, ahat_span
