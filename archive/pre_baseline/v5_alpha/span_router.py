"""§4.3 span 级稀疏路由 STE (v5.1 新增, D5 闭合)。

算法 (v5.1 §4.3 步骤 1-5):
  1. 路由 logits 仍 per-token: z[b,s,m] = x[b,s] · W_router^T
  2. span 内 mean-pool: alpha_span[b,j,m] = softmax_m(mean_{s∈span_j} z[b,s,m])
  3. 每 span 独立 Top-K + STE: alpha_hat_span[b,j,m] = alpha_span[b,j,m] · 𝟙[m∈TopK]
  4. 广播回 token: alpha_hat[b,s,m] = alpha_hat_span[b, span_id[b,s], m] (同 span 共享)
  5. 反向: autograd gather 反向 = sum-pool 回 span 级, 再乘 span 级 mask (STE)

STE 不变量 (三粒度统一): 非选中位置梯度为 0, 反向在对应粒度应用 mask, 不跨粒度泄漏。

输入:
    z:           [B, S, M] per-token 路由 logits (softmax 前)
    span_id:     [B, S] long, 每 token 所属 span 索引 (0..num_spans-1)
    num_spans:   int, 该 batch 内最大 span 数
    k:           int, Top-K

返回:
    alpha_hat:   [B, S, M] 广播后的路由权重 (同 span 内所有 token 共享)

边界 Token 划分: span 由 Tokenizer 层显式边界 Token ([Q_SEP]/[D_SEP]) 划分,
禁止基于注意力权重的学习切分 (保证梯度路径绝对确定可微)。见 build_span_id。
"""
from __future__ import annotations

import torch


def build_span_id(boundary_mask: torch.Tensor):
    """从边界 mask 构建 span_id。

    Args:
        boundary_mask: [B, S] bool, True 表示"新 span 起始"。
                       每行第一个 token 自动强制为边界 (保证至少一个 span)。

    Returns:
        span_id: [B, S] long, 每 token 的 span 索引 (0..N_span-1)
        num_spans: int, batch 内最大 span 数
    """
    B, S = boundary_mask.shape
    bm = boundary_mask.clone()
    bm[:, 0] = True  # 首 token 必为边界
    span_id = bm.cumsum(dim=1) - 1
    num_spans = int(span_id.max().item()) + 1
    return span_id.long(), num_spans


class SpanSparseRouterSTE(torch.autograd.Function):
    """span 级硬 Top-K 路由 + STE (v5.1 §4.3)。

    forward:  z [B,S,M] → z_span (mean-pool) → alpha_span (softmax)
              → alpha_hat_span (Top-K STE) → alpha_hat [B,S,M] (广播)
    backward: grad_output [B,S,M] → sum-pool 回 span 级 [B,num_spans,M]
              → STE mask → broadcast / span_cnt 回 token 级 [B,S,M]

    STE 不变量: 非 Top-K span 位置梯度为 0, 不跨粒度泄漏。
    """

    @staticmethod
    def forward(ctx, z, span_id, num_spans, k):
        B, S, M = z.shape
        dev = z.device
        z_span = torch.zeros(B, num_spans, M, device=dev, dtype=z.dtype)
        span_cnt = torch.zeros(B, num_spans, device=dev, dtype=z.dtype)
        ones = torch.ones(S, device=dev, dtype=z.dtype)
        for b in range(B):
            sid = span_id[b]  # [S]
            cnt = torch.zeros(num_spans, device=dev, dtype=z.dtype)
            ssum = torch.zeros(num_spans, M, device=dev, dtype=z.dtype)
            cnt.scatter_add_(0, sid, ones)
            ssum.scatter_add_(0, sid.unsqueeze(-1).expand(-1, M), z[b])
            z_span[b] = ssum / cnt.clamp(min=1).unsqueeze(-1)

        alpha_span = z_span.softmax(-1)
        _, idx = alpha_span.topk(k, dim=-1)
        mask_span = torch.zeros_like(alpha_span).scatter_(-1, idx, 1.0)
        alpha_hat_span = alpha_span * mask_span
        ctx.save_for_backward(mask_span, span_id, span_cnt)

        # 广播回 token 级: 同 span 内所有 token 共享路由
        alpha_hat = alpha_hat_span.gather(1, span_id.unsqueeze(-1).expand(-1, -1, M))
        return alpha_hat  # [B, S, M]

    @staticmethod
    def backward(ctx, grad_output):  # [B, S, M]
        mask_span, span_id, span_cnt = ctx.saved_tensors
        B, S, M = grad_output.shape
        grad_span = torch.zeros_like(mask_span)  # [B, num_spans, M]
        for b in range(B):
            sid = span_id[b]
            grad_span[b].scatter_add_(0, sid.unsqueeze(-1).expand(-1, M), grad_output[b])
        # STE: 仅 Top-K span 位置透传梯度
        grad_span = grad_span * mask_span
        # 反向 mean-pool: 每 token 得 span_grad / N_span
        cnt_gathered = span_cnt.gather(1, span_id).clamp(min=1).unsqueeze(-1)  # [B, S, 1]
        grad_z = grad_span.gather(1, span_id.unsqueeze(-1).expand(-1, -1, M)) / cnt_gathered
        return grad_z, None, None, None
