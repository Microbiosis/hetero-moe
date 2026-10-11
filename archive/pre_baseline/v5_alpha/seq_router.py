"""§2.2 序列级稀疏路由 STE (v5.0-α.1 errata V1/V2 修正版)。

Errata 闭合:
    V1 (forward): 返回 alpha_seq*mask (软权重), 非 binary mask —— 保留 softmax 概率,
                  使 §3.2 的 f_m = mean(α̂) 仍有负载信息。
    V2 (backward): 删除 mean(dim=1) —— 调用者 unsqueeze+expand 广播 [B,M]→[B,S,M],
                  autograd 的 expand 反向规则是 sum (非 mean), 故 backward 收到的
                  grad_output 已被自动 sum-pool 到 [B,M]; 仅需乘序列级 mask (STE)。

sum-pool 语义匹配 §2.2 文本 "sum-pooling 回序列级"。
"""
from __future__ import annotations

import torch


class SeqSparseRouterSTE(torch.autograd.Function):
    """序列级硬 Top-K 路由 + STE。

    前向: alpha_seq [B, M] -> alpha_seq * mask [B, M]   (软权重, V1 修正)
    反向: grad_output [B, M] (已 sum-pool) -> grad * mask  (V2 修正, 删 mean)

    调用者负责广播:
        alpha_hat_seq = SeqSparseRouterSTE.apply(alpha_seq, k)   # [B, M]
        alpha_hat = alpha_hat_seq.unsqueeze(1).expand(-1, S, -1)  # [B, S, M]
    反向时 autograd 将 [B,S,M] 梯度 sum 回 [B,M], 无需手动 mean。
    """

    @staticmethod
    def forward(ctx, alpha_seq, k):
        val, idx = torch.topk(alpha_seq, k, dim=-1)
        mask = torch.zeros_like(alpha_seq).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(mask)
        return alpha_seq * mask  # V1: 软权重 (非 binary)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None  # V2: grad_output 已 [B,M], 删 mean
