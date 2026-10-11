"""Rerank 排序损失 (v5.1 §4.3 + §6)。

list-wise InfoNCE: 每 query 一正样本对应多个负样本, 正样本在 (负样本池) 上
softmax 取最高 relevance。这是 cross-encoder reranker 的标准训练信号。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def listwise_infonce(scores: torch.Tensor, labels: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """list-wise InfoNCE 排序损失。

    Args:
        scores:   [B, N] 每 (query, doc) 对的 relevance 分数 (B = num_queries, N = docs/query)
        labels:   [B, N] 0/1 标注, 每行恰一个 1 (正样本)
        temperature: 温度

    Returns:
        标量损失。随机时 ≈ log(N-1)。
    """
    assert labels.sum(dim=1).max() == 1 and labels.sum(dim=1).min() == 1, \
        "每 query 恰一个正样本 (labels 每行恰一个 1)"
    pos_idx = labels.argmax(dim=1).long()  # [B]
    logits = scores / temperature
    # 屏蔽自匹配: 不筛, list-wise 直接在 N 个候选上 softmax
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -log_prob[torch.arange(logits.size(0)), pos_idx].mean()
    return loss
