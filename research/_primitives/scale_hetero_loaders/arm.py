"""v37.0 — 单臂训练 (4 臂: A/B/C/D).

c_frozen is None → β 自训练 (D 臂, 唯一新增的可训练入口组件)
其它臂: c_frozen 给定 (零 / 随机 / TabLDM 被动), 仅训 P+头
"""
import torch
import torch.nn as nn

from .entry_encoder import EntryEncoder
from .eval import knn_acc


def train_arm(h_base, X, y, c_frozen, tr_idx, te_idx, d_entry, d_big, n_cls,
             steps, seed):
    """训练单臂, 返回 (test_acc, heldout_knn)."""
    torch.manual_seed(seed)
    n_in = X.shape[1]
    E = EntryEncoder(n_in, d_entry)
    P = nn.Linear(d_entry, d_big)
    head = nn.Linear(d_big, n_cls)
    params = list(P.parameters()) + list(head.parameters()) + (
        list(E.parameters()) if c_frozen is None else [])
    opt = torch.optim.Adam(params, lr=1e-3)

    for _ in range(steps):
        b = tr_idx[torch.randperm(len(tr_idx))[:32]]
        cin = c_frozen[b] if c_frozen is not None else E(X[b])
        logits = head(h_base[b] + P(cin))
        loss = nn.functional.cross_entropy(logits, y[b])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        cin_all = c_frozen if c_frozen is not None else E(X)
        h_all = h_base + P(cin_all)
        logits_te = head(h_all[te_idx])
        acc_te = (logits_te.argmax(1) == y[te_idx]).float().mean().item()
        # 仅在留出集 te_idx 上评估 k-NN，避免混入已拟合的训练集造成记忆泄漏
        return acc_te, knn_acc(h_all[te_idx], y[te_idx])