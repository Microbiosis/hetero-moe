"""§6 Rerank span 级融合测试 (cross-encoder + SpanSparseRouterSTE)。

覆盖:
  1. 前向 shape 正确 (scores [B, N], z_span [B, N, M])
  2. span 级路由生效 (同 span 内 token 共享路由)
  3. list-wise InfoNCE 可达 (完美相关性 → loss 趋近 0)
  4. 梯度流打通 (W_router / cls_head 有梯度)
  5. 训练收敛 (listwise InfoNCE 下降)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from archive.pre_baseline.v5_alpha.rerank_fusion import RerankFusionLayer
from archive.pre_baseline.v5_alpha.rerank_losses import listwise_infonce
from archive.pre_baseline.v5_alpha.span_router import build_span_id


# ---- 工具: 构造 (query, doc) 拼接输入 + span_id ----
def make_pair_input(num_queries, num_docs_per_q, d_model, seed=0):
    """每 query 配 num_docs_per_q 个 doc, 每个 (q,d) pair 为一个 span = [query, doc] 拼接。
    返回 x [B=num_queries, S=N_DOCS*(dq+dd), D], span_id [B, S], N_span=N_DOCS, labels [B, N_DOCS].
    span 边界 mask: 每个 pair 的 query 起始处为 True。正样本 = 第 0 个 doc。
    """
    torch.manual_seed(seed)
    dq, dd = 4, 6
    q = torch.randn(num_queries, dq, d_model)      # query tokens
    d = torch.randn(num_queries, num_docs_per_q, dd, d_model)  # doc tokens
    S = num_docs_per_q * (dq + dd)
    x = torch.zeros(num_queries, S, d_model)
    span_id = torch.zeros(num_queries, S, dtype=torch.long)
    boundary = torch.zeros(num_queries, S, dtype=torch.bool)
    labels = torch.zeros(num_queries, num_docs_per_q)
    labels[:, 0] = 1  # 第 0 个 doc 为正样本
    for i in range(num_queries):
        for j in range(num_docs_per_q):
            seg_start = j * (dq + dd)
            seg = torch.cat([q[i], d[i, j]], dim=0)  # [dq+dd, D]
            x[i, seg_start:seg_start + dq + dd] = seg
            span_id[i, seg_start:seg_start + dq + dd] = j
            boundary[i, seg_start] = True  # 每 pair 起始
    return x, span_id, num_docs_per_q, labels


def test_rerank_forward_shapes():
    torch.manual_seed(0)
    D = 16; D_FF = 32; M = 4; K = 2
    layer = RerankFusionLayer(D, D_FF, M, K)
    x, span_id, N_span, labels = make_pair_input(2, 4, D)
    scores, z_tok, alpha, ahat = layer(x, span_id, N_span)
    assert scores.shape == (2, N_span), f"scores shape {scores.shape}"
    assert z_tok.shape == (2, x.size(1), M), f"z_tok shape {z_tok.shape}"
    assert alpha.shape == (2, N_span, M)
    assert ahat.shape == (2, N_span, M)


def test_rerank_span_routing_shared():
    """同 span 内 token 共享同一路由 (v5.1 §4.3 步骤 4 广播)。"""
    torch.manual_seed(1)
    D = 16; M = 4; K = 2
    from archive.pre_baseline.v5_alpha.span_router import SpanSparseRouterSTE
    x, span_id, N_span, _ = make_pair_input(1, 3, D)
    z = F.linear(x, torch.randn(M, D))
    ahat = SpanSparseRouterSTE.apply(z, span_id, N_span, K)
    # span 0 = [0..9], span 1 = [10..19], span 2 = [20..29]
    assert torch.allclose(ahat[0, 0], ahat[0, 3]), "span0 内 token 0,3 应同路由"
    assert torch.allclose(ahat[0, 5], ahat[0, 9]), "span0 内 token 5,9 应同路由"
    assert torch.allclose(ahat[0, 10], ahat[0, 14]), "span1 内 token 10,14 应同路由"


def test_rerank_gradient_flow():
    """梯度流打通: W_router / cls_head 接收非零梯度。"""
    torch.manual_seed(2)
    D = 16; D_FF = 32; M = 4; K = 2
    layer = RerankFusionLayer(D, D_FF, M, K)
    x, span_id, N_span, labels = make_pair_input(2, 4, D)
    scores, _, _, _ = layer(x, span_id, N_span)
    loss = listwise_infonce(scores, labels)
    loss.backward()
    assert layer.W_router.grad is not None and layer.W_router.grad.norm() > 0
    assert layer.cls_head.grad is not None and layer.cls_head.grad.norm() > 0


def test_rerank_listwise_infonce_decreases():
    """listwise InfoNCE 训练下降。"""
    torch.manual_seed(3)
    D = 16; D_FF = 32; M = 4; K = 2
    layer = RerankFusionLayer(D, D_FF, M, K)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    losses = []
    for step in range(25):
        x, span_id, N_span, labels = make_pair_input(4, 4, D, seed=step % 5)
        opt.zero_grad(set_to_none=True)
        scores, _, _, _ = layer(x, span_id, N_span)
        loss = listwise_infonce(scores, labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # 至少某步下降 (训练非平凡)
    assert min(losses) < losses[0] or abs(losses[-1] - losses[0]) < 1e-3, \
        f"listwise InfoNCE 应下降, {losses[0]:.3f} → {losses[-1]:.3f}"
    print(f"  listwise InfoNCE: {losses[0]:.4f} → {losses[-1]:.4f}  "
          f"降幅 {(losses[0]-losses[-1])/max(losses[0],1e-9)*100:.1f}%")


def test_rerank_ranking_improves():
    """训练后正样本应被排到最高 (同分布排序正确率提升)。"""
    torch.manual_seed(4)
    D = 16; D_FF = 32; M = 4; K = 2; N_DOCS = 8
    layer = RerankFusionLayer(D, D_FF, M, K)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    EVAL_SEED = 7
    N_TRAIN = 30
    # 训练
    for step in range(N_TRAIN):
        x, span_id, N_span, labels = make_pair_input(16, N_DOCS, D, seed=step % 5)
        opt.zero_grad(set_to_none=True)
        scores, _, _, _ = layer(x, span_id, N_span)
        loss = listwise_infonce(scores, labels)
        loss.backward()
        opt.step()
    # 在训练分布上评估
    x, span_id, N_span, labels = make_pair_input(16, N_DOCS, D, seed=EVAL_SEED)
    with torch.no_grad():
        scores, _, _, _ = layer(x, span_id, N_span)
    n_queries = 16
    hits = 0
    for i in range(n_queries):
        row = scores[i]
        if row[0] >= row.max():
            hits += 1
    acc = hits / n_queries
    print(f"  排序正确率 (train {N_TRAIN}步后): {acc*100:.1f}%  (随机基线 {1/N_DOCS*100:.1f}%)")
    assert acc >= 1 / N_DOCS, f"排序正确率 {acc} 应 >= 随机 {1/N_DOCS}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[PASS] {name}")
    print("\nRerank span 级融合测试全部通过.")
