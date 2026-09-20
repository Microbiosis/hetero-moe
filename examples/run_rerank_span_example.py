"""Rerank per-type 融合示例 (G=span, v5.1 §4.3 + 粒度匹配原理)。

rerank (cross-encoder) 场景:
  输入: query tokens + [SEP] + doc tokens → 两个 span (query span, doc span)
  路由: G=span, 每 span 独立 Top-K → query 路由到专精 query 的专家, doc 路由到专精 doc 的专家
  输出: mean-pool → linear head → scalar 相关性分数
  损失: MSE 到 ground-truth 相关性 (模拟)

验证粒度匹配原理: rerank 损失是 pair 级 (1 信号/pair),
但 query/doc 各自需要被不同专家处理 → span 粒度 (2 决策/pair)
比 token 粒度 (S 决策/pair) 更匹配, 且路由按 span 专化 (query≠doc 专家)。

配置: D=16, D_FF=32, M=4 专家 (模拟 4 个 rerank 变体), K=1,
      B=4 pairs, query 4 tokens, doc 4 tokens, S=9 (含 2 SEP), 30 步
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from archive.pre_baseline.v5_alpha.span_router import build_span_id
from archive.pre_baseline.v5_alpha.embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init


def make_rerank_data(B, query_len, doc_len, D, seed=11):
    """构造 rerank 输入与 ground-truth 相关性分数。

    query/doc 用不同随机基底模拟 (不同分布), ground-truth 相关性
    用 query 与 doc 基底向量的余弦相似度模拟 (有真信号, 可学习)。
    """
    g = torch.Generator().manual_seed(seed)
    S = query_len + 1 + doc_len  # +1 为 SEP
    x = torch.randn(B, S, D, generator=g)
    # query 基底 (前 query_len token 共享基底), doc 基底 (后 doc_len token)
    q_base = torch.randn(B, D, generator=g) * 0.5
    d_base = torch.randn(B, D, generator=g) * 0.5
    for b in range(B):
        x[b, :query_len] = q_base[b].unsqueeze(0) + 0.3 * torch.randn(query_len, D, generator=g)
        x[b, query_len + 1:] = d_base[b].unsqueeze(0) + 0.3 * torch.randn(doc_len, D, generator=g)
    # ground-truth 相关性: query 基底 vs doc 基底的余弦相似度 → [0,1]
    qn = F.normalize(q_base, dim=-1); dn = F.normalize(d_base, dim=-1)
    score = (qn * dn).sum(dim=-1).clamp(-1, 1) * 0.5 + 0.5  # → [0,1]
    x = x.detach()
    return x, score

import torch.nn.functional as F


def make_span_id_for_rerank(B, query_len, doc_len):
    """构造 rerank span_id: query span=[0..query_len-1], doc span=[query_len+1..S-1]。"""
    S = query_len + 1 + doc_len
    bm = torch.zeros(B, S, dtype=torch.bool)
    bm[:, 0] = True           # query span 起始
    bm[:, query_len + 1] = True  # doc span 起始 (SEP 后)
    return build_span_id(bm)


def build_model(D, D_FF, M, K, head_out=1, seed=7):
    torch.manual_seed(seed)
    layer = EmbeddingFusionLayer(D, D_FF, M, K, granularity="span")
    realistic_ffn_init(layer, D, D_FF)
    head = nn.Linear(D, head_out)
    head.weight.requires_grad_(False)  # 冻结输出头
    return layer, head


def main():
    torch.manual_seed(0)
    D, D_FF, M, K = 16, 32, 4, 1
    query_len, doc_len = 4, 4
    B, STEPS = 4, 30

    x, score = make_rerank_data(B, query_len, doc_len, D)
    span_id, num_spans = make_span_id_for_rerank(B, query_len, doc_len)
    assert num_spans == 2, f"rerank 应有 2 个 span (query+doc), 实际 {num_spans}"

    layer, head = build_model(D, D_FF, M, K)
    # 仅训 W_router (隔离路由能力, 与 H' 实验一致)
    opt = torch.optim.AdamW([layer.W_router], lr=1e-2)

    print(f"{'='*64}\nRerank per-type 融合 (G=span) 端到端\n{'='*64}")
    print(f"配置: D={D}, M={M}, K={K}, query={query_len}t + doc={doc_len}t, S={query_len+1+doc_len}, {B} pairs")
    print(f"span: query span=[0..{query_len-1}], doc span=[{query_len+1}..{query_len+doc_len}]")
    print(f"ground-truth 相关性: {[round(score[b].item(),3) for b in range(B)]}")

    losses = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y, z, ah, ahs_span = layer(x, span_id=span_id)
        e = y.mean(dim=1)
        pred = head(e).squeeze(-1)  # [B]
        loss = F.mse_loss(pred, score)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 7 == 0 or step == STEPS - 1:
            print(f"  step {step:2d}  loss={loss.item():.4f}  pred={[round(pred[b].item(),3) for b in range(B)]}")

    print(f"\n  MSE: {losses[0]:.4f} → {losses[-1]:.4f}  (降幅 {(losses[0]-losses[-1])/max(losses[0],1e-9)*100:.1f}%)")

    # ---- 验证 span 路由专化 (query span 与 doc span 路由到不同专家) ----
    layer.eval()
    with torch.no_grad():
        _, _, ah_final, _ = layer(x, span_id=span_id)
    picked = ah_final.argmax(dim=-1)  # [B, S] 每 token 选中的专家
    print(f"\n路由专化验证 (每个 pair 的 query span 与 doc span 选中专家):")
    same_count = 0
    for b in range(B):
        q_pick = picked[b, 0].item()  # query span 专家
        d_pick = picked[b, query_len + 1].item()  # doc span 专家
        same = (q_pick == d_pick)
        same_count += int(same)
        print(f"  pair {b}: query→expert{q_pick}  doc→expert{d_pick}  {'(同专家)' if same else '(不同专家)'}")
    diff_ratio = 1.0 - same_count / B
    print(f"\n路由按 span 专化比例: {diff_ratio*100:.0f}% query/doc 路由到不同专家")
    print(f"{'='*64}")
    if losses[-1] < losses[0] and diff_ratio >= 0.5:
        print("PASS: Rerank span 融合 — Loss 收敛 + 路由按 span 专化 (query≠doc 专家)")
    else:
        print("PARTIAL: Loss 收敛或路由专化不足, 需调参")


if __name__ == "__main__":
    main()
