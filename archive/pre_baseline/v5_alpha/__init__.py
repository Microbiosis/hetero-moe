"""v5.0-α.1 — 序列级异构融合 (Embedding 场景) + v5.1 span 路由。

可追溯性:
    span_router.build_span_id          → v5.1 §4.3 (边界 Token → span_id)
    span_router.SpanSparseRouterSTE    → v5.1 §4.3 (span 级 Top-K STE, D5 闭合)
    rerank_fusion.RerankFusionLayer       → v5.1 rerank 端到端 (cross-encoder + span 路由)
    rerank_losses.listwise_infonce        → §6 (rerank pair-wise 损失)
    infonce.infonce_loss              → §5.1 (句级 InfoNCE, 替代 L_LM)
    seq_losses.balance_loss_seq       → §3.2 (序列级 f_m·P_m, D1 闭合)
    seq_losses.z_loss_seq             → §3.3 (序列级 z-loss)
"""
from .span_router import build_span_id, SpanSparseRouterSTE
from .seq_router import SeqSparseRouterSTE
from .embedding_fusion import EmbeddingFusionLayer, realistic_ffn_init
from .rerank_fusion import RerankFusionLayer
from .rerank_losses import listwise_infonce
from .infonce import infonce_loss
from .seq_losses import balance_loss_seq, z_loss_seq

__all__ = [
    "build_span_id",
    "SpanSparseRouterSTE",
    "SeqSparseRouterSTE",
    "EmbeddingFusionLayer",
    "realistic_ffn_init",
    "RerankFusionLayer",
    "listwise_infonce",
    "infonce_loss",
    "balance_loss_seq",
    "z_loss_seq",
]
__version__ = "5.0.0a1"
