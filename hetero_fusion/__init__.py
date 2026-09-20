"""异构微观融合系统 (Heterogeneous Micro-Fusion) — v4.0 规范工程实现。

包结构:
    hetero_fusion.core   — 融合核心 (fusion/router/quant/losses/parallel)
    hetero_fusion.train  — 训练器 (trainer)
    hetero_fusion.infer  — 推理 (inference)

可追溯性矩阵（模块 → 规范条款）:
    core.fusion.HeteroFusionLayer    → §2.1-2.4 (分布对齐/截断/融合/残差)
    core.fusion.swiglu_forward       → §2.2
    core.router.SparseRouterSTE      → §2.3 (带 Mask 的 Top-K 稀疏路由 STE)
    core.router.capacity_route       → §5.1, §5.2
    core.quant.FakeQuantSTE          → §2.2, §6.2 (INT4 量化感知)
    core.losses.lm_loss              → §3.1
    core.losses.balance_loss         → §3.2
    core.losses.z_loss               → §3.3
    core.losses.total_loss           → §3
    core.parallel.tensor_parallel_split → §6.1
    train.HeteroTrainer              → §4.1-4.3
    infer.InferenceRunner            → §5
"""
from .core import (
    FakeQuantSTE,
    fake_quant,
    quantization_error,
    SparseRouterSTE,
    capacity_route,
    swiglu_forward,
    HeteroFusionLayer,
    lm_loss,
    balance_loss,
    z_loss,
    total_loss,
    tensor_parallel_split,
    EP_THRESHOLD,
)
from .train import HeteroTrainer, build_param_groups
from .infer import InferenceRunner

__all__ = [
    # core
    "FakeQuantSTE", "fake_quant", "quantization_error",
    "SparseRouterSTE", "capacity_route",
    "swiglu_forward", "HeteroFusionLayer",
    "lm_loss", "balance_loss", "z_loss", "total_loss",
    "tensor_parallel_split", "EP_THRESHOLD",
    # train
    "HeteroTrainer", "build_param_groups",
    # infer
    "InferenceRunner",
]

__version__ = "4.0.0"