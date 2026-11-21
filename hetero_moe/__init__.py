"""异构微观融合系统 (Heterogeneous Micro-Fusion) — v5.1 规范工程实现。

包结构:
    hetero_moe.core   — 融合核心 (experts/fusion/router/quant/losses/parallel)
    hetero_moe.train  — 训练器 (trainer)
    hetero_moe.infer  — 推理 (inference)
    hetero_moe.eval   — 反泄漏下游评测门禁 (gate)

v5.0 / v5.1 固有缺陷系统修复:
    - core.experts.{ExpertSpec, HeteroExpert, LoRALinearAdapter}: 真异构专家 + 深入非线性子层的内部 LoRA 破 β 墙
    - core.fusion.HeteroFusionLayer: 向量化预筛真稀疏分派、grad_checkpoint 真省显存、gate_rank 低秩向量门控、
      renorm_topk=True 训练推理幅值对齐、train_capacity_factor 训练推理容量对齐、权重+激活双目标 INT4 伪量化
    - core.parallel.{tp_swiglu_forward, TensorParallelSwiGLU}: 列并行+行并行+All-Reduce 可微张量并行算子
    - train.HeteroTrainer: Phase 1 同步归零 z_loss + Phase 1->2 无损保留 Adam 一阶/二阶动量状态
    - infer.InferenceRunner: 消除二次路由计算 + 修复多层 skip_rate 越界
    - eval.DownstreamEvalGate: 强制 train/val/test 零重叠 + 零参数/单底座/阴性对照 + Bootstrap CI
"""
from .core import (
    CapacityRouterSTE,
    DeCollapsedLocalInterface,
    EP_THRESHOLD,
    ExpertSpec,
    FakeQuantSTE,
    HeteroExpert,
    HeteroFusionLayer,
    LoRALinearAdapter,
    SparseRouterSTE,
    TensorParallelSwiGLU,
    balance_loss,
    capacity_route,
    capacity_route_ste,
    fake_quant,
    lm_loss,
    make_heterogeneous_specs,
    quantization_error,
    recommend_strategy,
    swiglu_forward,
    tensor_parallel_split,
    total_loss,
    tp_shard_ffn,
    tp_swiglu_forward,
    z_loss,
)
from .train import HeteroTrainer, build_param_groups
from .infer import InferenceRunner
from .eval import (
    DownstreamEvalGate,
    GateReport,
    IdentifiabilityGuard,
    SplitGuard,
    paired_bootstrap_ci,
)

__all__ = [
    # core
    "ExpertSpec", "HeteroExpert", "LoRALinearAdapter", "make_heterogeneous_specs",
    "DeCollapsedLocalInterface",
    "FakeQuantSTE", "fake_quant", "quantization_error",
    "SparseRouterSTE", "CapacityRouterSTE", "capacity_route", "capacity_route_ste",
    "swiglu_forward", "HeteroFusionLayer",
    "lm_loss", "balance_loss", "z_loss", "total_loss",
    "tensor_parallel_split", "tp_shard_ffn", "tp_swiglu_forward", "TensorParallelSwiGLU",
    "recommend_strategy", "EP_THRESHOLD",
    # train
    "HeteroTrainer", "build_param_groups",
    # infer
    "InferenceRunner",
    # eval
    "DownstreamEvalGate", "GateReport", "IdentifiabilityGuard", "SplitGuard", "paired_bootstrap_ci",
]

__version__ = "5.1.0"
