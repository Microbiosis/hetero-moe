"""异构微观融合系统 (Heterogeneous Micro-Fusion) — v5.0 规范工程实现。

包结构:
    hetero_moe.core   — 融合核心 (experts/fusion/router/quant/losses/parallel)
    hetero_moe.train  — 训练器 (trainer)
    hetero_moe.infer  — 推理 (inference)
    hetero_moe.eval   — 反泄漏下游评测门禁 (gate)

v5.0 固有缺陷系统修复:
    - core.experts.{ExpertSpec, HeteroExpert, LoRALinearAdapter}: 真异构专家 + 可选 LoRA 破 β 墙
    - core.fusion.HeteroFusionLayer: 真稀疏分派、grad_checkpoint 真省显存、gate_rank 低秩向量门控、
      renorm_topk=True 训练推理幅值对齐、零重复路由投影
    - train.HeteroTrainer: Phase 1 同步归零 z_loss 消除前层适配器污染
    - infer.InferenceRunner: 消除二次路由计算 + 修复多层 skip_rate 越界
    - eval.DownstreamEvalGate: 强制 train/val/test 零重叠 + 零参数/单底座/阴性对照 + Bootstrap CI
"""
from .core import (
    ExpertSpec,
    HeteroExpert,
    LoRALinearAdapter,
    make_heterogeneous_specs,
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
    "FakeQuantSTE", "fake_quant", "quantization_error",
    "SparseRouterSTE", "capacity_route",
    "swiglu_forward", "HeteroFusionLayer",
    "lm_loss", "balance_loss", "z_loss", "total_loss",
    "tensor_parallel_split", "EP_THRESHOLD",
    # train
    "HeteroTrainer", "build_param_groups",
    # infer
    "InferenceRunner",
    # eval
    "DownstreamEvalGate", "GateReport", "IdentifiabilityGuard", "SplitGuard", "paired_bootstrap_ci",
]

__version__ = "5.0.0"
