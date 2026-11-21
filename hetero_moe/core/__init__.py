"""hetero_moe.core — 核心融合层、异构专家、路由、量化、损失与并行工具 (v5.1)。"""
from .experts import ExpertSpec, HeteroExpert, LoRALinearAdapter, make_heterogeneous_specs
from .interface import DeCollapsedLocalInterface
from .quant import FakeQuantSTE, fake_quant, quantization_error
from .router import CapacityRouterSTE, SparseRouterSTE, capacity_route, capacity_route_ste
from .fusion import swiglu_forward, HeteroFusionLayer
from .losses import lm_loss, balance_loss, z_loss, total_loss
from .parallel import (
    EP_THRESHOLD,
    TensorParallelSwiGLU,
    recommend_strategy,
    tensor_parallel_split,
    tp_shard_ffn,
    tp_swiglu_forward,
)

__all__ = [
    "ExpertSpec",
    "HeteroExpert",
    "LoRALinearAdapter",
    "make_heterogeneous_specs",
    "DeCollapsedLocalInterface",
    "FakeQuantSTE",
    "fake_quant",
    "quantization_error",
    "SparseRouterSTE",
    "CapacityRouterSTE",
    "capacity_route",
    "capacity_route_ste",
    "swiglu_forward",
    "HeteroFusionLayer",
    "lm_loss",
    "balance_loss",
    "z_loss",
    "total_loss",
    "tensor_parallel_split",
    "tp_shard_ffn",
    "tp_swiglu_forward",
    "TensorParallelSwiGLU",
    "recommend_strategy",
    "EP_THRESHOLD",
]
