"""hetero_moe.core — 核心融合层、异构专家、路由、量化、损失与并行工具。"""
from .experts import ExpertSpec, HeteroExpert, LoRALinearAdapter, make_heterogeneous_specs
from .quant import FakeQuantSTE, fake_quant, quantization_error
from .router import SparseRouterSTE, capacity_route
from .fusion import swiglu_forward, HeteroFusionLayer
from .losses import lm_loss, balance_loss, z_loss, total_loss
from .parallel import tensor_parallel_split, EP_THRESHOLD

__all__ = [
    "ExpertSpec",
    "HeteroExpert",
    "LoRALinearAdapter",
    "make_heterogeneous_specs",
    "FakeQuantSTE",
    "fake_quant",
    "quantization_error",
    "SparseRouterSTE",
    "capacity_route",
    "swiglu_forward",
    "HeteroFusionLayer",
    "lm_loss",
    "balance_loss",
    "z_loss",
    "total_loss",
    "tensor_parallel_split",
    "EP_THRESHOLD",
]
