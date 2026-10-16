"""异构微观融合核心 — fusion / router / quant / losses / parallel

模块对应规范条款:
    fusion.HeteroFusionLayer → §2.1-2.4 (分布对齐/截断/融合/残差)
    fusion.swiglu_forward   → §2.2 (SwiGLU)
    router.SparseRouterSTE  → §2.3 (Top-K 稀疏路由 STE)
    router.capacity_route   → §5.1-5.2 (容量因子 + 溢出降级)
    quant.FakeQuantSTE      → §2.2, §6.2 (INT4 量化感知)
    losses.lm_loss          → §3.1
    losses.balance_loss     → §3.2 (f_m·P_m)
    losses.z_loss           → §3.3 (logsumexp²)
    losses.total_loss       → §3
    parallel.tensor_parallel_split → §6.1
"""
from .fusion import swiglu_forward, HeteroFusionLayer
from .router import SparseRouterSTE, capacity_route
from .quant import FakeQuantSTE, fake_quant, quantization_error
from .losses import lm_loss, balance_loss, z_loss, total_loss
from .parallel import tensor_parallel_split, EP_THRESHOLD

__all__ = [
    "swiglu_forward",
    "HeteroFusionLayer",
    "SparseRouterSTE",
    "capacity_route",
    "FakeQuantSTE",
    "fake_quant",
    "quantization_error",
    "lm_loss",
    "balance_loss",
    "z_loss",
    "total_loss",
    "tensor_parallel_split",
    "EP_THRESHOLD",
]