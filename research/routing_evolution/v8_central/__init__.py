"""v8.0 — 跨架构族 + 全局中枢 (Central Workspace)。

设计动机 (用户问题: "大模型融合是不是也需要一个中枢?"):
    v7.0 的 expert 是并联的, 但大脑皮层存在一个贯穿全脑的"中枢"
    (类比 Global Workspace Theory). v8.0 引入三层中枢:

    C-1 (central.py):
        中枢 token c ∈ R^{D_shared}, 全局可学习向量.
        路由时广播到所有 token: z += c · U^T
        每个 step 后 EMA 更新 c (聚合 expert 输出).

    C-2 (fusion.py):
        attn→ffn 单向协同: ffn 路由 logits 受 attn 输出影响.
        z_ffn = W_ffn · (x + attn_output)
        (默认同时启用 C-1)

    C-3 (layer.py):
        层级中枢: 每层都有一个"中枢 expert" (全连接 + LayerNorm),
        接收所有 expert 输出后做元认知聚合, 再分发.
        (默认同时启用 C-1 + C-2)

可追溯性:
    CentralWorkspace                       → C-1 全局中枢
    CentralAugmentedFusionLayer            → C-2 attn→ffn 协同 (继承 v7 部件)
    HierarchicalCentralLayer              → C-3 层级中枢
    V8Trainer                              → 中枢 EMA + per-expert param group

相对 v7.0 的关键差异:
    - 新增 CentralWorkspace (c + U)
    - 路由 logits 受中枢广播影响
    - attn 通路输出影响 ffn 路由
    - 层级中枢 expert
    - V8Trainer.step() 含 c.ema_update()
"""
from .central import CentralWorkspace
from .fusion import CentralAugmentedFusionLayer
from .layer import HierarchicalCentralLayer
from .trainer import V8Trainer, build_v8_param_groups

__all__ = [
    "CentralWorkspace",
    "CentralAugmentedFusionLayer",
    "HierarchicalCentralLayer",
    "V8Trainer",
    "build_v8_param_groups",
]
__version__ = "8.0.0"
