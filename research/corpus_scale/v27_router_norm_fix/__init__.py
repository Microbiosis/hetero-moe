"""v27.0 — router-norm / adaptive-ema held-out 修正补丁 (第三个跨版本集成包).

设计:
    v26 在 fully-real-corpus 上发现 router-norm 和 adaptive-ema 在 held-out 上退步:
        - training: gate > adaptive-ema > router-norm > broadcast
        - held-out: gate > broadcast > router-norm ≈ adaptive-ema (退步!)

    v22 现有 patch (freeze_u / warmup / set_u_init_scale / clip_grad) 无法解决.
    v27 提出 4 个新 patch 分别从梯度和 EMA 两个通道入手:

        Patch A: apply_l2_to_c     - c 加 L2 正则 (weight decay)
        Patch B: router_dropout    - router logits 加 Dropout
        Patch C: extend_warmup_steps - AdaptiveEMA warmup 10 → 100
        Patch D: noise_inject_c    - forward 时 c 加 Gaussian 噪声

可追溯性:
    RegularizedTheoryLayer    -> 继承 v25 RealCorpusTheoryLayer, 应用 4 个补丁
    compute_total_loss        -> 自动加 L2 正则到 MSE
    apply_l2_to_c / router_dropout / extend_warmup_steps / noise_inject_c
                              -> 4 个独立补丁函数
    DEFAULT_PATCH_CONFIGS     -> 5 个标准 patch 配置 (none / l2 / dropout / warmup / noise)
    PATCH_NAMES               -> 端到端脚本遍历用

验证环境: 复用 v26 fully-real-corpus (text/code/image 全部真实)
端到端规模: 5 patch × 4 mode × 5 seeds = 80 run
"""
from .patches import (
    apply_l2_to_c,
    get_l2_to_c_loss,
    router_dropout,
    extend_warmup_steps,
    noise_inject_c,
    apply_all_patches,
    DEFAULT_PATCH_CONFIGS,
    PATCH_NAMES,
)
from .layer import RegularizedTheoryLayer, compute_total_loss

__all__ = [
    # 4 个补丁
    "apply_l2_to_c",
    "get_l2_to_c_loss",
    "router_dropout",
    "extend_warmup_steps",
    "noise_inject_c",
    "apply_all_patches",
    "DEFAULT_PATCH_CONFIGS",
    "PATCH_NAMES",
    # 集成层
    "RegularizedTheoryLayer",
    "compute_total_loss",
]
__version__ = "27.0.0"
