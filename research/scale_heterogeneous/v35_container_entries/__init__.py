"""v35.0-α — 大模型容器 + 异质入口注入 (Container + Heterogeneous Entries)。

可追溯性:
    MSA_DIR/EMB_DIR/RER_DIR/TABLDM_CKPT → 路径解析 (paths)
    get_tabular_repr                      → TabLDM 表格入口 (hook 取 representations, D=512) (entries)
    load_backbone                        → embedding/reranker 文本入口 (D=1024) (entries)
    load_big_container / big_forward     → MSA-4B 浅层容器 + prefix prior 注入 (entries)
    ProjHead / make_P                    → 注入头与正交投影 (inject)

核心命题: 大模型作为容器(主干), 不同类型小模型作为"原生入口", 产出表征经投影后
作为 prior 注入大模型容器内部 (inputs_embeds), 使融合优于随机 prior baseline。

结果性质: **负结果 (-19.5%)** — 简单任务 + 强容器下, 入口注入是扰动非增量。
边界见 docs/V35_0_SPEC.md。与 v33/v34 (多专家路由融合) 是**不同的融合范式**。

依赖: 需 tabldm (Xiaomi-TabLDM) 可导入; 支持 V35_TABLDM_SRC 指定源码目录。
"""
from .paths import (
    MSA_DIR, EMB_DIR, RER_DIR, TABLDM_CKPT, first_existing,
)
from .entries import (
    get_tabular_repr,
    load_backbone,
    load_big_container,
    big_forward,
)
from .inject import ProjHead, make_P

__all__ = [
    "MSA_DIR",
    "EMB_DIR",
    "RER_DIR",
    "TABLDM_CKPT",
    "first_existing",
    "get_tabular_repr",
    "load_backbone",
    "load_big_container",
    "big_forward",
    "ProjHead",
    "make_P",
]
__version__ = "35.0.0"
