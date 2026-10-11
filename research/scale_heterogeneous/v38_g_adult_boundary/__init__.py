"""v38.0 — γ 真实数据跨模态验证 (Adult XOR + 5 臂)。

可追溯性:
    data.adult_loader  → UCI Adult 数据集 (n≈48k, 6 数值列 + 8 类别列)
    load_adult_crossmodal(n, seed, decouple=True) → XOR 标签:
        y = 1 iff (职业 ∈ 高薪集 [只在文本侧]) XOR (工时 ≥ 45 [只在表格侧])
        → 单侧信息近零, 跨模态必需.
    load_big_container    → 复用 v36 的 MSA-4B 浅层加载
    tabldm_passive_repr   → α2 对照: TabLDM 被动中间量 (逐样本)
    EntryEncoder / knn_acc → 复用 v37 的 β 自训练组件 (E 可训练)

五臂 (相对 v37 三臂扩到五臂):
    A 纯文本(无入口)  h = h_base                    (训头)
    B 随机入口        c = randn 冻结                (训 P+头)
    C 被动TabLDM(α2)  c = TabLDM 中间表征 冻结      (训 P+头)  ← v36 对照
    D 自训练(β)      c = E(x_table) 可训练          (训 E+P+头) ← v37 对照
    E 先验+自训练    c = E 蒸馏初始化到 TabLDM → 任务微调 (训 E+P+头) ← γ 核心

γ 价值: 在 Adult XOR 上 C/D/E 三者收敛到同一平台 0.755-0.767 (差异被噪声淹没).
        结论 = "在真实复杂数据上, 自训练/被动/先验+自训练三者打平".
        这是 v37 结论的**边界** (β 适用于简单可学任务, 真实复杂任务不成立).
"""
from .paths import MSA_DIR, TABLDM_CKPT, first_existing
from .data import adult_loader
from research._primitives.scale_hetero_loaders import load_big_container
from research._primitives.scale_hetero_loaders import (
    EntryEncoder,
    knn_acc,
)

__all__ = [
    "MSA_DIR",
    "TABLDM_CKPT",
    "first_existing",
    "adult_loader",
    "load_big_container",
    "EntryEncoder",
    "knn_acc",
]
__version__ = "38.0.0"