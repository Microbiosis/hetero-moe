"""v10.0 — 语义对齐器 (Semantic Aligner) 路线。

设计动机 (用户问题: "能不能使用一个巧妙的工具将不同的底座联系为一个新的底座?"):
    v6/v7/v8/v9 的 P_m (矩形投影) 是线性映射, 跨架构的非线性差异被压平.
    v10.0 提出语义对齐器 (SemanticAligner): 跨底座共享 K/V 的注意力,
    让异构底座的"相同概念"在共享空间自然对齐.

    这是与 v9 平行的**新路线**:
        v8 → 探索中枢层级
        v9 → 修正单独中枢 token
        v10 → 探索更好的跨架构对齐 (而非融合层设计)

可追溯性:
    SemanticAligner                      → T_mean / T_attn 两种实现
    AlignedFusionLayer                   → 替换 P_m 为 aligner, 复用 v8/v9 中枢
    v10 的产物仍是"协调层", 不是单模型
        (蒸馏到单模型是 v10.1 / v11 目标)

相对 v9.0 的关键差异:
    - P_m (矩形线性投影) → SemanticAligner (跨架构注意力)
    - 共享 K/V 空间让异构底座"概念对齐"
    - 中枢 c 接收对齐后的统一表示 (而非各底座独立投影)
"""
from .aligner import SemanticAligner, CrossArchMeanAligner, CrossArchAttnAligner
from .fusion import AlignedFusionLayer

__all__ = [
    "SemanticAligner",
    "CrossArchMeanAligner",
    "CrossArchAttnAligner",
    "AlignedFusionLayer",
]
__version__ = "10.0.0"