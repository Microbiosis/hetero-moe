"""v25.0 — v22 CentralTheory + v18 MiniCorpus 跨版本集成 (首个跨版本包).

设计:
    - 复用 v22_central_theory.CentralTheory (不重写中枢逻辑)
    - 复用 v18_real_corpus.MiniCorpus (不重写语料)
    - RealCorpusTheoryLayer 继承 v22 TheoryLayer 的全部结构与稳定性 patch
    - 仅替换数据准备函数 (text 模态用真实 Wikipedia 句子)

公平性声明:
    - 仅 text 模态用 MiniCorpus.train_sentences (真实 Wikipedia, 16 句)
    - code/image 模态保留 v22 合成扰动 (无真实语料库)
    - 评估: 训练集 + held-out 双评估 (仿 v18)

可追溯性:
    RealCorpusTheoryLayer     -> 委托 v22 TheoryLayer, text 模态用 MiniCorpus
    build_v25_param_groups    -> 与 v22 build_theory_param_groups 完全一致
    prepare_real_corpus_data  -> 真实 Wikipedia 句子 + 合成 code/image
    prepare_held_out_data     -> corpus.held_out 6 句 (真实泛化评估)
"""
from .layer import (
    RealCorpusTheoryLayer,
    build_v25_param_groups,
    prepare_real_corpus_data,
    prepare_held_out_data,
)

__all__ = [
    "RealCorpusTheoryLayer",
    "build_v25_param_groups",
    "prepare_real_corpus_data",
    "prepare_held_out_data",
]
__version__ = "25.0.0"
