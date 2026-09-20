"""research._primitives.real_corpus — 共享原语: 真实 Wikipedia 语料

来源: research/distillation/v18_real_corpus/corpus.py
原 v18_real_corpus 的 MiniCorpus 被 2 个研究包复用 (v25/v26),
   提升为 _primitives 后, 这些研究包不再依赖 v18_real_corpus.

API:
    MiniCorpus    — 真实 Wikipedia 句子语料 (16 训练 + 6 held-out)
    cycle_batches — batch 循环生成器
"""
from .corpus import MiniCorpus, cycle_batches

__all__ = ["MiniCorpus", "cycle_batches"]
__research_line__ = "_primitives/real_corpus"
__lifted_from__ = "research/distillation/v18_real_corpus"
__is_primitive__ = True

