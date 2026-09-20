"""v18.0 — Real Corpus (MiniCorpus) 实验的 thin re-export shim.

v18 的 MiniCorpus + cycle_batches 已全部提升为共享原语
   (research._primitives.real_corpus). 本包保留作为 thin re-export, 保持历史 import 兼容.

API (推荐 — 直接从 _primitives):
    from research._primitives.real_corpus import MiniCorpus, cycle_batches
"""
from research._primitives.real_corpus import MiniCorpus, cycle_batches

__all__ = ["MiniCorpus", "cycle_batches"]
__version__ = "18.0.0"
__lifted_to__ = "research._primitives.real_corpus"
__is_re_export_shim__ = True
