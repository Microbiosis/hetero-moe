"""v29.0 — 大型真实语料 (text/code/image 2-4× v26) 跨版本集成包.

设计:
    v18 MiniCorpus (16 text) + v26 MiniCodeCorpus (16 code) + v26 MiniImageCorpus (16 image)
    是 v22/v27/v28 端到端的基线.
    v27 + v28 共同证明 router-norm/adaptive-ema 在 16+6 小语料下 held-out 退步是数据规模问题.

    v29 扩展到 64 text + 32 code + 50 image, 重跑 v26/v27/v28 验证:
        - router-norm/adaptive-ema 在大语料下是否仍退步?
        - v27 patch 和 v28 架构变体在大语料下是否有效?
        - gate 是否仍是最优?

可追溯性:
    LargeTextCorpus   -> v18 MiniCorpus 接口克隆 (64 train + 24 held-out, 4×)
    LargeCodeCorpus   -> v26 MiniCodeCorpus 接口克隆 (32 train + 12 held-out, 2×)
    LargeImageCorpus  -> v26 MiniImageCorpus 接口克隆 (50 train + 16 held-out, 3×)
                          + 2 种新形状 (hexagon / star), 共 6 种
    prepare_v29_real_corpus_data / prepare_v29_held_out_data
                        -> 集成层, 对齐 v26 接口

验证环境: v26 fully-real 框架 + STEPS 100 → 500 (v19 H3 发现)
端到端规模: 3 个脚本, 共 ~175 run (v26 大语料 20 + v27 大语料 100 + v28 大语料 55)
"""
from .text_corpus import (
    LargeTextCorpus,
    cycle_batches as cycle_text_batches,
)
from .code_corpus import (
    LargeCodeCorpus,
    cycle_batches as cycle_code_batches,
)
from .image_corpus import (
    LargeImageCorpus,
    generate_wiki_image,
    WIKI_COLORS_LARGE,
    cycle_batches as cycle_image_batches,
)
from .layer import (
    prepare_v29_real_corpus_data,
    prepare_v29_held_out_data,
)

__all__ = [
    # 3 个大语料
    "LargeTextCorpus",
    "cycle_text_batches",
    "LargeCodeCorpus",
    "cycle_code_batches",
    "LargeImageCorpus",
    "generate_wiki_image",
    "WIKI_COLORS_LARGE",
    "cycle_image_batches",
    # 集成层
    "prepare_v29_real_corpus_data",
    "prepare_v29_held_out_data",
]
__version__ = "29.0.0"
