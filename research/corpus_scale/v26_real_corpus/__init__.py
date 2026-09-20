"""v26.0 — fully-real-corpus (Python 片段池 + Wikipedia 图像) 跨版本集成.

设计:
    - 复用 v18 MiniCorpus (text) + v22 CentralTheory + v25 RealCorpusTheoryLayer
    - 新建 MiniCodeCorpus (16 真实 Python 片段 + 6 held-out)
    - 新建 MiniImageCorpus (16 Wikipedia 主题结构图 + 6 held-out)
    - 集成层 layer.py 委托 v25 RealCorpusTheoryLayer, 仅替换 code/image 数据源

公平性声明:
    - text 模态: MiniCorpus.train_sentences (真实 Wikipedia, 16 句, 来自 v18)
    - code 模态: MiniCodeCorpus.train_snippets (真实 Python 片段, 16 个)
    - image 模态: MiniImageCorpus.train_themes (Wikipedia 主题结构图, 16 个)
    - **3 个模态全部用真实语料 (fully-real)**

可追溯性:
    MiniCodeCorpus        -> v18 MiniCorpus 接口克隆 (Python 片段 tokenize)
    MiniImageCorpus       -> v18 MiniCorpus 接口克隆 (numpy 主题图)
    generate_wiki_image   -> 单主题 → numpy 224x224x3 uint8 图像
    prepare_v26_real_corpus_data  -> 集成 v18 text + v26 code + v26 image
    prepare_v26_held_out_data     -> 集成 held-out 三个模态
"""
from .code_corpus import (
    MiniCodeCorpus,
    cycle_batches as cycle_code_batches,
)
from .image_corpus import (
    MiniImageCorpus,
    generate_wiki_image,
    WIKI_COLORS,
    cycle_batches as cycle_image_batches,
)
from .layer import (
    prepare_v26_real_corpus_data,
    prepare_v26_held_out_data,
)

__all__ = [
    # code corpus
    "MiniCodeCorpus",
    "cycle_code_batches",
    # image corpus
    "MiniImageCorpus",
    "generate_wiki_image",
    "WIKI_COLORS",
    "cycle_image_batches",
    # integration
    "prepare_v26_real_corpus_data",
    "prepare_v26_held_out_data",
]
__version__ = "26.0.0"
