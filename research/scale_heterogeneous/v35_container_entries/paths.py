"""research.scale_heterogeneous.v35_container_entries.paths — 路径解析 (已提升为共享原语)

背景:
    v35 原本自带一份 paths.py, 其中模型权重路径是占位符
    (入口 tabular-LM / embedding-LM / reranker-LM 与容器 large-LM backbone
     均为 proprietary, 作者不公开), 但本项目实际使用的就是
    research/_primitives/scale_hetero_loaders/paths.py 里那批已解析出真实位置的权重:
        MSA_DIR     -> /workspace/models/MSA-4B                        (即 large-LM 容器)
        EMB_DIR     -> /workspace/models/skillcorpus-embedding-0.6b    (即 embedding-LM)
        RER_DIR     -> /workspace/models/skillcorpus-reranker-0.6b     (即 reranker-LM)
        TABLDM_CKPT -> Xiaomi-TabLDM checkpoints                        (即 tabular-LM)

    故本文件改为从共享原语 re-export, 与 v33/v36/v37 的做法一致
    (见 research/_primitives/README.md 的"原始位置 -> 提升后位置"表)。

兼容性:
    1) 同时导出旧名 (BIG_LM_DIR / EMBED_LM_DIR / RERANK_LM_DIR / TAB_LM_CKPT),
       使本包的 entries.py 无需改动即可工作;
    2) ⚠ 环境变量名随之改变: V35_BIG_LM_DIR / V35_EMBED_LM_DIR /
       V35_RERANK_LM_DIR / V35_TAB_LM_CKPT  -> V36_MSA_DIR / V36_EMB_DIR /
       V36_RER_DIR / V36_TABLDM_CKPT。若需指定路径, 请改用 V36_* 变量名。
    3) `_v35_paths_summary()` 保留, 键名沿用旧名以兼容既有调用方。
"""
from research._primitives.scale_hetero_loaders.paths import (  # noqa: F401
    MSA_DIR,
    EMB_DIR,
    RER_DIR,
    TABLDM_CKPT,
    TABLDM_SRC,
    first_existing,
)

# --- 旧名别名 (供本包 entries.py 使用, 亦兼容外部调用方) ---
BIG_LM_DIR = MSA_DIR
EMBED_LM_DIR = EMB_DIR
RERANK_LM_DIR = RER_DIR
TAB_LM_CKPT = TABLDM_CKPT

__all__ = [
    "MSA_DIR", "EMB_DIR", "RER_DIR", "TABLDM_CKPT", "TABLDM_SRC",
    "BIG_LM_DIR", "EMBED_LM_DIR", "RERANK_LM_DIR", "TAB_LM_CKPT",
    "first_existing",
]


def _v35_paths_summary():
    """保留旧键名, 兼容既有调用方。"""
    return {
        "BIG_LM_DIR": MSA_DIR,
        "EMBED_LM_DIR": EMB_DIR,
        "RERANK_LM_DIR": RER_DIR,
        "TAB_LM_CKPT": TABLDM_CKPT,
    }
