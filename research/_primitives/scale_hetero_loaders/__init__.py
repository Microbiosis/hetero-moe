"""research._primitives.scale_hetero_loaders — 共享原语: 规模异构 + 容器范式加载器

合并自:
    research/scale_heterogeneous/v33_scale_hetero/paths.py
    research/scale_heterogeneous/v33_scale_hetero/encoders.py
    research/scale_heterogeneous/v36_alpha2_knn/paths.py
    research/scale_heterogeneous/v36_alpha2_knn/entries.py
    research/scale_heterogeneous/v36_alpha2_knn/eval.py
    research/scale_heterogeneous/v37_b_selftrained_entry/entry_encoder.py
    research/scale_heterogeneous/v37_b_selftrained_entry/arm.py

v33 的底座加载器 (MSA/EMB/RER)、v36 的大模型容器加载、TABLDM 入口、
   v37 的自训练 EntryEncoder 全部被 scale_heterogeneous 整条线复用,
   提升为 _primitives 后, v34-v39 都不再依赖 v33/v36/v37 的具体实现.

依赖 (运行期):
    torch, safetensors, transformers (Qwen3*)

API:
    paths:        MSA_DIR, EMB_DIR, RER_DIR, TABLDM_CKPT, first_existing
    encoders:     load_msa_big, load_small, load_tokenizer, encode
    entries:      load_big_container, big_forward, get_tabldm_repr,
                  encode_baseline_text, D_BIG, B_DEFAULT, S_DEFAULT
    entry_encoder: EntryEncoder, gen_table_classification,
                   container_text_repr, tabldm_passive_repr
    eval:         knn_acc, concat_norm_then_knn
    arm:          train_arm
"""
from .paths import (
    MSA_DIR, EMB_DIR, RER_DIR, TABLDM_CKPT, first_existing,
)
from .encoders import (
    load_msa_big, load_small, load_tokenizer, encode,
)
from .entries import (
    load_big_container, big_forward, get_tabldm_repr, encode_baseline_text,
    D_BIG, B_DEFAULT, S_DEFAULT,
)
from .entry_encoder import (
    EntryEncoder, gen_table_classification,
    container_text_repr, tabldm_passive_repr,
)
from .eval import knn_acc, concat_norm_then_knn
from .arm import train_arm

__all__ = [
    # paths
    "MSA_DIR", "EMB_DIR", "RER_DIR", "TABLDM_CKPT", "first_existing",
    # encoders (v33)
    "load_msa_big", "load_small", "load_tokenizer", "encode",
    # entries (v36)
    "load_big_container", "big_forward", "get_tabldm_repr", "encode_baseline_text",
    "D_BIG", "B_DEFAULT", "S_DEFAULT",
    # entry_encoder (v37)
    "EntryEncoder", "gen_table_classification",
    "container_text_repr", "tabldm_passive_repr",
    # eval (v36)
    "knn_acc", "concat_norm_then_knn",
    # arm (v37)
    "train_arm",
]
__research_line__ = "_primitives/scale_hetero_loaders"
__lifted_from__ = "research/scale_heterogeneous/{v33_scale_hetero, v36_alpha2_knn, v37_b_selftrained_entry}"
__is_primitive__ = True