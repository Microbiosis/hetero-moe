"""research._primitives.scale_hetero_loaders.paths — 统一路径解析

合并自:
    v33_scale_hetero/paths.py     (MSA_DIR, EMB_DIR, RER_DIR)
    v36_alpha2_knn/paths.py       (MSA_DIR, TABLDM_CKPT)
    v37_b_selftrained_entry/paths.py (MSA_DIR, TABLDM_CKPT, 副本)

注意: 本项目实际使用 EverMind-AI 的 skillcorpus-embedding-0.6b / skillcorpus-reranker-0.6b
      (不是 BGE 系列). EMB_DIR/RER_DIR 已对齐到实际下载位置.

所有路径解析:
    - MSA_DIR         → /workspace/models/MSA-4B          (EverMind-AI/MSA-4B)
    - EMB_DIR         → /workspace/models/skillcorpus-embedding-0.6b
    - RER_DIR         → /workspace/models/skillcorpus-reranker-0.6b
    - TABLDM_CKPT     → /workspace/models/Xiaomi-TabLDM/checkpoints/clf_default.ckpt
"""
import os
import sys


def first_existing(candidates, env_key):
    """环境变量 > 现存候选 > 首选候选 (兼容模型目录迁移)。"""
    cands = ([os.environ[env_key]] if os.environ.get(env_key) else []) + candidates
    for c in cands:
        if c and os.path.exists(c):
            return c
    return cands[0]


MSA_DIR = first_existing(
    ["/workspace/models/MSA-4B", "/workspace/MSA-4B"], "V36_MSA_DIR")
EMB_DIR = first_existing(
    ["/workspace/models/skillcorpus-embedding-0.6b",
     "/workspace/.cache/huggingface/embedding"], "V36_EMB_DIR")
RER_DIR = first_existing(
    ["/workspace/models/skillcorpus-reranker-0.6b",
     "/workspace/.cache/huggingface/reranker"], "V36_RER_DIR")
TABLDM_CKPT = first_existing(
    ["/workspace/models/Xiaomi-TabLDM/checkpoints/clf_default.ckpt",
     "/root/.cache/huggingface/hub/models--occams--Xiaomi-TabLDM/snapshots/0e10427273bc862fb60fe7df84c89e5298e16cc7/checkpoints/clf_default.ckpt"],
    "V36_TABLDM_CKPT")

# TabLDM 源码目录 (若已 pip install tabldm 则无需此路径)
TABLDM_SRC = first_existing(
    ["/workspace/models/Xiaomi-TabLDM"], "V36_TABLDM_SRC")
if not os.path.exists(TABLDM_SRC):
    # 在 PYTHONPATH 里临时挂载源码目录
    sys.path.insert(0, TABLDM_SRC)