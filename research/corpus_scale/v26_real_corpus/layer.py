"""v26.0 — fully-real-corpus 集成层 (委托 v25 + 替换 code/image 数据源).

设计:
    - text 模态: 委托 v25 prepare_real_corpus_data (Wikipedia 真实文本)
    - code 模态: 用 v26 MiniCodeCorpus (真实 Python 片段)
    - image 模态: 用 v26 MiniImageCorpus (Wikipedia 主题结构图)

    集成层只暴露 prepare_v26_real_corpus_data 和 prepare_v26_held_out_data,
    与 v25 prepare_* 接口对齐, 让 v26 端到端脚本可无缝接入.

不重写 v25 layer.py:
    - v25 RealCorpusTheoryLayer 结构保持不变
    - 仅在数据准备侧替换 code/image 数据源
    - text 模态保持 v18 MiniCorpus (v25 已有)
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch

# 委托 v18 (text) + v26 自有 (code + image) + v25 (集成模式参考)
from research._primitives.real_corpus import MiniCorpus
from research.corpus_scale.v25_corpus_central.layer import (
    RealCorpusTheoryLayer,
    build_v25_param_groups,
    prepare_real_corpus_data as _v25_prepare_real_corpus_data,
    prepare_held_out_data as _v25_prepare_held_out_data,
)

from .code_corpus import MiniCodeCorpus
from .image_corpus import MiniImageCorpus


def _encode_code_from_strings(code_strings, llama_tok, llama):
    """用 llama tokenizer + llama model 把 code strings → hidden states."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from examples.run_v8_full import encode_code
    return encode_code(code_strings, llama_tok, llama)


def _encode_image_from_array(img_array, vit_proc, vit):
    """用 ViT processor + ViT model 把 HxWx3 uint8 numpy → hidden states."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from examples.run_v8_full import encode_image
    return encode_image(img_array, vit_proc, vit)


def prepare_v26_real_corpus_data(
    seed: int,
    encoders,
    text_corpus: MiniCorpus,
    code_corpus: MiniCodeCorpus,
    image_corpus: MiniImageCorpus,
    n_per_class: int = 6,
    S: int = 16,
    D_SHARED: int = 256,
    N_CLS: int = 3,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    """v26 fully-real 数据准备: text + code + image 全部用真实语料.

    Args:
        seed:        训练 seed (用于 text/wiki 采样)
        encoders:    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _)
        text_corpus: v18 MiniCorpus (Wikipedia 真实文本)
        code_corpus: v26 MiniCodeCorpus (Python 片段)
        image_corpus: v26 MiniImageCorpus (Wikipedia 主题图)
        n_per_class: 每类样本数 (默认 6, 与 v25 一致)
        S, D_SHARED, N_CLS: 与 v25 prepare_real_corpus_data 一致

    Returns:
        (modal_seqs, modal_indices, targets)
    """
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # ---- text 模态: 委托 v25 (Wikipedia 真实文本) ----
    modal_seqs_text, modal_indices_text, targets_text = _v25_prepare_real_corpus_data(
        seed=seed, encoders=encoders, corpus=text_corpus,
        n_per_class=n_per_class, S=S, D_SHARED=D_SHARED, N_CLS=N_CLS,
    )
    # 取 text 部分 (cls=0 的 6 个样本)
    h_text_indices = [i for i, m in enumerate(modal_indices_text) if m == 0]
    h_text_list = [modal_seqs_text[i] for i in h_text_indices]
    t_text_list = [targets_text[i] for i in h_text_indices]
    h_text = torch.stack(h_text_list, dim=0)  # [6, S, D_bert]

    # ---- code 模态: v26 MiniCodeCorpus (真实 Python 片段) ----
    # 用 seed 控制采样, 让不同 seed 看到不同 Python 片段
    g_code = torch.Generator().manual_seed(seed + 1000)
    n_train_code = len(code_corpus.train_snippets)
    idx_code = torch.randint(0, n_train_code, (n_per_class,), generator=g_code).tolist()
    code_strings = [code_corpus.train_snippets[i] for i in idx_code]
    h_code = _encode_code_from_strings(code_strings, llama_tok, llama)  # [6, S, D_llama]

    # ---- image 模态: v26 MiniImageCorpus (Wikipedia 主题结构图) ----
    g_img = np.random.default_rng(seed + 2000)
    n_train_img = len(image_corpus.train_themes)
    idx_img = g_img.integers(0, n_train_img, size=n_per_class).tolist()
    themes = [image_corpus.train_themes[i] for i in idx_img]
    h_img_list = []
    for theme in themes:
        img_array = image_corpus.train_themes  # placeholder, real gen below
        from .image_corpus import generate_wiki_image
        img = generate_wiki_image(theme, image_size=image_corpus.image_size)
        h = _encode_image_from_array(img, vit_proc, vit)  # [1, S, D_vit]
        h_img_list.append(h.squeeze(0))
    h_img = torch.stack(h_img_list, dim=0)  # [6, S, D_vit]

    # ---- 拼接为 v25 格式 (按模态分组, 每模态 6 个) ----
    modal_seqs = []
    modal_indices = []
    targets = []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(n_per_class):
            src = h_block[i % n]
            g2 = torch.Generator().manual_seed(seed * 10 + cls * n_per_class + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g2))
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(n_per_class):
            targets.append(t)

    return modal_seqs, modal_indices, targets


def prepare_v26_held_out_data(
    encoders,
    text_corpus: MiniCorpus,
    code_corpus: MiniCodeCorpus,
    image_corpus: MiniImageCorpus,
    S: int = 16,
    D_SHARED: int = 256,
    N_CLS: int = 3,
    n_per_class: int = 6,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    """v26 held-out 数据: text/code/image 全部用真实 held-out 语料.

    与 v25 prepare_held_out_data 区别: code/image 不再合成, 用真实 held-out.
    """
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # ---- text held-out: 委托 v25 ----
    modal_seqs_text, modal_indices_text, targets_text = _v25_prepare_held_out_data(
        encoders=encoders, corpus=text_corpus,
        S=S, D_SHARED=D_SHARED, N_CLS=N_CLS, n_per_class=n_per_class,
    )
    h_text_indices = [i for i, m in enumerate(modal_indices_text) if m == 0]
    h_text_list = [modal_seqs_text[i] for i in h_text_indices]
    h_text = torch.stack(h_text_list, dim=0)  # [6, S, D_bert]

    # ---- code held-out: 全部 held-out 片段 (6 个) ----
    h_code_list = []
    for s in code_corpus.held_out:
        h = _encode_code_from_strings([s], llama_tok, llama).squeeze(0)  # [S, D_llama]
        h_code_list.append(h)
    h_code = torch.stack(h_code_list, dim=0)  # [6, S, D_llama]

    # ---- image held-out: 全部 held-out 主题图 (6 个) ----
    from .image_corpus import generate_wiki_image
    h_img_list = []
    for theme in image_corpus.held_out:
        img = generate_wiki_image(theme, image_size=image_corpus.image_size)
        h = _encode_image_from_array(img, vit_proc, vit).squeeze(0)  # [S, D_vit]
        h_img_list.append(h)
    h_img = torch.stack(h_img_list, dim=0)  # [6, S, D_vit]

    # ---- 拼接为 v25 格式 ----
    modal_seqs = []
    modal_indices = []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(n_per_class):
            src = h_block[i % n]
            modal_seqs.append(src)
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(n_per_class):
            targets.append(t)
    return modal_seqs, modal_indices, targets
