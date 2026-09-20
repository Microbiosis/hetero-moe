"""v29.0 — 数据准备层 (委托 v26 prepare_v26_real_corpus_data + 大语料).

设计:
    - text: LargeTextCorpus.train_sentences → 64 句 (4× v18)
    - code: LargeCodeCorpus.train_snippets → 32 片段 (2× v26)
    - image: LargeImageCorpus.train_themes → 50 主题 (3× v26)
    - STEPS: 端到端脚本中默认 500 (5× v18, 来自 v19 H3 发现)

    接口对齐 v26:
        - prepare_v29_real_corpus_data (text + code + image 真实语料)
        - prepare_v29_held_out_data (held-out 真实语料)

    委托 v25 prepare_real_corpus_data 处理 text (Wikipedia 真实), v29 仅替换 code/image 数据源.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch

# 委托 v18 (text) + v29 自有 (code + image) + v25 (集成模式参考)
from research.corpus_scale.v25_corpus_central.layer import (
    RealCorpusTheoryLayer,
    build_v25_param_groups,
    prepare_real_corpus_data as _v25_prepare_real_corpus_data,
    prepare_held_out_data as _v25_prepare_held_out_data,
)

from .text_corpus import LargeTextCorpus
from .code_corpus import LargeCodeCorpus
from .image_corpus import LargeImageCorpus


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


def prepare_v29_real_corpus_data(
    seed: int,
    encoders,
    text_corpus: LargeTextCorpus,
    code_corpus: LargeCodeCorpus,
    image_corpus: LargeImageCorpus,
    n_per_class: int = 6,
    S: int = 16,
    D_SHARED: int = 256,
    N_CLS: int = 3,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    """v29 大语料数据准备: text + code + image 全部用真实语料 (规模 v26 的 2-4×).

    与 v26 prepare_v26_real_corpus_data 接口对齐, 让 v29 端到端脚本可无缝接入.
    """
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # ---- text 模态: 委托 v25 (Wikipedia 真实文本) ----
    modal_seqs_text, modal_indices_text, targets_text = _v25_prepare_real_corpus_data(
        seed=seed, encoders=encoders, corpus=text_corpus,
        n_per_class=n_per_class, S=S, D_SHARED=D_SHARED, N_CLS=N_CLS,
    )
    h_text_indices = [i for i, m in enumerate(modal_indices_text) if m == 0]
    h_text_list = [modal_seqs_text[i] for i in h_text_indices]
    t_text_list = [targets_text[i] for i in h_text_indices]
    h_text = torch.stack(h_text_list, dim=0)

    # ---- code 模态: v29 LargeCodeCorpus (32 真实 Python 片段) ----
    g_code = torch.Generator().manual_seed(seed + 1000)
    n_train_code = len(code_corpus.train_snippets)
    idx_code = torch.randint(0, n_train_code, (n_per_class,), generator=g_code).tolist()
    code_strings = [code_corpus.train_snippets[i] for i in idx_code]
    h_code = _encode_code_from_strings(code_strings, llama_tok, llama)

    # ---- image 模态: v29 LargeImageCorpus (50 Wikipedia 主题) ----
    g_img = np.random.default_rng(seed + 2000)
    n_train_img = len(image_corpus.train_themes)
    idx_img = g_img.integers(0, n_train_img, size=n_per_class).tolist()
    h_img_list = []
    for theme in [image_corpus.train_themes[i] for i in idx_img]:
        img = image_corpus.train_themes  # placeholder, real gen below
        from .image_corpus import generate_wiki_image
        img = generate_wiki_image(theme, image_size=image_corpus.image_size)
        h = _encode_image_from_array(img, vit_proc, vit)
        h_img_list.append(h.squeeze(0))
    h_img = torch.stack(h_img_list, dim=0)

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


def prepare_v29_held_out_data(
    encoders,
    text_corpus: LargeTextCorpus,
    code_corpus: LargeCodeCorpus,
    image_corpus: LargeImageCorpus,
    S: int = 16,
    D_SHARED: int = 256,
    N_CLS: int = 3,
    n_per_class: int = 6,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    """v29 held-out 数据: text/code/image 全部用真实 held-out 语料."""
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # ---- text held-out: 委托 v25 ----
    modal_seqs_text, modal_indices_text, targets_text = _v25_prepare_held_out_data(
        encoders=encoders, corpus=text_corpus,
        S=S, D_SHARED=D_SHARED, N_CLS=N_CLS, n_per_class=n_per_class,
    )
    h_text_indices = [i for i, m in enumerate(modal_indices_text) if m == 0]
    h_text_list = [modal_seqs_text[i] for i in h_text_indices]
    h_text = torch.stack(h_text_list, dim=0)

    # ---- code held-out: 全部 held-out 片段 (12 个) ----
    h_code_list = []
    for s in code_corpus.held_out:
        h = _encode_code_from_strings([s], llama_tok, llama).squeeze(0)
        h_code_list.append(h)
    h_code = torch.stack(h_code_list, dim=0)

    # ---- image held-out: 全部 held-out 主题 (16 个) ----
    from .image_corpus import generate_wiki_image
    h_img_list = []
    for theme in image_corpus.held_out:
        img = generate_wiki_image(theme, image_size=image_corpus.image_size)
        h = _encode_image_from_array(img, vit_proc, vit).squeeze(0)
        h_img_list.append(h)
    h_img = torch.stack(h_img_list, dim=0)

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
