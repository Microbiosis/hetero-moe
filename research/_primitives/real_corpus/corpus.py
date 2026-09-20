"""v18.0 — 真实语料 (内置 8-16 句 Wikipedia 风格文本).

设计:
    不依赖网络下载, 内置 16 句英文 Wikipedia 风格句子供训练.
    用真实 tokenizer.encode() 得到真实 input_ids.
    评估时用全新句子 (held-out) 真实泛化.
"""
from __future__ import annotations

from typing import List, Iterator, Tuple

import torch


# 内置 16 句 Wikipedia 风格英文文本 (每句 8-20 个词, 多领域)
_MINI_CORPUS = [
    "The cat sat on the mat near the window.",
    "Albert Einstein developed the theory of relativity.",
    "Mount Everest is the highest mountain on Earth.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Eiffel Tower was built in eighteen eighty nine.",
    "Shakespeare wrote thirty seven plays during his career.",
    "DNA contains the genetic instructions of every organism.",
    "The Pacific Ocean covers more area than all land masses.",
    "Mathematics is the language in which the universe is written.",
    "The Roman Empire lasted for over a thousand years.",
    "Mozart composed his first symphony at the age of eight.",
    "Black holes are formed when massive stars collapse.",
    "The Amazon rainforest produces twenty percent of Earth oxygen.",
    "Quantum mechanics describes nature at the smallest scales.",
    "The Great Wall of China stretches over twenty thousand kilometers.",
    "Antibiotics revolutionized medicine in the twentieth century.",
]

# 6 句 held-out (评估时不参与训练)
_HELD_OUT = [
    "The human brain contains billions of neurons.",
    "Climate change affects ecosystems worldwide.",
    "Vaccines train the immune system against pathogens.",
    "Volcanoes form at tectonic plate boundaries.",
    "The speed of light is constant in vacuum.",
    "renewable energy sources include solar and wind power.",
]


class MiniCorpus:
    """真实语料 (内置 16 句训练 + 6 句 held-out)."""

    def __init__(self, tokenizer, max_length: int = 16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_sentences = _MINI_CORPUS
        self.held_out = _HELD_OUT

    def train_batch(self, batch_size: int = 4) -> torch.Tensor:
        """随机采样 batch_size 句, tokenize 到 [batch_size, max_length]."""
        idx = torch.randint(0, len(self.train_sentences), (batch_size,)).tolist()
        sentences = [self.train_sentences[i] for i in idx]
        enc = self.tokenizer(
            sentences, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return enc["input_ids"]

    def eval_batch(self, batch_size: int = 6) -> torch.Tensor:
        """返回 held-out 句子 tokenize."""
        enc = self.tokenizer(
            self.held_out[:batch_size], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return enc["input_ids"]

    def all_eval(self) -> torch.Tensor:
        """返回所有 held-out 句子的 tokenize."""
        return self.eval_batch(len(self.held_out))


def cycle_batches(corpus: MiniCorpus, batch_size: int = 4) -> Iterator[torch.Tensor]:
    """无限循环训练 batch."""
    while True:
        yield corpus.train_batch(batch_size)