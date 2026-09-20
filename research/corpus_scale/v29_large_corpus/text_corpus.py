"""v29.0 — 大型 Wikipedia 文本语料 (64 train + 24 held-out, 4× v18).

设计:
    v18 MiniCorpus 只有 16 train + 6 held-out (小规模, 可能导致 router-norm/adaptive-ema 退步).
    v29 LargeTextCorpus 扩展到 64 train + 24 held-out (4×), 验证数据规模修正假设.

    接口对齐 v18 MiniCorpus:
        - train_sentences: List[str]
        - held_out: List[str]
        - train_batch(batch_size): tokenize → [B, max_length]
        - eval_batch(batch_size): held-out tokenize
        - all_eval(): all held-out

    内容覆盖 8 个领域:
        - science (科学)
        - history (历史)
        - literature (文学)
        - technology (技术)
        - geography (地理)
        - arts (艺术)
        - philosophy (哲学)
        - biology (生物)
"""
from __future__ import annotations

from typing import Iterator, List

import torch


# 64 句 Wikipedia 风格英文文本 (8 领域 × 8 句, 每句 8-20 词)
_LARGE_TEXT_CORPUS: List[str] = [
    # === Science (8 句) ===
    "Albert Einstein developed the theory of relativity.",
    "Newton formulated three laws of motion.",
    "The periodic table organizes chemical elements by atomic number.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Quantum mechanics describes nature at the smallest scales.",
    "DNA contains the genetic instructions of every organism.",
    "Black holes are formed when massive stars collapse.",
    "The speed of light is constant in vacuum.",
    # === History (8 句) ===
    "The Roman Empire lasted for over a thousand years.",
    "Shakespeare wrote thirty seven plays during his career.",
    "Mozart composed his first symphony at the age of eight.",
    "The Eiffel Tower was built in eighteen eighty nine.",
    "The Great Wall of China stretches over twenty thousand kilometers.",
    "Antibiotics revolutionized medicine in the twentieth century.",
    "World War Two ended in nineteen forty five.",
    "Cleopatra was the last active ruler of the Ptolemaic Kingdom.",
    # === Geography (8 句) ===
    "Mount Everest is the highest mountain on Earth.",
    "The Pacific Ocean covers more area than all land masses.",
    "The Amazon rainforest produces twenty percent of Earth oxygen.",
    "The Sahara is the largest hot desert in the world.",
    "Antarctica contains about seventy percent of Earth fresh water.",
    "The Nile River flows through eleven countries in Africa.",
    "Tokyo is the most populous metropolitan area in the world.",
    "Madagascar is the fourth largest island in the world.",
    # === Literature (8 句) ===
    "Tolstoy wrote War and Peace during the Russian Empire.",
    "Shakespeare's Hamlet explores themes of revenge and mortality.",
    "The Iliad tells the story of the Trojan War.",
    "Don Quixote is widely considered the first modern novel.",
    "Kafka's works often explore themes of alienation and bureaucracy.",
    "Jane Austen published six major novels during her lifetime.",
    "Dostoevsky's Crime and Punishment explores psychological torment.",
    "Homer is credited as the author of the Odyssey.",
    # === Technology (8 句) ===
    "The internet was invented in the late twentieth century.",
    "Artificial intelligence aims to simulate human cognition.",
    "Quantum computers use qubits instead of classical bits.",
    "The World Wide Web was invented by Tim Berners Lee.",
    "Linux is a widely used open source operating system kernel.",
    "Blockchain technology underpins many modern cryptocurrencies.",
    "Machine learning algorithms improve through experience.",
    "Cloud computing delivers services over the internet.",
    # === Arts (8 句) ===
    "Picasso pioneered the Cubist movement in the early twentieth century.",
    "Beethoven composed nine famous symphonies.",
    "Rembrandt was a Dutch master of light and shadow.",
    "Impressionism began as a radical art movement in France.",
    "The Mona Lisa is one of the most valuable paintings in the world.",
    "Shakespeare wrote plays that are still performed worldwide.",
    "Michelangelo painted the ceiling of the Sistine Chapel.",
    "Bach composed over one thousand musical works.",
    # === Biology (8 句) ===
    "Mitochondria generate most of the chemical energy in cells.",
    "Evolution by natural selection shapes species over time.",
    "The human brain contains billions of neurons.",
    "Photosynthesis occurs in chloroplasts of plant cells.",
    "Ribosomes synthesize proteins from amino acids.",
    "Mitosis is the process of cell division in eukaryotic cells.",
    "Vaccines train the immune system against pathogens.",
    "Enzymes catalyze biochemical reactions in living organisms.",
    # === Philosophy (8 句) ===
    "Socrates is considered the father of Western philosophy.",
    "Plato founded the Academy in ancient Athens.",
    "Aristotle tutored Alexander the Great in philosophy.",
    "Descartes is famous for the statement I think therefore I am.",
    "Kant wrote the Critique of Pure Reason in seventeen eighty one.",
    "Nietzsche declared God is dead in the late nineteenth century.",
    "Hegel's dialectic shaped modern continental philosophy.",
    "Existentialism emerged as a movement in the twentieth century.",
]


# 24 句 held-out (与 train 不重叠, 全新领域/句子)
_LARGE_TEXT_HELD_OUT: List[str] = [
    "Volcanoes form at tectonic plate boundaries.",
    "Climate change affects ecosystems worldwide.",
    "The human genome contains about three billion base pairs.",
    "Renewable energy sources include solar and wind power.",
    "Pyramids of Giza stand as ancient engineering marvels.",
    "Mars once had liquid water on its surface.",
    "Gravity pulls objects toward the center of the Earth.",
    "Coral reefs host about twenty five percent of marine species.",
    "The brain uses about twenty percent of body energy.",
    "Shakespeare wrote sonnets in addition to his plays.",
    "DNA replication occurs before cell division.",
    "Glaciers store about sixty nine percent of fresh water.",
    "Mount Fuji is the tallest mountain in Japan.",
    "Leonardo da Vinci painted the Last Supper.",
    "Atoms consist of protons neutrons and electrons.",
    "The universe is about thirteen point eight billion years old.",
    "Galileo improved the telescope and observed Jupiter moons.",
    "Penicillin was discovered by Alexander Fleming.",
    "The violin has four strings tuned in perfect fifths.",
    "Photosynthesis releases oxygen as a byproduct.",
    "Mount Kilimanjaro is the highest peak in Africa.",
    "Carbon dating measures the age of ancient artifacts.",
    "Bach was a master of counterpoint in Baroque music.",
    "The heart pumps blood throughout the human body.",
]


class LargeTextCorpus:
    """大型 Wikipedia 文本语料 (64 train + 24 held-out).

    接口对齐 v18 MiniCorpus, 让 v29 端到端脚本可无缝接入.
    """

    def __init__(self, tokenizer, max_length: int = 16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_sentences = _LARGE_TEXT_CORPUS
        self.held_out = _LARGE_TEXT_HELD_OUT

    def train_batch(self, batch_size: int = 4) -> torch.Tensor:
        """随机采样 batch_size 句, tokenize → [batch_size, max_length]."""
        idx = torch.randint(0, len(self.train_sentences), (batch_size,)).tolist()
        sentences = [self.train_sentences[i] for i in idx]
        enc = self.tokenizer(
            sentences, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return enc["input_ids"]

    def eval_batch(self, batch_size: int = 6) -> torch.Tensor:
        """返回 held-out 句子的 tokenize."""
        enc = self.tokenizer(
            self.held_out[:batch_size], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return enc["input_ids"]

    def all_eval(self) -> torch.Tensor:
        """返回所有 held-out 句子的 tokenize."""
        return self.eval_batch(len(self.held_out))

    def train_strings(self, n: int = None) -> List[str]:
        """返回训练文本列表 (供 encode_text 调用)."""
        return self.train_sentences if n is None else self.train_sentences[:n]

    def eval_strings(self, n: int = None) -> List[str]:
        """返回 held-out 文本列表."""
        return self.held_out if n is None else self.held_out[:n]


def cycle_batches(corpus: LargeTextCorpus, batch_size: int = 4) -> Iterator[torch.Tensor]:
    """无限循环训练 batch."""
    while True:
        yield corpus.train_batch(batch_size)
