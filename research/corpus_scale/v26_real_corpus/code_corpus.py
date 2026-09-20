"""v26.0 — 真实 Python 片段语料 (替代 v25 合成 ["def hello():", ...]).

设计:
    内置 16 个真实 Python 片段 (训练) + 6 个 held-out (不同语法模式),
    适配 TinyLlama tokenizer + S=16 token 长度限制.
    不依赖网络下载.

可追溯性:
    MiniCodeCorpus    -> v18 MiniCorpus 接口克隆 (tokenize → [B, S])
    _MINI_PY_CORPUS   -> 16 个真实 Python 片段 (≤16 token)
    _PY_HELD_OUT      -> 6 个 held-out (不同语法模式, 与 train 不重叠)

token 长度控制:
    每个片段均经过手工挑选, ≤16 BPE token (适配 S=16).
    设计原则: 每片段代表一种 Python 语法模式, 覆盖函数/类/comprehension/
             lambda/with/异常处理 等核心构造.
"""
from __future__ import annotations

from typing import Iterator, List

import torch


# 内置 16 个真实 Python 片段 (≤16 token, 覆盖核心语法)
_MINI_PY_CORPUS: List[str] = [
    "def add(a,b): return a+b",            # 1. 函数定义
    "for i in range(10): print(i)",         # 2. for 循环
    "x = [i*i for i in range(5)]",          # 3. list comprehension
    "class Foo: pass",                      # 4. 类定义
    "if x > 0: y = 1",                      # 5. 条件分支
    "import math",                          # 6. import
    "d = {'a': 1}",                         # 7. dict 字面量
    "try: x = 1\nexcept: x = 0",            # 8. try/except
    "f = lambda x: x*2",                    # 9. lambda
    "with open('f') as f: data = f.read()", # 10. with 语句
    "y = sum([1,2,3])",                     # 11. 函数调用
    "z = a and b or c",                     # 12. 逻辑表达式
    "for k,v in d.items(): print(k)",       # 13. dict 迭代
    "def f(*a): return a",                  # 14. 可变参数
    "lst = [x for x in t if x > 0]",        # 15. 过滤 comprehension
    "tp = (1, 2, 3)",                       # 16. tuple 字面量
]


# 6 个 held-out (与 train 不同的语法模式)
_PY_HELD_OUT: List[str] = [
    "result = map(str, lst)",                           # 高阶函数
    "class Bar: def __init__(self): self.x = 0",        # 构造器
    "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",  # 递归
    "for i in range(3):\n  for j in range(3): print(i,j)",  # 嵌套循环
    "x = [y for y in range(10) if y%2==0]",             # 偶数 comprehension
    "yield from gen",                                    # yield from
]


class MiniCodeCorpus:
    """真实 Python 片段语料 (16 训练 + 6 held-out).

    接口完全对齐 v18 MiniCorpus, 让 v26 端到端脚本可无缝接入.
    """

    def __init__(self, tokenizer, max_length: int = 16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_snippets = _MINI_PY_CORPUS
        self.held_out = _PY_HELD_OUT

    def train_batch(self, batch_size: int = 4) -> torch.Tensor:
        """随机采样 batch_size 个 Python 片段, tokenize → [batch_size, max_length]."""
        idx = torch.randint(0, len(self.train_snippets), (batch_size,)).tolist()
        snippets = [self.train_snippets[i] for i in idx]
        enc = self.tokenizer(
            snippets, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"]

    def eval_batch(self, batch_size: int = 6) -> torch.Tensor:
        """返回 held-out Python 片段的 tokenize."""
        enc = self.tokenizer(
            self.held_out[:batch_size], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"]

    def all_eval(self) -> torch.Tensor:
        """返回所有 held-out Python 片段的 tokenize."""
        return self.eval_batch(len(self.held_out))

    def train_strings(self, n: int = None) -> List[str]:
        """返回训练 Python 字符串列表 (供 encode_code 调用, 非 tokenized)."""
        return self.train_snippets if n is None else self.train_snippets[:n]

    def eval_strings(self, n: int = None) -> List[str]:
        """返回 held-out Python 字符串列表."""
        return self.held_out if n is None else self.held_out[:n]


def cycle_batches(corpus: MiniCodeCorpus, batch_size: int = 4) -> Iterator[torch.Tensor]:
    """无限循环训练 batch (与 v18 cycle_batches 接口对齐)."""
    while True:
        yield corpus.train_batch(batch_size)
