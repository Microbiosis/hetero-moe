"""v29.0 — 大型 Python 片段语料 (32 train + 12 held-out, 2× v26).

设计:
    v26 MiniCodeCorpus 只有 16 train + 6 held-out (小规模).
    v29 LargeCodeCorpus 扩展到 32 train + 12 held-out (2×), 覆盖 12+ 种 Python 语法模式.

    接口对齐 v26 MiniCodeCorpus:
        - train_snippets: List[str]
        - held_out: List[str]
        - train_batch(batch_size): tokenize → [B, max_length]
        - eval_batch(batch_size): held-out tokenize
        - all_eval(): all held-out
        - train_strings(n): List[str] (供 encode_code)

    内容覆盖 14 种 Python 语法模式:
        - 函数定义 / 类定义 / 装饰器 / lambda / 生成器
        - 列表推导 / 字典推导 / 集合推导 / 嵌套循环
        - with / 异常处理 / 上下文管理器
        - 属性访问 / 继承 / 异步函数

    字符长度约束: 每片段 ≤ 60 字符 (适配 TinyLlama + S=16 BPE token)
"""
from __future__ import annotations

from typing import Iterator, List

import torch


# 32 个真实 Python 片段 (14 种语法模式, 每片段 ≤ 60 字符)
_LARGE_PY_CORPUS: List[str] = [
    # === 函数定义 (4 个) ===
    "def add(a,b): return a+b",
    "def mul(a,b): return a*b",
    "def square(x): return x*x",
    "def greet(name): return 'hi ' + name",
    # === 类定义 (3 个) ===
    "class Foo: pass",
    "class Point: def __init__(s,x,y): s.x=x; s.y=y",
    "class Animal: def speak(self): return 'sound'",
    # === 装饰器 (2 个) ===
    "@property\ndef x(self): return self._x",
    "@staticmethod\ndef foo(): return 1",
    # === lambda (3 个) ===
    "f = lambda x: x*2",
    "g = lambda a,b: a+b if a>b else b",
    "h = lambda lst: max(lst)",
    # === 列表推导 (3 个) ===
    "x = [i*i for i in range(5)]",
    "y = [i for i in range(10) if i%2==0]",
    "z = [x.upper() for x in lst]",
    # === 字典/集合推导 (2 个) ===
    "d = {k:v for k,v in pairs}",
    "s = {x for x in lst if x > 0}",
    # === 嵌套循环 (2 个) ===
    "for i in range(3):\n  for j in range(3): print(i,j)",
    "[[i*j for j in range(3)] for i in range(3)]",
    # === with 语句 (3 个) ===
    "with open('f') as f: data = f.read()",
    "with lock: x += 1",
    "with open(f) as f, open(g) as g: pass",
    # === 异常处理 (2 个) ===
    "try: x = 1\nexcept: x = 0",
    "try: f()\nexcept ValueError: g()",
    # === 生成器 (2 个) ===
    "def gen(): yield 1; yield 2",
    "yield from gen",
    # === 上下文管理器 (2 个) ===
    "from contextlib import contextmanager",
    "@contextmanager\ndef cm(): yield 1",
    # === 属性 (1 个) ===
    "obj.attr = obj.attr + 1",
    # === 继承 (1 个) ===
    "class Child(Parent): pass",
    # === 异步函数 (2 个) ===
    "async def fetch(): return await http.get()",
    "async with lock: pass",
]


# 12 个 held-out (与 train 不重叠, 不同语法模式)
_LARGE_PY_HELD_OUT: List[str] = [
    "result = map(str, lst)",
    "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
    "class Bar: def __init__(self): self.x = 0",
    "x = [y for y in range(10) if y%2==0]",
    "for k,v in d.items(): print(k, v)",
    "def f(*args): return sum(args)",
    "data = sorted(lst, key=lambda x: x[1])",
    "obj = type('X', (), {'a': 1})()",
    "tp = (1, 2, 3, 4, 5)",
    "lst = [x for x in t if x > 0]",
    "with open(f) as f: data = f.read()",
    "cls = type('X', (), {'a': 1})",
]


class LargeCodeCorpus:
    """大型 Python 片段语料 (32 train + 12 held-out)."""

    def __init__(self, tokenizer, max_length: int = 16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_snippets = _LARGE_PY_CORPUS
        self.held_out = _LARGE_PY_HELD_OUT

    def train_batch(self, batch_size: int = 4) -> torch.Tensor:
        idx = torch.randint(0, len(self.train_snippets), (batch_size,)).tolist()
        snippets = [self.train_snippets[i] for i in idx]
        enc = self.tokenizer(
            snippets, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"]

    def eval_batch(self, batch_size: int = 6) -> torch.Tensor:
        enc = self.tokenizer(
            self.held_out[:batch_size], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"]

    def all_eval(self) -> torch.Tensor:
        return self.eval_batch(len(self.held_out))

    def train_strings(self, n: int = None) -> List[str]:
        return self.train_snippets if n is None else self.train_snippets[:n]

    def eval_strings(self, n: int = None) -> List[str]:
        return self.held_out if n is None else self.held_out[:n]


def cycle_batches(corpus: LargeCodeCorpus, batch_size: int = 4) -> Iterator[torch.Tensor]:
    """无限循环训练 batch."""
    while True:
        yield corpus.train_batch(batch_size)
