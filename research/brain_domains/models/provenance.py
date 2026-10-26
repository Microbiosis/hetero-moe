"""provenance.py — 组装变体的"诚实归因"载体。

每个重组变体都必须能回答：**这个 block 来自哪个基模的哪一层？权重是搬来的还是重初始化的？许可是什么？**

没有这张表，"重组带来的提升"就无法归因 —— ROADMAP §3.3 要解决的正是这个问题。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# 权重状态
W_REUSE = "reuse"        # 权重直接搬用（维度与结构都兼容）
W_PROJECTED = "projected"  # 前面插了学习到的线性投影后搬用
W_FRESH = "fresh"        # 只借结构，权重重新初始化
W_ARCH_ONLY = "arch-only"  # 连结构都只是近似，不保证等价


@dataclass
class BlockProv:
    """一个 block 的来源记录。

    `weights` 是**整块**的权重状态（取各组件里最弱的那个）。
    `components` 逐组件记状态（mixer / ffn / proj）——
    因为「插了学习投影 + mixer 权重是新的」和「权重原样搬来」是两回事，
    只记一个总标签会把差别抹掉。
    """
    index: int                      # 在组装链里的序号
    role: str                       # patch_embed / mixer / ffn / norm / head
    source_model: str               # 基模名（如 "cbramod"）；"scratch" 表示自建
    source_layer: int | str | None  # 基模里的层号/子模块名
    weights: str                    # W_REUSE / W_PROJECTED / W_FRESH / W_ARCH_ONLY
    in_dim: int | None = None
    out_dim: int | None = None
    license: str | None = None      # 权重来源的许可（继承上游最严者）
    note: str = ""
    components: dict = field(default_factory=dict)  # 逐组件权重状态

    def to_dict(self):
        return asdict(self)


@dataclass
class VariantProv:
    """一个组装变体的完整来源记录。"""
    variant_id: str
    host_model: str                          # 组装宿主（提供 token 维度与骨架）
    blocks: list[BlockProv] = field(default_factory=list)
    domain: dict = field(default_factory=dict)      # 通道/采样率/窗口策略
    extra_components: list[dict] = field(default_factory=list)  # 记忆模块/MoE 等外挂
    notes: str = ""

    def add(self, b: BlockProv):
        self.blocks.append(b)
        return self

    def weight_summary(self):
        """权重来源统计 —— 一眼看出"这个变体有多少是搬来的"。"""
        out = {}
        for b in self.blocks:
            out[b.weights] = out.get(b.weights, 0) + 1
        return out

    def models_used(self):
        return sorted({b.source_model for b in self.blocks if b.source_model})

    def licenses_used(self):
        return sorted({b.license for b in self.blocks if b.license})

    def to_dict(self):
        d = asdict(self)
        d["weight_summary"] = self.weight_summary()
        d["models_used"] = self.models_used()
        d["licenses_used"] = self.licenses_used()
        return d

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def merge_license(licenses):
    """取最严者：只要有一个 NC ⇒ 整条链就是 NC。

    "检查点许可 ≠ 预训练语料许可" —— 上游 eegdino 模型卡明写衍生权重继承
    预训练语料许可（部分 braindecode 语料为 CC-BY-NC）。
    """
    ls = [str(x).upper() for x in licenses if x]
    if not ls:
        return None
    for x in ls:
        if "NC" in x or "NONCOMMERCIAL" in x:
            return "NC"
    if any("CC-BY-NC" in x for x in ls):
        return "NC"
    return sorted(set(ls))[0]
