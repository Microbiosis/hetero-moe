"""interfaces.py — 统一组件接口（D1「逐 block 拆」的载体）。

## 为什么需要它

7 个基模的内部结构各不相同（CBraMod = 二维 Criss-Cross 注意力；BIOT = 一维注意力 +
cls token；LaBraM = 层级金字塔；EEGDINO = DINOv2 式层级自蒸馏；MIRepNet = CNN+Transformer；
Signal-JEPA = JEPA 预测器；BrainTokenizer = 分词器）。

要"拆开来重装"，就必须有一个**共同的接口**，否则组件搬不动。

## 共同点

它们都是：

    EEG (B, C, T) ──[patch embed]──▶ tokens (B, N, D) ──[block]×L──▶ tokens ──[head]──▶ out

`block` 内部是「mixer（序列混合器）+ ffn（计算模块）+ norm」。
① 序列混合器、② 记忆模块、③ 计算模块、④ 域注意力
—— ROADMAP §3.1 的四个组件，正好落在这四个位置。

## 关键约束（来自 D1「逐 block 拆」的代价）

1. **token 维度 D 必须沿链一致** ⇒ 维度不同的 block 前插**学习到的线性投影**。
2. **token 必须携带空间布局 (rows, cols)** ⇒ 否则 CBraMod 的**二维** Criss-Cross
   注意力无法在别的模型的 token 上复用。
3. **归一化位置不同**（pre-LN / post-LN）⇒ 统一为 pre-LN；不兼容者标 `arch-only`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TokenBatch:
    """一批 token + 它的空间布局。

    layout 为 (rows, cols) 时 token 可视作二维网格（CBraMod 的 Criss-Cross
    需要）；为 None 时视作一维序列。
    """
    tokens: torch.Tensor          # (B, N, D)
    layout: Optional[tuple] = None  # (rows, cols) 或 None

    def __len__(self):
        return self.tokens.shape[0]

    @property
    def dim(self):
        return self.tokens.shape[-1]

    def mean_pool(self):
        """池化规则（预先写死）：**对全部 token（含全局 token）取均值**。"""
        return self.tokens.mean(dim=1)

    def replace(self, tokens):
        return TokenBatch(tokens, self.layout)


# ---------------------------------------------------------------------------
# Mixer 家族：统一签名 forward(TokenBatch) -> TokenBatch
# ---------------------------------------------------------------------------
class MixerBase(nn.Module):
    kind = "base"

    def forward(self, x: TokenBatch) -> TokenBatch:  # pragma: no cover
        raise NotImplementedError


class IdentityMixer(MixerBase):
    kind = "identity"

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: TokenBatch):
        return x


class SelfAttention1D(MixerBase):
    """一维自注意力（BIOT / EEGDINO / MIRepNet 的 transformer 块里那种）。"""
    kind = "attn1d"

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)

    def forward(self, x: TokenBatch):
        out, _ = self.attn(x.tokens, x.tokens, x.tokens, need_weights=False)
        return x.replace(out)


class CrissCross2D(MixerBase):
    """二维 Criss-Cross 注意力（CBraMod 的核心）。

    沿 token 网格的**行**与**列**各做一次注意力。
    需要 `layout=(rows, cols)`；没有布局就退化成 1D 注意力并说明。
    """
    kind = "crisscross2d"

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.d_model = d_model

    def forward(self, x: TokenBatch):
        if x.layout is None:
            out, _ = self.attn(x.tokens, x.tokens, x.tokens, need_weights=False)
            return x.replace(out)
        B, N, D = x.tokens.shape
        rows, cols = x.layout
        if rows * cols != N:
            out, _ = self.attn(x.tokens, x.tokens, x.tokens, need_weights=False)
            return x.replace(out)
        t = x.tokens.view(B, rows, cols, D)
        # 行方向
        t = t.permute(0, 2, 1, 3).reshape(B * cols, rows, D)
        t, _ = self.attn(t, t, t, need_weights=False)
        t = t.view(B, cols, rows, D).permute(0, 2, 1, 3).reshape(B, rows * cols, D)
        # 列方向
        t = t.view(B, rows, cols, D).permute(0, 2, 1, 3).reshape(B * rows, cols, D)
        t, _ = self.attn(t, t, t, need_weights=False)
        t = t.view(B, rows, cols, D).permute(0, 2, 1, 3).reshape(B, rows * cols, D)
        return x.replace(t)


class LinearSSM(MixerBase):
    """线性 SSM 占位（对应 ROADMAP 的"Mamba/GDN 线性层"）。

    **默认是占位实现**（nearest 投影），不是真正的 Mamba/GDN。
    真正的线性层需要 `mamba_ssm` / `fla` 依赖（环境里没有）。
    装配器遇到这个 mixer 时必须把 provenance 标成 `fresh`，
    并在报告里说明"这是结构占位，不是 Mamba"——**不得让占位冒充真身**。
    """
    kind = "linear_ssm_placeholder"

    def __init__(self, d_model, d_inner=None):
        super().__init__()
        d_inner = d_inner or d_model
        self.proj_in = nn.Linear(d_model, d_inner)
        self.proj_out = nn.Linear(d_inner, d_model)

    def forward(self, x: TokenBatch):
        return x.replace(self.proj_out(self.proj_in(x.tokens)))


class TimeFreqCross(MixerBase):
    """时频交叉注意力（ROADMAP ④ 域注意力）。

    token 网格的行轴=时间块、列轴=频率块：
    **每个时间块去看整张频率轴的全局上下文，每个频率块去看整张时间轴的全局上下文**。
    （这与 CrissCross2D 的"轴内逐位置注意"不同：这里是轴间的全局互注意。）
    没有 layout 时退化为 1D 注意力并说明。
    """
    kind = "timefreq"

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)

    def forward(self, x: TokenBatch):
        if x.layout is None:
            out, _ = self.attn(x.tokens, x.tokens, x.tokens, need_weights=False)
            return x.replace(out)
        B, N, D = x.tokens.shape
        rows, cols = x.layout
        if rows * cols != N:
            out, _ = self.attn(x.tokens, x.tokens, x.tokens, need_weights=False)
            return x.replace(out)
        g = x.tokens.view(B, rows, cols, D)
        # 全局频率上下文：把同一列的 rows 个 token 沿时间轴平均 → (B, cols, D)
        ctx_freq = g.mean(dim=1)                 # (B, cols, D)
        ctx_time = g.mean(dim=2)                 # (B, rows, D)
        row_ctx, _ = self.attn(g.mean(dim=2), ctx_freq, ctx_freq, need_weights=False)
        col_ctx, _ = self.attn(g.mean(dim=1), ctx_time, ctx_time, need_weights=False)
        # 把轴级上下文广播回每个 token
        row_ctx = row_ctx.unsqueeze(2).expand(B, rows, cols, D)
        col_ctx = col_ctx.unsqueeze(1).expand(B, rows, cols, D)
        return x.replace((g + row_ctx + col_ctx).reshape(B, rows * cols, D))


MIXER_KINDS = {
    "identity": IdentityMixer,
    "attn1d": SelfAttention1D,
    "crisscross2d": CrissCross2D,
    "linear_ssm": LinearSSM,
    "timefreq": TimeFreqCross,
}


# ---------------------------------------------------------------------------
# Block 与 FFN
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    """计算模块（③）。普通 FFN 或小 MoE。"""
    kind = "ffn"

    def __init__(self, d_model, d_hidden, n_experts=1, top_k=1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        if n_experts <= 1:
            self.experts = nn.ModuleList([nn.Sequential(
                nn.Linear(d_model, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_model))])
            self.gate = None
        else:
            self.experts = nn.ModuleList([nn.Sequential(
                nn.Linear(d_model, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_model)) for _ in range(n_experts)])
            self.gate = nn.Linear(d_model, n_experts)

    def forward(self, h):
        if self.n_experts <= 1:
            return self.experts[0](h)
        logits = self.gate(h)
        w, idx = torch.topk(logits, self.top_k, dim=-1)
        w = torch.softmax(w, dim=-1)
        out = torch.zeros_like(h)
        for e in range(self.n_experts):
            mask = (idx == e).any(dim=-1, keepdim=True)
            if mask.any():
                contrib = self.experts[e](h) * mask * w.gather(-1, (idx == e).int().argmax(-1, keepdim=True))
                out = out + contrib
        return out


class Projection(nn.Module):
    """维度不一致时插的学习投影（provenance = `projected`）。"""
    kind = "projection"

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, h):
        return self.lin(h)


class Block(nn.Module):
    """pre-LN 的 block：x = x + mixer(norm(x)); x = x + ffn(norm(x))。"""
    kind = "block"

    def __init__(self, mixer, ffn, d_model, norm_first=True):
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: TokenBatch):
        if self.norm_first:
            h = self.norm1(x.tokens)
            x = x.replace(x.tokens + self.mixer(x.replace(h)).tokens)
            h = self.norm2(x.tokens)
            x = x.replace(x.tokens + self.ffn(h))
        else:
            x = x.replace(x.tokens + self.mixer(x).tokens)
            h = self.norm1(x.tokens)
            x = x.replace(x.tokens + self.ffn(h))
        return x
