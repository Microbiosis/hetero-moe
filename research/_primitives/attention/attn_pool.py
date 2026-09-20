"""v7.0 — 原生 Attention + 矩形 P_m 投影 Q/K/V 的封装。

设计动机:
    v6.0 仅路由 FFN, 原生 Attention 被完全绕过 (`last_hidden_state` 直接抽取)。
    v7.0 让 Attention 也成为可路由对象:
      - 每个 attn 专家对应一个底座的 attention (TinyBERT/TinyLlama/ViT 等)
      - Q/K/V/O 用底座原生权重, 通过 P_m 矩形投影与 D_shared 空间互转
      - v7 阶段采用"手动实现 MHA 但权重从 HF 抽取"的策略, 绕开 HF 接口差异

P_m 投影约定:
    P_m: R^{D_shared × D_m} (正交初始化, 冻结, 与 v6 一致)
    x_shared (D_shared) -> P_m^T · x_shared -> 原生 attn 期望的 D_m 空间
    attn_out (D_m) -> attn_out · P_m -> 回到 D_shared
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_multi_head_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """手动多头 self-attention (单 head 时退化为普通 attn)。

    与 HF 接口对齐:
        HF 的 query/key/value 是合并的 `W_qkv: [3*D, D]`,
        本函数拆成 3 个独立权重 [D, D] (与 v6 §3 §6.1 的拆分约定一致)
        w_o: [D, D] 输出投影

    Args:
        x:        [B, S, D]
        w_q/k/v:  [D, D]
        w_o:      [D, D]
        num_heads: 头数

    Returns:
        [B, S, D]
    """
    B, S, D = x.shape
    H = num_heads
    Dh = D // H
    assert D % H == 0, f"D={D} 必须能被 num_heads={H} 整除"

    def split_heads(t):
        # [B, S, D] -> [B, H, S, Dh]
        return t.view(B, S, H, Dh).transpose(1, 2)

    q = split_heads(F.linear(x, w_q))  # [B, H, S, Dh]
    k = split_heads(F.linear(x, w_k))
    v = split_heads(F.linear(x, w_v))
    # scaled dot-product attention (标准实现, 不用 torch.nn.functional.scaled_dot_product_attention 以保兼容性)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)  # [B, H, S, S]
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [B, H, S, Dh]
    out = out.transpose(1, 2).contiguous().view(B, S, D)
    return F.linear(out, w_o)


class AttnPool(nn.Module):
    """单 attn 专家: 原生 attn 权重 + P_m 矩形投影。

    持有:
        P_m:    D_shared × D_m 正交矩阵 (冻结)
        W_q/k/v: D_m × D_m 原生 attn Q/K/V (冻结, 从 HF 抽取)
        W_o:   D_shared × D_shared 输出投影 (可训练, lr=LR_ADAPTER)
        num_heads: 头数 (与底座一致)

    forward(x_shared):
        x_m = x_shared · P_m               (D_shared → D_m)
        attn_out = manual_mha(x_m, W_q, W_k, W_v, W_o_native_proj)  # [B,S,D_m]
        attn_back = attn_out · P_m^T       (D_m → D_shared)
        attn_final = attn_back · W_o       (D_shared → D_shared)
    """

    def __init__(
        self,
        D_m: int,
        D_shared: int,
        num_heads: int,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o_native: torch.Tensor,
    ):
        super().__init__()
        assert w_q.shape == (D_m, D_m), f"w_q 形状应为 [{D_m},{D_m}], 实测 {tuple(w_q.shape)}"
        assert w_k.shape == (D_m, D_m)
        assert w_v.shape == (D_m, D_m)
        assert w_o_native.shape == (D_m, D_m), (
            f"w_o_native 应为 [{D_m},{D_m}], 实测 {tuple(w_o_native.shape)}"
        )
        self.D_m = D_m
        self.D_shared = D_shared
        self.num_heads = num_heads

        # P_m: D_shared × D_m, 正交初始化, 冻结
        self.P_m = nn.Parameter(torch.empty(D_shared, D_m), requires_grad=False)
        nn.init.orthogonal_(self.P_m)

        # 原生 Q/K/V/O 冻结权重 (从 HF 抽取)
        self.register_buffer("W_q", w_q.detach().clone())
        self.register_buffer("W_k", w_k.detach().clone())
        self.register_buffer("W_v", w_v.detach().clone())
        self.register_buffer("W_o_native", w_o_native.detach().clone())

        # W_o: D_shared × D_shared 可训练 (路由需要的输出投影)
        self.W_o = nn.Parameter(torch.empty(D_shared, D_shared))
        nn.init.orthogonal_(self.W_o)

    def forward(self, x_shared: torch.Tensor) -> torch.Tensor:
        """x_shared: [B, S, D_shared] -> attn_out: [B, S, D_shared]"""
        B, S, _ = x_shared.shape
        # 1. 投影到原生 D_m 空间: x_shared [B,S,D_shared] · P_m^T [D_m, D_shared] -> [B,S,D_m]
        x_m = F.linear(x_shared, self.P_m.t())  # [B, S, D_m]
        # 2. 在原生空间做 MHA (注意: 此时 W_o 是 D_m × D_m)
        attn_m = manual_multi_head_attention(
            x_m, self.W_q, self.W_k, self.W_v, self.W_o_native, self.num_heads
        )  # [B, S, D_m]
        # 3. 投影回 D_shared 空间: attn_m [B,S,D_m] · P_m [D_shared, D_m] -> [B,S,D_shared]
        attn_back = F.linear(attn_m, self.P_m)  # [B, S, D_shared]
        # 4. 可训练 W_o
        return F.linear(attn_back, self.W_o)
