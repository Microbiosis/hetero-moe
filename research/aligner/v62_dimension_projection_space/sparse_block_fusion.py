"""重构A：稀疏块融合层（Sparse Block Fusion）。

替代 v62 的稠密 1024→4096→1024 MoE（50.3M 参数，v54 E4 证其在拟合噪声）。
把 4096 维隐藏层切成 n_blocks 个 block_ff 维的「概念块」，
每样本按内容只激活 top_blocks 个块，其余置零。

设计动机：
  - 稀疏是硬正则，块结构提供归纳偏置，抑制「任意权重组合拟合单样本」
  - 块激活的跨样本一致性可作为新评价器（同类别是否稳定激活同几块）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUBlock(nn.Module):
    """一个 1024 → block_ff → 1024 的 SwiGLU 概念块。"""

    def __init__(self, d_space: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(d_ff, d_space) * 0.02)
        self.w_up = nn.Parameter(torch.randn(d_ff, d_space) * 0.02)
        self.w_down = nn.Parameter(torch.randn(d_space, d_ff) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.w_gate)) * F.linear(x, self.w_up),
                        self.w_down)


class SparseBlockFusion(nn.Module):
    """块稀疏 MoE：每样本选 top-k 块，其余不计算。"""

    def __init__(self, d_space: int = 1024, n_blocks: int = 16,
                 block_ff: int = 256, top_blocks: int = 3,
                 num_experts: int = 4):
        super().__init__()
        self.d_space = d_space
        self.n_blocks = n_blocks
        self.block_ff = block_ff
        self.top_blocks = top_blocks

        self.blocks = nn.ModuleList([SwiGLUBlock(d_space, block_ff)
                                     for _ in range(n_blocks)])
        # 块门控：只看输入分布（detach），基于样本级均值
        self.block_gate = nn.Linear(d_space, n_blocks)
        # 保留 γ/β 仿射（与 v62 专家接口一致）
        self.gammas = nn.Parameter(torch.ones(d_space))
        self.betas = nn.Parameter(torch.zeros(d_space))
        # 块输出权重（可学习，替代原 alpha）
        self.block_alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x: torch.Tensor):
        """
        x: [B,S,d_space]
        returns: (y [B,S,d_space], aux dict with block_load)
        """
        B, S, D = x.shape
        h = x * self.gammas + self.betas
        h_mean = h.detach().mean(dim=1)                  # [B,D] 样本级统计
        g = F.softmax(self.block_gate(h_mean), dim=-1)   # [B,n_blocks]

        topg, topi = torch.topk(g, self.top_blocks, dim=-1)  # [B,k]
        out = torch.zeros_like(x)
        # 逐块计算，只算被选中的（稀疏）
        for b in range(self.n_blocks):
            mask = (topi == b).any(dim=-1)               # [B] 哪些样本选了块b
            if not mask.any():
                continue
            sel = mask.nonzero(as_tuple=True)[0]
            w_b = (g[sel, b] * self.block_alpha).view(-1, 1, 1)  # [n_sel,1,1]
            out[sel] = out[sel] + w_b * self.blocks[b](h[sel])
        return x + out, {'block_load': g.detach(),
                         'top_blocks': topi.detach()}

    def forward_with_routing(self, x: torch.Tensor):
        """兼容 FusionOverProjSpace 接口：返回 (y, alpha)。"""
        y, aux = self.forward(x)
        B, S, _ = x.shape
        # 稀疏块无专家概念，alpha 用块门控的 top-k 归一化占位
        alpha = torch.zeros(B, S, self.n_blocks, device=x.device)
        alpha.scatter_(-1, aux['top_blocks'].unsqueeze(1).expand(-1, S, -1),
                       aux['block_load'].gather(1, aux['top_blocks'])
                       .unsqueeze(1).expand(-1, S, -1))
        return y, alpha

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return ('SparseBlockFusion\n'
                '  d_space=%d → %d blocks × %d → %d\n'
                '  top_blocks=%d (稀疏激活)\n'
                '  参数=%d（稠密MoE为 %d）'
                % (self.d_space, self.n_blocks, self.block_ff, self.d_space,
                   self.top_blocks, n,
                   4 * 3 * 4 * self.d_space * self.d_space))
