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
        self.num_experts = num_experts
        self.last_router_probs = None

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
        returns: (y [B,S,d_space], aux dict with differentiable block_probs and block_load)
        """
        B, S, D = x.shape
        h = x * self.gammas + self.betas
        h_mean = h.detach().mean(dim=1)                  # [B,D] 样本级统计
        g = F.softmax(self.block_gate(h_mean), dim=-1)   # [B,n_blocks]
        self.last_router_probs = g

        topg, topi = torch.topk(g, self.top_blocks, dim=-1)  # [B,k]
        # 避免重叠切片原地写 (out[sel] = out[sel] + ...) 触发 autograd 版本冲突
        out = torch.zeros_like(x)
        # 逐块计算，只算被选中的（稀疏）
        for b in range(self.n_blocks):
            mask = (topi == b).any(dim=-1)               # [B] 哪些样本选了块b
            if not mask.any():
                continue
            sel = mask.nonzero(as_tuple=True)[0]
            w_b = (g[sel, b] * self.block_alpha).view(-1, 1, 1)  # [n_sel,1,1]
            contrib = (w_b * self.blocks[b](h[sel])).to(out.dtype)
            out = out.index_add(0, sel, contrib)
        return x + out, {'block_probs': g,
                         'block_load': g.detach(),
                         'top_blocks': topi.detach()}

    def forward_with_routing(self, x: torch.Tensor):
        """兼容 FusionOverProjSpace 接口：返回 (y, alpha)，保留 block_gate 梯度流。"""
        y, aux = self.forward(x)
        B, S, _ = x.shape
        # 用保留梯度的 block_probs 构造 top-k 归一化路由权重，避免负载均衡损失梯度被截断
        g = aux['block_probs']                           # [B, n_blocks] (可微)
        topi = aux['top_blocks']                         # [B, k]
        topv = g.gather(1, topi)                         # [B, k]
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        alpha_b = torch.zeros(B, self.n_blocks, device=x.device, dtype=g.dtype).scatter(-1, topi, topv)
        alpha = alpha_b.unsqueeze(1).expand(-1, S, -1)
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
