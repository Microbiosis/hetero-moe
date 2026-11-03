"""路线4：Hebbian 槽对齐（Hebbian Slot Alignment）。

在推理时根据三模态在某槽上的激活相关性，按 Hebbian 规则实时更新
模态间耦合强度，替代纯静态的 alignment_loss。

维护一个可学习的 3×3 耦合矩阵 W_couple（Qwen/BERT/ViT），
前向时根据当前 batch 的跨模态余弦相似度做 in-place 更新。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianSlotAlignment(nn.Module):
    """模态耦合矩阵 + Hebbian 推理时更新。"""

    def __init__(self, models=('Qwen', 'BERT', 'ViT'),
                 eta: float = 0.01, sim_threshold: float = 0.9):
        super().__init__()
        self.models = list(models)
        self.n = len(self.models)
        self.eta = eta
        self.sim_threshold = sim_threshold
        # 3×3 耦合矩阵（对角线为1，初始对称）
        W = torch.eye(self.n)
        self.coupling = nn.Parameter(W, requires_grad=False)

    @torch.no_grad()
    def hebbian_update(self, feats: Dict[str, torch.Tensor]):
        """feats: {model: [B,S,d]}，按当前 batch 更新耦合矩阵。

        规则：耦合[i,j] += eta * <h_i, h_j> / (|h_i||h_j|)
        当余弦 > threshold 时增强，< 0 时衰减（简化版）。
        """
        vecs = []
        for m in self.models:
            if m in feats:
                v = feats[m].mean(dim=(0, 1))            # [d] 全局均值
                v = F.normalize(v, dim=0)
                vecs.append(v)
            else:
                vecs.append(torch.zeros_like(self.coupling[0]))
        V = torch.stack(vecs)                            # [n,d]
        sim = V @ V.T                                    # [n,n] 余弦相似度
        # Hebbian：同步增强，异步抑制
        delta = self.eta * sim
        self.coupling.add_(delta)
        # 保持对角线为1（自耦合不变）
        self.coupling.fill_diagonal_(1.0)
        return sim

    def gate_for_slot(self, slot_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """根据当前耦合矩阵生成该槽的模态融合权重 [n]。"""
        idx = [i for i, m in enumerate(self.models) if m in slot_feats]
        w = self.coupling[idx, :][:, idx].sum(dim=1)
        return F.softmax(w, dim=0)

    def forward(self, feats: Dict[str, torch.Tensor]):
        """推理入口：更新耦合并返回当前耦合矩阵。"""
        with torch.no_grad():
            sim = self.hebbian_update(feats)
        return self.coupling.detach(), sim

    def extra_repr(self):
        return ('models=%s, eta=%.3f, thresh=%.2f'
                % (self.models, self.eta, self.sim_threshold))
