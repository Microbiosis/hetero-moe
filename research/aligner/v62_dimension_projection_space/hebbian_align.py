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
        present = [m for m in self.models if m in feats]
        if not present:
            return torch.eye(self.n, device=self.coupling.device, dtype=self.coupling.dtype)
        ref = feats[present[0]]
        d = ref.shape[-1]
        vecs = []
        present_mask = torch.zeros(self.n, device=self.coupling.device, dtype=torch.bool)
        for i, m in enumerate(self.models):
            if m in feats:
                v = feats[m].mean(dim=(0, 1)).to(self.coupling.device, self.coupling.dtype)
                v = F.normalize(v, dim=0)
                vecs.append(v)
                present_mask[i] = True
            else:
                vecs.append(torch.zeros(d, device=self.coupling.device, dtype=self.coupling.dtype))
        V = torch.stack(vecs)                            # [n,d]
        sim = (V @ V.T).clamp(-1.0, 1.0)                 # [n,n] 余弦相似度
        # 仅更新双方均存在的模态对；超出 sim_threshold 的强相关对按原值增强，缺失模态对保持原耦合不变
        pair_mask = present_mask.unsqueeze(1) & present_mask.unsqueeze(0)
        eff_sim = torch.where(sim > self.sim_threshold, sim, sim * 0.5)
        updated = self.coupling * (1.0 - self.eta) + self.eta * eff_sim
        self.coupling.copy_(torch.where(pair_mask, updated, self.coupling))
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
