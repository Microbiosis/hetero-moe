"""路线 7：自修复槽机制（Self-Repairing Slot）。

v62.3 实测：Q 槽平均权重仅 0.003，模型几乎单方面依赖 K/V/O。
根因是 router 学到了「不看 Q」，而传统训练没有任何机制检测或纠正这一点。

本模块在训练循环中监测四槽贡献度（router 输出的 EMA），
当某槽长期低于阈值时，自动触发修复：
  1. 重新初始化 router 对应输出列（std=0.02，与初始化一致）
  2. 提升 slot_gate 中该槽的值（+2.0，使 softmax 不被压制）
  3. 对该槽投影矩阵注入极小噪声（std=0.005，帮助跳出局部最优）
  4. 重置该槽的 EMA，进入观察期

设计约束：
- 只在**训练时**介入，推理路径不变
- 修复有冷却期（默认 200 步），避免震荡
- 不改变梯度流，只是 in-place 权重修正
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SlotHealthMonitor:
    """监测槽贡献度 EMA，低贡献槽自动触发权重修复。"""

    def __init__(self,
                 n_slots: int,
                 slot_names: List[str],
                 threshold: float = 0.01,
                 ema_decay: float = 0.99,
                 cooldown: int = 200,
                 device: str = 'cpu'):
        self.n_slots = n_slots
        self.slot_names = slot_names
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.cooldown = cooldown
        self.ema = torch.ones(n_slots, device=device) / n_slots
        self.steps = 0
        self.last_repair_step = -cooldown
        self.repair_log: List[Dict] = []

    @torch.no_grad()
    def update(self, alpha: torch.Tensor) -> None:
        """alpha: [B,S,n_slots]（router 输出）或 [n_slots]。"""
        if alpha.dim() == 3:
            m = alpha.detach().mean(dim=(0, 1))
        elif alpha.dim() == 1:
            m = alpha.detach()
        else:
            raise ValueError('alpha 形状应为 [B,S,n_slots] 或 [n_slots]，得到 %s'
                             % tuple(alpha.shape))
        m = m.to(self.ema.device)                  # 对齐设备（ema 固定 CPU）
        self.ema.mul_(self.ema_decay).add_(m * (1 - self.ema_decay))
        self.steps += 1

    def needs_repair(self) -> List[int]:
        if self.steps - self.last_repair_step < self.cooldown:
            return []
        return [i for i in range(self.n_slots) if float(self.ema[i]) < self.threshold]

    @torch.no_grad()
    def repair(self, module: nn.Module, optimizer: Optional[object] = None) -> List[str]:
        """对低贡献槽执行自修复，返回被修复的槽名列表。"""
        idxs = self.needs_repair()
        if not idxs:
            return []

        repaired: List[str] = []
        for idx in idxs:
            name = self.slot_names[idx]
            ema_before = float(self.ema[idx])

            # 1. router 输出列重新初始化（仅当该槽参与路由时）
            if hasattr(module, 'router') and module.router is not None:
                nn.init.normal_(module.router.weight[idx, :], std=0.02)
                if optimizer is not None:
                    st = optimizer.state.get(module.router.weight)
                    if st is not None:
                        for k, v in st.items():
                            if k == 'step':
                                continue
                            if v.shape == module.router.weight.shape:
                                v[idx, :].zero_()

            # 2. slot_gate 提升，确保 softmax 后不被其他槽压制
            if hasattr(module, 'slot_gate'):
                module.slot_gate.data[idx] += 2.0

            # 3. 该槽投影矩阵注入极小噪声，帮助跳出局部最优
            if hasattr(module, 'proj'):
                for key in module.proj:
                    if key.startswith(name + '|'):
                        w = module.proj[key].weight
                        w.add_(torch.randn_like(w) * 0.005)

            self.ema[idx] = 1.0 / self.n_slots
            self.last_repair_step = self.steps
            repaired.append(name)
            self.repair_log.append({
                'step': self.steps,
                'slot': name,
                'ema_before': ema_before,
                'action': 'router_reinit + slot_gate_boost + proj_noise'
            })
        return repaired

    def status(self) -> Dict[str, float]:
        return {name: round(float(self.ema[i]), 5)
                for i, name in enumerate(self.slot_names)}

    def summary(self) -> str:
        s = self.status()
        worst = min(s, key=s.get)
        return ('slot_ema: %s | 最低 %s=%.5f | 已修复 %d 次'
                % (s, worst, s[worst], len(self.repair_log)))
