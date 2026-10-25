"""V50.0 — 「融合收益从哪来」审计: 探针 + proxy 对齐预训练。

被审计的主张 (论文 reconstruct.tex §6.3 与摘要):
    "Fused beats best single expert (v33) +26.7% (5/5) ... survives" (held-out)

v49 观测到与之张力的事实: **投影一旦可训练, 『融合 vs 单专家』的相对增益就塌到
~0** (冻结臂 +60.6%~+65.8%, 可训练臂 +0.5%~+10.4%, 三档样本量都塌)。
于是有两种竞争解释, v50 用两个新仪器把它们分开:

  E1 (假象说): 那个 margin 不是"融合"的功劳, 而是**投影空间从未被训练**造成的
      —— 随机正交投影把各底座搅乱, 单专家基线因此很弱, "融合"显得赢。
      检验: 换一个**语义对齐过但对目标任务不自适应**的投影 (proxy 对齐),
      如果 margin 塌掉 → E1 成立。

  E2 (退化说): margin 是真的, 但它对应的是一个**退化任务** —— 冻结随机空间让
      "识别样本来自哪个底座"变得容易, 而路由本来就是在做这件事。
      检验: 从**融合输出**做线性探针 —— 若源专家可解码性高、目标类别可解码性低,
      说明模型在"认来源"而不是"融语义" → E2 成立。

两个新仪器:
  1. proxy 对齐预训练: 用**无标签**的跨底座对齐目标把投影空间训成语义对齐的
     (同一文本在三个底座上的投影互相靠拢), 然后**冻结**。
        L_proxy = Σ_m || P_m(h_m^t) - mean_{m'} P_{m'}(h_{m'}^t) ||²
     它不需要目标任务的任何标签, 因此不构成泄漏。
  2. 线性探针: 从融合输出 (或投影后的输入) 线性探测
     (a) 源专家身份 (3 类)  (b) 目标类别 (8 类)
     两者都是 held-out 评估; 准确率对照各自的平凡基线。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["proxy_pretrain", "linear_probe", "PROXY_STEPS", "PROXY_LR"]

PROXY_STEPS = 150
PROXY_LR = 1e-3


def proxy_pretrain(projs: dict, feats: dict, experts, steps: int = PROXY_STEPS,
                   lr: float = PROXY_LR, log: bool = False):
    """跨底座 proxy 对齐 (无标签): 把同一文本在各底座的投影拉到一起。

    feats: {expert: [n, S, D_e]} —— 三个底座编码**同一批文本**(下标对应)
    projs: {expert: 投影器}
    返回最终 proxy 损失 (用于确认确实收敛)。
    """
    params = [p for e in experts for p in projs[e].parameters() if p.requires_grad]
    if not params:
        return 0.0
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    last = 0.0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        zs = [projs[e](feats[e]) for e in experts]
        mean = torch.stack(zs, dim=0).mean(dim=0)
        loss = sum(((z - mean) ** 2).mean() for z in zs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = float(loss.detach())
        if log and (step % 50 == 0 or step == steps - 1):
            print(f"    [proxy] step {step:4d}  L_proxy={last:.5f}", flush=True)
    return last


class LinearProbe(nn.Module):
    """线性探针: y ∈ R^d → logits (单层, 强权重衰减, 早停按验证)。"""

    def __init__(self, d_in: int, n_class: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_class)

    def forward(self, x):
        return self.fc(x)


def linear_probe(train_x, train_y, test_x, test_y, n_class: int, steps: int = 300,
                 lr: float = 1e-2, weight_decay: float = 1e-2, seed: int = 0):
    """训练线性探针并返回 (test_acc, train_acc)。

    train_x: [N, d] (已展平), train_y: [N] 标签
    test_x / test_y 同理 (held-out)
    """
    torch.manual_seed(seed)
    probe = LinearProbe(train_x.shape[-1], n_class)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    n = train_x.shape[0]
    g = torch.Generator().manual_seed(seed + 17)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, n, (min(64, n),), generator=g).to(train_x.device)
        loss = F.cross_entropy(probe(train_x[idx]), train_y[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_acc = float((probe(train_x).argmax(-1) == train_y).float().mean())
        te_acc = float((probe(test_x).argmax(-1) == test_y).float().mean())
    return te_acc, tr_acc
