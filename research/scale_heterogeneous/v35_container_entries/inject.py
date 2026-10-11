"""v35.0-α — 注入头与投影工具。

ProjHead: D_BIG → D_shared 可训练投影头 (入口注入 + baseline 共用结构)
make_P:   正交投影 D_in → D_out (冻结)
"""
import torch
import torch.nn as nn


class ProjHead(nn.Module):
    """投影头: D_BIG → D_shared, 可训练 (入口注入后 + baseline 共用结构)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        return self.fc(x)


def make_P(d_in, d_out):
    """正交投影 D_in → D_out (冻结)."""
    P = nn.Parameter(torch.empty(d_out, d_in))
    nn.init.orthogonal_(P)
    P.requires_grad_(False)
    return P
