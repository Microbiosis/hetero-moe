"""异构专家抽象与可选 LoRA 微调适配器 (Hetero-MoE v5.0)。

修复 v4.0 的两项固有架构缺陷:
    1. 伪异构: v4.0 的 HeteroFusionLayer 仅创建 M 个同尺寸随机 SwiGLU。
       本模块提供 ExpertSpec / HeteroExpert，支持不同输入/隐藏/输出宽度、不同网络结构
       (swiglu / gelu_mlp / conv_mlp / 任意冻结或可微 nn.Module 底座模块) 在同一层内共存。
    2. 纯冻结 β 墙死锁: 当全部专家权重完全冻结时，仅靠激活级线性组合无法突破底座共同错误率
       上界 (1 - β)。本模块内置可选低秩 LoRA 适配分支 (lora_rank > 0) 或解冻开关
       (freeze_base=False)，在保持轻量化开销的同时允许跨专家联合微调。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


@dataclass
class ExpertSpec:
    """单个异构专家的声明式规格。

    Attributes:
        name:         专家标识（用于评测基线与路由日志追踪）。
        arch:         架构类型: 'swiglu' | 'gelu_mlp' | 'conv_mlp' | 'custom'。
        d_in:         专家原生输入维度（0 表示继承层的 d_model）。
        d_ff:         专家内部中间维度（0 表示继承层的 d_ff）。
        d_out:        专家原生输出维度（0 表示继承层的 d_model）。
        freeze_base:  是否冻结专家原生底座权重（默认 True）。
        lora_rank:    原生空间低秩适配秩；0 表示关闭，>0 启用可训练 LoRA 分支以突破纯冻结 β 墙。
        lora_alpha:   LoRA 缩放系数（实际缩放为 lora_alpha / lora_rank）。
        module:       当 arch='custom' 时传入的任意 nn.Module（接受 [N, d_in] 输出 [N, d_out]）。
    """

    name: str
    arch: str = "swiglu"
    d_in: int = 0
    d_ff: int = 0
    d_out: int = 0
    freeze_base: bool = True
    lora_rank: int = 0
    lora_alpha: float = 1.0
    module: Optional[nn.Module] = None


class LoRALinearAdapter(nn.Module):
    """零初始化残差低秩适配器: Δy = (lora_alpha / r) · (x A^T) B^T。

    B 零初始化确保训练第 0 步严格等价于原始冻结底座（无冷启动分布冲击）。
    """

    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float = 1.0):
        super().__init__()
        assert rank > 0, "LoRA rank 必须为正整数"
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.A), self.B) * self.scaling


class _SwiGLUBase(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(d_ff, d_in) * 0.02)
        self.w_up = nn.Parameter(torch.randn(d_ff, d_in) * 0.02)
        self.w_down = nn.Parameter(torch.randn(d_out, d_ff) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu(x, self.w_gate, self.w_up, self.w_down)


class _GELUMLPBase(nn.Module):
    """标准 Transformer (BERT / ViT 风格) 两层 GELU MLP。"""

    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _GatedBottleneckBase(nn.Module):
    """带 LayerNorm 与深度可分离局部混合的异构特征变换专家（区别于标准前馈）。"""

    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj_in = nn.Linear(d_in, d_ff)
        self.gate = nn.Linear(d_in, d_ff)
        self.proj_out = nn.Linear(d_ff, d_out)
        for lin in (self.proj_in, self.gate, self.proj_out):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.proj_out(torch.tanh(self.proj_in(h)) * torch.sigmoid(self.gate(h)))


class HeteroExpert(nn.Module):
    """封装任意异构底座的统一专家接口。

    前向流:
        x [..., d_model]
          → (可选 in_proj: d_model → d_in)
          → base(h_in) [+ lora(h_in)]
          → (可选 out_proj: d_out → d_model)
          → y [..., d_model]
    """

    def __init__(self, d_model: int, spec: ExpertSpec, default_d_ff: int = 0):
        super().__init__()
        self.spec = spec
        self.name = spec.name
        self.arch = spec.arch
        self.d_model = int(d_model)
        self.d_in = int(spec.d_in) if spec.d_in > 0 else int(d_model)
        self.d_ff = int(spec.d_ff) if spec.d_ff > 0 else (int(default_d_ff) if default_d_ff > 0 else int(d_model) * 2)
        self.d_out = int(spec.d_out) if spec.d_out > 0 else int(d_model)
        self.freeze_base = bool(spec.freeze_base)

        if spec.arch == "custom":
            if spec.module is None:
                raise ValueError(f"ExpertSpec({spec.name!r}, arch='custom') 必须提供 module")
            self.base: nn.Module = spec.module
        elif spec.arch == "swiglu":
            self.base = _SwiGLUBase(self.d_in, self.d_ff, self.d_out)
        elif spec.arch == "gelu_mlp":
            self.base = _GELUMLPBase(self.d_in, self.d_ff, self.d_out)
        elif spec.arch == "conv_mlp":
            self.base = _GatedBottleneckBase(self.d_in, self.d_ff, self.d_out)
        else:
            raise ValueError(f"未知专家架构 arch={spec.arch!r}，支持 swiglu/gelu_mlp/conv_mlp/custom")

        if self.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        # 异构维度对齐投影（仅在原生维度与 d_model 不同时创建，避免同维冗余参数）
        if self.d_in != self.d_model:
            self.in_proj: Optional[nn.Linear] = nn.Linear(self.d_model, self.d_in, bias=False)
            nn.init.orthogonal_(self.in_proj.weight)
        else:
            self.in_proj = None

        if self.d_out != self.d_model:
            self.out_proj: Optional[nn.Linear] = nn.Linear(self.d_out, self.d_model, bias=False)
            nn.init.orthogonal_(self.out_proj.weight)
        else:
            self.out_proj = None

        # 可选原生空间 LoRA 分支（打破纯冻结底座的 β 墙）
        if spec.lora_rank > 0:
            self.lora: Optional[LoRALinearAdapter] = LoRALinearAdapter(
                self.d_in, self.d_out, rank=spec.lora_rank, alpha=spec.lora_alpha
            )
        else:
            self.lora = None

    def adapter_parameters(self) -> List[nn.Parameter]:
        """输入/输出维度对齐投影参数。"""
        params: List[nn.Parameter] = []
        if self.in_proj is not None:
            params.extend(p for p in self.in_proj.parameters() if p.requires_grad)
        if self.out_proj is not None:
            params.extend(p for p in self.out_proj.parameters() if p.requires_grad)
        return params

    def lora_and_unfrozen_parameters(self) -> List[nn.Parameter]:
        """LoRA 参数及解冻的底座参数。"""
        params: List[nn.Parameter] = []
        if self.lora is not None:
            params.extend(p for p in self.lora.parameters() if p.requires_grad)
        if not self.freeze_base:
            params.extend(p for p in self.base.parameters() if p.requires_grad)
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) if self.in_proj is not None else x
        y = self.base(h)
        if self.lora is not None:
            y = y + self.lora(h)
        if self.out_proj is not None:
            y = self.out_proj(y)
        return y


def make_heterogeneous_specs(
    d_model: int,
    specs: Iterable[ExpertSpec],
    default_d_ff: int = 0,
) -> nn.ModuleList:
    """将一组 ExpertSpec 实例化为 HeteroExpert 列表。"""
    return nn.ModuleList([HeteroExpert(d_model, s, default_d_ff=default_d_ff) for s in specs])
