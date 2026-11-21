"""异构专家抽象与内部子层 LoRA 微调适配器 (Hetero-MoE v5.1)。

修复 v4.0 / v5.0 的架构缺陷:
    1. 伪异构: v4.0 的 HeteroFusionLayer 仅创建 M 个同尺寸随机 SwiGLU。
       本模块提供 ExpertSpec / HeteroExpert，支持不同输入/隐藏/输出宽度、不同网络结构
       (swiglu / gelu_mlp / conv_mlp / 任意冻结或可微 nn.Module 底座模块) 在同一层内共存。
    2. 纯冻结 β 墙死锁与表层线性旁路缺陷:
       - 当全部专家权重完全冻结时，仅靠激活级线性组合无法突破底座共同错误率上界 (1 - β)。
       - v5.1 将 LoRA 从专家外围的线性跳跃旁路深入到专家**内部非线性子层**
         (internal_lora=True: 直接调制 SwiGLU 的 w_gate/w_up/w_down、GELU MLP 的 fc1/fc2、
          GatedBottleneck 的 proj_in/gate/proj_out 以及 custom 模块内部的全部 nn.Linear 子层)，
         同时保持 self.base.parameters() 严格只含冻结底座参数。
    3. 专家权重级 INT4 伪量化支持:
       支持在专家内部矩阵乘前对冻结底座权重施加 FakeQuantSTE，对齐真实 MoE 权重压缩误差。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import FakeQuantSTE


def _maybe_quant_weight(w: torch.Tensor, quant_bits: int, quant_group_size: int) -> torch.Tensor:
    if quant_bits > 0 and quant_bits < 16:
        return FakeQuantSTE.apply(w, quant_bits, quant_group_size)
    return w


def _swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    sublayer_loras: Optional[nn.ModuleDict] = None,
    quant_bits: int = 0,
    quant_group_size: int = 128,
) -> torch.Tensor:
    wg = _maybe_quant_weight(w_gate, quant_bits, quant_group_size)
    wu = _maybe_quant_weight(w_up, quant_bits, quant_group_size)
    wd = _maybe_quant_weight(w_down, quant_bits, quant_group_size)
    h_gate = F.linear(x, wg)
    h_up = F.linear(x, wu)
    if sublayer_loras is not None:
        if "w_gate" in sublayer_loras:
            h_gate = h_gate + sublayer_loras["w_gate"](x)
        if "w_up" in sublayer_loras:
            h_up = h_up + sublayer_loras["w_up"](x)
    h_act = F.silu(h_gate) * h_up
    out = F.linear(h_act, wd)
    if sublayer_loras is not None and "w_down" in sublayer_loras:
        out = out + sublayer_loras["w_down"](h_act)
    return out


@dataclass
class ExpertSpec:
    """单个异构专家的声明式规格。

    Attributes:
        name:          专家标识（用于评测基线与路由日志追踪）。
        arch:          架构类型: 'swiglu' | 'gelu_mlp' | 'conv_mlp' | 'custom'。
        d_in:          专家原生输入维度（0 表示继承层的 d_model）。
        d_ff:          专家内部中间维度（0 表示继承层的 d_ff）。
        d_out:         专家原生输出维度（0 表示继承层的 d_model）。
        freeze_base:   是否冻结专家原生底座权重（默认 True）。
        lora_rank:     原生空间低秩适配秩；0 表示关闭，>0 启用可训练 LoRA 分支以突破纯冻结 β 墙。
        lora_alpha:    LoRA 缩放系数（实际缩放为 lora_alpha / lora_rank）。
        module:        当 arch='custom' 时传入的任意 nn.Module（接受 [N, d_in] 输出 [N, d_out]）。
        internal_lora: 是否将 LoRA 注入到专家内部非线性子层（v5.1 默认 True）。
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
    internal_lora: bool = True


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

    def forward(
        self,
        x: torch.Tensor,
        sublayer_loras: Optional[nn.ModuleDict] = None,
        quant_bits: int = 0,
        quant_group_size: int = 128,
    ) -> torch.Tensor:
        return _swiglu(x, self.w_gate, self.w_up, self.w_down, sublayer_loras, quant_bits, quant_group_size)


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

    def forward(
        self,
        x: torch.Tensor,
        sublayer_loras: Optional[nn.ModuleDict] = None,
        quant_bits: int = 0,
        quant_group_size: int = 128,
    ) -> torch.Tensor:
        w1 = _maybe_quant_weight(self.fc1.weight, quant_bits, quant_group_size)
        w2 = _maybe_quant_weight(self.fc2.weight, quant_bits, quant_group_size)
        h1 = F.linear(x, w1, self.fc1.bias)
        if sublayer_loras is not None and "fc1" in sublayer_loras:
            h1 = h1 + sublayer_loras["fc1"](x)
        h_act = F.gelu(h1)
        out = F.linear(h_act, w2, self.fc2.bias)
        if sublayer_loras is not None and "fc2" in sublayer_loras:
            out = out + sublayer_loras["fc2"](h_act)
        return out


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

    def forward(
        self,
        x: torch.Tensor,
        sublayer_loras: Optional[nn.ModuleDict] = None,
        quant_bits: int = 0,
        quant_group_size: int = 128,
    ) -> torch.Tensor:
        h = self.norm(x)
        wi = _maybe_quant_weight(self.proj_in.weight, quant_bits, quant_group_size)
        wg = _maybe_quant_weight(self.gate.weight, quant_bits, quant_group_size)
        wo = _maybe_quant_weight(self.proj_out.weight, quant_bits, quant_group_size)
        p_in = F.linear(h, wi, self.proj_in.bias)
        g_in = F.linear(h, wg, self.gate.bias)
        if sublayer_loras is not None:
            if "proj_in" in sublayer_loras:
                p_in = p_in + sublayer_loras["proj_in"](h)
            if "gate" in sublayer_loras:
                g_in = g_in + sublayer_loras["gate"](h)
        h_mid = torch.tanh(p_in) * torch.sigmoid(g_in)
        out = F.linear(h_mid, wo, self.proj_out.bias)
        if sublayer_loras is not None and "proj_out" in sublayer_loras:
            out = out + sublayer_loras["proj_out"](h_mid)
        return out


class HeteroExpert(nn.Module):
    """封装任意异构底座的统一专家接口 (v5.1)。

    前向流:
        x [..., d_model]
          → (可选 in_proj: d_model → d_in)
          → base(h_in, sublayer_loras=...) [+ 外围残差 lora(h_in)]
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
        self.internal_lora = bool(spec.internal_lora)

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

        # 可选原生空间 LoRA 分支（包含深入非线性子层的 sublayer_loras 与外围残差 lora）
        self.sublayer_loras: Optional[nn.ModuleDict] = None
        self._custom_linear_keys: List[Tuple[str, nn.Linear]] = []
        if spec.lora_rank > 0:
            self.lora: Optional[LoRALinearAdapter] = LoRALinearAdapter(
                self.d_in, self.d_out, rank=spec.lora_rank, alpha=spec.lora_alpha
            )
            if self.internal_lora:
                r, a = spec.lora_rank, spec.lora_alpha
                if spec.arch == "swiglu":
                    self.sublayer_loras = nn.ModuleDict({
                        "w_gate": LoRALinearAdapter(self.d_in, self.d_ff, r, a),
                        "w_up": LoRALinearAdapter(self.d_in, self.d_ff, r, a),
                        "w_down": LoRALinearAdapter(self.d_ff, self.d_out, r, a),
                    })
                elif spec.arch == "gelu_mlp":
                    self.sublayer_loras = nn.ModuleDict({
                        "fc1": LoRALinearAdapter(self.d_in, self.d_ff, r, a),
                        "fc2": LoRALinearAdapter(self.d_ff, self.d_out, r, a),
                    })
                elif spec.arch == "conv_mlp":
                    self.sublayer_loras = nn.ModuleDict({
                        "proj_in": LoRALinearAdapter(self.d_in, self.d_ff, r, a),
                        "gate": LoRALinearAdapter(self.d_in, self.d_ff, r, a),
                        "proj_out": LoRALinearAdapter(self.d_ff, self.d_out, r, a),
                    })
                elif spec.arch == "custom":
                    sub_dict: Dict[str, nn.Module] = {}
                    for nm, mod in self.base.named_modules():
                        if isinstance(mod, nn.Linear):
                            key = nm.replace(".", "__") if nm else "root_linear"
                            sub_dict[key] = LoRALinearAdapter(mod.in_features, mod.out_features, r, a)
                            self._custom_linear_keys.append((key, mod))
                    if sub_dict:
                        self.sublayer_loras = nn.ModuleDict(sub_dict)
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
        """内部子层 LoRA、外围 LoRA 参数及解冻的底座参数。"""
        params: List[nn.Parameter] = []
        if self.sublayer_loras is not None:
            params.extend(p for p in self.sublayer_loras.parameters() if p.requires_grad)
        if self.lora is not None:
            params.extend(p for p in self.lora.parameters() if p.requires_grad)
        if not self.freeze_base:
            params.extend(p for p in self.base.parameters() if p.requires_grad)
        return params

    def forward(
        self,
        x: torch.Tensor,
        quant_weight_bits: int = 0,
        quant_group_size: int = 128,
    ) -> torch.Tensor:
        h = self.in_proj(x) if self.in_proj is not None else x
        if self.arch in ("swiglu", "gelu_mlp", "conv_mlp"):
            y = self.base(
                h,
                sublayer_loras=self.sublayer_loras,
                quant_bits=quant_weight_bits,
                quant_group_size=quant_group_size,
            )
        else:
            # custom 模块: 通过瞬时 forward hook 将内部子层 LoRA 注入到每个 nn.Linear 子模块
            handles = []
            if self.sublayer_loras is not None and self._custom_linear_keys:
                for key, lin_mod in self._custom_linear_keys:
                    adapter = self.sublayer_loras[key]

                    def _make_hook(adp: nn.Module):
                        def _hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
                            return out + adp(inp[0])

                        return _hook

                    handles.append(lin_mod.register_forward_hook(_make_hook(adapter)))
            try:
                y = self.base(h)
            finally:
                for hdl in handles:
                    hdl.remove()
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
    """按规格列表构建异构专家 ModuleList。"""
    return nn.ModuleList([HeteroExpert(d_model, s, default_d_ff=default_d_ff) for s in specs])
