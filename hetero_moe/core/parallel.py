"""§6.1 / v5.1 — 并行策略约束与张量并行 (TP) 可微前向归约算子。

规范要点:
    - M=4 等小规模系统禁用专家并行 (EP)，优先张量并行 (TP)。
    - 将每个专家的 W_gate/W_up/W_down 沿 D_ff 维拆分到多卡，避免跨节点 All-to-All。
    - 仅当 M >= 8 或单专家参数量超单卡显存时才启用 EP。

v5.1 升级:
    - 不仅提供权重切分，还提供真正可执行前向/反向归约的 `tp_swiglu_forward` 与 `TensorParallelSwiGLU`：
      列并行切分 W_gate / W_up (沿 d_ff)，行并行切分 W_down (沿 d_ff)，并在输出端执行
      `torch.distributed.all_reduce`（当传入已初始化的分布式 process_group 时）或本地精确分片求和归约，
      在数学上与未切分 `swiglu_forward` 严格等价。
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# §6.1 阈值
EP_THRESHOLD = 8


def tensor_parallel_split(
    weight: torch.Tensor, num_shards: int, dim: int
) -> Tuple[torch.Tensor, ...]:
    """沿指定维将权重均匀切分为 num_shards 份 (§6.1)。

    约定:
        W_gate / W_up: [d_ff, d_model] → 沿 dim=0 (d_ff) 拆分 (Column-Parallel)
        W_down:       [d_model, d_ff] → 沿 dim=1 (d_ff) 拆分 (Row-Parallel)
    拆分后各卡只持有 D_ff/M_shard 列(行)的子矩阵，前向分块计算后 All-Reduce。
    """
    assert weight.shape[dim] % num_shards == 0, (
        f"权重第 {dim} 维 {weight.shape[dim]} 不能被 {num_shards} 整除, "
        "TP 切分要求 D_ff 为 shard 数整数倍。"
    )
    return tuple(torch.chunk(weight, num_shards, dim=dim))


def recommend_strategy(
    num_experts: int, single_expert_bytes: int, single_gpu_bytes: int
) -> Tuple[str, str]:
    """§6.1 EP/TP 决策。"""
    if num_experts >= EP_THRESHOLD or single_expert_bytes > single_gpu_bytes:
        return "EP", f"M={num_experts}≥{EP_THRESHOLD} 或单专家超单卡显存"
    return "TP", f"M={num_experts}<{EP_THRESHOLD}, 小规模优先 TP 避免 All-to-All"


def tp_shard_ffn(
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, num_shards: int
) -> List[dict]:
    """对一个专家的 FFN 三权重做 TP 切分 (§6.1)。"""
    gates = tensor_parallel_split(w_gate, num_shards, dim=0)  # [d_ff/shard, d_model]
    ups = tensor_parallel_split(w_up, num_shards, dim=0)      # [d_ff/shard, d_model]
    downs = tensor_parallel_split(w_down, num_shards, dim=1)  # [d_model, d_ff/shard]
    return [
        {"w_gate": g, "w_up": u, "w_down": d}
        for g, u, d in zip(gates, ups, downs)
    ]


def tp_swiglu_forward(
    x: torch.Tensor,
    shards: List[dict],
    process_group: Optional[Any] = None,
    local_rank: Optional[int] = None,
) -> torch.Tensor:
    """执行张量并行 (Column-Parallel Gate/Up + Row-Parallel Down + All-Reduce) SwiGLU 前向 (v5.1)。

    - 若处于已初始化的 `torch.distributed` 进程组且指定了 `local_rank`，仅计算当前 rank 的分片并调用
      `torch.distributed.all_reduce(partial_out, group=process_group)`。
    - 若在单设备验证模式，逐分片独立执行列并行激活与行并行下投影，再对分片部分和做 All-Reduce 归约。
    """
    assert len(shards) >= 1, "shards 不可为空"
    if (
        local_rank is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        sh = shards[int(local_rank)]
        h_act = F.silu(F.linear(x, sh["w_gate"])) * F.linear(x, sh["w_up"])
        partial = F.linear(h_act, sh["w_down"])
        torch.distributed.all_reduce(partial, group=process_group)
        return partial

    # 单设备多卡分片仿真 All-Reduce: Σ_s [ (SiLU(x W_gate^{(s)T}) ⊙ (x W_up^{(s)T})) W_down^{(s)T} ]
    partials = []
    for sh in shards:
        h_act = F.silu(F.linear(x, sh["w_gate"])) * F.linear(x, sh["w_up"])
        partials.append(F.linear(h_act, sh["w_down"]))
    return torch.stack(partials, dim=0).sum(dim=0)


class TensorParallelSwiGLU(nn.Module):
    """封装分片权重的张量并行 SwiGLU 模块 (v5.1)。"""

    def __init__(self, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, num_shards: int):
        super().__init__()
        self.num_shards = int(num_shards)
        raw_shards = tp_shard_ffn(w_gate, w_up, w_down, self.num_shards)
        self.w_gates = nn.ParameterList([nn.Parameter(s["w_gate"].clone()) for s in raw_shards])
        self.w_ups = nn.ParameterList([nn.Parameter(s["w_up"].clone()) for s in raw_shards])
        self.w_downs = nn.ParameterList([nn.Parameter(s["w_down"].clone()) for s in raw_shards])

    def forward(
        self,
        x: torch.Tensor,
        process_group: Optional[Any] = None,
        local_rank: Optional[int] = None,
    ) -> torch.Tensor:
        shards = [
            {"w_gate": g, "w_up": u, "w_down": d}
            for g, u, d in zip(self.w_gates, self.w_ups, self.w_downs)
        ]
        return tp_swiglu_forward(x, shards, process_group=process_group, local_rank=local_rank)
