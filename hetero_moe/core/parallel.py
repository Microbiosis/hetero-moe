"""§6.1 — 并行策略约束。

规范要点:
    - M=4 等小规模系统禁用专家并行 (EP)，优先张量并行 (TP)。
    - 将每个专家的 W_gate/W_up/W_down 沿 D_ff 维拆分到多卡，避免跨节点 All-to-All。
    - 仅当 M >= 8 或单专家参数量超单卡显存时才启用 EP。

本模块提供 TP 分块工具与 EP/TP 决策函数。真实多卡部署需配合 device_mesh /
nccl，此处提供单设备可验证的逻辑切片（返回各卡分片张量列表）。
"""
from __future__ import annotations

from typing import List, Tuple

import torch

# §6.1 阈值
EP_THRESHOLD = 8


def tensor_parallel_split(
    weight: torch.Tensor, num_shards: int, dim: int
) -> Tuple[torch.Tensor, ...]:
    """沿指定维将权重均匀切分为 num_shards 份 (§6.1)。

    约定:
        W_gate / W_up: [d_ff, d_model] → 沿 dim=0 (d_ff) 拆分
        W_down:       [d_model, d_ff] → 沿 dim=1 (d_ff) 拆分
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
    """§6.1 EP/TP 决策。

    Returns:
        (strategy, rationale)
    """
    if num_experts >= EP_THRESHOLD or single_expert_bytes > single_gpu_bytes:
        return "EP", f"M={num_experts}≥{EP_THRESHOLD} 或单专家超单卡显存"
    return "TP", f"M={num_experts}<{EP_THRESHOLD}, 小规模优先 TP 避免 All-to-All"


def tp_shard_ffn(
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, num_shards: int
) -> List[dict]:
    """对一个专家的 FFN 三权重做 TP 切分 (§6.1)。

    Returns:
        list[dict]: 每张卡持有 {w_gate_shard, w_up_shard, w_down_shard}。
    """
    gates = tensor_parallel_split(w_gate, num_shards, dim=0)  # [d_ff/shard, d_model]
    ups = tensor_parallel_split(w_up, num_shards, dim=0)
    downs = tensor_parallel_split(w_down, num_shards, dim=1)  # [d_model, d_ff/shard]
    return [
        {"w_gate": g, "w_up": u, "w_down": d}
        for g, u, d in zip(gates, ups, downs)
    ]
