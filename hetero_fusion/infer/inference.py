"""§5 — 推理部署规范：容量与溢出。

实现容量感知推理路径 (§5.1 容量因子 / §5.2 溢出降级 + 动态重归一化)
以及 §8.3 推理基准测试 (C=1.0/1.25/1.5 对照)。
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..core.fusion import HeteroFusionLayer
from ..core.router import capacity_route


class InferenceRunner:
    """容量感知推理执行器 (§5)。

    训练期路由用 SparseRouterSTE (硬 Top-K + STE)；推理期改用 capacity_route
    (容量因子 + 溢出降级 + 重归一化)，二者通过 HeteroFusionLayer 的
    routed_weights 入口注入同一份融合权重，避免维护两份逻辑。
    """

    def __init__(
        self,
        layers: Sequence[HeteroFusionLayer],
        capacity_factor: float = 1.25,  # §5.1 推荐 C=1.25
        lm_head: Optional[Callable] = None,
    ):
        self.layers = list(layers)
        self.capacity_factor = capacity_factor
        self.lm_head = lm_head

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """单次推理前向, 返回 (logits, per_layer_stats)。"""
        y = x
        stats: List[Dict] = []
        for layer in self.layers:
            z = F.linear(y, layer.W_router)
            alpha = F.softmax(z, dim=-1)
            # §5.1/§5.2 容量感知路由
            alpha_tilde, valid, st = capacity_route(
                alpha, layer.top_k, self.capacity_factor
            )
            # 注入融合层 (推理路径, §5)
            y, _, _ = layer(y, routed_weights=alpha_tilde)
            stats.append(st)
        logits = self.lm_head(y) if self.lm_head is not None else y
        return logits, stats

    @torch.no_grad()
    def benchmark(
        self,
        x: torch.Tensor,
        capacity_factors: Sequence[float] = (1.0, 1.25, 1.5),  # §8.3 三组对照
        warmup: int = 2,
        repeat: int = 5,
    ) -> Dict[float, Dict]:
        """§8.3 推理基准: 测量溢出率与端到端延迟。

        确认动态重归一化逻辑是否避免了输出分布偏移 (§8.3 验收点)。
        """
        results: Dict[float, Dict] = {}
        prev_C = self.capacity_factor
        for C in capacity_factors:
            self.capacity_factor = float(C)
            # warmup
            for _ in range(warmup):
                self.forward(x)
            # timed
            times = []
            last_stats = None
            for _ in range(repeat):
                t0 = time.perf_counter()
                _, last_stats = self.forward(x)
                times.append(time.perf_counter() - t0)
            times_t = torch.tensor(times)
            # 聚合各层溢出统计
            total_tokens = last_stats[0]["total_tokens"]
            agg_overflow = sum((s["per_expert_overflow"] for s in last_stats), [])
            agg_req = sum((s["per_expert_requested"] for s in last_stats), [])
            skip_total = sum(s["skip_tokens"] for s in last_stats)
            results[float(C)] = {
                "capacity_factor": float(C),
                "capacity_per_expert": last_stats[0]["capacity_per_expert"],
                "latency_mean_ms": float(times_t.mean() * 1e3),
                "latency_std_ms": float(times_t.std(unbiased=False) * 1e3),
                "total_overflow_tokens": sum(agg_overflow),
                "total_requested_tokens": sum(agg_req),
                "overall_overflow_rate": float(
                    sum(agg_overflow) / max(1, sum(agg_req))
                ),
                "skip_tokens": skip_total,
                "skip_rate": float(skip_total / max(1, total_tokens)),
            }
        self.capacity_factor = prev_C
        return results
