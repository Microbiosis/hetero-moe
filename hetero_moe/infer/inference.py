"""§5 / v5.0 — 推理引擎：容量感知调度、无重复路由计算与多层口径校准基准。

v5.0 修复:
    1. 消除重复路由矩阵乘: forward 将预计算的 (alpha_tilde, z) 一并传给 layer，
       HeteroFusionLayer 不再二次计算 F.linear(y, W_router)。
    2. 修复多层 skip_rate 越界 bug: v4.0 将 L 层 skip_tokens 求和后除以单层 total_tokens，
       多层时可能 > 1.0；v5.0 除以 total_tokens * len(last_stats)，严格保证 skip_rate ∈ [0, 1]，
       并同时输出逐层 skip_rate 与实际稀疏计算 token 数。
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..core.fusion import HeteroFusionLayer
from ..core.router import capacity_route


class InferenceRunner:
    """容量感知推理执行器 (§5 / v5.0)。"""

    def __init__(
        self,
        layers: Sequence[HeteroFusionLayer],
        capacity_factor: float = 1.25,
        lm_head: Optional[Callable] = None,
    ):
        assert capacity_factor > 0, "capacity_factor 必须为正"
        self.layers = list(layers)
        self.capacity_factor = float(capacity_factor)
        self.lm_head = lm_head

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """单次推理前向, 返回 (logits, per_layer_stats)。"""
        y = x
        stats: List[Dict] = []
        for layer in self.layers:
            z = F.linear(y, layer.W_router)
            alpha = F.softmax(z, dim=-1)
            alpha_tilde, _valid, st = capacity_route(
                alpha, layer.top_k, self.capacity_factor
            )
            # 同时传入 routed_weights 与 routed_logits，彻底避免层内二次计算路由投影
            y, _, _ = layer(y, routed_weights=alpha_tilde, routed_logits=z)
            st = dict(st)
            st["active_expert_tokens"] = sum(layer.last_expert_token_counts)
            stats.append(st)
        logits = self.lm_head(y) if self.lm_head is not None else y
        return logits, stats

    @torch.no_grad()
    def benchmark(
        self,
        x: torch.Tensor,
        capacity_factors: Sequence[float] = (1.0, 1.25, 1.5),
        warmup: int = 2,
        repeat: int = 5,
    ) -> Dict[float, Dict]:
        """§8.3 推理基准: 测量溢出率、多层口径正确的 skip_rate、稀疏计算量与延迟。"""
        results: Dict[float, Dict] = {}
        prev_C = self.capacity_factor
        for C in capacity_factors:
            self.capacity_factor = float(C)
            for _ in range(warmup):
                self.forward(x)
            times = []
            last_stats = None
            for _ in range(repeat):
                t0 = time.perf_counter()
                _, last_stats = self.forward(x)
                times.append(time.perf_counter() - t0)
            times_t = torch.tensor(times)
            n_layers = max(1, len(last_stats))
            total_tokens = last_stats[0]["total_tokens"]
            total_layer_tokens = total_tokens * n_layers
            agg_overflow = sum((s["per_expert_overflow"] for s in last_stats), [])
            agg_req = sum((s["per_expert_requested"] for s in last_stats), [])
            skip_total = sum(s["skip_tokens"] for s in last_stats)
            per_layer_skip = [
                float(s["skip_tokens"] / max(1, s["total_tokens"])) for s in last_stats
            ]
            active_expert_tokens = sum(s.get("active_expert_tokens", 0) for s in last_stats)
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
                "skip_rate": float(skip_total / max(1, total_layer_tokens)),
                "per_layer_skip_rates": per_layer_skip,
                "active_expert_tokens": active_expert_tokens,
            }
        self.capacity_factor = prev_C
        return results
