"""hetero_moe.core.interface — 跨架构/跨模态局部化去常数坍缩与幅度锁相接口算子 (v5.1)。

统一解决主线 (v77–v83 Qwen L3/L4 Massive Activations: max=816.0, top-1 跨样本常数方向占比 >95%)
与幻脑线 (brain_domains exp102/105/106: 条件 token 跨概念一致性 0.9905、范数 76.41 vs 宿主 0.1)
共同暴露的数学瓶颈:
    1. 局部功能区/子空间分块编码 (Localized Functional Modules)
    2. 跨样本公共常数方向正交剥离 (Common-Mode Orthogonal Stripping, 支持 Batch + EMA 推理统计)
    3. 宿主目标模态 RMS 幅度锁相 (Target-Modality RMS Phase-Locking)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeCollapsedLocalInterface(nn.Module):
    """局部化 + 去常数坍缩 + 跨模态幅度锁相的可微稀疏接口算子 (主线与幻脑线共用核心组件)。"""

    def __init__(
        self,
        d_brain: int = 1024,
        n_tokens: int = 16,
        d_tok: int = 768,
        num_regions: int = 4,
        bottleneck_rank: int = 32,
        common_mode_rank: int = 1,
        default_target_norm: float = 1.0,
        gain_bound: float = 0.25,
        ema_momentum: float = 0.05,
    ) -> None:
        super().__init__()
        assert d_brain >= num_regions and d_brain % num_regions == 0, "d_brain 需能被 num_regions 整除"
        assert n_tokens >= num_regions and n_tokens % num_regions == 0, "n_tokens 需能被 num_regions 整除"
        assert bottleneck_rank >= 1 and common_mode_rank >= 0

        self.d_brain = int(d_brain)
        self.n_tokens = int(n_tokens)
        self.d_tok = int(d_tok)
        self.num_regions = int(num_regions)
        self.d_region = self.d_brain // self.num_regions
        self.tokens_per_region = self.n_tokens // self.num_regions
        self.bottleneck_rank = int(bottleneck_rank)
        self.common_mode_rank = int(common_mode_rank)
        self.default_target_norm = float(default_target_norm)
        self.gain_bound = float(gain_bound)
        self.ema_momentum = float(ema_momentum)

        self.region_norms = nn.ModuleList([nn.LayerNorm(self.d_region) for _ in range(self.num_regions)])
        self.region_down = nn.ModuleList(
            [nn.Linear(self.d_region, self.bottleneck_rank, bias=False) for _ in range(self.num_regions)]
        )
        self.region_mid = nn.ModuleList(
            [nn.Linear(self.bottleneck_rank, self.bottleneck_rank) for _ in range(self.num_regions)]
        )
        self.region_up = nn.ModuleList(
            [
                nn.Linear(self.bottleneck_rank, self.tokens_per_region * self.d_tok, bias=False)
                for _ in range(self.num_regions)
            ]
        )
        for down, mid, up in zip(self.region_down, self.region_mid, self.region_up):
            nn.init.orthogonal_(down.weight)
            nn.init.orthogonal_(mid.weight)
            nn.init.zeros_(mid.bias)
            nn.init.orthogonal_(up.weight)

        self.region_gates = nn.Parameter(torch.ones(self.num_regions, 1, 1))
        self.slot_gain_raw = nn.Parameter(torch.zeros(1, self.n_tokens, 1))
        self.slot_residual_proj = nn.Linear(self.d_region, self.d_tok, bias=False)
        nn.init.orthogonal_(self.slot_residual_proj.weight)

        self.register_buffer("ema_mean", torch.zeros(1, self.n_tokens, self.d_tok))
        self.register_buffer("ema_spread", torch.ones(1, self.n_tokens, 1))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

    def _encode_regions(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z_regions = z.reshape(B, self.num_regions, self.d_region)
        region_tokens = []
        for g in range(self.num_regions):
            zr = self.region_norms[g](z_regions[:, g, :])
            h = F.gelu(self.region_mid[g](self.region_down[g](zr)))
            tok_g = self.region_up[g](h).reshape(B, self.tokens_per_region, self.d_tok)
            anchor = self.slot_residual_proj(zr).unsqueeze(1)
            tok_g = (tok_g + 0.5 * anchor) * self.region_gates[g]
            region_tokens.append(tok_g)
        return torch.cat(region_tokens, dim=1)

    def strip_common_mode_and_lock_norm(
        self,
        raw_tokens: torch.Tensor,
        target_token_norm: Optional[float] = None,
        host_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, K, _ = raw_tokens.shape
        if B >= 2:
            batch_mean = raw_tokens.mean(dim=0, keepdim=True)
            dev = raw_tokens - batch_mean
            if self.training:
                with torch.no_grad():
                    if not bool(self.ema_initialized.item()):
                        self.ema_mean.copy_(batch_mean)
                        self.ema_initialized.fill_(True)
                    else:
                        self.ema_mean.lerp_(batch_mean, self.ema_momentum)
        else:
            dev = raw_tokens - self.ema_mean

        if self.common_mode_rank > 0 and B >= 4 and K >= 2:
            slot_common = dev.mean(dim=1, keepdim=True)
            dev = dev - 0.35 * (slot_common - slot_common.mean(dim=0, keepdim=True))

        tok_norms = dev.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if B >= 2 and self.training:
            with torch.no_grad():
                self.ema_spread.lerp_(tok_norms.mean(dim=0, keepdim=True), self.ema_momentum)

        unit_tok = dev / tok_norms
        if B >= 2:
            unit_tok = unit_tok - unit_tok.mean(dim=0, keepdim=True)
            unit_tok = unit_tok / unit_tok.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if host_tokens is not None:
            tgt_norm = float(host_tokens.detach().norm(dim=-1).median().item())
        elif target_token_norm is not None:
            tgt_norm = float(target_token_norm)
        else:
            tgt_norm = self.default_target_norm

        gain = 1.0 + self.gain_bound * torch.tanh(self.slot_gain_raw[:, :K, :])
        locked_tokens = unit_tok * (tgt_norm * gain)

        with torch.no_grad():
            mu_after = locked_tokens.mean(dim=0)
            norm_after = locked_tokens.norm(dim=-1)
            consistency_after = float(
                (mu_after.norm(dim=-1) / norm_after.mean(dim=0).clamp(min=1e-8)).mean().item()
            )
            median_norm_after = float(norm_after.median().item())
            amp_ratio = float(median_norm_after / max(tgt_norm, 1e-8))

        diag = {
            "consistency_after": consistency_after,
            "token_norm_median": median_norm_after,
            "target_norm": tgt_norm,
            "amplitude_ratio": amp_ratio,
        }
        return locked_tokens, diag

    def forward(
        self,
        z: torch.Tensor,
        target_token_norm: Optional[float] = None,
        host_tokens: Optional[torch.Tensor] = None,
        return_diag: bool = False,
    ):
        raw_tokens = self._encode_regions(z)
        locked, diag = self.strip_common_mode_and_lock_norm(
            raw_tokens,
            target_token_norm=target_token_norm,
            host_tokens=host_tokens,
        )
        if return_diag:
            return locked, diag
        return locked
