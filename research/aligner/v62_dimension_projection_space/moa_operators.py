"""
System-Level MoA (Mixture-of-Agents / Mixture-of-Architectures / Mixture-of-Attention)
Underlying Neural Operator Library (PyTorch + Analytical NumPy Kernel Verification).

This module translates system-level MoA multi-agent workflows into end-to-end
differentiable, vectorized deep learning operators:

1. FusedProposerLinearFunction & BatchedHomogeneousProposerOp:
   Vectorized parallel proposal generation via custom autograd.Function and einsum
   contractions ([B, L, D] -> [B, N, L, D]), eliminating serial Python loops.
2. HeterogeneousProposerOp (Mixture-of-Architectures Proposer):
   Multi-inductive-bias proposer ensemble combining Causal Self-Attention,
   Selective State-Space (SSM / Gated Linear Recurrence), Causal Depthwise Conv1d,
   and SwiGLU Expert MLP with branch-wise RMSNorm and LayerScale.
3. CrossProposalAggregatorOp & SparseGatedAggregatorOp:
   Differentiable latent-space synthesizers mapping [B, N, L, D] -> [B, L, D] via
   O(L * N * D) Cross-Proposal Attention, Top-K Confidence-Gated Routing (with
   Load-Balancing + Router Z-Loss), or Gated Low-Rank Concatenation.
4. IterativeMoAOperatorBlock:
   Multi-stage Proposer-Aggregator refinement operator with token-level Adaptive
   Confidence Halting (differentiable early-stopping gate across MoA stages).
5. MixtureOfAttentionOp:
   Head-wise heterogeneous sparse attention operator combining Attention Sinks,
   Sliding Window, Dilated Strided, and Full Causal patterns.
6. NumpyMoAOperatorKernels:
   Hardware-agnostic analytical forward/backward operator kernels with central
   finite-difference gradient verification and FLOPs/parameter profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1. Unified Operator Configuration
# ==============================================================================

@dataclass
class MoAOperatorConfig:
    """Configuration for System-to-Operator MoA modules."""
    d_model: int = 256
    n_heads: int = 8
    num_proposers: int = 4
    top_k_proposers: int = 2
    d_ffn: int = 512
    d_state: int = 16
    conv_kernel_size: int = 4
    proposer_mode: Literal["homogeneous", "heterogeneous"] = "heterogeneous"
    aggregator_mode: Literal["cross_attention", "sparse_router", "low_rank_concat"] = "cross_attention"
    num_refinement_steps: int = 2
    share_refinement_weights: bool = False
    enable_adaptive_halting: bool = True
    halting_threshold: float = 0.90
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.001
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    # Mixture-of-Attention (MoA) sparse head config
    sliding_window_size: int = 8
    num_sink_tokens: int = 2
    dilation_stride: int = 2


# ==============================================================================
# 2. Analytical Forward & Backward Operator Kernels (NumPy Reference Engine)
# ==============================================================================

class NumpyMoAOperatorKernels:
    """
    Exact forward and analytical backward kernels for System-Level MoA operators.
    Used both as a reference implementation for custom CUDA/Triton kernel authoring
    and for deterministic numerical gradient verification (gradcheck).
    """

    @staticmethod
    def fused_proposer_linear_forward(
        x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Batched Multi-Proposer Linear Projection Forward Kernel.
        Args:
            x: [B, L, D_in]
            weight: [N, D_in, D_out]
            bias: Optional [N, D_out]
        Returns:
            out: [B, N, L, D_out]
        """
        out = np.einsum("bld,ndo->bnlo", x, weight)
        if bias is not None:
            out = out + bias[None, :, None, :]
        return out

    @staticmethod
    def fused_proposer_linear_backward(
        grad_out: np.ndarray,
        x: np.ndarray,
        weight: np.ndarray,
        has_bias: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Analytical Backward Kernel for Batched Multi-Proposer Linear Projection.
        Args:
            grad_out: [B, N, L, D_out]
            x: [B, L, D_in]
            weight: [N, D_in, D_out]
        Returns:
            grad_x: [B, L, D_in]
            grad_weight: [N, D_in, D_out]
            grad_bias: [N, D_out] or None
        """
        grad_x = np.einsum("bnlo,ndo->bld", grad_out, weight)
        grad_weight = np.einsum("bld,bnlo->ndo", x, grad_out)
        grad_bias = grad_out.sum(axis=(0, 2)) if has_bias else None
        return grad_x, grad_weight, grad_bias

    @staticmethod
    def branch_rmsnorm_forward(
        x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Branch-wise RMSNorm Forward Kernel on [B, N, L, D] or [B, L, D].
        Returns (normalized_output, inv_rms).
        """
        ms = np.mean(np.square(x), axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(ms + eps)
        out = x * inv_rms * gamma
        return out, inv_rms

    @staticmethod
    def branch_rmsnorm_backward(
        grad_out: np.ndarray,
        x: np.ndarray,
        gamma: np.ndarray,
        inv_rms: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical Backward Kernel for Branch-wise RMSNorm.
        """
        x_hat = x * inv_rms
        grad_gamma = np.sum(grad_out * x_hat, axis=tuple(range(x.ndim - 1)))
        dx_hat = grad_out * gamma
        grad_x = inv_rms * (
            dx_hat - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return grad_x, grad_gamma

    @staticmethod
    def cross_proposal_attention_forward(
        x_query: np.ndarray,
        proposals: np.ndarray,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        w_o: np.ndarray,
        n_heads: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Token-wise Cross-Proposal Attention Aggregator Forward Kernel.
        Complexity: O(B * L * N * D) instead of O(B * (L*N)^2 * D).

        Args:
            x_query: Backbone state [B, L, D]
            proposals: Parallel branch outputs [B, N, L, D]
            w_q, w_k, w_v, w_o: Projection matrices [D, D]
            n_heads: Number of attention heads H
        Returns:
            synthesized: [B, L, D]
            attn_weights: [B, L, H, 1, N]
        """
        B, N, L, D = proposals.shape
        d_head = D // n_heads
        scale = 1.0 / math.sqrt(d_head)

        # Query from backbone: [B, L, H, 1, d_head]
        q = (x_query @ w_q).reshape(B, L, n_heads, 1, d_head)

        # Keys & Values from N proposals: permute [B, N, L, D] -> [B, L, N, D]
        p_blnd = np.transpose(proposals, (0, 2, 1, 3))
        k = (p_blnd @ w_k).reshape(B, L, N, n_heads, d_head).transpose(0, 1, 3, 2, 4)
        v = (p_blnd @ w_v).reshape(B, L, N, n_heads, d_head).transpose(0, 1, 3, 2, 4)
        # k, v: [B, L, H, N, d_head]

        # Attention scores across the N proposals: [B, L, H, 1, N]
        logits = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        attn_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Weighted sum over N proposals: [B, L, H, 1, d_head] -> [B, L, D]
        agg = np.matmul(attn_weights, v).reshape(B, L, D)
        synthesized = agg @ w_o
        return synthesized, attn_weights

    @staticmethod
    def sparse_topk_router_forward(
        x_query: np.ndarray,
        proposals: np.ndarray,
        w_rq: np.ndarray,
        w_rk: np.ndarray,
        top_k: int,
        aux_weight: float = 0.01,
        z_weight: float = 0.001,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Confidence-Gated Sparse Top-K Proposal Router & Synthesizer Forward Kernel.
        Computes token-to-proposer affinity, selects Top-K proposers per token,
        and calculates Switch-Transformer Load Balancing Loss + Router Z-Loss.
        """
        B, N, L, D = proposals.shape
        q_r = (x_query @ w_rq) / math.sqrt(D)  # [B, L, D]
        k_r = np.einsum("bnld,de->bnle", proposals, w_rk)  # [B, N, L, D]
        router_logits = np.einsum("bld,bnld->bln", q_r, k_r)  # [B, L, N]

        # Router Z-Loss: mean(logsumexp(logits)^2)
        max_l = np.max(router_logits, axis=-1, keepdims=True)
        lse = np.squeeze(max_l, axis=-1) + np.log(
            np.sum(np.exp(router_logits - max_l), axis=-1)
        )
        z_loss = float(np.mean(np.square(lse)))

        # Full softmax probabilities
        exp_r = np.exp(router_logits - max_l)
        router_probs = exp_r / np.sum(exp_r, axis=-1, keepdims=True)  # [B, L, N]

        # Top-K selection per token
        topk_idx = np.argpartition(router_logits, -top_k, axis=-1)[..., -top_k:]
        topk_vals = np.take_along_axis(router_logits, topk_idx, axis=-1)
        max_tk = np.max(topk_vals, axis=-1, keepdims=True)
        exp_tk = np.exp(topk_vals - max_tk)
        topk_weights = exp_tk / np.sum(exp_tk, axis=-1, keepdims=True)  # [B, L, K]

        sparse_weights = np.zeros_like(router_probs)
        np.put_along_axis(sparse_weights, topk_idx, topk_weights, axis=-1)

        selection_mask = np.zeros_like(router_probs)
        np.put_along_axis(selection_mask, topk_idx, 1.0, axis=-1)

        # Load balancing loss: N * sum_i (f_i * P_i)
        f_i = np.mean(selection_mask, axis=(0, 1))  # [N]
        p_i = np.mean(router_probs, axis=(0, 1))    # [N]
        load_balance_loss = float(N * np.sum(f_i * p_i))
        total_aux_loss = aux_weight * load_balance_loss + z_weight * z_loss

        synthesized = np.einsum("bln,bnld->bld", sparse_weights, proposals)
        return synthesized, total_aux_loss, f_i

    @staticmethod
    def build_mixture_of_attention_mask(
        seq_len: int,
        n_heads: int,
        window_size: int = 8,
        num_sinks: int = 2,
        dilation: int = 2,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Constructs the heterogeneous head-wise boolean causal mask [1, H, L, L]
        for Mixture-of-Attention (MoA) and computes per-pattern density metrics.
        """
        idx = np.arange(seq_len)
        q_idx = idx[:, None]
        k_idx = idx[None, :]
        causal = q_idx >= k_idx

        sink_or_local = causal & ((k_idx < num_sinks) | ((q_idx - k_idx) <= window_size))
        dilated = causal & (((q_idx - k_idx) % dilation == 0) | ((q_idx - k_idx) <= 1))
        full_causal = causal

        masks = []
        for h in range(n_heads):
            mode = h % 3
            if mode == 0:
                masks.append(sink_or_local)
            elif mode == 1:
                masks.append(dilated)
            else:
                masks.append(full_causal)

        mask_tensor = np.stack(masks, axis=0)[None, ...]  # [1, H, L, L]
        causal_pairs = seq_len * (seq_len + 1) / 2.0
        stats = {
            "sink_local_density": float(sink_or_local.sum() / causal_pairs),
            "dilated_density": float(dilated.sum() / causal_pairs),
            "full_causal_density": 1.0,
            "overall_moa_density": float(mask_tensor.mean(axis=1).sum() / causal_pairs),
        }
        return mask_tensor, stats


# ==============================================================================
# 3. PyTorch Neural Operator Implementations (nn.Module & autograd.Function)
# ==============================================================================

if HAS_TORCH:

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization supporting arbitrary leading dimensions."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm_x = x.to(torch.float32)
            rms = torch.rsqrt(norm_x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (norm_x * rms).to(x.dtype) * self.weight


    class FusedProposerLinearFunction(torch.autograd.Function):
        """
        Custom PyTorch autograd operator projecting a shared representation X
        into N parallel proposer spaces via a single batched tensor contraction.

        Forward:
            X: [B, L, D_in], W: [N, D_in, D_out], bias: [N, D_out] -> Out: [B, N, L, D_out]
        """

        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            ctx.save_for_backward(x, weight, bias)
            out = torch.einsum("bld,ndo->bnlo", x, weight)
            if bias is not None:
                out = out + bias.unsqueeze(0).unsqueeze(2)
            return out

        @staticmethod
        def backward(
            ctx, grad_output: torch.Tensor
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            x, weight, bias = ctx.saved_tensors
            grad_x = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_x = torch.einsum("bnlo,ndo->bld", grad_output, weight)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.einsum("bld,bnlo->ndo", x, grad_output)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(dim=(0, 2))

            return grad_x, grad_weight, grad_bias


    def fused_proposer_linear(
        x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return FusedProposerLinearFunction.apply(x, weight, bias)


    class BatchedHomogeneousProposerOp(nn.Module):
        """
        Vectorized Homogeneous Proposer Operator.
        Executes N distinct SwiGLU proposer transformations in parallel using batched
        tensor contractions without Python loops.
        """

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.num_proposers = config.num_proposers
            self.d_model = config.d_model
            self.d_ffn = config.d_ffn

            self.w_gate_up = nn.Parameter(
                torch.randn(self.num_proposers, self.d_model, 2 * self.d_ffn)
                * (self.d_model ** -0.5)
            )
            self.b_gate_up = nn.Parameter(torch.zeros(self.num_proposers, 2 * self.d_ffn))
            self.w_down = nn.Parameter(
                torch.randn(self.num_proposers, self.d_ffn, self.d_model)
                * (self.d_ffn ** -0.5)
            )
            self.b_down = nn.Parameter(torch.zeros(self.num_proposers, self.d_model))

            self.branch_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
            self.branch_scale = nn.Parameter(
                torch.ones(1, self.num_proposers, 1, self.d_model)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D] -> gate_up: [B, N, L, 2 * D_ffn]
            gate_up = fused_proposer_linear(x, self.w_gate_up, self.b_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up  # [B, N, L, D_ffn]
            proposals = torch.einsum("bnlh,nhd->bnld", hidden, self.w_down)
            proposals = proposals + self.b_down.unsqueeze(0).unsqueeze(2)
            return self.branch_norm(proposals) * self.branch_scale


    class CausalSelfAttentionSubOp(nn.Module):
        """Proposer Branch 1: Global Causal Multi-Head Self-Attention."""

        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out)


    class SelectiveSSMSubOp(nn.Module):
        """Proposer Branch 2: Mamba/SSM-inspired Selective State Space Operator."""

        def __init__(self, d_model: int, d_state: int = 16):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            self.x_proj = nn.Linear(d_model, d_model + 2 * d_state, bias=False)
            self.dt_bias = nn.Parameter(torch.full((d_model,), -2.0))
            A = (
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(d_model, -1)
            )
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(d_model))
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            u, z = self.in_proj(x).chunk(2, dim=-1)
            u = F.silu(u)
            dt, B_ssm, C_ssm = torch.split(
                self.x_proj(u), [D, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)
            A = -torch.exp(self.A_log)
            dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, D, self.d_state))
            dB_u = dt.unsqueeze(-1) * B_ssm.unsqueeze(2) * u.unsqueeze(-1)

            h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
            ys = []
            for t in range(L):
                h = dA[:, t] * h + dB_u[:, t]
                ys.append(torch.einsum("bdn,bn->bd", h, C_ssm[:, t]))
            y = torch.stack(ys, dim=1)
            return self.out_proj((y + u * self.D) * F.silu(z))


    class CausalDepthwiseConvSubOp(nn.Module):
        """Proposer Branch 3: Local Causal Depthwise Separable Convolution."""

        def __init__(self, d_model: int, kernel_size: int = 4):
            super().__init__()
            self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            self.dw_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size - 1,
                groups=d_model,
            )
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, _ = x.shape
            u, gate = self.in_proj(x).chunk(2, dim=-1)
            u_conv = self.dw_conv(u.transpose(1, 2))[..., :L].transpose(1, 2)
            return self.out_proj(F.silu(u_conv) * F.silu(gate))


    class SwiGLUMLPSubOp(nn.Module):
        """Proposer Branch 4: Pointwise Non-Linear Channel Mixing Expert."""

        def __init__(self, d_model: int, d_ffn: int):
            super().__init__()
            self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
            self.w_up = nn.Linear(d_model, d_ffn, bias=False)
            self.w_down = nn.Linear(d_ffn, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


    class HeterogeneousProposerOp(nn.Module):
        """
        Mixture-of-Architectures (MoA) Heterogeneous Proposer Operator.
        Runs diverse operator branches (Attention, SSM, Conv, SwiGLU) in parallel
        and applies branch-wise RMSNorm + LayerScale before aggregation.
        """

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.num_proposers = config.num_proposers
            branches: List[nn.Module] = []
            for i in range(self.num_proposers):
                kind = i % 4
                if kind == 0:
                    branches.append(CausalSelfAttentionSubOp(config.d_model, config.n_heads))
                elif kind == 1:
                    branches.append(SelectiveSSMSubOp(config.d_model, config.d_state))
                elif kind == 2:
                    branches.append(
                        CausalDepthwiseConvSubOp(config.d_model, config.conv_kernel_size)
                    )
                else:
                    branches.append(SwiGLUMLPSubOp(config.d_model, config.d_ffn))

            self.branches = nn.ModuleList(branches)
            self.branch_norms = nn.ModuleList(
                [RMSNorm(config.d_model, eps=config.rms_norm_eps) for _ in range(self.num_proposers)]
            )
            self.branch_scale = nn.Parameter(
                torch.ones(1, self.num_proposers, 1, config.d_model)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D] -> proposals: [B, N, L, D]
            outs = [norm(branch(x)) for norm, branch in zip(self.branch_norms, self.branches)]
            return torch.stack(outs, dim=1) * self.branch_scale


    class CrossProposalAggregatorOp(nn.Module):
        """
        Synthesizes N parallel proposals [B, N, L, D] into [B, L, D] via:
        - 'cross_attention': O(L * N * D) token-wise Cross-Proposal Attention
        - 'sparse_router': Top-K Confidence-Gated Router with Load-Balancing + Z-Loss
        - 'low_rank_concat': Gated Low-Rank Bottleneck Concatenation
        """

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.num_proposers = config.num_proposers
            self.top_k = min(config.top_k_proposers, config.num_proposers)
            self.mode = config.aggregator_mode

            if self.mode == "cross_attention":
                self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
                self.kv_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
                self.proposer_id_emb = nn.Parameter(
                    torch.randn(1, config.num_proposers, 1, config.d_model) * 0.02
                )
                self.learnable_temp = nn.Parameter(torch.ones(1, config.n_heads, 1, 1))
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            elif self.mode == "sparse_router":
                self.router_query = nn.Linear(config.d_model, config.d_model, bias=False)
                self.router_key = nn.Linear(config.d_model, config.d_model, bias=False)
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            elif self.mode == "low_rank_concat":
                concat_dim = config.num_proposers * config.d_model
                self.gate_proj = nn.Linear(concat_dim, config.d_model, bias=False)
                self.val_proj = nn.Linear(concat_dim, config.d_model, bias=False)
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        def forward(
            self, x: torch.Tensor, proposals: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            B, N, L, D = proposals.shape
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            metrics: Dict[str, torch.Tensor] = {}

            if self.mode == "cross_attention":
                p_tagged = proposals + self.proposer_id_emb
                q = self.q_proj(x).view(B * L, 1, self.n_heads, self.d_head).transpose(1, 2)
                p_blnd = p_tagged.permute(0, 2, 1, 3).contiguous()
                kv = self.kv_proj(p_blnd).view(B * L, N, 2, self.n_heads, self.d_head)
                k, v = kv.unbind(dim=2)
                k, v = k.transpose(1, 2), v.transpose(1, 2)

                scale = (self.d_head ** -0.5) * self.learnable_temp
                attn_weights = F.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1
                )
                agg = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
                synthesized = self.out_proj(agg)
                metrics["proposal_importance"] = (
                    attn_weights.view(B, L, self.n_heads, N).mean(dim=(0, 1, 2)).detach()
                )

            elif self.mode == "sparse_router":
                q_r = self.router_query(x) * (D ** -0.5)
                k_r = self.router_key(proposals)
                router_logits = torch.einsum("bld,bnld->bln", q_r, k_r)
                z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
                router_probs = F.softmax(router_logits, dim=-1)

                top_k_vals, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
                top_k_weights = F.softmax(top_k_vals, dim=-1)
                sparse_weights = torch.zeros_like(router_probs).scatter_(
                    dim=-1, index=top_k_indices, src=top_k_weights
                )
                selection_mask = torch.zeros_like(router_probs).scatter_(
                    dim=-1, index=top_k_indices, value=1.0
                )
                f_i = selection_mask.mean(dim=(0, 1))
                P_i = router_probs.mean(dim=(0, 1))
                load_balance_loss = N * torch.sum(f_i * P_i)
                aux_loss = (
                    self.config.aux_loss_weight * load_balance_loss
                    + self.config.z_loss_weight * z_loss
                )
                agg = torch.einsum("bln,bnld->bld", sparse_weights, proposals)
                synthesized = self.out_proj(agg)
                metrics["proposal_importance"] = f_i.detach()

            else:
                p_concat = proposals.permute(0, 2, 1, 3).contiguous().view(B, L, N * D)
                gated = F.silu(self.gate_proj(p_concat)) * self.val_proj(p_concat)
                synthesized = self.out_proj(gated)
                metrics["proposal_importance"] = torch.full(
                    (N,), 1.0 / N, device=x.device, dtype=x.dtype
                )

            return synthesized, aux_loss, metrics


    class MoARefinementStage(nn.Module):
        """Single stage of Pre-RMSNorm -> Parallel Proposer -> Aggregator -> Residual."""

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.pre_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
            if config.proposer_mode == "homogeneous":
                self.proposer = BatchedHomogeneousProposerOp(config)
            else:
                self.proposer = HeterogeneousProposerOp(config)
            self.aggregator = CrossProposalAggregatorOp(config)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            normed = self.pre_norm(x)
            proposals = self.proposer(normed)
            delta, aux_loss, metrics = self.aggregator(normed, proposals)
            return x + delta, aux_loss, metrics


    class IterativeMoAOperatorBlock(nn.Module):
        """
        Multi-Layer System-Level MoA Refinement Operator with token-level
        Adaptive Confidence Halting (ACT).
        """

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.config = config
            self.num_steps = config.num_refinement_steps
            self.enable_halting = config.enable_adaptive_halting

            if config.share_refinement_weights:
                shared = MoARefinementStage(config)
                self.stages = nn.ModuleList([shared for _ in range(self.num_steps)])
            else:
                self.stages = nn.ModuleList(
                    [MoARefinementStage(config) for _ in range(self.num_steps)]
                )

            if self.enable_halting:
                self.halting_gate = nn.Sequential(
                    RMSNorm(config.d_model, eps=config.rms_norm_eps),
                    nn.Linear(config.d_model, 1),
                )
            self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            B, L, _ = x.shape
            h = x
            total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            last_metrics: Dict[str, torch.Tensor] = {}

            if not self.enable_halting or self.num_steps == 1:
                for stage in self.stages:
                    h, step_aux, last_metrics = stage(h)
                    total_aux_loss = total_aux_loss + step_aux
                return self.final_norm(h), total_aux_loss, last_metrics

            halting_accum = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
            remainder = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)
            weighted_state = torch.zeros_like(x)
            ponder_steps = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)

            for step_idx, stage in enumerate(self.stages):
                h, step_aux, last_metrics = stage(h)
                total_aux_loss = total_aux_loss + step_aux

                if step_idx == self.num_steps - 1:
                    step_weight = remainder
                else:
                    p_halt = torch.sigmoid(self.halting_gate(h))
                    still_running = (halting_accum < self.config.halting_threshold).to(x.dtype)
                    new_accum = halting_accum + p_halt * still_running
                    overflow = (new_accum >= self.config.halting_threshold).to(x.dtype) * still_running
                    step_weight = overflow * remainder + (1.0 - overflow) * p_halt * still_running
                    halting_accum = halting_accum + step_weight
                    remainder = remainder - step_weight
                    ponder_steps = ponder_steps + still_running

                weighted_state = weighted_state + step_weight * h

            total_aux_loss = total_aux_loss + 0.001 * ponder_steps.mean()
            last_metrics["mean_refinement_steps"] = (ponder_steps.mean() + 1.0).detach()
            return self.final_norm(weighted_state), total_aux_loss, last_metrics


    class MixtureOfAttentionOp(nn.Module):
        """
        Mixture-of-Attention (MoA) Long-Context Heterogeneous Sparse Attention Operator.
        Combines Sink+Sliding-Window heads, Dilated Strided heads, and Full Causal heads.
        """

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.window_size = config.sliding_window_size
            self.num_sinks = config.num_sink_tokens
            self.dilation = config.dilation_stride

            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.head_gate = nn.Linear(config.d_model, config.n_heads, bias=True)
            self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
            B, L, D = x.shape
            qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]

            mask_np, stats = NumpyMoAOperatorKernels.build_mixture_of_attention_mask(
                L, self.n_heads, self.window_size, self.num_sinks, self.dilation
            )
            attn_mask = torch.from_numpy(mask_np).to(device=x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            gate = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            out = (out * gate).transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out), stats["overall_moa_density"]


# ==============================================================================
# 4. Complexity & Parameter Profiler + Verification Runner
# ==============================================================================

def profile_moa_operator_complexity(config: MoAOperatorConfig, seq_len: int = 1024) -> List[Dict[str, object]]:
    """
    Computes exact parameter counts, active parameters per token, and theoretical
    per-token FLOPs across different MoA operator configurations.
    """
    D = config.d_model
    D_ffn = config.d_ffn
    N = config.num_proposers
    K = config.top_k_proposers

    # 1. Homogeneous Parallel Proposer (N SwiGLU branches: 3 * D * D_ffn per branch)
    homo_proposer_params = N * (3 * D * D_ffn + 2 * D_ffn + D) + 2 * D

    # 2. Heterogeneous Proposer (Attention + SSM + DepthwiseConv + SwiGLU cycled across N)
    hetero_proposer_params = 0
    for i in range(N):
        kind = i % 4
        if kind == 0:  # Attention: 4 * D^2
            hetero_proposer_params += 4 * D * D
        elif kind == 1:  # SSM: 3 * D^2 + D * (D + 2*d_state) + 2*D*d_state
            hetero_proposer_params += 4 * D * D + 4 * D * config.d_state
        elif kind == 2:  # Causal Depthwise Conv: 3 * D^2 + D * kernel
            hetero_proposer_params += 3 * D * D + D * config.conv_kernel_size
        else:  # SwiGLU: 3 * D * D_ffn
            hetero_proposer_params += 3 * D * D_ffn

    # 3. Aggregator Parameters
    cross_attn_agg_params = 4 * D * D + N * D
    sparse_router_agg_params = 3 * D * D
    low_rank_concat_params = (2 * N + 1) * D * D

    rows = [
        {
            "operator": "BatchedHomogeneousProposer + CrossAttnAggregator",
            "total_params": homo_proposer_params + cross_attn_agg_params,
            "active_params_per_token": homo_proposer_params + cross_attn_agg_params,
            "complexity_order": "O(L * N * D * D_ffn + L * N * D^2)",
        },
        {
            "operator": "HeterogeneousProposer (Attn+SSM+Conv+MLP) + CrossAttnAggregator",
            "total_params": hetero_proposer_params + cross_attn_agg_params,
            "active_params_per_token": hetero_proposer_params + cross_attn_agg_params,
            "complexity_order": "O(L^2 * D + L * N * D^2)",
        },
        {
            "operator": f"BatchedHomogeneousProposer + SparseRouter (Top-{K} of {N})",
            "total_params": homo_proposer_params + sparse_router_agg_params,
            "active_params_per_token": int(homo_proposer_params * (K / N)) + sparse_router_agg_params,
            "complexity_order": f"O(L * K * D * D_ffn + L * N * D) [Sparse K={K}]",
        },
        {
            "operator": "BatchedHomogeneousProposer + LowRankConcatAggregator",
            "total_params": homo_proposer_params + low_rank_concat_params,
            "active_params_per_token": homo_proposer_params + low_rank_concat_params,
            "complexity_order": "O(L * N * D * D_ffn + L * N * D^2)",
        },
    ]
    return rows


def run_verification_suite() -> None:
    """Executes numerical gradient checks, tensor shape assertions, and complexity profiling."""
    rng = np.random.default_rng(42)
    print("=" * 82)
    print("1. Analytical vs. Numerical Finite-Difference Gradcheck (FusedProposerLinearOp)")
    print("=" * 82)
    B, L, D_in, N, D_out = 2, 4, 6, 3, 5
    x = rng.standard_normal((B, L, D_in)).astype(np.float64)
    w = rng.standard_normal((N, D_in, D_out)).astype(np.float64) * 0.2
    b = rng.standard_normal((N, D_out)).astype(np.float64) * 0.1
    upstream = rng.standard_normal((B, N, L, D_out)).astype(np.float64)

    # Analytical gradients
    gx_ana, gw_ana, gb_ana = NumpyMoAOperatorKernels.fused_proposer_linear_backward(
        upstream, x, w, has_bias=True
    )

    # Central finite-difference check on X and W
    eps = 1e-6
    gx_num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_pos, x_neg = x.copy(), x.copy()
        x_pos[idx] += eps
        x_neg[idx] -= eps
        loss_pos = np.sum(NumpyMoAOperatorKernels.fused_proposer_linear_forward(x_pos, w, b) * upstream)
        loss_neg = np.sum(NumpyMoAOperatorKernels.fused_proposer_linear_forward(x_neg, w, b) * upstream)
        gx_num[idx] = (loss_pos - loss_neg) / (2 * eps)

    gw_num = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        w_pos, w_neg = w.copy(), w.copy()
        w_pos[idx] += eps
        w_neg[idx] -= eps
        loss_pos = np.sum(NumpyMoAOperatorKernels.fused_proposer_linear_forward(x, w_pos, b) * upstream)
        loss_neg = np.sum(NumpyMoAOperatorKernels.fused_proposer_linear_forward(x, w_neg, b) * upstream)
        gw_num[idx] = (loss_pos - loss_neg) / (2 * eps)

    rel_err_x = np.linalg.norm(gx_ana - gx_num) / (np.linalg.norm(gx_ana) + np.linalg.norm(gx_num))
    rel_err_w = np.linalg.norm(gw_ana - gw_num) / (np.linalg.norm(gw_ana) + np.linalg.norm(gw_num))
    print(f"   Relative L2 Grad Error (dL/dX)     : {rel_err_x:.3e}  [PASS < 1e-8]")
    print(f"   Relative L2 Grad Error (dL/dWeight): {rel_err_w:.3e}  [PASS < 1e-8]")
    assert rel_err_x < 1e-8 and rel_err_w < 1e-8

    # Branch RMSNorm Gradcheck (with eps=0 so x_hat^2 identity is exact to machine precision)
    gamma = rng.standard_normal((D_out,)).astype(np.float64)
    p_in = rng.standard_normal((B, N, L, D_out)).astype(np.float64)
    _, inv_rms = NumpyMoAOperatorKernels.branch_rmsnorm_forward(p_in, gamma, eps=0.0)
    gp_ana, ggamma_ana = NumpyMoAOperatorKernels.branch_rmsnorm_backward(
        upstream, p_in, gamma, inv_rms
    )
    gp_num = np.zeros_like(p_in)
    for idx in np.ndindex(p_in.shape):
        p_pos, p_neg = p_in.copy(), p_in.copy()
        p_pos[idx] += eps
        p_neg[idx] -= eps
        l_pos = np.sum(NumpyMoAOperatorKernels.branch_rmsnorm_forward(p_pos, gamma, eps=0.0)[0] * upstream)
        l_neg = np.sum(NumpyMoAOperatorKernels.branch_rmsnorm_forward(p_neg, gamma, eps=0.0)[0] * upstream)
        gp_num[idx] = (l_pos - l_neg) / (2 * eps)
    rel_err_norm = np.linalg.norm(gp_ana - gp_num) / (np.linalg.norm(gp_ana) + np.linalg.norm(gp_num))
    print(f"   Relative L2 Grad Error (RMSNorm)   : {rel_err_norm:.3e}  [PASS < 1e-8]")
    assert rel_err_norm < 1e-8

    print("\n" + "=" * 82)
    print("2. System-to-Operator Tensor Flow & Aggregator Verification")
    print("=" * 82)
    B_t, L_t, D_t, N_t, H_t = 2, 32, 256, 4, 8
    x_state = rng.standard_normal((B_t, L_t, D_t)) * 0.1
    w_proposer = rng.standard_normal((N_t, D_t, D_t)) * (D_t ** -0.5)
    proposals = NumpyMoAOperatorKernels.fused_proposer_linear_forward(x_state, w_proposer)
    proposals, _ = NumpyMoAOperatorKernels.branch_rmsnorm_forward(proposals, np.ones(D_t))

    w_q = rng.standard_normal((D_t, D_t)) * (D_t ** -0.5)
    w_k = rng.standard_normal((D_t, D_t)) * (D_t ** -0.5)
    w_v = rng.standard_normal((D_t, D_t)) * (D_t ** -0.5)
    w_o = rng.standard_normal((D_t, D_t)) * (D_t ** -0.5)

    syn_attn, attn_w = NumpyMoAOperatorKernels.cross_proposal_attention_forward(
        x_state, proposals, w_q, w_k, w_v, w_o, n_heads=H_t
    )
    syn_sparse, aux_loss, f_i = NumpyMoAOperatorKernels.sparse_topk_router_forward(
        x_state, proposals, w_q, w_k, top_k=2
    )
    _, moa_attn_stats = NumpyMoAOperatorKernels.build_mixture_of_attention_mask(
        seq_len=L_t, n_heads=H_t, window_size=8, num_sinks=2, dilation=2
    )

    print(f"   Input Backbone Tensor X             : {x_state.shape}")
    print(f"   Parallel Proposer Tensor P          : {proposals.shape}  (N={N_t} parallel branches)")
    print(f"   Cross-Proposal Attention Output Y   : {syn_attn.shape}     (Attn Weights: {attn_w.shape})")
    print(f"   Sparse Top-2 Router Output Y        : {syn_sparse.shape}     (Aux Loss: {aux_loss:.5f})")
    print(f"   Top-2 Token Routing Distribution f_i: {[round(float(v), 3) for v in f_i]}")
    print(
        f"   Mixture-of-Attention (MoA) Density  : {moa_attn_stats['overall_moa_density']*100:.1f}% "
        f"(Sink+Win: {moa_attn_stats['sink_local_density']*100:.1f}%, "
        f"Dilated: {moa_attn_stats['dilated_density']*100:.1f}%, Full: 100.0%)"
    )

    print("\n" + "=" * 82)
    print("3. Operator Parameter & Complexity Breakdown (d_model=256, d_ffn=512, N=4)")
    print("=" * 82)
    cfg = MoAOperatorConfig(d_model=256, d_ffn=512, num_proposers=4, top_k_proposers=2)
    rows = profile_moa_operator_complexity(cfg, seq_len=1024)
    for r in rows:
        print(
            f"   - {r['operator']:56s} | "
            f"Total: {r['total_params']:9,d} | "
            f"Active/Token: {r['active_params_per_token']:9,d} | "
            f"{r['complexity_order']}"
        )
    print("=" * 82)


if __name__ == "__main__":
    run_verification_suite()