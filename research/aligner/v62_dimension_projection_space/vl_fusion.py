"""v70 VL 跨模态融合算子（从 neuro-physical-hierarchical-moa-backbone 抽出最小依赖）
FusedCrossModalBilinearGateFunction + CrossModalVisionLanguageFusionSubOp
锚点共注意力(零距离衰减) + Hadamard-Bilinear 门控，显式视觉/文本两路交互。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedCrossModalBilinearGateFunction(torch.autograd.Function):
    """
    Custom PyTorch Autograd Function encapsulating the Fused Cross-Modal
    Vision-Language Hadamard-Bilinear Gated Synthesis Operator.
    """

    @staticmethod
    def forward(
        ctx,
        v_vis: torch.Tensor,
        t_txt: torch.Tensor,
        c_align: torch.Tensor,
        alpha_vis: torch.Tensor,
        alpha_txt: torch.Tensor,
        beta_bilin: torch.Tensor,
        lam_norm: float,
    ) -> torch.Tensor:
        av_b = alpha_vis.view(1, 1, -1)
        at_b = alpha_txt.view(1, 1, -1)
        bb_b = beta_bilin.view(1, 1, -1)

        g_cm = torch.sigmoid(av_b * v_vis + at_b * t_txt)
        one_m_g = 1.0 - g_cm
        y_interp = g_cm * v_vis + one_m_g * t_txt

        v_sq = v_vis.pow(2)
        t_sq = t_txt.pow(2)
        norm_arg = 1.0 + lam_norm * (v_sq + t_sq)
        inv_norm = torch.rsqrt(norm_arg)
        b_num = v_vis * t_txt + c_align
        bilin_unit = b_num * inv_norm

        out = y_interp + bb_b * bilin_unit
        ctx.save_for_backward(
            v_vis, t_txt, alpha_vis, alpha_txt, beta_bilin,
            g_cm, one_m_g, norm_arg, inv_norm, b_num, bilin_unit
        )
        ctx.lam_norm = lam_norm
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            v_vis, t_txt, alpha_vis, alpha_txt, beta_bilin,
            g_cm, one_m_g, norm_arg, inv_norm, b_num, bilin_unit
        ) = ctx.saved_tensors
        lam_norm = ctx.lam_norm

        av_b = alpha_vis.view(1, 1, -1)
        at_b = alpha_txt.view(1, 1, -1)
        bb_b = beta_bilin.view(1, 1, -1)

        d_betabilin = (grad_out * bilin_unit).sum(dim=(0, 1))
        d_bunit = grad_out * bb_b

        d_bnum = d_bunit * inv_norm
        d_invnorm = d_bunit * b_num
        d_normarg = d_invnorm * (-0.5 * inv_norm / norm_arg)
        d_calign = d_bnum

        d_gcm = grad_out * (v_vis - t_txt)
        d_glogit = d_gcm * g_cm * one_m_g
        d_alphavis = (d_glogit * v_vis).sum(dim=(0, 1))
        d_alphatxt = (d_glogit * t_txt).sum(dim=(0, 1))

        d_vvis = (
            grad_out * g_cm
            + d_glogit * av_b
            + d_bnum * t_txt
            + d_normarg * (2.0 * lam_norm * v_vis)
        )
        d_ttxt = (
            grad_out * one_m_g
            + d_glogit * at_b
            + d_bnum * v_vis
            + d_normarg * (2.0 * lam_norm * t_txt)
        )
        return d_vvis, d_ttxt, d_calign, d_alphavis, d_alphatxt, d_betabilin, None


class CrossModalVisionLanguageFusionSubOp(nn.Module):
    """
    Purpose-Built Vision-Language (Image-Text) Cross-Modal Fusion Proposer Branch
    for Vision-Language (Image-Text) Multimodal Tasks:
    - Decomposes input into 2D Spatial Visual stream (`V_vis`, with bidirectional
      depthwise spatial context) and Semantic Textual stream (`T_txt`).
    - Performs Low-Rank Cross-Modal Latent Anchor Co-Attention (`O(L * R * D)`)
      with ZERO distance decay between early image patches and late text tokens.
    - Applies `FusedCrossModalBilinearGateFunction` in a single fused kernel!
    """

    def __init__(
        self,
        d_model: int,
        num_anchors: int = 16,
        alpha_vis_init: float = 0.80,
        alpha_txt_init: float = 0.80,
        beta_bilinear_init: float = 0.65,
        lam_norm: float = 0.35,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.lam_norm = lam_norm

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        # Bidirectional 2D-aware local spatial neighborhood conv (padding='same' / symmetric)
        self.spatial_dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model
        )
        self.cross_modal_anchors = nn.Parameter(
            torch.randn(num_anchors, d_model) * (d_model ** -0.5)
        )
        self.alpha_vis = nn.Parameter(torch.full((d_model,), alpha_vis_init))
        self.alpha_txt = nn.Parameter(torch.full((d_model,), alpha_txt_init))
        self.beta_bilin = nn.Parameter(torch.full((d_model,), beta_bilinear_init))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_from_projected(self, v_raw: torch.Tensor, t_raw: torch.Tensor) -> torch.Tensor:
        # Bidirectional spatial neighborhood mixing on visual branch (no 1D causal blindness!)
        v_vis = F.silu(self.spatial_dw_conv(v_raw.transpose(1, 2)).transpose(1, 2))
        t_txt = F.silu(t_raw)

        # Low-rank Cross-Modal Anchor Co-Attention (Zero distance decay across L)
        scale = self.d_model ** -0.5
        joint_kv = 0.5 * (v_vis + t_txt)
        attn_pool = F.softmax(
            torch.einsum("rd,bld->brl", self.cross_modal_anchors, joint_kv) * scale,
            dim=-1,
        )
        z_anchors = torch.einsum("brl,bld->brd", attn_pool, v_vis)
        attn_bc = F.softmax(
            torch.einsum("bld,brd->blr", t_txt, z_anchors) * scale,
            dim=-1,
        )
        c_align = torch.einsum("blr,brd->bld", attn_bc, z_anchors)

        fused = FusedCrossModalBilinearGateFunction.apply(
            v_vis, t_txt, c_align,
            self.alpha_vis, self.alpha_txt, self.beta_bilin, self.lam_norm
        )
        return self.out_proj(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_raw, t_raw = self.in_proj(x).chunk(2, dim=-1)
        return self.forward_from_projected(v_raw, t_raw)



