"""V64.0 — MoA 异构 proposer 融合层 单元测试。

实现文件在 v62 包内（moa_operators.py / moa_fusion.py），它们用**平铺 import**
（`from moa_operators import ...`），故必须把 v62 包目录加入 sys.path 后按普通模块导入；
走包内相对导入会 ModuleNotFoundError（实测）。

本代无端到端数字（见 results/metrics.json），测试只验证**机制语义**：
分支齐备且异构、稀疏聚合真的稀疏、aux_loss 非负、迭代精炼可配、
numpy 参考核与 torch 前向/反向逐位一致。

每个 test_* 必须在末尾 return True（run_suite 以返回真值判定通过）。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)
_v62 = os.path.join(os.path.dirname(_pkg), "v62_dimension_projection_space")
_root = _pkg
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _v62 not in sys.path:
    sys.path.insert(0, _v62)

import numpy as np
import torch

import moa_operators as MOA
import moa_fusion as MF

TITLE = "V64.0 MoA 异构 proposer 融合层 — 单元测试"

_B, _S, _D = 2, 12, 64


def _cfg(**kw):
    base = dict(d_model=_D, n_heads=4, num_proposers=4, top_k_proposers=2,
                proposer_mode='heterogeneous', aggregator_mode='sparse_router',
                num_refinement_steps=1, d_state=4, d_ffn=128)
    base.update(kw)
    return MOA.MoAOperatorConfig(**base)


def _proposals(n=None, seed=0):
    torch.manual_seed(seed)
    n = n or 4
    return torch.randn(_B, n, _S, _D)


# ---------------------------------------------------------------- 1
def test_three_heterogeneous_branch_kinds():
    """1. 异构 proposer 的分支类型按 i%3 循环：Attn / Conv / SwiGLU 三种归纳偏置"""
    op = MOA.HeterogeneousProposerOp(_cfg(num_proposers=4))
    kinds = [type(b).__name__ for b in op.branches]
    assert kinds == ['CausalSelfAttentionSubOp', 'CausalDepthwiseConvSubOp',
                     'SwiGLUMLPSubOp', 'CausalSelfAttentionSubOp'], kinds
    assert len(set(kinds)) == 3, '分支类型应异构（3 种），实测 %s' % kinds
    print('[PASS] test_three_heterogeneous_branch_kinds (%s)' % ','.join(kinds))
    return True


# ---------------------------------------------------------------- 2
def test_proposer_output_shape_and_layerscale():
    """2. proposer 输出 [B,N,L,D]，且 LayerScale 存在（每个分支可独立缩放）"""
    op = MOA.HeterogeneousProposerOp(_cfg(num_proposers=4))
    x = torch.randn(_B, _S, _D)
    p = op(x)
    assert p.shape == (_B, 4, _S, _D), p.shape
    assert op.branch_scale.shape == (1, 4, 1, _D), op.branch_scale.shape
    assert torch.isfinite(p).all(), 'proposer 输出含 NaN/Inf'
    print('[PASS] test_proposer_output_shape_and_layerscale (输出 %s)' % (tuple(p.shape),))
    return True


# ---------------------------------------------------------------- 3
def test_sparse_router_aggregates_and_sparsifies():
    """3. sparse_router 聚合 [B,N,L,D] → [B,L,D]，且 proposal_importance 只标记 top-k"""
    cfg = _cfg(num_proposers=4, top_k_proposers=2)
    op = MOA.CrossProposalAggregatorOp(cfg)
    x = torch.randn(_B, _S, _D)
    props = _proposals(4)
    y, aux, metrics = op(x, props)
    assert y.shape == (_B, _S, _D), y.shape
    imp = metrics.get('proposal_importance')
    assert imp is not None, 'sparse_router 未返回 proposal_importance'
    nz = int((imp > 0).sum())
    k = min(cfg.top_k_proposers, cfg.num_proposers)
    assert nz <= k * _B * _S, \
        '非零 proposer 数 %d 超过 top_k=%d 上界' % (nz, k)
    print('[PASS] test_sparse_router_aggregates_and_sparsifies (nonzero %d <= top_k*BS=%d)'
          % (nz, k * _B * _S))
    return True


# ---------------------------------------------------------------- 4
def test_aux_loss_nonnegative():
    """4. aux_loss（Load-Balancing + Z-Loss）非负"""
    op = MOA.CrossProposalAggregatorOp(_cfg(aggregator_mode='sparse_router'))
    x = torch.randn(_B, _S, _D)
    props = _proposals(4)
    _, aux, _ = op(x, props)
    val = float(aux.detach()) if torch.is_tensor(aux) else float(aux)
    assert val >= 0.0, 'aux_loss 应非负，实测 %r' % val
    print('[PASS] test_aux_loss_nonnegative (%.6f)' % val)
    return True


# ---------------------------------------------------------------- 5
def test_all_three_aggregator_modes_forward():
    """5. 三种聚合模式（cross_attention / sparse_router / low_rank_concat）均前向通过"""
    x = torch.randn(_B, _S, _D)
    props = _proposals(4)
    for mode in ('cross_attention', 'sparse_router', 'low_rank_concat'):
        op = MOA.CrossProposalAggregatorOp(_cfg(aggregator_mode=mode))
        y, aux, metrics = op(x, props)
        assert y.shape == (_B, _S, _D), (mode, y.shape)
        assert torch.isfinite(y).all(), mode
    print('[PASS] test_all_three_aggregator_modes_forward (3 种模式均通过)')
    return True


# ---------------------------------------------------------------- 6
def test_mixture_of_attention_mask_builder():
    """6. MoA 头级异构掩码可构造：sink+local / dilated / full-causal 三模式按 h%3 轮转"""
    mask, stats = MOA.NumpyMoAOperatorKernels.build_mixture_of_attention_mask(
        seq_len=16, n_heads=4, window_size=4, num_sinks=2, dilation=2)
    assert mask.shape == (1, 4, 16, 16), mask.shape
    for h in range(mask.shape[1]):
        m = mask[0, h]
        assert m.diagonal().sum() > 0, 'head %d 对角线应可见（因果）' % h
        assert np.all(np.triu(m, 1) == 0), 'head %d 上三角应被屏蔽' % h
    for key in ('sink_local_density', 'dilated_density',
                'full_causal_density', 'overall_moa_density'):
        assert key in stats, stats.keys()
        assert 0.0 < stats[key] <= 1.0, (key, stats[key])
    print('[PASS] test_mixture_of_attention_mask_builder (掩码 %s, 密度 %.3f)'
          % (tuple(mask.shape), stats['overall_moa_density']))
    return True


# ---------------------------------------------------------------- 7
def test_numpy_reference_forward_matches_torch():
    """7. numpy 参考核与 torch 前向逐位一致（算子正确性交叉验证）

    参考核契约: x [B,L,D_in] × weight [N,D_in,D_out] -> out [B,N,L,D_out]
    （einsum "bld,ndo->bnlo"，weight 的 d 轴对齐 x 的 D_in，torch 侧需转置后给 F.linear）
    （einsum "bld,ndo->bnlo"：x 的 D_in 对应 ndo 的 d，weight 是 [N,D_in,D_out]）
    """
    torch.manual_seed(0)
    B, L, Din, Dout, N = 2, 6, 8, 5, 3
    x = torch.randn(B, L, Din)
    w = torch.randn(N, Din, Dout) * 0.1
    b = torch.randn(N, Dout) * 0.1
    ref = MOA.NumpyMoAOperatorKernels.fused_proposer_linear_forward(
        x.numpy(), w.numpy(), b.numpy())
    assert ref.shape == (B, N, L, Dout), ref.shape
    # torch 侧逐 proposer 复算
    for n in range(N):
        t = torch.nn.functional.linear(x, w[n].t(), b[n])
        diff = float(abs(t.numpy() - ref[:, n]).max())
        assert diff < 1e-5, 'proposer %d 前向不一致，max diff=%g' % (n, diff)
    print('[PASS] test_numpy_reference_forward_matches_torch (max diff<1e-5)')
    return True


# ---------------------------------------------------------------- 8
def test_numpy_reference_backward_matches_torch():
    """8. numpy 解析反向核与 torch autograd 逐位一致"""
    torch.manual_seed(0)
    B, L, Din, Dout, N = 2, 5, 7, 4, 3
    x = torch.randn(B, L, Din, requires_grad=True)
    w = (torch.randn(N, Din, Dout) * 0.1).requires_grad_(True)
    upstream = torch.randn(B, N, L, Dout)
    for n in range(N):
        torch.nn.functional.linear(x, w[n].t()).backward(upstream[:, n], retain_graph=True)
    gx, gw, gb = MOA.NumpyMoAOperatorKernels.fused_proposer_linear_backward(
        upstream.numpy(), x.detach().numpy(), w.detach().numpy(), has_bias=False)
    dx = float(abs(x.grad.numpy() - gx).max())
    dw = float(abs(w.grad.numpy() - gw).max())
    assert max(dx, dw) < 1e-4, '梯度不一致 dx=%g dw=%g' % (dx, dw)
    assert gb is None, 'has_bias=False 时应返回 None'
    print('[PASS] test_numpy_reference_backward_matches_torch (dx=%.1e dw=%.1e)'
          % (dx, dw))
    return True


# ---------------------------------------------------------------- 9
def test_moa_fusion_interface_compatible_with_v62():
    """9. MoAFusionOverProjSpace 接口与 v62 FusionOverProjSpace 一致（可无痛替换）"""
    f = MF.MoAFusionOverProjSpace(d_space=_D, num_proposers=4, top_k=2)
    x = torch.randn(_B, _S, _D)
    y, alpha = f.forward_with_routing(x)
    assert y.shape == (_B, _S, _D), y.shape
    assert alpha.shape == (_B, _S, 4), alpha.shape
    assert torch.isfinite(y).all(), '输出含 NaN/Inf'
    assert float(f.last_aux_loss.detach()) >= 0.0, 'aux_loss 应非负'
    assert 'MoAFusionOverProjSpace' in f.describe()
    print('[PASS] test_moa_fusion_interface_compatible_with_v62 (y%s alpha%s)'
          % (tuple(y.shape), tuple(alpha.shape)))
    return True


TESTS = [
    test_three_heterogeneous_branch_kinds,
    test_proposer_output_shape_and_layerscale,
    test_sparse_router_aggregates_and_sparsifies,
    test_aux_loss_nonnegative,
    test_all_three_aggregator_modes_forward,
    test_mixture_of_attention_mask_builder,
    test_numpy_reference_forward_matches_torch,
    test_numpy_reference_backward_matches_torch,
    test_moa_fusion_interface_compatible_with_v62,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
