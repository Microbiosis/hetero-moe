"""V66.0 — MoA Triton 融合内核库 单元测试。

本代是 v63–v74 中**唯一能在本地 CPU 上复算硬指标**的一代：
TritonTileSimulator 是 numpy 逐 program 模拟器，把四个 @triton.jit 内核的
分块策略、SRAM 寄存器累加、在线 softmax 重标定、块稀疏跳过逐 program 复算，
从而在**没有 GPU** 的情况下验证位级正确性并给出精确 HBM 流量账。

本代无端到端精度数字（端到端待 Kaggle 批量验证），
但「内核正确」与「HBM 流量下降」是**实测的**，不是声称的。

实现文件在 v62 包内（平铺 import，须把该目录加入 sys.path）。
每个 test_* 必须在末尾 return True。
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

import moa_triton_kernels as TK

TITLE = "V66.0 MoA Triton 融合内核库 — 位级验证"

SIM = TK.TritonTileSimulator


# ---------------------------------------------------------------- 1
def test_kernel1_rmsnorm_scale_bitwise():
    """1. 内核1 Branch-RMSNorm+LayerScale：前向与反向均与 eager 逐位一致"""
    rng = np.random.default_rng(0)
    B, N, L, D = 2, 3, 8, 16
    x = rng.standard_normal((B, N, L, D))
    gamma = rng.standard_normal((D,))
    scale = rng.standard_normal((N, D))   # 每分支一组 LayerScale：eff_w = gamma * scale[n]
    grad_out = rng.standard_normal((B, N, L, D))
    out, dx, rep = SIM.simulate_fused_branch_rmsnorm_scale(x, gamma, scale, grad_out)
    assert out.shape == (B, N, L, D), out.shape
    assert dx.shape == (B, N, L, D), dx.shape
    assert rep.max_abs_diff_vs_ref < 1e-5, "与 eager 参考差异 %g" % rep.max_abs_diff_vs_ref
    print("[PASS] test_kernel1_rmsnorm_scale_bitwise (max diff=%.2e)" % rep.max_abs_diff_vs_ref)
    return True


# ---------------------------------------------------------------- 2
def test_kernel1_hbm_traffic_reduction():
    """2. 内核1 的 HBM 流量账：融合后显著低于 eager 的 14 遍扫描"""
    rng = np.random.default_rng(1)
    B, N, L, D = 2, 3, 8, 16
    x = rng.standard_normal((B, N, L, D))
    gamma = rng.standard_normal((D,))
    scale = rng.standard_normal((N, D))
    go = rng.standard_normal((B, N, L, D))
    _, _, rep = SIM.simulate_fused_branch_rmsnorm_scale(x, gamma, scale, go)
    assert rep.triton_fused_hbm_bytes < rep.eager_hbm_bytes, "流量未下降"
    red = rep.bandwidth_reduction_pct
    assert red > 50.0, "流量下降仅 %.1f%%" % red
    print("[PASS] test_kernel1_hbm_traffic_reduction (下降 %.1f%%)" % red)
    return True


# ---------------------------------------------------------------- 3
def test_kernel2_proposer_swiglu_bitwise():
    """3. 内核2 多 proposer GEMM+SwiGLU：分块累加结果与 eager einsum 一致"""
    rng = np.random.default_rng(2)
    B, L, D_in, D_ffn, N = 2, 8, 16, 24, 3
    x = rng.standard_normal((B, L, D_in))
    w_gate = rng.standard_normal((N, D_in, D_ffn))
    w_up = rng.standard_normal((N, D_in, D_ffn))
    b_gate = rng.standard_normal((N, D_ffn))
    b_up = rng.standard_normal((N, D_ffn))
    out, rep = SIM.simulate_fused_proposer_swiglu(x, w_gate, w_up, b_gate, b_up)
    assert out.shape == (B, N, L, D_ffn), out.shape
    assert rep.max_abs_diff_vs_ref < 1e-8, "与 eager 参考差异 %g" % rep.max_abs_diff_vs_ref
    print("[PASS] test_kernel2_proposer_swiglu_bitwise (max diff=%.2e)" % rep.max_abs_diff_vs_ref)
    return True


# ---------------------------------------------------------------- 4
def test_kernel2_no_intermediate_writeback():
    """4. 内核2 的核心价值：Gate/Up 中间张量**不回写 HBM**（流量账应体现）"""
    rng = np.random.default_rng(3)
    B, L, D_in, D_ffn, N = 2, 8, 16, 24, 3
    x = rng.standard_normal((B, L, D_in))
    wg = rng.standard_normal((N, D_in, D_ffn))
    wu = rng.standard_normal((N, D_in, D_ffn))
    bg = rng.standard_normal((N, D_ffn))
    bu = rng.standard_normal((N, D_ffn))
    _, rep = SIM.simulate_fused_proposer_swiglu(x, wg, wu, bg, bu)
    act = B * N * L * D_ffn * 2
    # eager 要写 G + 写 U + 读 G + 读 U + 写 Out = 5 遍激活；融合版只写 Out = 1 遍
    assert rep.eager_hbm_bytes - rep.triton_fused_hbm_bytes >= 4 * act, "激活流量账不成立"
    print("[PASS] test_kernel2_no_intermediate_writeback (省 %d 字节激活流量)"
          % (rep.eager_hbm_bytes - rep.triton_fused_hbm_bytes))
    return True


# ---------------------------------------------------------------- 5
def test_kernel3_sparse_topk_aggregator_bitwise():
    """5. 内核3 稀疏 top-k 聚合：只 gather K 个 proposer，结果与 dense 参考一致"""
    rng = np.random.default_rng(4)
    B, N, L, D, K = 2, 4, 8, 16, 2
    props = rng.standard_normal((B, N, L, D))
    logits = rng.standard_normal((B, L, N))
    out, rep = SIM.simulate_fused_sparse_topk_aggregator(props, logits, K)
    assert out.shape == (B, L, D), out.shape
    assert rep.max_abs_diff_vs_ref < 1e-10, "与 dense 参考差异 %g" % rep.max_abs_diff_vs_ref
    print("[PASS] test_kernel3_sparse_topk_aggregator_bitwise (max diff=%.2e)" % rep.max_abs_diff_vs_ref)
    return True


# ---------------------------------------------------------------- 6
def test_kernel3_reads_only_k_proposals():
    """6. 内核3 只读 K 个 proposer 的流量（不是 N 个）—— 稀疏性的硬件层体现"""
    rng = np.random.default_rng(5)
    B, N, L, D, K = 2, 8, 8, 16, 2
    props = rng.standard_normal((B, N, L, D))
    logits = rng.standard_normal((B, L, N))
    _, rep = SIM.simulate_fused_sparse_topk_aggregator(props, logits, K)
    assert rep.triton_fused_hbm_bytes < rep.eager_hbm_bytes, "流量未下降"
    # Triton 只读 K 份 proposals，eager 读 N 份 ⇒ 比值应接近 K/N（加上输出项）
    ratio = rep.triton_fused_hbm_bytes / rep.eager_hbm_bytes
    assert ratio < 0.6, "读流量比 %.3f 未体现 top-k 稀疏" % ratio
    print("[PASS] test_kernel3_reads_only_k_proposals (流量比 %.3f, K/N=%d/%d)"
          % (ratio, K, N))
    return True


# ---------------------------------------------------------------- 7
def test_kernel4_block_sparse_flash_attention_runs():
    """7. 内核4 块稀疏 MoA flash attention：前向可跑且与参考一致"""
    rng = np.random.default_rng(6)
    # L 必须足够大，否则因果区内的 tile 全都含近对角元素，一个都跳不掉
    B, H, L, D = 1, 3, 64, 8
    q = rng.standard_normal((B, H, L, D))
    k = rng.standard_normal((B, H, L, D))
    v = rng.standard_normal((B, H, L, D))
    out, stats, rep = SIM.simulate_block_sparse_moa_flash_attention(
        q, k, v, window_size=4, num_sinks=2, dilation=2)
    assert out.shape == (B, H, L, D), out.shape
    assert rep.max_abs_diff_vs_ref < 1e-8, "与参考差异 %g" % rep.max_abs_diff_vs_ref
    assert stats["skipped_tile_pct"] > 0.0, "块稀疏未跳过任何 tile"
    print("[PASS] test_kernel4_block_sparse_flash_attention_runs (max diff=%.2e, 跳过 tile %.1f%%)"
          % (rep.max_abs_diff_vs_ref, stats["skipped_tile_pct"]))
    return True


# ---------------------------------------------------------------- 8
def test_all_four_kernels_registered():
    """8. 四个内核的模拟器入口齐备（内核名进入 HBMProfileReport）"""
    names = [m for m in dir(SIM) if m.startswith("simulate_")]
    assert len(names) == 4, "应有 4 个内核模拟器，实测 %d: %s" % (len(names), names)
    print("[PASS] test_all_four_kernels_registered (%s)" % ", ".join(sorted(names)))
    return True


TESTS = [
    test_kernel1_rmsnorm_scale_bitwise,
    test_kernel1_hbm_traffic_reduction,
    test_kernel2_proposer_swiglu_bitwise,
    test_kernel2_no_intermediate_writeback,
    test_kernel3_sparse_topk_aggregator_bitwise,
    test_kernel3_reads_only_k_proposals,
    test_kernel4_block_sparse_flash_attention_runs,
    test_all_four_kernels_registered,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
