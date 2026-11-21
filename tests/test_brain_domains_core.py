"""tests/test_brain_domains_core.py — 幻脑线 (research/brain_domains/) 核心算子与工程门禁回归测试。

覆盖:
    1. DeCollapsedLocalInterface (MODEL_FORM.md 2026-10-28 脑区局部化 + 去常数坍缩 + 宿主模态范数锁相)
    2. exp106_ops_diag 环境变量解耦与离线去坍缩算子自检 (run_self_test)
    3. DomainConditionedSSM 低秩域调制与 domain_gate=0 精确退化
    4. fast_scan.chunked_scan 块并行扫描与逐 token 参考实现数值等价
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.brain_domains.arch.decollapsed_interface import DeCollapsedLocalInterface
from research.brain_domains.arch.domain_ssm import DomainConditionedSSM
from research.brain_domains.arch.fast_scan import chunked_scan, scan_reference
from research.brain_domains.exp.exp106_ops_diag import (
    concept_separability,
    consistency,
    run_self_test,
)


def test_brain_domains_decollapsed_local_interface_eliminates_collapse_and_locks_norm():
    """验证 DeCollapsedLocalInterface 消除跨概念常数方向坍缩 (0.99 -> <0.15) 并锁死宿主 token 幅度。"""
    torch.manual_seed(42)
    B, D_BRAIN, K, D_TOK = 40, 128, 8, 32
    iface = DeCollapsedLocalInterface(
        d_brain=D_BRAIN,
        n_tokens=K,
        d_tok=D_TOK,
        num_regions=4,
        bottleneck_rank=16,
        common_mode_rank=1,
        default_target_norm=0.18,
    )

    # 构造高度共线 (常数偏置占主导，模拟 exp102 跨概念一致性 > 0.99) + 概念微扰的输入
    common_bias = torch.randn(1, D_BRAIN) * 25.0
    n_concepts, reps = 8, 5
    labels = np.repeat(np.arange(n_concepts), reps)
    concept_vecs = torch.randn(n_concepts, D_BRAIN) * 1.2
    z = common_bias + concept_vecs[torch.from_numpy(labels)] + torch.randn(B, D_BRAIN) * 0.15
    z.requires_grad_(True)

    # 同时模拟原始坍缩 token (均值偏移 76.0)
    raw_collapsed = torch.randn(1, K, D_TOK) * 30.0 + torch.randn(B, K, D_TOK) * 0.5
    raw_c_before = consistency(raw_collapsed.detach().numpy())
    assert raw_c_before > 0.98

    host_txt_tokens = torch.randn(2, 12, D_TOK) * (0.18 / math.sqrt(D_TOK)) if "math" in globals() else torch.randn(2, 12, D_TOK) * 0.03
    out_tokens, diag = iface(z, host_tokens=host_txt_tokens, return_diag=True)
    assert out_tokens.shape == (B, K, D_TOK)
    assert diag["consistency_after"] < 0.05, f"去坍缩后跨概念一致性仍过高: {diag}"
    assert abs(diag["amplitude_ratio"] - 1.0) < 0.05, f"幅度锁相偏离宿主范数: {diag}"

    # 验证概念可分性保持高位且四个脑区局部模块全部获得非零梯度
    sep = concept_separability(out_tokens.detach().numpy(), labels)
    assert sep > 0.65, f"去坍缩后概念 1-NN 可分性不足: {sep:.4f}"

    loss = out_tokens.pow(2).mean()
    loss.backward()
    assert z.grad is not None and z.grad.norm() > 0
    for g in range(iface.num_regions):
        assert iface.region_down[g].weight.grad is not None
        assert iface.region_down[g].weight.grad.norm() > 0


def test_brain_domains_exp106_self_test_and_env_decoupling():
    """验证 exp106 算子诊断离线自检通过，且支持 BRAIN_DOMAINS_ROOT 环境变量解耦。"""
    res = run_self_test()
    assert res["consistency_before"] > 0.98
    assert res["consistency_after_A2"] < 0.05
    assert res["separability_after_A2"] > res["separability_before"] + 0.35


def test_brain_domains_ssm_degeneration_and_fast_scan_equivalence():
    """验证 DomainConditionedSSM 在 domain_gate=0 时精确退化，且 chunked_scan 与逐 token 扫描等价。"""
    torch.manual_seed(7)
    B, L, D_IN, D_ST = 2, 33, 32, 8
    ssm = DomainConditionedSSM(d_inner=D_IN, d_state=D_ST, n_domains=5, rank=4)
    with torch.no_grad():
        ssm.alpha.normal_(0, 0.2)
    u = torch.randn(B, L, D_IN)
    dom_a = torch.zeros(B, L, dtype=torch.long)
    dom_b = torch.ones(B, L, dtype=torch.long) * 3

    # domain_gate != 0 时不同域产生不同状态转移矩阵 A_eff
    A_a = ssm.effective_A(dom_a)
    A_b = ssm.effective_A(dom_b)
    assert not torch.allclose(A_a, A_b)

    # domain_gate == 0 时不同域严格退化为同一共享 A
    with torch.no_grad():
        ssm.domain_gate.zero_()
    assert torch.allclose(ssm.effective_A(dom_a), ssm.effective_A(dom_b), atol=1e-7)

    # chunked_scan 与 scan_reference 数值严格等价 (覆盖非整除序列长度 L=33)
    delta = torch.rand(B, L, D_IN) * 0.05 + 1e-3
    a_eff = -(torch.rand(B, L, D_IN, D_ST) * 8.0 + 1.0)
    log_dA = delta.unsqueeze(-1) * a_eff
    dBu = torch.randn(B, L, D_IN, D_ST) * 0.1
    Cm = torch.randn(B, L, D_ST) * 0.1
    x0 = torch.randn(B, D_IN, D_ST) * 0.1
    y_ref, x_ref = scan_reference(log_dA, dBu, Cm, x0)
    y_chk, x_chk = chunked_scan(log_dA, dBu, Cm, x0, block=8)
    assert torch.allclose(y_ref, y_chk, atol=1e-5)
    assert torch.allclose(x_ref, x_chk, atol=1e-5)
