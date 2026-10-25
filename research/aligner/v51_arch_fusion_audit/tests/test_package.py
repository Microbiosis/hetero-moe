"""V51.0 架构异构线审计 — 单元测试 (7 项)。不加载模型。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V51.0 架构异构线审计 — 单元测试 (7 项)"

import numpy as np
import torch
import importlib.util

ROOT = _root
_spec = importlib.util.spec_from_file_location("v51run", os.path.join(ROOT, "examples/run_v51_arch_fusion_audit.py"))
v51 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v51)


def test_arms_form_2x2():
    """四臂必须是 {冻结, 可训练} × {对齐, 置换} 的完整 2×2。"""
    grid = {(k, p) for k, p in v51.ARMS.values()}
    for k in ("frozen", "trainable"):
        for p in (False, True):
            assert (k, p) in grid, (k, p)
    assert len(v51.ARMS) == 4
    print("[PASS] 四臂构成完整的 2×2 (空间类型 × 目标对齐)")
    return True


def test_aligned_target_is_modality_block():
    """v7 的原协议: 模态 m 的目标块就是 m (这正是被审计的事实)。"""
    idx = [0, 1, 2, 0, 1, 2]
    t = v51.targets_for(idx, perm=None)
    blk = t.shape[-1] // v51.N_CLS
    for i, m in enumerate(idx):
        seg = t[i, 0]
        assert seg[m * blk:(m + 1) * blk].min() > 0.9
        rest = torch.cat([seg[:m * blk], seg[(m + 1) * blk:]])
        assert rest.abs().max() < 1e-6
    print("[PASS] 对齐目标 = 模态下标对应的块 (v7 原协议)")
    return True


def test_permuted_target_is_derangement():
    """置换必须无不动点, 否则对照不彻底; 且目标块仍然一一对应 (难度不变)。"""
    perm = [(m + 1) % v51.N_CLS for m in range(v51.N_CLS)]
    assert all(perm[m] != m for m in range(v51.N_CLS)), perm
    assert sorted(perm) == list(range(v51.N_CLS))
    idx = [0, 1, 2]
    t = v51.targets_for(idx, perm=perm)
    blk = t.shape[-1] // v51.N_CLS
    for i, m in enumerate(idx):
        seg = t[i, 0]
        assert seg[perm[m] * blk:(perm[m] + 1) * blk].min() > 0.9
        rest = torch.cat([seg[:perm[m] * blk], seg[(perm[m] + 1) * blk:]])
        assert rest.abs().max() < 1e-6
    print("[PASS] 置换目标无不动点, 且块与模态仍一一对应 (难度不变)")
    return True


def test_permuted_and_aligned_targets_are_distinct():
    """对齐与置换两组目标必须不同 (否则对照无效)。"""
    idx = [0, 1, 2]
    a = v51.targets_for(idx, perm=None)
    b = v51.targets_for(idx, perm=[(m + 1) % v51.N_CLS for m in range(v51.N_CLS)])
    assert not torch.allclose(a, b)
    print("[PASS] 对齐目标 ≠ 置换目标")
    return True


def test_permutation_preserves_target_energy():
    """置换只改标签对应, 不改目标的能量/结构 (难度必须可比)。"""
    idx = [0, 1, 2]
    a = v51.targets_for(idx, perm=None)
    b = v51.targets_for(idx, perm=[(m + 1) % v51.N_CLS for m in range(v51.N_CLS)])
    assert torch.allclose(a.sum(-1), b.sum(-1)), "置换后目标的每 token 能量应相同"
    assert torch.allclose((a > 0).float().sum(-1), (b > 0).float().sum(-1))
    print("[PASS] 置换保持目标能量与非零维数 (难度可比)")
    return True


def test_probe_selfcheck():
    """探针自检: 可分数据上高准确率, 噪声上接近随机。"""
    from research.aligner.v50_fusion_gain_audit.mechanism import linear_probe
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    y = (x[:, 0] > 0).long()
    te, tr = linear_probe(x, y, x, y, n_class=2, steps=200)
    assert tr > 0.9, tr
    x2 = torch.randn(300, 16)
    y2 = torch.randint(0, 3, (300,))
    te2, _ = linear_probe(x2, y2, x2, y2, n_class=3, steps=200)
    assert te2 < 0.6, te2
    print(f"[PASS] 探针自检通过 (可分 train {tr:.2f} / 噪声 {te2:.2f})")
    return True


def test_v7_protocol_constants_unchanged():
    """必须沿用 v7 的协议常量 (D_shared=256, 3 类, 3 专家), 否则不是同一条线。"""
    assert v51.D_SHARED == 256
    assert v51.N_CLS == 3
    assert v51.M == 3
    assert v51.S == 16
    print("[PASS] 协议常量与 v7 一致 (D_shared=256, 3 类, 3 专家, S=16)")
    return True


TESTS = [
    test_arms_form_2x2,
    test_aligned_target_is_modality_block,
    test_permuted_target_is_derangement,
    test_permuted_and_aligned_targets_are_distinct,
    test_permutation_preserves_target_energy,
    test_probe_selfcheck,
    test_v7_protocol_constants_unchanged,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
