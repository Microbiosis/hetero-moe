"""V53.0 第三审 — 单元测试 (7 项)。不加载模型。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V53.0 第三审 — 单元测试 (7 项)"

import numpy as np
import torch

from research.aligner.v53_third_audit.audit import (
    ARMS, PREDICTIONS, CLASSES, N_CLASSES, ResidualLift, LinearLift,
    make_split, build_targets, probe_accuracy_all, consistency,
)


def test_domain_split_disjoint():
    """值域分割: 训练域与留出域的文本与槽位取值都必须不相交。"""
    tr, ho = make_split(12, seed=0)
    tr_t = {t for _, t in tr}
    ho_t = {t for _, t in ho}
    assert not (tr_t & ho_t), sorted(tr_t & ho_t)[:4]
    # 类内: 训练域用 (前8×前8), 留出域用 (后8×后8) → 槽位取值在类内也不相交
    for ci, (adj, nouns) in enumerate(CLASSES.values()):
        tr_words = {w for c, t in tr if c == ci for w in t.split()}
        ho_words = {w for c, t in ho if c == ci for w in t.split()}
        assert not (tr_words & ho_words), (ci, sorted(tr_words & ho_words)[:4])
    print(f"[PASS] 值域分割: 文本零交集 ({len(tr_t)}/{len(ho_t)}), 类内槽位取值零交集")
    return True


def test_targets_shape_and_randomization():
    """目标 [n, T, d], 类别块正确; 随机化目标与之不同。"""
    labels = [0, 2, 5]
    d = 192
    t = build_targets(labels, d, N_CLASSES, torch.device("cpu"), n_tokens=4)
    blk = d // N_CLASSES
    assert t.shape == (3, 4, d), t.shape
    for i, c in enumerate(labels):
        assert t[i, :, c * blk:(c + 1) * blk].min() > 0.9
        rest = torch.cat([t[i, 0, :c * blk], t[i, 0, (c + 1) * blk:]])
        assert rest.numel() == 0 or rest.abs().max() < 1e-6
    r = build_targets(labels, d, N_CLASSES, torch.device("cpu"), randomized=True, seed=0)
    assert not torch.allclose(r, t)
    print("[PASS] 目标块正确且随机化目标不同")
    return True


def test_three_probe_families_learn():
    """三族探针在可分数据上都必须学得动 (否则不能用来下"读不出"的结论)。"""
    torch.manual_seed(0)
    x = torch.randn(240, 10)
    y = (x[:, 0] > 0).long()
    acc = probe_accuracy_all(x, y, 2, seed=0)
    for fam, (te, tr) in acc.items():
        assert tr > 0.85, (fam, tr)
    print("[PASS] 三族探针均有效: " + " ".join(f"{k}(train {v[1]:.2f})" for k, v in acc.items()))
    return True


def test_three_probe_families_chance_on_noise():
    """噪声上三族探针都必须接近随机。"""
    torch.manual_seed(1)
    x = torch.randn(300, 16)
    y = torch.randint(0, 3, (300,))
    acc = probe_accuracy_all(x, y, 3, seed=0)
    for fam, (te, tr) in acc.items():
        assert te < 0.60, (fam, te)
    print("[PASS] 噪声上三族探针接近随机: " + " ".join(f"{k}(test {v[0]:.2f})" for k, v in acc.items()))
    return True


def test_residual_lift_capacity_matched():
    """残差抬升的参数量必须与线性臂对齐 (否则 N 轴被容量混淆)。"""
    for D_m in (2560, 1024):
        lin = sum(p.numel() for p in LinearLift(256, D_m).parameters())
        res = sum(p.numel() for p in ResidualLift(256, D_m).parameters())
        assert abs(res - lin) / lin < 0.01, (D_m, lin, res)
    print("[PASS] 残差抬升与线性臂参数量对齐 (<1%)")
    return True


def test_consistency_metric():
    """一致性指标: 相同向量 → 1; 正交 → 0。"""
    a = torch.randn(5, 3, 4)
    same = consistency({"x": a, "y": a.clone()}, None)
    assert abs(same - 1.0) < 1e-4, same
    b = torch.randn(5, 3, 4)
    orth = consistency({"x": a, "y": b}, None)
    assert abs(orth) < 0.4, orth
    print(f"[PASS] 跨底座一致性指标正确 (相同 {same:.3f}, 随机 {orth:+.3f})")
    return True


def test_predictions_and_arms_present():
    """七条预登记判决与四个臂必须齐全 (否则审计不可判定)。"""
    for k in ("D1a", "D1b", "D1c", "D2", "D3", "D4", "D5"):
        assert k in PREDICTIONS, k
    assert set(ARMS) == {"frozen_random", "permuted_target", "learned_lift", "randomized_target"}
    print("[PASS] 七条预登记判决 + 四臂齐全")
    return True


TESTS = [
    test_domain_split_disjoint,
    test_targets_shape_and_randomization,
    test_three_probe_families_learn,
    test_three_probe_families_chance_on_noise,
    test_residual_lift_capacity_matched,
    test_consistency_metric,
    test_predictions_and_arms_present,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
