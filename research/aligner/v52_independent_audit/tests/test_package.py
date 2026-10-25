"""V52.0 独立复算 — 单元测试 (7 项)。不加载模型。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V52.0 独立复算 — 单元测试 (7 项)"

import numpy as np
import torch

from research.aligner.v52_independent_audit.audit import (
    CLASSES, ARMS, PREDICTIONS, MLPLift, LinearLift, make_split, build_targets,
    probe_accuracy, RidgeProbe, N_CLASSES,
)


def test_train_heldout_combinations_disjoint():
    """留出必须是**未见组合** (奇偶划分) —— 比"换扰动种子"更严。"""
    tr, ho = make_split(8, seed=0)
    tr_t = {t for _, t in tr}
    ho_t = {t for _, t in ho}
    assert not (tr_t & ho_t), sorted(tr_t & ho_t)[:5]
    assert len(tr_t) == len(tr) and len(ho_t) == len(ho), "每个类内的组合应互不重复"
    print(f"[PASS] 训练/留出组合零交集 (各 {len(tr_t)}/{len(ho_t)} 条)")
    return True


def test_targets_block_and_randomization():
    """类别目标落在对应块; 随机化目标必须与类别无关。"""
    labels = [0, 3, 7]
    d = 256
    t = build_targets(labels, d, N_CLASSES, torch.device("cpu"), n_tokens=4)
    blk = d // N_CLASSES
    assert t.shape == (len(labels), 4, d), t.shape
    for i, c in enumerate(labels):
        assert t[i, :, c * blk:(c + 1) * blk].min() > 0.9
        rest = torch.cat([t[i, 0, :c * blk], t[i, 0, (c + 1) * blk:]])
        assert rest.numel() == 0 or rest.abs().max() < 1e-6
    r = build_targets(labels, d, N_CLASSES, torch.device("cpu"), randomized=True, seed=0)
    assert not torch.allclose(r, t)
    print("[PASS] 类别目标块正确, 随机化目标与之不同")
    return True


def test_ridge_probe_learns_separable():
    """岭回归探针在可分数据上必须学得动。"""
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    y = (x[:, 0] > 0).long()
    acc = probe_accuracy(x, y, 2, seed=0)
    assert acc > 0.9, acc
    print(f"[PASS] 岭回归探针在可分数据上有效 (acc {acc:.2f})")
    return True


def test_ridge_probe_chance_on_noise():
    """噪声上必须接近随机 (否则探针会给出假的高准确率)。"""
    torch.manual_seed(1)
    x = torch.randn(400, 16)
    y = torch.randint(0, 3, (400,))
    acc = probe_accuracy(x, y, 3, seed=0)
    assert acc < 0.55, acc
    print(f"[PASS] 噪声上探针接近随机 (acc {acc:.2f}, 随机 0.33)")
    return True


def test_ridge_probe_is_deterministic():
    """闭式探针必须无随机性 (与 v50/v51 的 SGD 探针不同的一点)。"""
    torch.manual_seed(2)
    x = torch.randn(120, 6)
    y = torch.randint(0, 3, (120,))
    a1 = probe_accuracy(x, y, 3, seed=0)
    a2 = probe_accuracy(x, y, 3, seed=0)
    assert abs(a1 - a2) < 1e-9, (a1, a2)
    print(f"[PASS] 岭回归探针确定性 (两次同为 {a1:.3f})")
    return True


def test_mlp_lift_capacity_matched_to_linear():
    """非线性抬升的参数量必须与线性臂对齐 (否则 N 轴会被容量混淆)。"""
    for D_m in (2560, 1024):
        lin = sum(p.numel() for p in LinearLift(256, D_m).parameters())
        mlp = sum(p.numel() for p in MLPLift(256, D_m).parameters())
        gap = abs(mlp - lin) / lin
        assert gap < 0.01, (D_m, lin, mlp, gap)
    print("[PASS] MLP 抬升与线性臂参数量对齐 (<1%)")
    return True


def test_arms_and_predictions_present():
    """四个臂与五条预登记判决必须齐全 (否则审计不可判定)。"""
    assert set(ARMS) == {"frozen_random", "permuted_target", "learned_lift", "randomized_target"}
    for k in ("C1a", "C1b", "C1c", "C2", "C3"):
        assert k in PREDICTIONS, k
    print("[PASS] 四臂 + 五条预登记判决齐全")
    return True


TESTS = [
    test_train_heldout_combinations_disjoint,
    test_targets_block_and_randomization,
    test_ridge_probe_learns_separable,
    test_ridge_probe_chance_on_noise,
    test_ridge_probe_is_deterministic,
    test_mlp_lift_capacity_matched_to_linear,
    test_arms_and_predictions_present,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
