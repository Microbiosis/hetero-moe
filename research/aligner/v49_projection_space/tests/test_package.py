"""V49.0 投影空间 — 单元测试 (8 项)。不加载真实模型, 用合成特征。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V49.0 投影空间 — 单元测试 (8 项)"

import numpy as np
import torch
import torch.nn.functional as F

from research.aligner.v49_projection_space.data import Split, SAMPLE_SIZES, CLASSES
from research.aligner.v49_projection_space.mechanism import (
    ARMS, build_projector, param_report, matched_hidden, permute_cond,
)

sys.path.insert(0, os.path.join(_root, "examples"))
from importlib import import_module
_run = import_module("run_v49_projection_space")   # 取 slice_indices / build_targets


def test_split_nested_and_disjoint():
    """样本量档必须嵌套, 且训练/留出文本零交集。"""
    a = Split(SAMPLE_SIZES["S32"], seed=0)
    b = Split(SAMPLE_SIZES["S256"], seed=0)
    c = Split(SAMPLE_SIZES["S1024"], seed=0)
    assert set(a.texts("train")) < set(b.texts("train")) < set(c.texts("train"))
    assert set(c.texts("train")).isdisjoint(set(c.texts("heldout")))
    assert len(c.texts("train")) == 8 * SAMPLE_SIZES["S1024"]
    print(f"[PASS] 档位嵌套 (32⊂256⊂1024) 且训练/留出文本零交集")
    return True


def test_slot_words_unique():
    """槽位词必须互不重复 (重复会造成训练/留出文本相撞)。"""
    for k, (a, n) in CLASSES.items():
        assert len(set(a)) == 16 and len(set(n)) == 16, (k, len(set(a)), len(set(n)))
        assert not (set(a) & set(n)), k
    print("[PASS] 8 个类的槽位词各自唯一且两族不交")
    return True


def test_slice_indices_picks_right_texts():
    """按类切片后的全局下标必须指向**同一类**的文本 (第一版的 bug)。"""
    sp = Split(SAMPLE_SIZES["S256"], seed=0)
    idx = _run.slice_indices(sp.train, SAMPLE_SIZES["S32"])
    picked = [sp.train[i] for i in idx]
    assert len(picked) == 8 * SAMPLE_SIZES["S32"], len(picked)
    # 每个类取到的条数正确
    cnt = {}
    for c, _ in picked:
        cnt[c] = cnt.get(c, 0) + 1
    assert set(cnt.values()) == {SAMPLE_SIZES["S32"]}, cnt
    # 且下标确实指向该类 (不是别的类)
    for i, (c, _) in zip(idx, picked):
        assert sp.train[i][0] == c
    print(f"[PASS] 切片下标指向正确类别 (每类 {SAMPLE_SIZES['S32']} 条)")
    return True


def test_targets_align_with_labels():
    """目标块必须落在该样本所属类别的那一段。"""
    labels = [0, 3, 7, 1]
    t = _run.build_targets(labels, torch.device("cpu"))
    blk = t.shape[-1] // len(CLASSES)
    for i, c in enumerate(labels):
        seg = t[i, 0]
        assert seg[c * blk:(c + 1) * blk].min() > 0.9
        others = torch.cat([seg[:c * blk], seg[(c + 1) * blk:]])
        assert others.abs().max() < 1e-6
    print("[PASS] 目标块与类别标签一一对齐")
    return True


def test_capacity_matched_linear_vs_nonlinear():
    """N 轴上两臂参数量必须几乎相等 (v41 容量混杂的排除)。"""
    for D_m in (2560, 1024):
        lin = param_report(build_projector("L_static", 256, D_m))["total"]
        nl = param_report(build_projector("NL_static", 256, D_m))["total"]
        gap = abs(nl - lin) / lin
        assert gap < 0.005, (D_m, lin, nl, gap)
    print("[PASS] 线性 vs 非线性 臂参数量差 <0.5% (N 轴容量已对齐)")
    return True


def test_arms_form_factorial():
    """臂集合必须覆盖 N(线性/非线性) × C(静态/条件) 的 2×2。"""
    grid = {(lift, cond) for lift, cond in ARMS.values()}
    for lift in ("linear", "nonlinear"):
        for cond in (False, True):
            assert (lift, cond) in grid, (lift, cond)
    print("[PASS] 四臂构成完整的 2×2 析因 (非线性 × 条件依赖)")
    return True


def test_cond_modulation_behaviour():
    """条件轴: U=0 时打乱条件无影响; U≠0 时打乱条件必须改变输出。"""
    x = torch.randn(6, 4, 1024)
    p = build_projector("L_cond", 256, 1024)
    c = p.condition_from(x)
    y1, y2 = p(x, c), p(x, permute_cond(c, 3))
    assert float((y1 - y2).abs().max()) < 1e-9, "U=0 时条件不应有影响"
    with torch.no_grad():
        p.U.weight.normal_(0, 0.05)
    y1, y2 = p(x, c), p(x, permute_cond(c, 3))
    assert float((y1 - y2).abs().max()) > 1e-8, "U≠0 时打乱条件必须改变输出"
    print("[PASS] 条件调制行为正确 (初始无影响, 训练后影响输出)")
    return True


def test_permute_cond_is_permutation():
    """打乱条件必须是**置换** (不改值, 只改归属)。"""
    c = torch.arange(20).float().unsqueeze(1)
    p = permute_cond(c, 5)
    assert sorted(p.flatten().tolist()) == list(range(20))
    assert not torch.equal(p, c)
    print("[PASS] 条件打乱是保持值域的置换")
    return True


TESTS = [
    test_split_nested_and_disjoint,
    test_slot_words_unique,
    test_slice_indices_picks_right_texts,
    test_targets_align_with_labels,
    test_capacity_matched_linear_vs_nonlinear,
    test_arms_form_factorial,
    test_cond_modulation_behaviour,
    test_permute_cond_is_permutation,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
