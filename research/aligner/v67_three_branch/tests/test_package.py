"""V67.0 — 去掉 SSM 分支（3 异构 proposer）权威测试。

本代只做一件事：把 MoA proposer 的分支轮转从 `i % 4` 改成 `i % 3`，
去掉 `SelectiveSSMSubOp` 分支。

理由（两条，缺一不可）:
    1. 机制冲突 —— SSM 的状态维度与显存、并行扫描构成三角冲突，
       各种扫描实现接连 OOM（见 v66 SPEC §4.1）；
    2. 实测更差 —— zero-OOM 版 4 分支 held-out 仅 0.53，输给 3 分支的 0.64
       （git commit c496d6d）。

被锁住的契约:
    1. 轮转确实是 `i % 3`（不是 `i % 4`）
    2. proposer 里不再实例化 SSM 分支
    3. 三种归纳偏置按序轮转（Attn / Conv / SwiGLU）
    4. proposer 前向仍可跑
    5. `SelectiveSSMSubOp` 类本身仍在文件里 —— 删的是「接线」，不是「研究」
    6. zero-OOM 变体里的 SSM 通路仍在（v68 的实验底座没被一起删掉）

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

import ast

import torch

import moa_operators as MOA
import moa_operators_zerooom as MOAZ
import _audit_kit as K

TITLE = "V67.0 去 SSM 分支 — 3 异构 proposer"

OPS = "moa_operators.py"
_B, _S, _D = 2, 12, 64


def _cfg(**kw):
    base = dict(d_model=_D, n_heads=4, num_proposers=4, top_k_proposers=2,
                proposer_mode='heterogeneous', aggregator_mode='sparse_router',
                num_refinement_steps=1, d_state=4, d_ffn=128)
    base.update(kw)
    return MOA.MoAOperatorConfig(**base)


def _kind_seg():
    """HeterogeneousProposerOp 里给 kind 赋值的语句片段。"""
    node = K.class_node(OPS, "HeterogeneousProposerOp")
    text = K.src(OPS)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "kind" for t in sub.targets):
            return ast.get_source_segment(text, sub)
    return None


# ---------------------------------------------------------------- 1
def test_branch_rotation_is_mod_three():
    """1. 分支轮转是 i % 3（v67 的核心改动本身）"""
    seg = _kind_seg()
    assert seg is not None, "未找到 kind 赋值语句"
    assert "% 3" in seg, "轮转应为 i %% 3，实际: %r" % seg
    assert "% 4" not in seg, "轮转不应再是 i %% 4: %r" % seg
    print("[PASS] test_branch_rotation_is_mod_three (%s)" % seg.strip())
    return True


# ---------------------------------------------------------------- 2
def test_proposer_no_longer_instantiates_ssm():
    """2. proposer 里不再实例化 SSM 分支（删的是接线）"""
    node = K.class_node(OPS, "HeterogeneousProposerOp")
    hits = [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "SelectiveSSMSubOp"]
    assert not hits, "proposer 里仍实例化 SSM 分支 %d 处" % len(hits)
    print("[PASS] test_proposer_no_longer_instantiates_ssm (0 处 SSM 实例化)")
    return True


# ---------------------------------------------------------------- 3
def test_three_inductive_biases_rotate_in_order():
    """3. 三种归纳偏置按序轮转：Attention / CausalConv / SwiGLU"""
    op = MOA.HeterogeneousProposerOp(_cfg(num_proposers=6))
    kinds = [type(b).__name__ for b in op.branches]
    assert kinds == ['CausalSelfAttentionSubOp', 'CausalDepthwiseConvSubOp',
                     'SwiGLUMLPSubOp', 'CausalSelfAttentionSubOp',
                     'CausalDepthwiseConvSubOp', 'SwiGLUMLPSubOp'], kinds
    assert len(set(kinds)) == 3, '分支类型应异构（3 种），实测 %s' % kinds
    assert not any('SSM' in k for k in kinds), kinds
    print("[PASS] test_three_inductive_biases_rotate_in_order (%s)" % ','.join(kinds))
    return True


# ---------------------------------------------------------------- 4
def test_proposer_forward_still_runs():
    """4. 去掉一个分支后，proposer 前向仍可跑、形状不变"""
    op = MOA.HeterogeneousProposerOp(_cfg(num_proposers=4))
    p = op(torch.randn(_B, _S, _D))
    assert p.shape == (_B, 4, _S, _D), p.shape
    assert torch.isfinite(p).all(), '输出含 NaN/Inf'
    print("[PASS] test_proposer_forward_still_runs (输出 %s)" % (tuple(p.shape),))
    return True


# ---------------------------------------------------------------- 5
def test_ssm_class_still_present_but_unwired():
    """5. SelectiveSSMSubOp 类本身仍在文件里 —— 删的是接线，不是研究

    这条是本代与「直接删代码」的分野：实现保留，便于后续换回；
    断开的只是 proposer 的接线。
    """
    node = K.class_node(OPS, "SelectiveSSMSubOp")
    assert node is not None, "SelectiveSSMSubOp 类已被删除"
    assert node.lineno < K.class_node(OPS, "HeterogeneousProposerOp").lineno, \
        "SSM 类应仍在 proposer 之前定义"
    print("[PASS] test_ssm_class_still_present_but_unwired (类在 line %d, 未接线)" % node.lineno)
    return True


# ---------------------------------------------------------------- 6
def test_zero_oom_variant_keeps_ssm_path():
    """6. zero-OOM 变体里的 SSM 通路仍在（v68 的四分支实验底座完整）"""
    node = K.class_node("moa_operators_zerooom.py", "HeterogeneousProposerOp")
    text = K.src("moa_operators_zerooom.py")
    segs = [ast.get_source_segment(text, n) for n in ast.walk(node)
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and "branch_kinds" in (ast.get_source_segment(text, n) or "")]
    assert segs, "zero-OOM 变体里未找到 branch_kinds 赋值"
    assert any("% 4" in s for s in segs), \
        "zero-OOM 变体应保留 4 分支轮转，实测 %s" % segs
    ssm_calls = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "SelectiveSSMSubOp"]
    assert ssm_calls, "zero-OOM 变体里未实例化 SSM 分支"
    assert hasattr(MOAZ.NumpyMoAOperatorKernels, "ssm_zero_oom_checkpointed_forward"), \
        "zero-OOM SSM 前向核缺失"
    print("[PASS] test_zero_oom_variant_keeps_ssm_path (%s)" % segs[0].strip())
    return True


TESTS = [
    test_branch_rotation_is_mod_three,
    test_proposer_no_longer_instantiates_ssm,
    test_three_inductive_biases_rotate_in_order,
    test_proposer_forward_still_runs,
    test_ssm_class_still_present_but_unwired,
    test_zero_oom_variant_keeps_ssm_path,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
