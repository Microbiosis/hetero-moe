"""V68.0 — 架构图 + zero-OOM SSM 四分支 + 单模态消融 权威测试。

本代三件事:
    1. 恢复四分支并配 **zero-OOM SSM** 路径（autograd checkpointing + 状态 tile），
       目标是让 `d_state=16` 的 SSM 能在 T4 上跑起来；
    2. 交付三模态异构融合 pipeline 架构图（`hetero_arch_v68.png`）；
    3. 加单模态消融开关（`ABL=text/all/image`），用来回答
       「融合的增益到底来自视觉通道，还是来自多一个文本底座」。

⚠️ 本代同时是**一次被实测否掉的路线**：
    四分支（含 zero-OOM SSM）held-out 仅 **0.53**，输给三分支的 **0.64**
    （git commit c496d6d）⇒ 后续 v67 去掉 SSM 分支。本代保留这份负结果。

被锁住的契约:
    1. zero-OOM 前向与串行参考实现一致
    2. zero-OOM 解析反向与**数值有限差分**一致
    3. zero-OOM 峰值显存低于 SSD 与 Kogge-Stone 两种旧实现
    4. 架构图是合法 PNG 且尺寸足够
    5. ABL 消融开关真的能拿掉 ViT 通道
    6. 四分支（含 SSM）实验底座完整保留在 zero-OOM 变体里

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
import struct

import numpy as np

import moa_operators_zerooom as MOAZ
import _audit_kit as K

TITLE = "V68.0 zero-OOM SSM + 架构图 + 单模态消融"

KER = MOAZ.NumpyMoAOperatorKernels
ZERO = "moa_operators_zerooom.py"
TRAIN = "train_v2.py"

# 小尺寸：够触发多 chunk + 多 state tile，又能让有限差分跑得快
_B, _L, _D, _N = 1, 8, 3, 2
_CHUNK, _TILE = 4, 2


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((_B, _L, _D))
    dt = np.abs(rng.standard_normal((_B, _L, _D))) * 0.1 + 0.05
    A = -np.abs(rng.standard_normal((_D, _N))) * 0.5
    B_ssm = rng.standard_normal((_B, _L, _N))
    C_ssm = rng.standard_normal((_B, _L, _N))
    return u, dt, A, B_ssm, C_ssm


def _forward(u, dt, A, B_ssm, C_ssm):
    y, hb, stats = KER.ssm_zero_oom_checkpointed_forward(
        u, dt, A, B_ssm, C_ssm, chunk_size=_CHUNK, state_tile=_TILE)
    return y, hb, stats


# ---------------------------------------------------------------- 1
def test_zero_oom_forward_matches_sequential_reference():
    """1. zero-OOM 前向与串行参考实现一致（自研扫描必须等价于朴素扫描）"""
    u, dt, A, B_ssm, C_ssm = _inputs(0)
    y, hb, stats = _forward(u, dt, A, B_ssm, C_ssm)
    # 参考：朴素 O(L) 串行扫描（用物化的 dA / dB_u）
    dA = np.exp(dt[:, :, :, None] * A[None, None, :, :])
    dB_u = dt[:, :, :, None] * u[:, :, :, None] * B_ssm[:, :, None, :]
    y_ref, _launches = KER.ssm_sequential_scan_ref(dA, dB_u, C_ssm)
    diff = float(np.max(np.abs(y - y_ref)))
    assert y.shape == (_B, _L, _D), y.shape
    assert diff < 1e-10, "与串行参考差异 %g" % diff
    print("[PASS] test_zero_oom_forward_matches_sequential_reference (max diff=%.2e)" % diff)
    return True


# ---------------------------------------------------------------- 2
def test_zero_oom_backward_matches_finite_difference():
    """2. zero-OOM 解析反向与数值有限差分一致（五个参数全查）"""
    u, dt, A, B_ssm, C_ssm = _inputs(1)
    w = np.random.default_rng(7).standard_normal((_B, _L, _D))

    def loss_of(u_, dt_, A_, B_, C_):
        y, _, _ = _forward(u_, dt_, A_, B_, C_)
        return float(np.sum(y * w))

    y, hb, _ = _forward(u, dt, A, B_ssm, C_ssm)
    grads = KER.ssm_zero_oom_checkpointed_backward(
        w, u, dt, A, B_ssm, C_ssm, hb, chunk_size=_CHUNK, state_tile=_TILE)
    for key in ("d_u", "d_dt", "d_A", "d_B", "d_C"):
        assert key in grads, "反向未返回 %s，实际 %s" % (key, sorted(grads))

    eps = 1e-6
    rng = np.random.default_rng(3)
    checks = [
        ("d_u", u, (0, 2, 1)),
        ("d_dt", dt, (0, 3, 2)),
        ("d_A", A, (1, 0)),
        ("d_B", B_ssm, (0, 1, 1)),
        ("d_C", C_ssm, (0, 5, 0)),
    ]
    worst = 0.0
    for key, arr, idx in checks:
        orig = arr[idx]
        arr[idx] = orig + eps
        lp = loss_of(u, dt, A, B_ssm, C_ssm)
        arr[idx] = orig - eps
        lm = loss_of(u, dt, A, B_ssm, C_ssm)
        arr[idx] = orig
        fd = (lp - lm) / (2 * eps)
        an = float(grads[key][idx])
        err = abs(fd - an) / max(abs(fd), 1e-8)
        worst = max(worst, err)
        assert err < 1e-4, "%s[%s] 解析梯度 %g vs 有限差分 %g（相对误差 %g）" % (
            key, idx, an, fd, err)
    print("[PASS] test_zero_oom_backward_matches_finite_difference (最大相对误差 %.2e)" % worst)
    return True


# ---------------------------------------------------------------- 3
def test_zero_oom_peak_below_ssd_and_kogge_stone():
    """3. zero-OOM 峰值显存低于 SSD 与 Kogge-Stone 两种旧实现（本代的核心主张）"""
    u, dt, A, B_ssm, C_ssm = _inputs(2)
    _y, _hb, stats = _forward(u, dt, A, B_ssm, C_ssm)
    new_kb = stats["new_zero_oom_peak_kb"]
    ssd_kb = stats["old_ssd_peak_kb"]
    ks_kb = stats["old_kogge_stone_peak_kb"]
    assert new_kb < ssd_kb, "zero-OOM 峰值 %d KB 未低于 SSD 的 %d KB" % (new_kb, ssd_kb)
    assert new_kb < ks_kb, "zero-OOM 峰值 %d KB 未低于 Kogge-Stone 的 %d KB" % (new_kb, ks_kb)
    print("[PASS] test_zero_oom_peak_below_ssd_and_kogge_stone (new %dKB < ssd %dKB, ks %dKB)"
          % (new_kb, ssd_kb, ks_kb))
    return True


# ---------------------------------------------------------------- 4
def test_arch_diagram_is_valid_png():
    """4. 架构图是合法 PNG 且尺寸足够（不是占位空文件）"""
    path = os.path.join(_v62, "hetero_arch_v68.png")
    assert os.path.exists(path), "架构图缺失: %s" % path
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "不是合法 PNG"
    w, h = struct.unpack(">II", data[16:24])
    assert w >= 1000 and h >= 500, "尺寸过小: %dx%d" % (w, h)
    assert len(data) > 50000, "文件过小，可能是占位: %d 字节" % len(data)
    print("[PASS] test_arch_diagram_is_valid_png (%dx%d, %d 字节)" % (w, h, len(data)))
    return True


# ---------------------------------------------------------------- 5
def test_abl_switch_removes_vit_channel():
    """5. ABL 消融开关真的能拿掉 ViT 通道（单模态消融的前提）"""
    src = K.def_source(TRAIN, "_feats")
    ns_text = src
    defs = {}
    for abl in ("all", "text", "image"):
        g = {"ABL": abl}
        exec(compile(ns_text, "_feats", "exec"), g)
        defs[abl] = g["_feats"]
    out_all = defs["all"](1, 2, 3)
    out_text = defs["text"](1, 2, 3)
    assert "ViT" in out_all, "ABL=all 应保留 ViT，实测 %s" % sorted(out_all)
    assert "ViT" not in out_text, "ABL=text 应拿掉 ViT，实测 %s" % sorted(out_text)
    assert "Qwen" in out_text and "BERT" in out_text, sorted(out_text)
    ns = K.exec_assigns(TRAIN, ["ABL"])
    assert ns["ABL"] == "all", "默认应是 all，实测 %r" % ns["ABL"]
    ns = K.exec_assigns(TRAIN, ["ABL"], env={"ABL": "text"})
    assert ns["ABL"] == "text", ns["ABL"]
    print("[PASS] test_abl_switch_removes_vit_channel (all/text/image 三档, 默认 all)")
    return True


# ---------------------------------------------------------------- 6
def test_four_branch_ssm_path_preserved():
    """6. 四分支（含 SSM）的实验底座完整保留 —— 本代是负结果，底座不能删"""
    node = K.class_node(ZERO, "HeterogeneousProposerOp")
    text = K.src(ZERO)
    segs = [ast.get_source_segment(text, n) for n in ast.walk(node)
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and "branch_kinds" in (ast.get_source_segment(text, n) or "")]
    assert segs and any("% 4" in s for s in segs), \
        "zero-OOM 变体应保留 4 分支轮转，实测 %s" % segs
    ssm_calls = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "SelectiveSSMSubOp"]
    assert ssm_calls, "zero-OOM 变体里未实例化 SSM 分支"
    for fn in ("ssm_zero_oom_checkpointed_forward", "ssm_zero_oom_checkpointed_backward",
               "ssm_sequential_scan_ref", "ssm_chunkwise_ssd_scan"):
        assert hasattr(KER, fn), "缺 %s" % fn
    print("[PASS] test_four_branch_ssm_path_preserved (4 分支轮转 + 4 个 SSM 核齐备)")
    return True


TESTS = [
    test_zero_oom_forward_matches_sequential_reference,
    test_zero_oom_backward_matches_finite_difference,
    test_zero_oom_peak_below_ssd_and_kogge_stone,
    test_arch_diagram_is_valid_png,
    test_abl_switch_removes_vit_channel,
    test_four_branch_ssm_path_preserved,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
