"""V70.0 — VL 跨模态融合旁路 权威测试。

本代把「视觉与文本的交互」从四槽对齐阶段**旁路**出来：
四槽把三模态预混合进 canonical，视觉/文本的细粒度交互在进融合层前就被抹平了。
VL 旁路直接吃分离的文本流（Qwen+BERT）与视觉流（ViT），
用「低秩跨模态锚点共注意力（零距离衰减）+ Hadamard-Bilinear 门控」做显式两路交互，
输出以**残差**方式并联回 canonical（不破坏四槽结构）。

实现文件 `vl_fusion.py` 在 v62 包内（平铺 import）。

被锁住的契约:
    1. 算子前向形状正确（in_proj 二分后融合再投影回去）
    2. `forward_from_projected` 两路序列长度必须相等（内部要逐位相加）
    3. beta=0 时输出是视觉与文本的**凸组合**（落在两者之间）
    4. 锚点共注意力**零距离衰减**：末位文本 token 会影响首位视觉位置
    5. 门控参数（alpha_vis / alpha_txt / beta_bilin）梯度非零
    6. USE_VL 环境变量可切
    7. VL 输出被投到 canonical 维度再残差（v70c 修的维度 bug）

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

import torch
import torch.nn.functional as F

import vl_fusion as VF
import _audit_kit as K

TITLE = "V70.0 VL 跨模态融合旁路"

OP = VF.CrossModalVisionLanguageFusionSubOp
TRAIN = "train_v2.py"
_B, _L, _D = 2, 12, 64


def _op(seed=0):
    torch.manual_seed(seed)
    return OP(d_model=_D, num_anchors=8)


def _vis_stream(op, v_raw):
    """复算视觉支路（内部中间量，测试需要它才能验证凸组合性质）。"""
    return F.silu(op.spatial_dw_conv(v_raw.transpose(1, 2)).transpose(1, 2))


# ---------------------------------------------------------------- 1
def test_cross_modal_op_forward_shape():
    """1. 算子前向形状正确（in_proj 二分 -> 融合 -> out_proj）"""
    op = _op(0)
    x = torch.randn(_B, _L, _D)
    y = op(x)
    assert y.shape == (_B, _L, _D), y.shape
    assert torch.isfinite(y).all(), '输出含 NaN/Inf'
    print("[PASS] test_cross_modal_op_forward_shape (输出 %s)" % (tuple(y.shape),))
    return True


# ---------------------------------------------------------------- 2
def test_forward_from_projected_requires_equal_lengths():
    """2. forward_from_projected 两路序列长度必须相等（内部逐位相加）"""
    op = _op(1)
    v_raw = torch.randn(_B, _L, _D)
    t_raw = torch.randn(_B, _L, _D)
    y = op.forward_from_projected(v_raw, t_raw)
    assert y.shape == (_B, _L, _D), y.shape
    bad = 0
    try:
        op.forward_from_projected(v_raw, torch.randn(_B, _L + 3, _D))
    except RuntimeError:
        bad = 1
    assert bad == 1, "长度不等时应当报错（内部 v_vis + t_txt 逐位相加）"
    print("[PASS] test_forward_from_projected_requires_equal_lengths (不等长报错)")
    return True


# ---------------------------------------------------------------- 3
def test_gate_is_convex_combination_when_beta_zero():
    """3. beta=0 时输出是视觉与文本的**凸组合**（逐维落在两者之间）"""
    op = _op(2)
    with torch.no_grad():
        op.beta_bilin.zero_()
        # out_proj 是最后一步线性投影；把它置成恒等，
        # 这样「凸组合」这条门控性质才能直接在输出上验证
        op.out_proj.weight.copy_(torch.eye(_D))
    v_raw = torch.randn(_B, _L, _D)
    t_raw = torch.randn(_B, _L, _D)
    with torch.no_grad():
        out = op.forward_from_projected(v_raw, t_raw)
        v_vis = _vis_stream(op, v_raw)
        t_txt = F.silu(t_raw)
    lo = torch.minimum(v_vis, t_txt)
    hi = torch.maximum(v_vis, t_txt)
    tol = 1e-5
    assert bool((out >= lo - tol).all()), "凸组合下界被突破"
    assert bool((out <= hi + tol).all()), "凸组合上界被突破"
    print("[PASS] test_gate_is_convex_combination_when_beta_zero (逐维落在两路之间)")
    return True


# ---------------------------------------------------------------- 4
def test_anchor_attention_has_zero_distance_decay():
    """4. 锚点共注意力**零距离衰减**：末位文本 token 也要影响首位视觉位置"""
    op = _op(3)
    v_raw = torch.randn(_B, _L, _D)
    t_raw = torch.randn(_B, _L, _D)
    with torch.no_grad():
        base = op.forward_from_projected(v_raw, t_raw)
        t2 = t_raw.clone()
        t2[:, -1, :] += 3.0
        moved = op.forward_from_projected(v_raw, t2)
    delta = float((moved[:, 0, :] - base[:, 0, :]).abs().max())
    assert delta > 1e-4, "末位文本改动未影响首位视觉输出（存在距离衰减？）delta=%g" % delta
    print("[PASS] test_anchor_attention_has_zero_distance_decay (首位输出变化 %.4g)" % delta)
    return True


# ---------------------------------------------------------------- 5
def test_gradients_flow_to_gate_parameters():
    """5. 门控参数（alpha_vis / alpha_txt / beta_bilin）梯度非零"""
    op = _op(4)
    v_raw = torch.randn(_B, _L, _D, requires_grad=True)
    t_raw = torch.randn(_B, _L, _D, requires_grad=True)
    out = op.forward_from_projected(v_raw, t_raw)
    out.sum().backward()
    for name in ("alpha_vis", "alpha_txt", "beta_bilin"):
        p = getattr(op, name)
        assert p.grad is not None, "%s 无梯度" % name
        assert float(p.grad.abs().sum()) > 0.0, "%s 梯度为 0" % name
    assert v_raw.grad is not None and float(v_raw.grad.abs().sum()) > 0.0
    assert t_raw.grad is not None and float(t_raw.grad.abs().sum()) > 0.0
    print("[PASS] test_gradients_flow_to_gate_parameters (三组门控参数 + 两路输入梯度均非零)")
    return True


# ---------------------------------------------------------------- 6
def test_use_vl_env_flag():
    """6. USE_VL 环境变量可切"""
    ns = K.exec_assigns(TRAIN, ["_USE_VL"])
    assert ns["_USE_VL"] is False, ns["_USE_VL"]
    ns = K.exec_assigns(TRAIN, ["_USE_VL"], env={"USE_VL": "1"})
    assert ns["_USE_VL"] is True, ns["_USE_VL"]
    print("[PASS] test_use_vl_env_flag (默认 off, 可切 on)")
    return True


# ---------------------------------------------------------------- 7
def test_vl_output_projected_to_canonical_dim():
    """7. VL 输出先投到 canonical 维度再残差（v70c 修的 256/1024 维度 bug）"""
    src = K.src(TRAIN)
    assert "s.vl_out_proj = nn.Linear(d_shared, s.d_space, bias=False)" in src, \
        "未找到 vl_out_proj 投到 d_space 的定义"
    assert "canon = canon + s.vl_out_proj(vl_out)" in src, \
        "VL 输出未以残差方式并联进 canonical"
    print("[PASS] test_vl_output_projected_to_canonical_dim (vl_out -> d_space 后残差)")
    return True


TESTS = [
    test_cross_modal_op_forward_shape,
    test_forward_from_projected_requires_equal_lengths,
    test_gate_is_convex_combination_when_beta_zero,
    test_anchor_attention_has_zero_distance_decay,
    test_gradients_flow_to_gate_parameters,
    test_use_vl_env_flag,
    test_vl_output_projected_to_canonical_dim,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
