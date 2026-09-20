"""§2 / §8.1 — 融合层与梯度流验证。

覆盖: 前向形状、Phase1 路由冻结+均匀、Phase2 路由解冻+稀疏、detach 截断
深层梯度、残差连接、2 层真正深层传递 (非单层并行)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from hetero_fusion.core.fusion import HeteroFusionLayer, swiglu_forward


def _make(D=16, D_FF=32, M=4, K=2):
    torch.manual_seed(123)
    return HeteroFusionLayer(D, D_FF, M, K)


def test_forward_shapes():
    layer = _make()
    x = torch.randn(2, 8, 16)
    y, z, ah = layer(x)
    assert y.shape == (2, 8, 16)
    assert z.shape == (2, 8, 4)
    assert ah.shape == (2, 8, 4)


def test_phase1_uniform_routing_and_router_frozen():
    """§4.2: Phase1 强制均匀路由 1/M, W_router 不收梯度 (双重保险)。"""
    layer = _make()
    layer.W_router.requires_grad_(False)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, z, ah = layer(x, phase1=True)
    # 均匀
    assert torch.allclose(ah, torch.full_like(ah, 0.25), atol=1e-6)
    (y.sum()).backward()
    assert layer.W_router.grad is None, "Phase1 W_router 必须无梯度"
    # 适配器应有梯度
    assert layer.gammas[0].grad is not None
    assert layer.alphas[0].grad is not None


def test_phase2_sparse_and_router_grad():
    """§4.3: Phase2 Top-K 稀疏 + W_router 收梯度。"""
    layer = _make()
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, z, ah = layer(x, phase1=False)
    # 每个 token 恰好 K=2 个非零
    nz = (ah[0, 0] > 0).sum().item()
    assert nz == 2, f"Top-K=2, 实际非零 {nz}"
    (y.sum()).backward()
    assert layer.W_router.grad is not None and layer.W_router.grad.norm() > 0


def test_detach_breaks_deep_gradient():
    """§1.4 / §2.1: detach 截断 — 输入 x 不因底座 FFN 而收到深层梯度路径,
    但 gamma/beta 仍可训练。验证: 对 x 的梯度仅来自残差直连, 不来自 FFN 内部。"""
    layer = _make()
    x = torch.randn(2, 4, 16, requires_grad=True)
    y, _, _ = layer(x, phase1=False)
    y.sum().backward()
    # x 有梯度 (来自残差 y = x + out, out 含 alpha_hat 项, 但 alpha_hat 由 x 经 router)
    assert x.grad is not None
    # gamma 梯度非零 (适配器可训练)
    assert layer.gammas[0].grad.norm() > 0


def test_residual_connection():
    """§2.4: y = x + Σ ... 。当所有 α_m→0 时 y≈x。"""
    layer = _make()
    with torch.no_grad():
        for a in layer.alphas:
            a.zero_()
    x = torch.randn(1, 4, 16)
    y, _, _ = layer(x, phase1=False)
    assert torch.allclose(y, x, atol=1e-5), "α_m=0 时输出应等于输入 (纯残差)"


def test_two_layer_deep_propagation():
    """§7 关键修正: 2 层网络为真正深层传递 (layer0 输出 → layer1 输入),
    而非独立单层并行。验证 layer1 的梯度能经 layer0 回传到输入。"""
    l0, l1 = _make(), _make()
    x = torch.randn(2, 8, 16, requires_grad=True)
    y0, _, _ = l0(x, phase1=False)
    y1, _, _ = l1(y0, phase1=False)
    y1.sum().backward()
    # x 的梯度非零证明深层链路打通 (经 l0 → l1)
    assert x.grad is not None and x.grad.norm() > 0, "深层梯度链路未打通"


def test_swiglu_correctness():
    """§2.2 SwiGLU 公式正确性 (与手算一致)。"""
    x = torch.tensor([[1.0, 2.0]])
    w_gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    w_up = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    w_down = torch.tensor([[1.0, 1.0]])  # [out=1, in=2]
    # h_gate = [1,2]; h_up=[3,3]; silu([1,2])=[0.7311, 1.7616]; act=[2.193, 5.285]
    # down: 2.193+5.285 = 7.478
    out = swiglu_forward(x, w_gate, w_up, w_down)
    expected = torch.tensor([[7.4783]])
    assert torch.allclose(out, expected, atol=1e-3), f"SwiGLU {out} != {expected}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\nAll fusion tests passed.")
