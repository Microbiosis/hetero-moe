"""§2.2 / §6.2 / §8.2 — 量化算子验证。

覆盖: per-row 与 group_size 路径等价性、分块独立性、STE 梯度直通、
量化误差随位宽/分块单调下降 (§8.2 PPL 对齐诊断依据)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hetero_fusion.core.quant import FakeQuantSTE, fake_quant, quantization_error


def test_per_row_matches_prototype():
    """group_size=None 等价于规范 §7 原型的 per-row 行为。"""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    # 原型 per-row: scale = max(|x|, -1) / 7
    scale = x.abs().amax(dim=-1, keepdim=True) / 7
    scale = torch.clamp(scale, min=1e-5)
    ref = torch.round(x / scale) * scale
    out = fake_quant(x, bits=4, group_size=None)
    assert torch.allclose(out, ref, atol=1e-6), "per-row 路径与原型不一致"


def test_group_degenerates_when_ge_D():
    """group_size >= 末维 D 时退化为 per-row (向后兼容)。"""
    torch.manual_seed(1)
    x = torch.randn(2, 8, 16)
    a = fake_quant(x, bits=4, group_size=None)
    b = fake_quant(x, bits=4, group_size=128)  # 128 >= 16
    assert torch.allclose(a, b, atol=1e-6)


def test_group_quant_partition_independence():
    """D=256, group_size=128 → 2 组各自独立 scale (§6.2 真实实现)。"""
    torch.manual_seed(2)
    x = torch.randn(1, 1, 256)
    out = fake_quant(x, bits=4, group_size=128)
    g1, g2 = out[0, 0, :128], out[0, 0, 128:]
    # 每组应是自身 max 的整数倍
    s1 = x[0, 0, :128].abs().max() / 7
    s2 = x[0, 0, 128:].abs().max() / 7
    assert torch.allclose(g1, torch.round(x[0, 0, :128] / s1.clamp(min=1e-5)) * s1, atol=1e-6)
    assert torch.allclose(g2, torch.round(x[0, 0, 128:] / s2.clamp(min=1e-5)) * s2, atol=1e-6)
    # 两组 scale 不同 (随机数据几乎必然不同)
    assert abs(s1.item() - s2.item()) > 1e-3


def test_ste_gradient_passes_through_round():
    """STE: 反向梯度直通, round 不阻断 (§6.2)。"""
    x = (torch.randn(8) * 3).requires_grad_(True)  # 叶节点, 可接收梯度
    y = FakeQuantSTE.apply(x, 4, 128)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6), "STE 应恒等直通"


def test_higher_bits_lower_error():
    """§8.2 诊断: 位宽越高, 量化误差越小。INT8 < INT4。"""
    torch.manual_seed(3)
    x = torch.randn(4, 64, 256)
    e4 = quantization_error(x, bits=4, group_size=128)
    e8 = quantization_error(x, bits=8, group_size=128)
    assert e8 < e4, f"INT8 误差 {e8:.4f} 应小于 INT4 {e4:.4f}"


def test_smaller_group_lower_error():
    """§8.2: 更小 group_size → 更细 scale → 更低误差 (调大 group_size 反之)。"""
    torch.manual_seed(4)
    x = torch.randn(4, 64, 512)
    e_small = quantization_error(x, bits=4, group_size=64)
    e_large = quantization_error(x, bits=4, group_size=512)
    assert e_small < e_large, "更小分块应降低量化误差"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\nAll quant tests passed.")
