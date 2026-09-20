"""§4 / §5 / §8.1 / §8.3 — 训练协议与推理基准端到端验证。

§8.1 验收: Phase1 Loss 稳定下降, α_m 不坍缩为等值, Phase2 W_router 不退化。
§8.3 验收: C=1.0/1.25/1.5 溢出率与延迟对照, 重归一化避免分布偏移。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from hetero_fusion.core.fusion import HeteroFusionLayer
from hetero_fusion.train.trainer import HeteroTrainer
from hetero_fusion.infer.inference import InferenceRunner


def _build(D=32, D_FF=64, M=4, K=2, n_layers=2, V=100, seed=7):
    torch.manual_seed(seed)
    layers = nn.ModuleList([HeteroFusionLayer(D, D_FF, M, K) for _ in range(n_layers)])
    lm_head = nn.Linear(D, V, bias=False)
    lm_head.weight.requires_grad_(False)  # 冻结输出头
    return layers, lm_head


def test_phase1_loss_decreases():
    """§8.1: Phase1 Loss 稳定下降。"""
    layers, head = _build()
    trainer = HeteroTrainer(layers, lm_head=head)
    trainer.begin_phase(1)
    x = torch.randn(4, 16, 32)
    tgt = torch.randint(0, 100, (4, 16))
    losses = [trainer.step(x, tgt)["total"] for _ in range(30)]
    assert losses[-1] < losses[0], f"Phase1 Loss 未下降: {losses[0]:.3f}→{losses[-1]:.3f}"


def test_phase1_router_stays_frozen():
    """§4.2: 整个 Phase1 期间 W_router 不更新 (值不变)。"""
    layers, head = _build()
    trainer = HeteroTrainer(layers, lm_head=head)
    before = layers[0].W_router.detach().clone()
    trainer.begin_phase(1)
    x = torch.randn(4, 16, 32); tgt = torch.randint(0, 100, (4, 16))
    for _ in range(10):
        trainer.step(x, tgt)
    after = layers[0].W_router.detach()
    assert torch.allclose(before, after, atol=1e-10), "Phase1 期间 W_router 被修改"


def test_phase2_router_updates():
    """§8.1: Phase2 W_router 不退化为常量 (实际更新)。"""
    layers, head = _build()
    trainer = HeteroTrainer(layers, lm_head=head)
    trainer.begin_phase(2)
    before = layers[0].W_router.detach().clone()
    x = torch.randn(4, 16, 32); tgt = torch.randint(0, 100, (4, 16))
    for _ in range(15):
        trainer.step(x, tgt)
    after = layers[0].W_router.detach()
    assert not torch.allclose(before, after, atol=1e-8), "Phase2 W_router 未更新"


def test_phase2_loss_decreases():
    """§8.1: Phase2 Loss 下降。"""
    layers, head = _build()
    trainer = HeteroTrainer(layers, lm_head=head)
    trainer.begin_phase(2)
    x = torch.randn(4, 16, 32); tgt = torch.randint(0, 100, (4, 16))
    losses = [trainer.step(x, tgt)["total"] for _ in range(30)]
    assert losses[-1] < losses[0]


def test_alpha_not_collapsed():
    """§8.1: α_m 训练后不坍缩为等值 (存在区分度)。"""
    layers, head = _build()
    trainer = HeteroTrainer(layers, lm_head=head)
    trainer.begin_phase(2)
    x = torch.randn(8, 16, 32); tgt = torch.randint(0, 100, (8, 16))
    for _ in range(40):
        trainer.step(x, tgt)
    a = torch.cat([p.detach().flatten() for p in layers[0].alphas])
    spread = a.max().item() - a.min().item()
    assert spread > 1e-4, f"α_m 坍缩为等值, spread={spread}"


def test_inference_benchmark_three_C():
    """§8.3: C=1.0/1.25/1.5 三组对照, 输出可运行且溢出单调。"""
    layers, head = _build()
    runner = InferenceRunner(layers, capacity_factor=1.25, lm_head=head)
    x = torch.randn(4, 32, 32)
    res = runner.benchmark(x, capacity_factors=(1.0, 1.25, 1.5))
    assert set(res.keys()) == {1.0, 1.25, 1.5}
    # C 越大, 总溢出 token 数应非增
    ofs = [res[c]["total_overflow_tokens"] for c in (1.0, 1.25, 1.5)]
    assert ofs[2] <= ofs[0], f"溢出未随 C 增大而减少: {ofs}"


def test_renorm_avoids_magnitude_compression():
    """§8.3: 动态重归一化避免输出分布偏移 (幅值未被压缩)。
    对比: 启用重归一化 vs 禁用(直接截断不归一) 的输出范数。"""
    from hetero_fusion.core.router import capacity_route
    torch.manual_seed(11)
    layer = HeteroFusionLayer(32, 64, 4, 2)
    x = torch.randn(2, 16, 32)
    # 均衡路由 (不易溢出), 启用 renorm
    z = torch.randn(2, 16, 4)
    alpha = torch.softmax(z, dim=-1)
    alpha_t, _, _ = capacity_route(alpha, 2, 1.0)
    y_renorm, _, _ = layer(x, routed_weights=alpha_t)
    # 禁用 renorm: 直接用原始 sparse alpha (会压缩幅值)
    from hetero_fusion.core.router import SparseRouterSTE
    alpha_sparse = SparseRouterSTE.apply(alpha, 2)
    y_norenorm, _, _ = layer(x, routed_weights=alpha_sparse)
    # 输出范数: renorm 应 >= norenorm (未被压缩)
    norm_renorm = y_renorm.norm().item()
    norm_norenorm = y_norenorm.norm().item()
    assert norm_renorm >= norm_norenorm - 1e-3, (
        f"重归一化未避免幅值压缩: renorm={norm_renorm:.3f} < norenorm={norm_norenorm:.3f}"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\nAll train/infer tests passed.")
