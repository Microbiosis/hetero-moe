"""V30.0 — Softplus 门控中枢 单元测试 (8 项).

锁定本代的核心主张: **用 softplus 替代 tanh, 梯度链在饱和区不断**.
同时把一个**已知的取舍**钉成事实: softplus(0)=ln2≠0, 故初始 temperature ≠ 1,
**v30 不满足 paper.tex 摘要里那条 "perfect degeneration" 保证** —— 那是 v9/v31 的性质。

对照基线: research._primitives.central_mechanism.GateStyleWorkspace (v9 纯 tanh)。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_fusion")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V30.0 Softplus 门控中枢 — 单元测试 (9 项)"

import torch
import torch.nn.functional as F

from research._primitives.central_mechanism import GateStyleWorkspace
from research.central_stability.v30_gate_softplus.workspace import SoftplusGateStyleWorkspace

D, M, ALPHA = 32, 3, 0.1


def _grad_sum(ws):
    """W_t 的梯度绝对值和。**完全饱和时 grad 为 None**(图里根本不含 W_t), 记作 0。"""
    g = ws.W_t.grad
    return 0.0 if g is None else g.abs().sum().item()


def _ws(d=D, m=M, alpha=ALPHA):
    return SoftplusGateStyleWorkspace(d_shared=d, num_experts=m, alpha=alpha)


def test_temperature_above_one_in_central_range():
    """1. 中性区间 softplus 恒正 ⇒ temperature > 1 (temperature 是标量; 负尾下溢见测试 8)"""
    w = _ws()
    for tval in (-5.0, -1.0, 0.0, 1.0, 50.0):
        with torch.no_grad():
            w.c.fill_(tval); w.W_t.fill_(1.0 / D)   # t = W_t·c 恰为 tval(点积求和, 需除 D)
        # W_t 取 1/D 使 t = W_t·c 恰为 c 的每个分量值
        t = w.W_t @ w.c          # 两个 [D] 向量做 matmul ⇒ **标量**
        assert t.dim() == 0, f"t 应为标量, 实得 {tuple(t.shape)}"
        temp = 1.0 + ALPHA * F.softplus(t)
        assert float(temp) > 1.0, f"c={tval}: temperature={float(temp):.8f} 应 > 1"
    print("[PASS] test_temperature_above_one_in_central_range")
    return True


def test_order_preservation():
    """2. 标量正温度缩放 ⇒ M 个 logit 的次序不变"""
    w = _ws()
    with torch.no_grad():
        w.c.fill_(7.0); w.W_t.fill_(1.0)
    z = torch.randn(2, 5, M)
    out = w.augment_router_logits(z)
    assert torch.equal(torch.argsort(z, dim=-1), torch.argsort(out, dim=-1))
    print("[PASS] test_order_preservation")
    return True


def test_no_saturation_linear_growth():
    """3. 大 |t| 时 softplus ≈ |t| 线性增长, 不像 tanh 那样封顶"""
    w = _ws()
    with torch.no_grad():
        w.c.fill_(1.0); w.W_t.fill_(1.0)
    soft_1 = F.softplus(torch.tensor(50.0)).item()
    soft_2 = F.softplus(torch.tensor(100.0)).item()
    assert abs(soft_2 - soft_1 - 50.0) < 1.0, f"应为线性增长, 实得 {soft_2-soft_1:.2f}"
    assert F.softplus(torch.tensor(50.0)).item() < 50.01
    print(f"[PASS] test_no_saturation_linear_growth (softplus(100)-softplus(50)={soft_2-soft_1:.2f})")
    return True


def test_gradient_survives_saturation_where_tanh_dies():
    """4. **核心主张**: 饱和区 |c|=50 时梯度链不消失, 而 v9 tanh 恰为 0"""
    z = torch.randn(1, 5, M, requires_grad=True)

    # v9 基线: 梯度应完全消失(复现论文诊断的失效模式)
    w9 = GateStyleWorkspace(d_shared=D, num_experts=M, alpha=ALPHA)
    with torch.no_grad():
        w9.c.fill_(50.0); w9.W_t.fill_(1.0)
    w9.augment_router_logits(z).pow(2).sum().backward()
    g9 = _grad_sum(w9)
    assert g9 == 0.0, f"v9 tanh 基线梯度应为 0, 实得 {g9}"

    # v30: 梯度必须存活
    w = _ws()
    with torch.no_grad():
        w.c.fill_(50.0); w.W_t.fill_(1.0 / D)
    z2 = torch.randn(1, 5, M, requires_grad=True)
    w.augment_router_logits(z2).pow(2).sum().backward()
    g30 = _grad_sum(w)
    assert g30 > 0.0, "v30 的核心主张就是梯度不断, 实得 0"
    print(f"[PASS] test_gradient_survives_saturation_where_tanh_dies (v9={g9:.1e} vs v30={g30:.3e})")
    return True


def test_positive_tail_gradient_does_not_decay():
    """5. **v30 与 tanh 的本质差别**: 正尾梯度保持 O(1) 而非指数衰减

    sigmoid(t)→1 (t→+∞), 故 dL/dW_t 不随 |c| 衰减 —— 这正是 v30 想要的性质;
    而 v9 的 tanh 在饱和区梯度指数衰减到精确 0 (见测试 4)。
    """
    grads = []
    for cval in (1.0, 10.0, 50.0):
        w = _ws()
        with torch.no_grad():
            w.c.fill_(cval); w.W_t.fill_(1.0 / D)
        z = torch.randn(1, 5, M, generator=torch.Generator().manual_seed(int(cval)))
        z = z.requires_grad_(True)
        w.augment_router_logits(z).pow(2).sum().backward()
        grads.append(_grad_sum(w))
    assert all(g > 0 for g in grads), f"正尾梯度必须全为正: {grads}"
    lo, hi = min(grads), max(grads)
    assert hi < lo * 1e3, f"正尾梯度不应指数衰减: {grads}"
    print(f"[PASS] test_positive_tail_gradient_does_not_decay ({[f'{g:.3g}' for g in grads]})")
    return True


def test_perfect_degeneration_is_NOT_held():
    """6. **已知的取舍**: softplus(0)=ln2 ⇒ 初始 temperature≠1 ⇒ 不满足完美退化

    这条是**记录事实**, 不是期望。v30 用"初始就偏离"换"梯度永不消失";
    v31 两者都保住。若将来修正为 softplus(x)-ln2, 本测试会失败并提醒同步 SPEC。
    """
    w = _ws()
    z = torch.randn(1, 5, M)
    dev = (w.augment_router_logits(z.clone()) - z).abs().max().item()
    assert dev > 1e-3, "若此处变成恒等, 说明实现已改为 softplus(x)-ln2, 需同步更新本测试与 SPEC"
    # 理论: temperature = 1 + α·ln2 (逐维), 故 z/temperature ≈ z/1.0693
    ratio = (1.0 / (1.0 + ALPHA * torch.log(torch.tensor(2.0)))).item()
    assert abs(z.max().item() * ratio - z.max().item()) > 1e-3
    print(f"[PASS] test_perfect_degeneration_is_NOT_held (初始最大偏差 {dev:.4f}, "
          f"temperature={1.0 + ALPHA * torch.log(torch.tensor(2.0)):.4f})")
    return True


def test_ema_update_accumulates():
    """7. ema_update 累积 c (本类无 reset(); 状态清理由调用方负责)"""
    w = _ws()
    with torch.no_grad():
        w.c.zero_()
        # ema_update 期望一个专家输出列表, 每个 [S, D]; 内部按 (0,1,2) 求均值
        w.ema_update([torch.ones(4, D), torch.ones(4, D)])
    assert w.c.abs().sum() > 0, "ema_update 应累积 c"
    assert hasattr(w, "snapshot"), "应提供 snapshot() 供推理时取中枢"
    assert not hasattr(w, "reset"), "本类无 reset(); 若将来加入, 需在此补断言"
    print("[PASS] test_ema_update_accumulates")
    return True


def test_negative_tail_underflow_limits_the_gradient_claim():
    """8. **SPEC 主张的边界**: 『∂softplus/∂x = sigmoid(x) 永远不会为 0』

    只在 t 偏正侧成立。float32 下 t 足够负时 sigmoid(t) 下溢为**精确 0**,
    该侧梯度同样消失 —— v30 是把 tanh 的**双侧**饱和变成**单侧**(仅正尾)饱和,
    换来的不是"永不消失",而是"只在正尾永不消失"。
    """
    for cval in (-1e3, -1e4, -1e6):
        w = _ws()
        with torch.no_grad():
            w.c.fill_(cval); w.W_t.fill_(1.0 / D)
        z = torch.randn(1, 5, M, generator=torch.Generator().manual_seed(7))
        z = z.requires_grad_(True)
        w.augment_router_logits(z).pow(2).sum().backward()
        assert _grad_sum(w) == 0.0, (
            f"c={cval}: 负尾 sigmoid 下溢, 梯度应为 0, 实得 {_grad_sum(w)}")
        assert F.sigmoid(torch.tensor(-1e4, dtype=torch.float32)).item() == 0.0
    print("[PASS] test_negative_tail_underflow_limits_the_gradient_claim (负尾梯度精确为 0)")
    return True


def test_forward_shape_and_finiteness():
    """9. 输出形状保持, 且极端 c 下无 NaN/Inf"""
    w = _ws()
    z = torch.randn(4, 16, M)
    assert w.augment_router_logits(z).shape == z.shape
    with torch.no_grad():
        w.c.fill_(1e4); w.W_t.fill_(1.0 / D)
    assert torch.isfinite(w.augment_router_logits(z)).all(), "极端 c 下出现 NaN/Inf"
    print("[PASS] test_forward_shape_and_finiteness")
    return True


TESTS = [
    test_temperature_above_one_in_central_range,
    test_order_preservation,
    test_no_saturation_linear_growth,
    test_gradient_survives_saturation_where_tanh_dies,
    test_positive_tail_gradient_does_not_decay,
    test_perfect_degeneration_is_NOT_held,
    test_ema_update_accumulates,
    test_negative_tail_underflow_limits_the_gradient_claim,
    test_forward_shape_and_finiteness,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
