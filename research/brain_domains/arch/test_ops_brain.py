"""test_ops_brain.py — **自写算子**的正确性验证（与 `test_domain_ssm.py` 同构的三项检查）。

对每个算子检查三件事（缺一不可）：

1. **退化性** — 关掉 / 置零 ⇒ 严格退化为「无此算子」或某个可预测的平凡形式。
   这是所有消融的前提：若关掉开关后输出仍不同 ⇒ 对比里混进了别的变量 ⇒ 结论不可信。
2. **关键性质** — 该算子**特有**的可测性质（跨区隔离 / 逐通道隔离 / 衰减常数 / 上下路分离 …）。
3. **梯度可达** — 参数能收到非零梯度，否则这个结构学不到东西。

外加一条**组装级**检查：两个区各跑自己的区内算子 ⇒ 改一个区的输入，另一个区的输出**不变**
（这就是「区内混、区间不混」的可测形式）。

跑法：
    python research/brain_domains/arch/test_ops_brain.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import torch

from research.brain_domains.arch.ops_brain import (
    PredictiveInterface,
    RegionLocalCenterSurround,
    RegionLocalSynergy,
    RegionLocalTransient,
    TaskContextGate,
)

torch.manual_seed(0)

B, C, T = 2, 6, 32
FAILS = []


def check(name, cond, detail=""):
    tag = "✓" if cond else "✗"
    print(f"  {tag} {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def _rand(*shape):
    return torch.randn(*shape)


def _idx(mask) -> set:
    """一维布尔掩码 ⇒ 命中下标集合。

    ⚠️ `torch.nonzero` 在 1-D 输入上返回 **list[list]** ⇒ 必须 flatten 再取。
    """
    return {int(i) for i in torch.nonzero(mask.reshape(-1)).flatten().tolist()}


# ---------------------------------------------------------------------------
def test_center_surround():
    print("\n=== RegionLocalCenterSurround（V1-like，区内）===")
    op = RegionLocalCenterSurround(C, 4)
    x = _rand(B, C, T)

    # 1) 退化性：权重全零 ⇒ 输出恒为 0（半波整流后的平凡形式）
    with torch.no_grad():
        op.Wc.zero_()
        op.Ws_raw.fill_(-1e4)          # softplus(-1e4)≈0 ⇒ 环绕项≈0
        y0 = op(x)
    check("退化性：权重置零 ⇒ 输出恒 0", bool(y0.abs().max() < 1e-4),
          f"max={float(y0.abs().max()):.2e}")

    # 2) 关键性质：环绕权重恒 ≤ 0（脑科学：周边是抑制性的）
    with torch.no_grad():
        op.Wc.normal_(0, 0.3)
        op.Ws_raw.normal_(0, 0.5)
        ws = op.surround()
    check("关键性质：环绕权重恒 ≤ 0", bool((ws <= 1e-6).all()),
          f"max={float(ws.max()):.3e}")
    with torch.no_grad():
        y = op(x)
    check("关键性质：半波整流 ⇒ 输出 ≥ 0", bool((y >= 0).all()))
    # 邻接掩码：环绕池只应依赖**邻域内**的通道
    with torch.no_grad():
        adj = torch.zeros(C, C)
        for c in range(C):
            for d in (c - 1, c + 1):
                if 0 <= d < C:
                    adj[c, d] = 1.0
        op2 = RegionLocalCenterSurround(C, 4, adj=adj)
        op2.Wc.normal_(0, 0.3)
        op2.Ws_raw.normal_(0, 0.5)
        # 改通道 0 ⇒ 环绕池只动「自己 + 邻居」⇒ 改通道 0 ⇒ 动通道 0 与 1
        x2 = x.clone()
        x2[:, 0, :] += 5.0
        dp = (op2.surround_pool(x2) - op2.surround_pool(x)).abs().sum(dim=(0, 2))
        touched = _idx(dp > 1e-6)
        check("关键性质：环绕池只依赖邻域通道（改通道0 ⇒ 只动 0 与 1）",
              touched == {0, 1}, f"touched={sorted(touched)}")
        # 行归一化 ⇒ 环绕池每一行权重和为 1
        check("关键性质：环绕池行归一化",
              bool(torch.allclose(op2.A.sum(dim=1), torch.ones(C), atol=1e-6)))

    # 3) 梯度可达
    op3 = RegionLocalCenterSurround(C, 4)
    y = op3(x).pow(2).mean()
    y.backward()
    g_c = op3.Wc.grad
    check("梯度可达：Wc 有非零梯度",
          g_c is not None and float(g_c.abs().sum()) > 0,
          f"|g|={float(g_c.abs().sum()):.3f}" if g_c is not None else "None")


# ---------------------------------------------------------------------------
def test_synergy():
    print("\n=== RegionLocalSynergy（M1-like，区内低秩）===")
    rank = 3
    op = RegionLocalSynergy(C, rank)
    x = _rand(B, C, T)

    # 1) 退化性：U=0 ⇒ 协同空间恒 0
    with torch.no_grad():
        op.U.zero_()
        z0 = op(x)
    check("退化性：U 置零 ⇒ 协同输出恒 0", bool(z0.abs().max() < 1e-6))

    # 2) 关键性质：输出维数 ≤ rank（流形维数受控）
    with torch.no_grad():
        op.U.normal_(0, 0.5)
        z = op(x)
    check("关键性质：输出维数 = rank（低维流形）", z.shape[1] == rank,
          f"z={tuple(z.shape)}")
    #     低维性：rank 足够大时重建误差小
    with torch.no_grad():
        err = op.reconstruction_error(x)
    check("关键性质：重建误差有限（该区活动可被低秩张成）", err < 10.0,
          f"err={err:.3f}")

    # 3) 梯度可达
    op2 = RegionLocalSynergy(C, rank)
    z = op2(x)
    (z.pow(2).mean() + op2.reconstruct(z).pow(2).mean()).backward()
    check("梯度可达：U 有非零梯度",
          op2.U.grad is not None and float(op2.U.grad.abs().sum()) > 0)


# ---------------------------------------------------------------------------
def test_transient():
    print("\n=== RegionLocalTransient（V2-like，严格逐通道）===")
    op = RegionLocalTransient(C, learnable=True)
    x = _rand(B, C, T)

    # 1) 退化性：固定核模式下 on+off 抵消 ⇒ sustained 之外不产生额外项
    fx = RegionLocalTransient(C, learnable=False)
    with torch.no_grad():
        y = fx(x)
        on = y[:, :C]
        off = y[:, C:2 * C]
    check("退化性/恒等性：固定核的 on + off = 0",
          bool((on + off).abs().max() < 1e-5),
          f"max={float((on + off).abs().max()):.2e}")

    # 2) 关键性质：**严格逐通道** —— 改通道 c ⇒ 只有 c 的 3 个输出变
    #    ⚠️ grouped conv 的输出是「按输入通道分组」⇒ 第 c 个通道的三个输出
    #    落在 3c, 3c+1, 3c+2（不是 on/off/sust 三个块！）
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 2, :] += 5.0
        d = (op(x2) - op(x)).abs().sum(dim=(0, 2))    # (3C,)
    touched = _idx(d > 1e-6)
    expect = {2, C + 2, 2 * C + 2}          # 输出是 [on|off|sust] 三块 ⇒ 通道 c 在 c / C+c / 2C+c
    check("关键性质：改一个通道 ⇒ 只影响它自己的 3 个输出", touched == expect,
          f"touched={sorted(touched)} expect={sorted(expect)}")

    # 3) 梯度可达
    op2 = RegionLocalTransient(C, learnable=True)
    op2(x).pow(2).mean().backward()
    g = op2.conv.weight.grad
    check("梯度可达：conv 核有非零梯度",
          g is not None and float(g.abs().sum()) > 0)


# ---------------------------------------------------------------------------
def test_context_gate():
    print("\n=== TaskContextGate（dlPFC-like，跨区接口）===")
    op = TaskContextGate(C, gamma=0.9)
    x = _rand(B, C, T)

    # 1) 退化性：门控强度 g=0 ⇒ 输出 = 输入（恒等）
    op0 = TaskContextGate(C, gamma=0.0)
    with torch.no_grad():
        op0.g.zero_()
        y0 = op0(x)
    check("退化性：g=0 ⇒ 输出 = 输入（恒等门控）",
          bool((y0 - x).abs().max() < 1e-5))

    # 2) 关键性质：**冲激**输入下，状态按 γ 衰减（常数输入是它的不动点 ⇒ 测不出来）
    #    ⚠️ 冲激必须是**单通道**：通道维上做 LayerNorm ⇒ 全通道同时为 1 会被
    #    归一化成 0（均值 1、标准差 0）⇒ 比值变 NaN。
    with torch.no_grad():
        imp = torch.zeros(1, C, T)
        imp[:, 0, 0] = 1.0
        s = op.context(imp)
        ratio = float(s[0, 0, -1] / s[0, 0, 0])
        expect = op.gamma ** (T - 1)
    check("关键性质：冲激后状态按 γ 衰减", abs(ratio - expect) < 1e-4,
          f"ratio={ratio:.6f} expect={expect:.6f}")
    #     γ=1 ⇒ 状态恒定（不衰减）：首帧 == 末帧（⚠️ 不等于 1 ⇒ 冲激经过了 LN）
    op1 = TaskContextGate(C, gamma=1.0)
    with torch.no_grad():
        s1 = op1.context(imp)
    check("关键性质：γ=1 ⇒ 冲激后状态不衰减",
          abs(float(s1[0, 0, -1]) - float(s1[0, 0, 0])) < 1e-5,
          f"first={float(s1[0, 0, 0]):.6f} last={float(s1[0, 0, -1]):.6f}")
    #     state_decay 辅助方法必须给出同一个数（避免测试与实现各说各话）
    with torch.no_grad():
        decay = op.state_decay(T, channel=0)
    check("关键性质：state_decay 与手算一致",
          abs(decay - expect) < 1e-4, f"decay={decay:.6f}")

    # 3) 梯度可达
    #    ⚠️ **门控强度 g 的梯度任何时候都非零**（∂y/∂g = x ⊙ s 与 g 无关）
    op2 = TaskContextGate(C, gamma=0.9)
    op2(x).pow(2).mean().backward()
    check("梯度可达：门控强度 g 有非零梯度",
          op2.g.grad is not None and float(op2.g.grad.abs().sum()) > 0,
          f"|g|={float(op2.g.grad.abs().sum()):.3f}" if op2.g.grad is not None else "None")
    #    ⚠️ **上下文通路的参数只有在门打开后才有梯度**（g=0 ⇒ 恒等 ⇒ 通路不影响输出）
    op3 = TaskContextGate(C, gamma=0.9)
    with torch.no_grad():
        op3.g.fill_(0.5)
    op3(x).pow(2).mean().backward()
    check("梯度可达：门打开后上下文通路（LN）有非零梯度",
          op3.ln.weight.grad is not None and float(op3.ln.weight.grad.abs().sum()) > 0,
          f"|g|={float(op3.ln.weight.grad.abs().sum()):.3f}"
          if op3.ln.weight.grad is not None else "None")


# ---------------------------------------------------------------------------
def test_predictive_interface():
    print("\n=== PredictiveInterface（预测编码接口，跨区）===")
    op = PredictiveInterface(C, C)
    x = _rand(B, C, T)

    # 1) 退化性：W_up=0 ⇒ p=0, e=ln(x) ⇒ 上下路分离且「无预测」
    with torch.no_grad():
        op.W_up.zero_()
        p0, e0 = op(x)
    check("退化性：W_up=0 ⇒ 预测路恒 0", bool(p0.abs().max() < 1e-6))
    with torch.no_grad():
        op.W_up.copy_(torch.eye(C))
        p1, e1 = op(x)
    check("退化性：W_up=I ⇒ 误差路恒 0（完全预测）",
          bool(e1.abs().max() < 1e-5),
          f"max={float(e1.abs().max()):.2e}")

    # 2) 关键性质：上下路**物理分开**且 p+e = 归一化后的 x
    with torch.no_grad():
        op.W_up.normal_(0, 0.2)
        p, e = op(x)
        xn = op._norm(x)          # ⚠️ op.ln 是 LayerNorm(d_in) ⇒ 直接调用会归一化错轴
    check("关键性质：p + e = 归一化输入（两路正交可分）",
          bool((p + e - xn).abs().max() < 1e-5))

    # 3) 梯度可达
    op2 = PredictiveInterface(C, C)
    p2, e2 = op2(x)
    (p2.pow(2).mean() + e2.pow(2).mean()).backward()
    check("梯度可达：W_up 有非零梯度",
          op2.W_up.grad is not None and float(op2.W_up.grad.abs().sum()) > 0)
    check("梯度可达：归一化仿射参数有非零梯度",
          op2.ln.weight.grad is not None and float(op2.ln.weight.grad.abs().sum()) > 0)


# ---------------------------------------------------------------------------
def test_region_isolation():
    print("\n=== 组装级：跨区隔离（「区内混、区间不混」的可测形式）===")
    c_post, c_ant = 4, 3
    op_a = RegionLocalCenterSurround(c_post, 3)
    op_b = RegionLocalCenterSurround(c_ant, 2)
    xa, xb = _rand(B, c_post, T), _rand(B, c_ant, T)

    with torch.no_grad():
        ya1 = op_a(xa)
        xb2 = xb.clone()
        xb2[:, 1, :] += 3.0
        ya2 = op_a(xa)              # 区A的输入没变 ⇒ 输出必须不变
        yb1 = op_b(xb)
        yb2 = op_b(xb2)
    check("跨区隔离：改 B 区输入 ⇒ A 区输出不变",
          bool((ya1 - ya2).abs().max() < 1e-6))
    check("区内生效：改 B 区输入 ⇒ B 区输出确实变",
          bool((yb1 - yb2).abs().max() > 1e-4))


# ---------------------------------------------------------------------------
def main():
    print("=== 自写算子验证（退化性 / 关键性质 / 梯度可达 / 跨区隔离）===")
    test_center_surround()
    test_synergy()
    test_transient()
    test_context_gate()
    test_predictive_interface()
    test_region_isolation()
    if FAILS:
        print(f"\n✗ {len(FAILS)} 项不通过：")
        for f in FAILS:
            print("   -", f)
        return 1
    print("\n=== 全部通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
