"""test_ops_brain.py — **自写算子**的验证。

每个算子的测试要回答两件事（与 `arch/test_domain_ssm.py` 的三项检查同构）：

1. **退化性** — 关掉/置零 ⇒ 严格退化为「无此算子」或某个可预测的平凡形式。
   这是所有消融的前提：关掉后输出仍不同 ⇒ 对比里混进了别的变量 ⇒ 结论不可信。
2. **关键性质** — 该算子**特有**的可测性质，且要对应它的**使用场景**
   （例如：环绕池真的是「同组公共成分」而不是自己？on/off 的幅度对不对？
   偏置真的携带历史吗？）。
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
    RegionLocalLowRank,
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
    print("\n=== RegionLocalCenterSurround（V1-like，POST 组内）===")
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

    #    ⚠️ **环绕池 = 同组「其它」通道的公共成分**（不是自己！）
    #    若环绕池含自己 ⇒ y = (Wc+Ws)·x ⇒ 中心-周边退化成逐通道线性。
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 0, :] += 5.0
        dp = (op.surround_pool(x2) - op.surround_pool(x)).abs().sum(dim=(0, 2))
        touched = _idx(dp > 1e-6)
        expect = set(range(1, C))       # 除自己以外全部（公共成分）
    check("关键性质：环绕池 = 同组其它通道（不含自己 ⇒ 不是逐通道线性）",
          touched == expect, f"touched={sorted(touched)} expect={sorted(expect)}")
    check("关键性质：环绕池行归一化",
          bool(torch.allclose(op.A.sum(dim=1), torch.ones(C), atol=1e-6)))
    check("关键性质：环绕池对角为 0（自己不算自己的周边）",
          bool((op.A.diagonal().abs().max() < 1e-9)),
          f"max={float(op.A.diagonal().abs().max()):.2e}")

    # 3) 梯度可达
    op3 = RegionLocalCenterSurround(C, 4)
    y = op3(x).pow(2).mean()
    y.backward()
    g_c = op3.Wc.grad
    check("梯度可达：Wc 有非零梯度",
          g_c is not None and float(g_c.abs().sum()) > 0,
          f"|g|={float(g_c.abs().sum()):.3f}" if g_c is not None else "None")


# ---------------------------------------------------------------------------
def test_low_rank():
    print("\n=== RegionLocalLowRank（集群活动/共同驱动，CENT 组内）===")
    rank = 3
    op = RegionLocalLowRank(C, rank)
    x = _rand(B, C, T)

    # 1) 退化性：U=0 ⇒ 成分空间恒 0
    with torch.no_grad():
        op.U.zero_()
        z0 = op(x)
    check("退化性：U 置零 ⇒ 成分输出恒 0", bool(z0.abs().max() < 1e-6))

    # 2) 关键性质：输出维数 ≤ rank（共同成分个数受控）
    with torch.no_grad():
        op.U.normal_(0, 0.5)
        z = op(x)
    check("关键性质：输出维数 = rank（少数成分）", z.shape[1] == rank,
          f"z={tuple(z.shape)}")
    with torch.no_grad():
        err = op.reconstruction_error(x)
    check("关键性质：重建误差有限（该组活动可被少数成分张成）", err < 10.0,
          f"err={err:.3f}")

    # 3) 梯度可达
    op2 = RegionLocalLowRank(C, rank)
    z = op2(x)
    (z.pow(2).mean() + op2.reconstruct(z).pow(2).mean()).backward()
    check("梯度可达：U 有非零梯度",
          op2.U.grad is not None and float(op2.U.grad.abs().sum()) > 0)


# ---------------------------------------------------------------------------
def test_transient():
    print("\n=== RegionLocalTransient（ON/OFF + 持续，POST 组内逐通道）===")
    op = RegionLocalTransient(C, learnable=True)
    x = _rand(B, C, T)
    fx = RegionLocalTransient(C, learnable=False)

    # 1) 关键性质：固定核的 on/off **互斥且幅度正确**
    #    上升斜坡 ⇒ on>0、off=0；下降斜坡 ⇒ off>0、on=0
    with torch.no_grad():
        ramp_up = torch.arange(T, dtype=torch.float32).view(1, 1, T).expand(B, C, T).contiguous()
        ramp_dn = -ramp_up
        yu = fx(ramp_up)
        yd = fx(ramp_dn)
        on_u, off_u = yu[:, :C], yu[:, C:2 * C]
        on_d, off_d = yd[:, :C], yd[:, C:2 * C]
        inner = slice(1, T - 1)          # 避开零填充的两端
    check("关键性质：上升沿 ⇒ on>0 且 off=0",
          bool((on_u[:, :, inner] > 0).all() and (off_u[:, :, inner].abs().max() < 1e-6)),
          f"on_min={float(on_u[:, :, inner].min()):.3f} off_max={float(off_u[:, :, inner].abs().max()):.2e}")
    check("关键性质：下降沿 ⇒ off>0 且 on=0",
          bool((off_d[:, :, inner] > 0).all() and (on_d[:, :, inner].abs().max() < 1e-6)),
          f"off_min={float(off_d[:, :, inner].min()):.3f} on_max={float(on_d[:, :, inner].abs().max()):.2e}")

    # 2) 关键性质：**严格逐通道** —— 改通道 c ⇒ 只有 c 的 3 个输出变
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 2, :] += 5.0
        d = (op(x2) - op(x)).abs().sum(dim=(0, 2))    # (3C,)
    touched = _idx(d > 1e-6)
    expect = {2, C + 2, 2 * C + 2}      # 输出是 [on|off|sust] 三块 ⇒ 通道 c 在 c / C+c / 2C+c
    check("关键性质：改一个通道 ⇒ 只影响它自己的 3 个输出", touched == expect,
          f"touched={sorted(touched)} expect={sorted(expect)}")

    # 3) 关键性质：sustained = 平滑（常量输入 ⇒ 常量输出）
    with torch.no_grad():
        const = torch.ones(B, C, T)
        yc = fx(const)
        sust = yc[:, 2 * C:]
    check("关键性质：sustained 对常量输入保持常量",
          bool((sust[:, :, inner] - 1.0).abs().max() < 1e-5),
          f"max_dev={float((sust[:, :, inner] - 1.0).abs().max()):.2e}")

    # 4) 梯度可达
    op2 = RegionLocalTransient(C, learnable=True)
    op2(x).pow(2).mean().backward()
    g = op2.conv.weight.grad
    check("梯度可达：conv 核有非零梯度",
          g is not None and float(g.abs().sum()) > 0)


# ---------------------------------------------------------------------------
def test_context_gate():
    print("\n=== TaskContextGate（dlPFC-like，FRONT 组内维持 + 加性偏置）===")
    op = TaskContextGate(C, gamma=0.9)
    x = _rand(B, C, T)

    # 1) 退化性：偏置强度 g=0 ⇒ 输出 = 输入（恒等）
    op0 = TaskContextGate(C, gamma=0.0)
    with torch.no_grad():
        op0.g.zero_()
        y0 = op0(x)
    check("退化性：g=0 ⇒ 输出 = 输入（恒等）",
          bool((y0 - x).abs().max() < 1e-5))

    # 2) 关键性质：**冲激**输入下，状态按 γ 衰减（常数输入是它的不动点 ⇒ 测不出来）
    with torch.no_grad():
        imp = torch.zeros(1, C, T)
        imp[:, 0, 0] = 1.0
        s = op.context(imp)
        ratio = float(s[0, 0, -1] / s[0, 0, 0])
        expect = op.gamma ** (T - 1)
    check("关键性质：冲激后状态按 γ 衰减", abs(ratio - expect) < 1e-4,
          f"ratio={ratio:.6f} expect={expect:.6f}")
    with torch.no_grad():
        decay = op.state_decay(T, channel=0)
    check("关键性质：state_decay 与手算一致",
          abs(decay - expect) < 1e-4, f"decay={decay:.6f}")
    #     γ=0 ⇒ 无历史：状态 = 当帧
    op_g0 = TaskContextGate(C, gamma=0.0)
    with torch.no_grad():
        s0 = op_g0.context(x)
    check("关键性质：γ=0 ⇒ 状态 = 当帧（无历史）",
          bool((s0 - x).abs().max() < 1e-5))

    # 3) 关键性质：**偏置携带历史**（改最早的时间步 ⇒ 后面每一帧都变）
    op1 = TaskContextGate(C, gamma=0.9)
    with torch.no_grad():
        op1.g.fill_(0.5)
        y_a = op1(x)
        x2 = x.clone()
        x2[:, :, 0] += 5.0
        y_b = op1(x2)
    later = (y_b[:, :, 1:] - y_a[:, :, 1:]).abs().max()
    check("关键性质：偏置携带历史（改 t=0 ⇒ t>0 的输出也变）",
          bool(later > 1e-3), f"max_diff={float(later):.3f}")

    # 4) 关键性质：**逐通道**（改一个通道 ⇒ 只有它自己的输出变）
    with torch.no_grad():
        x3 = x.clone()
        x3[:, 2, :] += 3.0
        d = (op1(x3) - op1(x)).abs().sum(dim=(0, 2))
    check("关键性质：逐通道（改通道 2 ⇒ 只有通道 2 变）",
          _idx(d > 1e-6) == {2}, f"touched={sorted(_idx(d > 1e-6))}")

    # 5) 梯度可达（偏置强度 g 的梯度任何时候都非零：∂y/∂g = s 与 g 无关）
    op2 = TaskContextGate(C, gamma=0.9)
    op2(x).pow(2).mean().backward()
    check("梯度可达：偏置强度 g 有非零梯度",
          op2.g.grad is not None and float(op2.g.grad.abs().sum()) > 0,
          f"|g|={float(op2.g.grad.abs().sum()):.3f}" if op2.g.grad is not None else "None")


# ---------------------------------------------------------------------------
def test_predictive_interface():
    print("\n=== PredictiveInterface（预测编码接口，POST 组内层间）===")
    op = PredictiveInterface(C, C)
    x = _rand(B, C, T)

    # 1) 初始化：W_up = I ⇒ 强预测先验 ⇒ 预测路 = 归一化输入、误差路 ≈ 0
    with torch.no_grad():
        p0, e0 = op(x)
        xn0 = op._norm(x)
    check("初始化：预测路 = 归一化输入（层间强预测先验）",
          bool((p0 - xn0).abs().max() < 1e-5),
          f"max={float((p0 - xn0).abs().max()):.2e}")
    check("初始化：误差路 ≈ 0（新信息无处藏身）",
          bool(e0.abs().max() < 1e-5), f"max={float(e0.abs().max()):.2e}")

    # 2) 退化性：W_up=0 ⇒ p=0, e=ln(x)；W_up=I ⇒ e=0（完全预测）
    with torch.no_grad():
        op.W_up.zero_()
        pz, ez = op(x)
    check("退化性：W_up=0 ⇒ 预测路恒 0", bool(pz.abs().max() < 1e-6))
    with torch.no_grad():
        op.W_up.copy_(torch.eye(C))
        pi, ei = op(x)
    check("退化性：W_up=I ⇒ 误差路恒 0（完全预测）",
          bool(ei.abs().max() < 1e-5), f"max={float(ei.abs().max()):.2e}")

    # 3) 关键性质：上下路**物理分开**且 p+e = 归一化后的 x
    with torch.no_grad():
        op.W_up.normal_(0, 0.2)
        p, e = op(x)
        xn = op._norm(x)          # ⚠️ op.ln 是 LayerNorm(d_in) ⇒ 直接调用会归一化错轴
    check("关键性质：p + e = 归一化输入（两路正交可分）",
          bool((p + e - xn).abs().max() < 1e-5))

    # 4) 梯度可达
    op2 = PredictiveInterface(C, C)
    p2, e2 = op2(x)
    (p2.pow(2).mean() + e2.pow(2).mean()).backward()
    check("梯度可达：W_up 有非零梯度",
          op2.W_up.grad is not None and float(op2.W_up.grad.abs().sum()) > 0)
    check("梯度可达：归一化仿射参数有非零梯度",
          op2.ln.weight.grad is not None and float(op2.ln.weight.grad.abs().sum()) > 0)

    # 5) 关键性质：**必须等维**（跨组不等维的接口不属于本算子 ⇒ 明确报错，别静默算错）
    #    ⚠️ 只写**一次** check 调用（try/except 两个分支各写一次会让「调用数」
    #    与「实际执行数」不一致 ⇒ 守卫绑定的是调用数）
    err_msg = ""
    try:
        PredictiveInterface(4, 3)
    except ValueError as e:
        err_msg = str(e)
    check("关键性质：不等维 ⇒ 明确报错（而不是静默算错）", bool(err_msg),
          f"msg={err_msg[:44]}")


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
    test_low_rank()
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
