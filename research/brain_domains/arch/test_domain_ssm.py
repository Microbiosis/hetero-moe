"""域条件 SSM 的正确性验证.

三项检查, 对应三个必须成立的前提:

1. **退化性** — domain_gate=0 时, 域路由层严格等价于无域的 SSM。
   这是所有消融实验的前提: 若关掉开关后输出仍不同, 则「有/无域结构」
   的对比里混入了别的变量, 结论不可信。

2. **域敏感性** — 不同 domain_id 产生不同输出, 且差异随训练/扰动增长。
   若域标签不影响前向, 说明路由根本没接上。

3. **梯度可达** — 域参数 (alpha / U) 能收到非零梯度, 否则域结构学不到东西。

跑法:
    python research/brain_domains/arch/test_domain_ssm.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch

from research.brain_domains.arch.domain_ssm import DomainConditionedSSM

torch.manual_seed(0)

D_INNER = 64
D_STATE = 8
N_DOMAINS = 5
B, L = 2, 16


def _new_layer() -> DomainConditionedSSM:
    """每个用例用同一份初始权重, 保证比较公平."""
    torch.manual_seed(42)
    return DomainConditionedSSM(D_INNER, D_STATE, N_DOMAINS, rank=4)


def _rand_input() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    u = torch.randn(B, L, D_INNER, generator=g)
    domain_id = torch.randint(0, N_DOMAINS, (B, L), generator=g)
    return u, domain_id


def check_degenerate_to_plain_ssm() -> bool:
    """域偏移零初始化 + gate=0 时, A_eff 应与纯 SSM 完全一致."""
    layer = _new_layer()
    u, domain_id = _rand_input()

    a_shared = -torch.exp(layer.A_log)

    # gate=0 → alpha*0 = 0 → offset=0 → A_eff 全等于 a_shared
    with torch.no_grad():
        layer.domain_gate.zero_()
    a_eff = layer.effective_A(domain_id)
    ref = a_shared.unsqueeze(0).unsqueeze(0).expand_as(a_eff)
    same = torch.allclose(a_eff, ref, atol=1e-6)

    # 且此时前向应与「域标签打乱」无关
    layer.eval()
    with torch.no_grad():
        y1 = layer(u, domain_id)
        y2 = layer(u, domain_id.flip(-1))
    order_invariant = torch.allclose(y1, y2, atol=1e-5)

    print(f"[1] 退化性   A_eff==A_shared: {same}   关域后与标签无关: {order_invariant}")
    return same and order_invariant


def check_domain_sensitivity() -> bool:
    """开启域路由后, 不同域标签应产生可测差异."""
    layer = _new_layer()
    layer.eval()
    u, _ = _rand_input()

    with torch.no_grad():
        # 给域系数一个非零值, 模拟「学到域差异」之后的状态
        layer.alpha.normal_(0, 0.5)
        layer.U.normal_(0, 0.1)

        all_a = torch.zeros(B, L, dtype=torch.long)
        y_flat = layer(u, all_a)
        y_dom0 = layer(u, torch.full((B, L), 0, dtype=torch.long))
        y_dom3 = layer(u, torch.full((B, L), 3, dtype=torch.long))

    diff = (y_dom0 - y_dom3).abs().mean().item()
    base = y_flat.abs().mean().item()
    ratio = diff / (base + 1e-8)

    print(f"[2] 域敏感性 域0 vs 域3 平均差 {diff:.6f}  (输出量级 {base:.6f}, 比值 {ratio:.4f})")
    print(f"             差异非零: {diff > 1e-6}")
    return diff > 1e-6


def check_gradient_flow() -> bool:
    """域参数必须能收到梯度, 否则域结构学不出来.

    注意这是**分阶段**的: ∂offset/∂alpha ∝ U, ∂offset/∂U ∝ alpha。
    起点 alpha=0 而 U≠0, 所以第一步只有 alpha 收到梯度; 等 alpha 变为
    非零后, U 才拿到梯度。因此必须走两步才能验证「两者都在学」。
    """
    layer = _new_layer()
    u, domain_id = _rand_input()
    opt = torch.optim.SGD(layer.parameters(), lr=1e-2)

    # 第 1 步: alpha 应收到梯度 (U≠0)
    out = layer(u, domain_id)
    out.pow(2).mean().backward()
    na1 = layer.alpha.grad.abs().sum().item() if layer.alpha.grad is not None else 0.0
    nu1 = layer.U.grad.abs().sum().item() if layer.U.grad is not None else 0.0
    opt.step()
    opt.zero_grad()

    # 第 2 步: alpha 已更新为非零, U 应开始收到梯度
    out = layer(u, domain_id)
    out.pow(2).mean().backward()
    nu2 = layer.U.grad.abs().sum().item() if layer.U.grad is not None else 0.0

    ok_alpha = na1 > 0
    ok_u = nu2 > 0

    print(f"[3] 梯度可达 第1步 alpha.grad {na1:.3e} / U.grad {nu1:.3e} (U 此时应为 0)")
    print(f"             第2步 U.grad {nu2:.3e} (alpha 更新后应非零)")
    print(f"             分阶段学习成立: {ok_alpha and ok_u}")
    return ok_alpha and ok_u


def check_param_budget() -> None:
    """域路由引入的参数量应远小于基座 (130M)."""
    layer = DomainConditionedSSM(d_inner=1536, d_state=16, n_domains=17, rank=8)
    domain_params = layer.alpha.numel() + layer.U.numel()
    total = sum(p.numel() for p in layer.parameters())
    print(f"[4] 参数量   域专属 (alpha+U) = {domain_params:,}")
    print(f"             该层总计 {total:,}  → 域占比 {100 * domain_params / total:.3f}%")
    print(f"             24 层推算域参数 {domain_params * 24:,} (基座 130M 的 "
          f"{100 * domain_params * 24 / 130e6:.3f}%)")


if __name__ == "__main__":
    print("=" * 62)
    print("域条件 SSM 验证")
    print("=" * 62)
    r1 = check_degenerate_to_plain_ssm()
    print()
    r2 = check_domain_sensitivity()
    print()
    r3 = check_gradient_flow()
    print()
    check_param_budget()
    print()
    ok = r1 and r2 and r3
    print("=" * 62)
    print(f"结论: {'全部通过' if ok else '存在失败项'}")
    print("=" * 62)
    sys.exit(0 if ok else 1)
