"""规范 §7 原版原型（逐字节保留）—— 可追溯性基线锚点。

用途：作为 v4.0 规范 §7 "核心参考实现" 的冻结副本，验证规范本身定义的
梯度流（Phase 1 / Phase 2、detach 截断、STE）在当前环境可复现。

注意：本文件刻意不做任何"改进"，与规范文档 §7 代码块完全一致；任何工程增强
应放在 hetero_fusion/ 包内，并以本文件的输出作为回归基线。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantSTE(torch.autograd.Function):
    """模拟 INT4 量化噪声，反向传播使用 STE。"""
    @staticmethod
    def forward(ctx, x, bits=4, group_size=128):
        scale = x.abs().max(dim=-1, keepdim=True).values / (2 ** (bits - 1) - 1)
        scale = torch.clamp(scale, min=1e-5)
        return torch.round(x / scale) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class SparseRouterSTE(torch.autograd.Function):
    """带 Mask 的 Top-K 稀疏路由，防止非 Top-K 梯度泄漏"""
    @staticmethod
    def forward(ctx, alpha, k):
        val, idx = torch.topk(alpha, k, dim=-1)
        mask = torch.zeros_like(alpha).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(mask)
        return alpha * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


def swiglu_forward(x, w_gate, w_up, w_down):
    h_gate = F.linear(x, w_gate)
    h_up = F.linear(x, w_up)
    h_act = F.silu(h_gate) * h_up
    return F.linear(h_act, w_down)


class HeteroFusionLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 冻结底座
        self.w_gates = nn.ParameterList([
            nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)
        ])
        self.w_ups = nn.ParameterList([
            nn.Parameter(torch.randn(d_ff, d_model) * 0.02) for _ in range(num_experts)
        ])
        self.w_downs = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_ff) * 0.02) for _ in range(num_experts)
        ])
        for p_list in [self.w_gates, self.w_ups, self.w_downs]:
            for p in p_list:
                p.requires_grad = False

        # 可训练参数
        self.W_router = nn.Parameter(torch.randn(num_experts, d_model) * 0.01)
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(d_model)) for _ in range(num_experts)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(num_experts)
        ])
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.empty(1).uniform_(0.01, 0.1)) for _ in range(num_experts)
        ])

    def forward(self, x, phase1=False):
        B, S, D = x.shape
        x_residual = x

        z = F.linear(x, self.W_router)
        alpha = F.softmax(z, dim=-1)

        if phase1:
            alpha_hat = (torch.ones_like(alpha) / self.num_experts).detach()
        else:
            alpha_hat = SparseRouterSTE.apply(alpha, self.top_k)

        out_sum = torch.zeros_like(x)
        for m in range(self.num_experts):
            x_detached = x.detach()
            h_aligned = x_detached * self.gammas[m] + self.betas[m]
            f_m_exact = swiglu_forward(
                h_aligned, self.w_gates[m], self.w_ups[m], self.w_downs[m]
            )
            f_m_noisy = FakeQuantSTE.apply(f_m_exact)
            out_sum = out_sum + alpha_hat[..., m:m+1] * self.alphas[m] * f_m_noisy

        y = x_residual + out_sum
        return y, z, alpha_hat


def run_verification():
    torch.manual_seed(42)
    B, S, D, D_FF, M, K = 2, 8, 16, 32, 4, 2

    model = nn.Sequential(
        HeteroFusionLayer(D, D_FF, M, K),
        HeteroFusionLayer(D, D_FF, M, K)
    )

    x = torch.randn(B, S, D)
    target = torch.randn(B, S, D)

    # === Phase 1 ===
    print("=== Phase 1 验证 ===")
    for layer in model:
        layer.W_router.requires_grad = False

    opt_p1 = torch.optim.AdamW([
        {'params': [p for l in model for p in l.gammas] + [p for l in model for p in l.betas], 'lr': 1e-2},
        {'params': [p for l in model for p in l.alphas], 'lr': 1e-3},
    ])

    model.train()
    opt_p1.zero_grad()

    y = x  # 层间传递：第一层输出作为第二层输入
    for layer in model:
        y, z, _ = layer(y, phase1=True)

    loss = F.mse_loss(y, target) + 0.001 * (torch.logsumexp(z, dim=-1) ** 2).mean()
    loss.backward()
    opt_p1.step()

    l0 = model[0]
    print(f"Loss: {loss.item():.4f}")
    print(f"W_router.grad: {l0.W_router.grad}")
    print(f"gamma[0] grad: {l0.gammas[0].grad.norm().item():.2e}")
    print(f"alpha[0] grad: {l0.alphas[0].grad.norm().item():.2e}")

    # === Phase 2 ===
    print("\n=== Phase 2 验证 ===")
    for layer in model:
        layer.W_router.requires_grad = True

    opt_p2 = torch.optim.AdamW([
        {'params': [p for l in model for p in [l.W_router]], 'lr': 1e-4},
        {'params': [p for l in model for p in l.gammas] + [p for l in model for p in l.betas], 'lr': 1e-2},
        {'params': [p for l in model for p in l.alphas], 'lr': 1e-3},
    ])

    opt_p2.zero_grad()

    y = x  # 层间传递：第一层输出作为第二层输入
    for layer in model:
        y, z, ah = layer(y, phase1=False)

    lm = F.mse_loss(y, target)
    lz = (torch.logsumexp(z, dim=-1) ** 2).mean()
    lb = M * (ah.mean(dim=(0, 1)) * (ah > 0).float().mean(dim=(0, 1))).sum()

    loss = lm + 0.01 * lb + 0.001 * lz
    loss.backward()
    opt_p2.step()

    l0 = model[0]
    print(f"Loss: {loss.item():.4f}")
    print(f"W_router grad: {l0.W_router.grad.norm().item():.2e}")
    print(f"alpha[0] grad: {l0.alphas[0].grad.norm().item():.2e}")
    print(f"Sparsity: {(ah[0, 0] > 0).sum().item()} / {M}")


if __name__ == "__main__":
    run_verification()
