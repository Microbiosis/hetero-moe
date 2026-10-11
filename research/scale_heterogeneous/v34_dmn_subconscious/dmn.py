"""v34.0 — DMN 潜意识创造性联想层。

    Q  = a_big^t         (大专家当前时刻激活, 主导)
    KV = ℳ_t             (memory bank: 历史全专家)
    r^t = CrossAttn(Q, KV)  → 涌现联想表征 (≠ Σa_m, 参与梯度)
    C ← C + η·r^t        (突触稳态式慢变累积)
    r_eff = r^t − Ĉ      (稳态修正: 抵消已被累积整合的方向)

联想权重 (q/k/v/o_proj) 通过 loss 学习, 联想非随机。
"""
import torch
import torch.nn as nn


class DMNSubconscious(nn.Module):
    """DMN 创造性联想层: 跨专家×跨时刻, 大专家主导.

    - associate(big_summary): Q=大专家当前, KV=memory历史全专家 → r (参与梯度)
    - push_memory(summaries): 存本步各专家摘要到滑动窗口
    - C: 慢变 buffer, C <- C + eta*r
    - inject(r): r_eff = r − Ĉ, 把 C 接进前向 (完美退化: C=0 时等于 r)
    - route_bias_proj(r): g(r) 路由调制投影 D->M (参与梯度)
    """

    def __init__(self, d_shared, n_experts, window=8):
        super().__init__()
        self.d_shared = d_shared
        self.n_experts = n_experts
        self.window = window
        # 联想 cross-attention (单头省CPU) — 权重可学习
        self.q_proj = nn.Linear(d_shared, d_shared)
        self.k_proj = nn.Linear(d_shared, d_shared)
        self.v_proj = nn.Linear(d_shared, d_shared)
        self.o_proj = nn.Linear(d_shared, d_shared)
        self.norm = nn.LayerNorm(d_shared)
        # 路由调制 g(r): D -> M — 可学习
        self.route_bias_proj = nn.Linear(d_shared, n_experts)
        nn.init.zeros_(self.route_bias_proj.weight)
        nn.init.zeros_(self.route_bias_proj.bias)
        # 慢变状态 C (buffer, 不被 optimizer 更新, 仅记录累积)
        self.register_buffer("C", torch.zeros(d_shared))
        # memory bank [W, M, D]
        self.register_buffer("memory", torch.zeros(window, n_experts, d_shared))
        self.register_buffer("mem_count", torch.tensor(0, dtype=torch.long))

    def push_memory(self, expert_summaries):
        with torch.no_grad():
            idx = int(self.mem_count) % self.window
            self.memory[idx] = expert_summaries.detach()
            self.mem_count += 1

    def associate(self, big_summary):
        """创造性联想: Q=大专家当前 [D], KV=memory 历史全专家 → r [D] (参与梯度)."""
        n = min(int(self.mem_count), self.window)
        if n == 0:
            return torch.zeros(self.d_shared, device=big_summary.device)
        kv = self.memory[:n].reshape(n * self.n_experts, self.d_shared).unsqueeze(0)
        q = big_summary.unsqueeze(0).unsqueeze(0)  # [1,1,D]
        qh = self.q_proj(q)
        kh = self.k_proj(kv)
        vh = self.v_proj(kv)
        scores = torch.bmm(qh, kh.transpose(1, 2)) / (self.d_shared ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        r = torch.bmm(attn, vh)
        r = self.o_proj(r)
        r = self.norm(r.squeeze(0).squeeze(0))  # [D]
        return r

    def update_C(self, r, eta):
        with torch.no_grad():
            self.C += eta * r.detach()

    def inject(self, r, eps=1e-8):
        """突触稳态修正后的有效联想注入量 r_eff = r − Ĉ。

        Ĉ 取慢变累积态 C 的**方向**, 幅度对齐 ‖r‖:
            Ĉ = (C / (‖C‖ + eps)) · ‖r‖

        语义: C 累积的是"已经被反复注入、已经整合进慢状态"的方向。
        每次注入前抵消该方向 ⇒ 同一联想重复注入的边际效应递减
        (突触稳态 SHY: 慢变累积 / 保留重要 / 削弱无关)。

        三条性质:
        - **完美退化**: C=0 ⇒ Ĉ=0 ⇒ r_eff = r, 与未接 C 的 v34 完全一致
        - **数值稳定**: C 无界累积, 但幅度由 ‖r‖ 归一, 不会淹没 r
        - **纯前向门控**: Ĉ 同时经 ``C/‖C‖`` 与 ``r.detach().norm()`` 两条
          非可微路径构成, 故对 r 完全不可微 —— 稳态只改变**前向数值**,
          梯度场与未接 C 时逐元素相同, 模型无法通过操纵 ‖r‖ 绕过门控
        """
        c_norm = self.C.norm()
        if float(c_norm) <= eps:          # C 还是零向量 ⇒ 完美退化
            return r
        scale = r.detach().norm()          # detach: 尺度不可被梯度操纵
        c_hat = (self.C / (c_norm + eps)) * scale
        return r - c_hat

    def reset(self):
        self.C.zero_(); self.mem_count.zero_(); self.memory.zero_()
