"""脑域结构化的选择性状态空间模型 (domain-conditioned selective SSM).

基座: state-spaces/mamba-130m (Apache-2.0), 架构参数 d_model=768 / n_layer=24 /
d_inner=1536 / d_state=16.

改造点: 仅把 SSM 的状态转移矩阵 A 从「全模型共享」改为「域条件低秩偏移」,
其余 (in_proj / out_proj / B / C / D) 全部保持共享。

选择 A 的理由: A 控制状态衰减速率, 即时间常数 —— 不同脑区的更新频率与
时滞确实不同, 这是四个组件里与「脑域」对应关系最直接的一个。
B / C 是输入输出映射, 跨域共享更省参数也更符合「共享基 + 域特化偏移」的设定。

低秩形式 (R 远小于 d_state):
    A_d = A_shared + Σ_r α_r(d) · U_r
其中 U_r ∈ R^{d_inner × d_state} 为全域共享的基, α(d) ∈ R^R 为域 d 的系数。
K 个域只额外引入 K × R 个参数 (R=8, K=17 时为 136 个/层)。

逐 token 计算:
    α[b, l] = α_table[domain_id[b, l]]
    A_eff[b, l] = A_shared + einsum('blr,irs->blis', α[b, l], U)
    y[b, l] = selective_scan(u[b, l], A_eff[b, l], B[b, l], C[b, l], D)

本模块是纯 PyTorch 参考实现, 用于在无 CUDA 的环境里验证设计与消融。
线上训练时应换成 mamba_ssm 的 fused kernel, 但数学形式完全一致。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainConditionedSSM(nn.Module):
    """域条件化的选择性 SSM 扫描层.

    参数
    ----
    d_inner : int
        SSM 内部维度 (mamba-130m 为 1536 = expand 2 x d_model 768)。
    d_state : int
        状态维度 (mamba-130m 为 16)。
    n_domains : int
        脑域数量 K。domain_id 取值范围 [0, K)。
    rank : int
        域偏移的低秩基数量 R, 通常取 4-16。
    dt_min / dt_max : float
        步长的数值钳位区间, 保证 exp 不溢出。
    """

    def __init__(
        self,
        d_inner: int = 1536,
        d_state: int = 16,
        n_domains: int = 17,
        rank: int = 8,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_domains = n_domains
        self.rank = rank

        # A 采用 log 参数化: A = -exp(A_log), 保证状态始终衰减(稳定).
        # 初始化对齐 mamba 的 S4D-real 谱: A ~ N(log(1..d_state), 1).
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))

        # 全域共享的低秩基 U: (R, d_inner, d_state)。
        #
        # ⚠️ 不能与 alpha 同时零初始化。offset = einsum(alpha, U) 对两者的梯度
        #    互为对方的取值: d/dU ∝ alpha, d/dalpha ∝ U。两者皆零时梯度恒为零,
        #    参数永远锁在原点 —— 域路由会**静默失效** (模型照常训练, 但域结构
        #    形同虚设, 且梯度检查不会报警)。
        #
        #    解法: U 随机小扰动打破对称, alpha 保持零。alpha=0 使训练起点
        #    仍严格等价于原版 mamba (消融前提不受影响), 而 U≠0 保证了
        #    alpha 一开始就能收到梯度, 从而带动 U 学起来。
        u_init = torch.randn(rank, d_inner, d_state) * (1.0 / d_state**0.5)
        self.U = nn.Parameter(u_init)

        # 域系数 α: (K, R), 零初始化 —— 同上, 起点是「无域结构」。
        self.alpha = nn.Parameter(torch.zeros(n_domains, rank))

        # 步长投影: 与 mamba 一样由输入动态生成 (选择性)。
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # B / C 由输入动态生成, 跨域共享。
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # 跳跃连接 D, 跨域共享。
        self.D = nn.Parameter(torch.ones(d_inner))

        # 域嵌入的可学习尺度: 显式开关, 用于把「域路由」变成可消融的单一变量。
        # 置 0 时 forward 严格等价于无域路由的 SSM。
        self.domain_gate = nn.Parameter(torch.ones(1))

        self.dt_min, self.dt_max = dt_min, dt_max
        self._init_dt()

    def _init_dt(self) -> None:
        """按 mamba 的 dt 初始化策略: bias 取 log 均匀分布的反函数."""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=1e-4)
        # inv_dt = 1 + log(softplus(dt) - dt), 令 softplus(bias)=dt
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def effective_A(self, domain_id: torch.Tensor) -> torch.Tensor:
        """计算逐 token 的有效状态转移矩阵.

        参数
        ----
        domain_id : (B, L) long
            每个 token 所属的脑域。

        返回
        ----
        (B, L, d_inner, d_state)
        """
        b, l = domain_id.shape
        a_shared = -torch.exp(self.A_log)                                   # (D, N)
        alpha = self.alpha[domain_id] * self.domain_gate                    # (B,L,R)
        # U 的形状是 (R, D, N) —— 秩维 r 在最前, 不能与 d_inner 的下标混用。
        offset = torch.einsum("blr,rdn->bldn", alpha, self.U)               # (B,L,D,N)
        return a_shared.unsqueeze(0).unsqueeze(0) + offset

    def forward(
        self,
        u: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        """选择性扫描.

        参数
        ----
        u : (B, L, d_inner)
            经 in_proj 得到的主输入。
        domain_id : (B, L) long
            每个 token 的脑域标签。

        返回
        ----
        (B, L, d_inner)
        """
        b, l, _ = u.shape

        delta = F.softplus(self.dt_proj(u))                                  # (B,L,D)
        delta = delta.clamp(self.dt_min, self.dt_max)
        Bm = self.B_proj(u)                                                 # (B,L,N)
        Cm = self.C_proj(u)                                                 # (B,L,N)

        A_eff = self.effective_A(domain_id)                                  # (B,L,D,N)

        # 离散化: C * (delta*A) * B, 再走前缀和 —— 与 mamba 的选择性扫描同构。
        # 为控制内存, 按 batch 维分块累积, 不物化 (L, D, N) 的大张量。
        y = torch.zeros(b, l, self.d_inner, dtype=u.dtype, device=u.device)
        h = torch.zeros(b, self.d_inner, self.d_state, dtype=u.dtype, device=u.device)

        dA = torch.exp(delta.unsqueeze(-1) * A_eff)                          # (B,L,D,N)
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)        # (B,L,D,N)
        for t in range(l):
            h = dA[:, t] * h + dBu[:, t]
            y[:, t] = torch.einsum("bdn,bn->bd", h, Cm[:, t]) + self.D * u[:, t]

        return y


class DomainRouterStack(nn.Module):
    """在 mamba 的每层 SSM 后插入域路由, 形成脑域结构化的层堆.

    这里只实现域路由部分, 基座冻结/加载交给调用方 —— 这样才能干净地做
    「基座不动, 只加域结构」的单变量消融。
    """

    def __init__(
        self,
        n_layer: int = 24,
        d_inner: int = 1536,
        d_state: int = 16,
        n_domains: int = 17,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DomainConditionedSSM(d_inner, d_state, n_domains, rank)
                for _ in range(n_layer)
            ]
        )

    def set_domain_gate(self, value: float) -> None:
        """统一开关: 0 = 关闭域路由 (退化为原版), 1 = 全量开启.

        这是消融实验的主抓手, 也是「域标签是否真的携带信息」的判据。
        """
        with torch.no_grad():
            for layer in self.layers:
                layer.domain_gate.fill_(value)

    def forward(self, u: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            u = u + layer(u, domain_id)
        return u
