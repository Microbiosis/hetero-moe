"""ops_brain.py — 脑域线**自己写的算子**（按脑科学定义；**区内混、区间不混**）。

## 为什么要自己写算子（而不是只搬组件）

外部现成组件（LaBraM / NeuroLM / Mamba 家族）解决的是「怎么在大数据上压出一个好表征」，
**不解决**「按脑区边界切开的多个模块怎么各自成立、怎么连」。
⇒ 而我们自己的实测给出一条硬约束（exp94，n=10）：

    输入端混通道（`ch_mix`）**显著更差**：单试次 200-way
    现状骨干 0.0573 → 保留通道身份 0.0627（10/10 被试，p=0.0020；参数量匹配 ⇒ 不是容量的功劳）

⇒ 若照抄 V1 的空间中心-周边核（在我们这里就是**跨通道**混合），
**会直接踩掉自己的实证结论** ⇒ ⇒ 所以「自己写」不是权宜，是**必须**。

## 统一法则：区内混、区间不混

| | 依据 |
|---|---|
| **区内**可以混 | 脑科学：视觉区**内部**的空间滤波（中心-周边）是真实的 ⇒ 区内混合有脑科学依据 |
| **跨区**绝不混 | 我们：跨区混会把位置轴的分工抹掉（exp94）⇒ 跨区信息只能走**显式接口算子** |

⇒⇒ 本文件里每个**区内**算子都只在自己那个区的通道上作用；
跨区流动一律由 `TaskContextGate` / `PredictiveInterface` 两个**显式接口算子**承担。

## 算子清单

| 算子 | 脑科学对应 | 作用 | 混合粒度 |
|---|---|---|---|
| `RegionLocalCenterSurround` | V1 简单细胞：中心-周边 + 稀疏编码 | 区内空间滤波（中心 + 环绕，可选邻接掩码）| 区内 |
| `RegionLocalSynergy` | M1 肌肉协同：几十维活动由 ~10 个协同张成 | 区内低秩流形 | 区内 |
| `RegionLocalTransient` | V2 对瞬变/运动方向敏感 | 逐通道 on/off/持续 分解 | **逐通道（严格不混）** |
| `TaskContextGate` | dlPFC：维持任务集 + 自上而下偏置（top-down）| 衰减循环状态 + 乘法门控 | **跨组（这就是接口）** |
| `PredictiveInterface` | 预测编码：上行预测 / 下行误差分离 | 上下路分离 | **跨组（接口）** |

## 三条纪律（每个算子都必须可测）

1. **退化性**：关掉/置零 ⇒ 严格退化为「无此算子」⇒ 否则对比里混进了别的变量。
2. **关键性质**：区内算子**不能影响区外通道**（逐通道算子更严格：一个通道只影响自己）。
3. **梯度可达**：参数能收到非零梯度 ⇒ 否则这个结构学不到东西。

（与 `arch/test_domain_ssm.py` 的三项检查同构。）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 区内算子
# ---------------------------------------------------------------------------

class RegionLocalCenterSurround(nn.Module):
    """V1-like：区内可学习的**中心-周边**空间滤波 + 半波整流。

    脑科学：V1 简单细胞是中心-周边感受野（Olshausen & Field 1997 稀疏编码；
    Deco/Schilling/Schmidhuber 2010 的 V1→V2→V4→IT 层级模型）。

    ⚠️ **只作用在本区的通道上**（调用方负责把本区通道喂进来）⇒ 跨区不混。
    ⚠️ `adj` 传入真实 scalp 邻接时才是**真正的中心-周边**；不传（默认全 1）
    ⇒ 退化为「中心 + 全局环绕」，**这时它等价于一次加权平均**（仍不跨区）。

    参数
    ----
    in_ch : 本区通道数
    n_out : 本区输出通道数（“神经元”数）
    adj : (in_ch, in_ch) 邻接掩码（可选，∈{0,1}）⇒ **行归一化后作为「环绕池」**：
          环绕项只看**邻域内**通道的加权平均 ⇒ 才是真正的中心-周边。
          不传 ⇒ 环绕池 = 恒等（环绕项退化为「同通道的抑制性权重」，**不是**全局平均）。
    """

    def __init__(self, in_ch: int, n_out: int, adj: torch.Tensor | None = None):
        super().__init__()
        self.in_ch, self.n_out = in_ch, n_out
        self.Wc = nn.Parameter(torch.randn(n_out, in_ch) * (1.0 / max(in_ch, 1) ** 0.5))
        # 环绕项**强制非正**（脑科学：周边是抑制性的）⇒ 用 -softplus 保证 ≤ 0
        self.Ws_raw = nn.Parameter(torch.zeros(n_out, in_ch))
        if adj is not None:
            a = (adj.float() > 0).float()
            a = a + torch.eye(in_ch)          # 自己也在邻域里（否则孤立通道环绕池为空）
            a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)   # 行归一化
            self.register_buffer("A", a)
        else:
            self.register_buffer("A", torch.eye(in_ch))

    def surround(self) -> torch.Tensor:
        """环绕权重（恒 ≤ 0），形状 (n_out, in_ch)。"""
        return -F.softplus(self.Ws_raw)

    def surround_pool(self, x: torch.Tensor) -> torch.Tensor:
        """**邻域加权平均**（环绕池）：x (B, in_ch, T) ⇒ (B, in_ch, T)。

        ⚠️ 下标必须**共享通道轴**（`g` 同时出现在 x 与 A 里）⇒
        写成 "bct,gd->bdt" 会让 c 与 g 各求和一次 ⇒ 变成外积 ⇒ 全错。
        """
        return torch.einsum("bgt,dg->bdt", x, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) ⇒ (B, n_out, T)，半波整流后 ≥ 0。"""
        w_s = self.surround()
        y = torch.einsum("bct,oc->bot", x, self.Wc) \
            + torch.einsum("bct,oc->bot", self.surround_pool(x), w_s)
        return F.relu(y)

    @torch.no_grad()
    def sparsity(self, x: torch.Tensor) -> float:
        """输出稀疏度（**L0：非零元素占比**）——V1 稀疏编码的可测量。

        ⚠️ 稀疏度 = 非零比例（越小越稀疏）⇒ **不是**「非零值的均值」。
        """
        y = self(x)
        return float((y > 0).float().mean())


class RegionLocalSynergy(nn.Module):
    """M1-like：区内**低秩协同流形**。

    脑科学：运动皮层的活动是**低维**的——几十维肌电由 ~10 个肌肉协同张成
   （Todorov 的肌肉协同；Churchland 的 arm/hand manifold）⇒ ⇒ 算子 = 组内通道
    投影到 rank 维协同空间；重建时误差小 ⇒ 说明该区活动确实低维。

    参数
    ----
    in_ch : 本区通道数
    rank : 协同数（低维流形的维数）
    """

    def __init__(self, in_ch: int, rank: int):
        super().__init__()
        self.in_ch, self.rank = in_ch, rank
        self.U = nn.Parameter(torch.randn(in_ch, rank) / max(in_ch, 1) ** 0.5)
        self.W2 = nn.Parameter(torch.randn(rank, in_ch) / max(rank, 1) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) ⇒ z: (B, rank, T)（协同空间）。"""
        return torch.einsum("bct,cr->brt", x, self.U)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """从协同空间重建原始信号（用于测低维性）。"""
        return torch.einsum("brt,rc->bct", z, self.W2)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> float:
        """相对重建误差 ⇒ 越小说明该区越低维。"""
        z = self(x)
        rec = self.reconstruct(z)
        num = (rec - x).pow(2).sum()
        den = x.pow(2).sum().clamp_min(1e-12)
        return float(num / den)


class RegionLocalTransient(nn.Module):
    """V2-like：**逐通道**的瞬变/持续分解（on / off / sustained）。

    脑科学：V2 单元对**运动方向与瞬变**敏感 ⇒ 算子 = 每通道自己的时间差分核
    （grouped conv ⇒ **结构上就保证不混通道**）。

    ⚠️ 与前两个算子不同：**这个是严格逐通道的**（连区内都不混）⇒
    用来检验「时间结构要不要区内共享」这类问题。

    参数
    ----
    in_ch : 本区通道数
    learnable : True ⇒ 核可学；False ⇒ 固定为 [on, off, sustained]
    """

    def __init__(self, in_ch: int, learnable: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.learnable = learnable
        if learnable:
            self.conv = nn.Conv1d(in_ch, 3 * in_ch, kernel_size=3,
                                  padding=1, groups=in_ch, bias=True)
        else:
            # ⚠️ 固定核用**独立 buffer 名**（不能叫 conv ⇒ 会覆盖 Module ⇒ self.conv 变成张量）
            self.register_buffer("conv_fixed", _fixed_transient_kernel(in_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) ⇒ (B, 3*in_ch, T)，排列 [on | off | sustained]（逐通道）。

        ⚠️ **grouped conv 的输出通道是「按输入通道分组」的**（第 c 个输入通道
        产生第 3c..3c+2 个输出）⇒ 要得到 [on|off|sust] 三个块，必须先
        reshape 成 (B, C, 3, T) 再 permute ⇒ 否则下标会错位。
        """
        if self.learnable:
            y = self.conv(x)
        else:
            y = F.conv1d(x, self.conv_fixed, padding=1)
        b, c, t = x.shape
        return y.reshape(b, c, 3, t).permute(0, 2, 1, 3).reshape(b, 3 * c, t)


def _fixed_transient_kernel(in_ch: int) -> torch.Tensor:
    """固定核（**与 grouped conv 同样的「按输入通道分组」布局**）：
    第 c 个输入通道 ⇒ 权重行 3c, 3c+1, 3c+2 = [on, off, sustained]。"""
    w = torch.zeros(3 * in_ch, in_ch, 3)
    for c in range(in_ch):
        w[3 * c + 0, c, 1] = 1.0                  # on
        w[3 * c + 1, c, 1] = -1.0                 # off
        w[3 * c + 2, c, :] = 1.0 / 3.0            # sustained
    return w


# ---------------------------------------------------------------------------
# 跨区接口算子
# ---------------------------------------------------------------------------

class TaskContextGate(nn.Module):
    """dlPFC-like：**衰减循环状态** + 自上而下的**乘法门控**。

    脑科学：dlPFC 维持**任务集/上下文**并对下路做 top-down 偏置
    （Buschman & Miller 2007, *Science* 315:1860, DOI 10.1126/science.1138071：
    "top-down and bottom-up signals arise from the frontal and sensory cortex, respectively"；
    以及 C19 里的执行功能分解：策略/维度转移/抑制）。

    机制：s_t = γ·s_{t−1} + (1−γ)·x_t（沿时间指数衰减的循环状态）
          y_t = x_t ⊙ (1 + g·s_t)（乘法门控：上下文调制下路）
    ⇒ ⚠️ **这是跨区算子** ⇒ 它的输入可以是多区的拼接 ⇒ 它就是「接口层」的载体。

    退化性：**门控强度 g=0 ⇒ 输出 = 输入**（恒等）⇒ ⇒ 初始化 g=0 ⇒
    训练一开始这个算子**不改变信号** ⇒ 可以安全地挂上去再学。

    ⚠️ **由退化性直接推出的一个后果（不是 bug）**：g=0 时输出恒等于输入
    ⇒ 上下文通路（状态 s、以及它的归一化参数）对损失的梯度**恒为 0**
    ⇒ 「上下文通路的参数能不能学」必须在**门打开之后**才测
    （见 `arch/test_ops_brain.py`）；门控强度 g 自己的梯度则**任何时候都非零**
    （∂y/∂g = x ⊙ s 与 g 无关）。

    参数
    ----
    d_model : 状态维度
    gamma : 每步保留率（0 ⇒ 无记忆；1 ⇒ 恒定状态）
    """

    def __init__(self, d_model: int, gamma: float = 0.9):
        super().__init__()
        self.d_model, self.gamma = d_model, gamma
        # ⚠️ x 是 (B, D, T) ⇒ 要沿**通道**维归一化必须转置（见 context）
        self.ln = nn.LayerNorm(d_model)
        self.g = nn.Parameter(torch.zeros(1))      # 门控强度（0 ⇒ 恒等）

    def context(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T) ⇒ 上下文状态 s: (B, D, T)（沿时间衰减累加）。

        ⚠️ **不能用「zeros_like + 逐位赋值」**：那样会切断 autograd 图
        （状态对输入没有梯度 ⇒ 整个算子学不到东西）⇒ 这里用 stack 保留图。
        """
        g = self.gamma
        # ⚠️ LayerNorm 归一化**最后一维** ⇒ x.transpose(1,2) 后最后一维是通道维
        xn = self.ln(x.transpose(1, 2)).transpose(1, 2)
        acc = xn[:, :, 0]
        states = [acc]
        for t in range(1, x.shape[-1]):
            acc = g * acc + (1.0 - g) * xn[:, :, t]
            states.append(acc)
        return torch.stack(states, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T) ⇒ (B, D, T)，门控后的下路信号。"""
        s = self.context(x)
        return x * (1.0 + self.g * s)

    @torch.no_grad()
    def state_decay(self, T: int, channel: int = 0) -> float:
        """**冲激**输入下，末帧状态相对首帧的比值 ⇒ 检验衰减常数是否按 γ 走。

        ⚠️ 必须是**单通道**冲激：通道维上做 LayerNorm ⇒ 若所有通道同时为 1，
        会被归一化成 0（均值 1、标准差 0）⇒ 比值变 NaN。
        ⚠️ 常数输入**测不出衰减**（它是递归的不动点）⇒ 这里用冲激。
        """
        imp = torch.zeros(1, self.d_model, int(T))
        imp[:, channel, 0] = 1.0
        s = self.context(imp)
        s0 = s[:, channel, 0]
        denom = torch.where(s0.abs() < 1e-9, torch.full_like(s0, 1e-9), s0)
        return float((s[:, channel, -1] / denom).abs().mean())


class PredictiveInterface(nn.Module):
    """预测编码接口：**上行预测 / 下行误差** 分开。

    脑科学：预测编码（Rao & Ballard 1999, *Nature Neuroscience* 2:79-87；
    Friston 的自由能层级）——高层给出预测，低层给出**预测误差**。

    机制：p = W_up·x（上行预测）;  e = x − p（下行误差）
    ⇒ ⚠️ **这是跨区算子** ⇒ 它把「上一个区的输出」变成「下一个区的输入」，
    并让两条路**物理上分开**（不混在一条线上）。

    退化性：W_up = 0 ⇒ p=0, e=x（等于没有这个接口）；W_up = I ⇒ e=0（完全预测）。
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.W_up = nn.Parameter(torch.zeros(d_in, d_out))
        self.ln = nn.LayerNorm(d_in)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """**沿通道维**归一化（x 是 (B, D, T) ⇒ LayerNorm 的最后一维是 T ⇒ 必须转置）。

        ⚠️ 用 `self.ln`（带仿射参数）而不是裸 `F.layer_norm` ⇒ 否则这个 Module
        是死代码 ⇒ 它的参数永远拿不到梯度。
        """
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, D_in, T) ⇒ (p, e) 两个 (B, D_out, T)。

        ⚠️ einsum 必须**共享通道轴**（`c` 同时出现在 x 与 W_up 里）⇒
        写成 "bct,do->bot" 会把通道维也求和掉 ⇒ 全错。
        """
        xn = self._norm(x)
        p = torch.einsum("bct,co->bot", xn, self.W_up)
        e = xn - p
        return p, e
