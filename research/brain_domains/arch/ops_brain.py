"""ops_brain.py — 脑域线**自己写的算子**。

## 判据（两条，其余都是推论）

1. **符合使用场景**：算子必须回答「在我们的数据上、我们那个组里，它到底改变了什么」。
   我们只有 **63 通道 scalp EEG**、250 Hz、250 ms 窗，通道按脑区分组
   （POST 感觉/语义 34 / CENT 运动 14 / FRONT 执行 15）⇒ 任何在**视网膜或皮层表面**
   成立、但在 scalp 上不成立的空间机制，都不能照搬。
2. **符合脑科学定义下的作用**：算子的**名字承诺的机制必须在实现里兑现**。
   名字叫「中心-周边」却算成逐通道线性 = 形式主义（本文件里踩过一次，见下）。

⇒⇒ 由此推出唯一的结构性纪律：**算子全部是「区内」的**（只作用在自己那一组的通道上）；
**跨区**（FRONT→POST 的 top-down、POST→FRONT 的腹侧流）留给 exp90 的接口层定义，
不在算子里 ⇒ ⇒ 「区内混、区间不混」是**结构事实**，不是口号。

## 算子清单（每个都标了「用在哪 / 起什么作用 / 怎么被证伪」）

| 算子 | 用在哪 | 脑科学下的作用 | 我们这里的用法 |
|---|---|---|---|
| `RegionLocalCenterSurround` | POST 34 | V1 简单细胞中心-周边 + 稀疏编码 | 组内**群体相对**中心-周边：中心=本通道，环绕=同组其它通道的公共驱动；半波整流 ⇒ 稀疏 |
| `RegionLocalLowRank` | CENT 14 | 集群活动/共同驱动 | 该组活动投到 rank 维 ⇒ 是否少数成分在驱动（⚠️ 必须与同尺寸随机组比才有意义）|
| `RegionLocalTransient` | POST 34 | ON/OFF 通路 ⇒ V1 简单细胞 ON/OFF 亚群 + 持续成分 | 逐通道分出上升/下降瞬变与持续（250 Hz、250 ms 窗 ⇒ ON 约 100 ms）|
| `TaskContextGate` | FRONT 15 | dlPFC 维持任务集 + **加性** top-down 偏置 | 组内维持：衰减循环状态 ⇒ 偏置向量（g=0 ⇒ 恒等）|
| `PredictiveInterface` | POST 34 **组内层间**（V1→V2→V4→IT）| 预测编码 | 层间上行预测 / 下行误差**物理分开**（初始强预测 ⇒ 误差路只带新信息）|

## 踩过的坑（改掉前请先读）

- ⚠️ **环绕池不能含自己**：环绕池若取恒等（`A = I`），则
  `y = Wc·x + Ws·x = (Wc+Ws)·x` ⇒ **中心-周边退化成逐通道线性层，机制蒸发**
  ⇒ 环绕池必须**排除自己**、取同组其它通道的平均（群体相对版）。
- ⚠️ **on/off 必须用带符号的差分再取非负**：用固定核 `[0,1,0]` 只能取到**符号**，
  分不出幅度（on 与 off 的幅度都是同一绝对值）⇒ 差分核要带正负两支，再 `relu`。
- ⚠️ **einsum 在通道轴上必须共享下标**：`"bct,co->bot"`；写成 `"bct,do->bot"` 会把通道维也求和掉。
- ⚠️ **LayerNorm 归一化最后一维**：信号是 (B, D, T) ⇒ 要沿通道维归一化必须 `transpose(1, 2)`。
- ⚠️ **`zeros_like` + 逐位赋值会切断 autograd** ⇒ 状态累加要用 `torch.stack`。
- ⚠️ **grouped conv 的输出按输入通道分组**：第 c 个输入通道产生第 3c..3c+2 个输出
  ⇒ 要 [on|off|sust] 三个块必须 reshape→permute。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 区内算子
# ---------------------------------------------------------------------------

class RegionLocalCenterSurround(nn.Module):
    """V1-like：组内**群体相对**的中心-周边 + 半波整流。

    脑科学：视觉皮层简单细胞是中心-周边感受野（Olshausen & Field 1997 稀疏编码）。

    ⚠️ **我们没有视网膜/皮层表面，只有 scalp 通道** ⇒ 视觉皮层的「空间邻域」
    在我们这里**不存在**（POST 34 个通道在 scalp 上并不空间相邻）⇒ 所以采用
    **群体相对版**：中心 = 本通道自己；环绕 = **同组其它通道**的平均
    （该区的公共驱动 / 共同成分）⇒ 这才是我们能测、且 EEG 上真实存在的中心-周边。

    ⚠️ 环绕池**必须排除自己** ⇒ 否则中心-周边退化成逐通道线性（见文件头「踩过的坑」）。

    参数
    ----
    in_ch : 本组通道数
    n_out : 本组输出通道数（“神经元”数）
    adj : (in_ch, in_ch) 可选邻接掩码（∈{0,1}）⇒ 传了就是**空间**中心-周边
          （只在将来有真实 scalp 2D 邻接时用）；不传 ⇒ 群体相对版（环绕 = 同组其它通道）
    """

    def __init__(self, in_ch: int, n_out: int, adj: torch.Tensor | None = None):
        super().__init__()
        self.in_ch, self.n_out = in_ch, n_out
        self.Wc = nn.Parameter(torch.randn(n_out, in_ch) * (1.0 / max(in_ch, 1) ** 0.5))
        # 环绕项**强制非正**（脑科学：周边是抑制性的）⇒ 用 -softplus 保证 ≤ 0
        self.Ws_raw = nn.Parameter(torch.zeros(n_out, in_ch))
        if adj is not None:
            a = (adj.float() > 0).float()
            a = a * (1.0 - torch.eye(in_ch))      # ⚠️ 排除自己（否则退化成逐通道线性）
            a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
            self.register_buffer("A", a)
        else:
            # 群体相对版：环绕 = 同组**其它**通道的平均（对角为 0，行归一化）
            a = torch.ones(in_ch, in_ch) * (1.0 - torch.eye(in_ch))
            a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
            self.register_buffer("A", a)

    def surround(self) -> torch.Tensor:
        """环绕权重（恒 ≤ 0），形状 (n_out, in_ch)。"""
        return -F.softplus(self.Ws_raw)

    def surround_pool(self, x: torch.Tensor) -> torch.Tensor:
        """**同组其它通道**的加权平均（环绕池）：x (B, in_ch, T) ⇒ (B, in_ch, T)。

        ⚠️ 下标必须**共享通道轴**（`g` 同时出现在 x 与 A 里）⇒
        写成 "bct,gd->bdt" 会让 c 与 g 各求和一次 ⇒ 变成外积 ⇒ 全错。
        """
        return torch.einsum("bgt,dg->bdt", x, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) ⇒ (B, n_out, T)，半波整流后 ≥ 0。"""
        y = torch.einsum("bct,oc->bot", x, self.Wc) \
            + torch.einsum("bct,oc->bot", self.surround_pool(x), self.surround())
        return F.relu(y)

    @torch.no_grad()
    def sparsity(self, x: torch.Tensor) -> float:
        """输出稀疏度（**L0：非零元素占比**）——V1 稀疏编码的可测量。"""
        y = self(x)
        return float((y > 0).float().mean())


class RegionLocalLowRank(nn.Module):
    """集群活动 / 共同驱动：组内**低秩分解**（共同成分 + 特异成分）。

    脑科学：同一块皮层/同一群神经元的活动被少数几个成分驱动
    （population activity，Toron/Bressler 综述）。
    ⚠️ **不是肌肉协同**（Todorov）：肌肉协同要 EMG/ECoG；scalp EEG 上低秩
    来源更可能是共模/参考/共享体积传导 ⇒ 这里只敢叫「共同驱动」。

    ⚠️ **低秩本身不是证据**：任何有限信号都能被投影到低维 ⇒ 这个算子
    的价值**只能由与同尺寸随机组的对比给出**（exp96 的地板）⇒ 单独看
    reconstruction_error 没有意义。

    参数
    ----
    in_ch : 本组通道数
    rank : 成分数（低维流形的维数）
    """

    def __init__(self, in_ch: int, rank: int):
        super().__init__()
        self.in_ch, self.rank = in_ch, rank
        self.U = nn.Parameter(torch.randn(in_ch, rank) / max(in_ch, 1) ** 0.5)
        self.W2 = nn.Parameter(torch.randn(rank, in_ch) / max(rank, 1) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) ⇒ z: (B, rank, T)（共同成分空间）。"""
        return torch.einsum("bct,cr->brt", x, self.U)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """从成分空间重建原始信号（用于测低维性）。"""
        return torch.einsum("brt,rc->bct", z, self.W2)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> float:
        """相对重建误差 ⇒ 越小说明该组活动越能被少数成分张成。"""
        z = self(x)
        rec = self.reconstruct(z)
        num = (rec - x).pow(2).sum()
        den = x.pow(2).sum().clamp_min(1e-12)
        return float(num / den)


class RegionLocalTransient(nn.Module):
    """ON/OFF 通路 ⇒ V1 简单细胞的 ON/OFF 亚群 + 持续成分（**逐通道**）。

    脑科学：视网膜/LGN 分 ON 与 OFF 通路 ⇒ 皮层简单细胞相应分成 ON 与 OFF 亚群
    （另一半是持续反应）⇒ ⇒ 算子 = 每通道自己的「上升瞬变 / 下降瞬变 / 持续」。

    ⚠️ **不是 V2**：运动方向敏感是 MT/V5 的属性；V2 主要是整合与边缘/朝向。

    ⚠️ 与前两个算子不同：**严格逐通道**（连组内都不混）⇒ 用来检验
    「时间结构要不要在组内共享」这类问题。

    ⚠️ 使用场景：250 Hz、250 ms 窗 ⇒ ON 响应约 100 ms ⇒ 窗内事件数很少
    ⇒ 这个算子在我们的窗里**信息量有限**，别当主力。

    参数
    ----
    in_ch : 本组通道数
    learnable : True ⇒ 核可学（仍对 on/off 施加非负性）；False ⇒ 固定为差分+平滑
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
        y = y.reshape(b, c, 3, t).permute(0, 2, 1, 3).reshape(b, 3 * c, t)
        # on / off 是非负的稀疏响应 ⇒ 施加半波整流；sustained 保留原值
        on = F.relu(y[:, :c])
        off = F.relu(y[:, c:2 * c])
        sust = y[:, 2 * c:]
        return torch.cat([on, off, sust], dim=1)


def _fixed_transient_kernel(in_ch: int) -> torch.Tensor:
    """固定核（**与 grouped conv 同样的「按输入通道分组」布局**）：
    第 c 个输入通道 ⇒ 权重行 3c, 3c+1, 3c+2 = [上升差分, 下降差分, 持续]。

    ⚠️ 差分核**带正负两支**（中心差分 0.5·[x(t+1) − x(t−1)]）⇒ 上升为正、
    下降为负；配合 forward 里的 relu ⇒ on 块得到上升幅度、off 块得到下降幅度
    （若用 `[0,1,0]` 这种单支核，只能取到符号，on/off 幅度会相同）。
    """
    w = torch.zeros(3 * in_ch, in_ch, 3)
    for c in range(in_ch):
        w[3 * c + 0, c, 0] = -0.5                 # on：上升差分
        w[3 * c + 0, c, 2] = 0.5
        w[3 * c + 1, c, 0] = 0.5                  # off：下降差分
        w[3 * c + 1, c, 2] = -0.5
        w[3 * c + 2, c, :] = 1.0 / 3.0            # sustained：三点平滑
    return w


# ---------------------------------------------------------------------------
# 组内接口算子（跨区的接口留给 exp90，不在算子里）
# ---------------------------------------------------------------------------

class TaskContextGate(nn.Module):
    """dlPFC-like：组内**维持** + **加性** top-down 偏置。

    脑科学：dlPFC 维持任务集/规则并对下路做 top-down 偏置
    （Miller & Cohen 2001 的 PFC bias signal 是**加性**的 task-dependent bias；
    Buschman & Miller 2007, *Science* 315:1860, DOI 10.1126/science.1138071：
    "top-down and bottom-up signals arise from the frontal and sensory cortex, respectively"）。

    机制：s_t = γ·s_{t−1} + (1−γ)·x_t（沿时间指数衰减的循环状态）
          y_t = x_t + g·s_t（**加性**偏置：上下文调制下路）

    ⚠️ **逐通道**（状态与偏置都不混通道）⇒ 组内使用不违反「区间不混」。
    ⚠️ **不做归一化**：加了跨通道的 LayerNorm 会造成组间统计耦合 ⇒ 不要。
    ⚠️ 使用场景的诚实边界：scalp EEG 的 FRONT 15 通道上**没有**任务切换范式
    ⇒ 「任务集维持」没有直接对应物 ⇒ 这个算子能不能用，只能由 exp90 的
    接口判据（偏置是否携带信息）回答，不能靠 dlPFC 的名义回答。

    退化性：**偏置强度 g=0 ⇒ 输出 = 输入**（恒等）⇒ 初始化 g=0 ⇒
    训练一开始这个算子**不改变信号** ⇒ 可以安全地挂上去再学。
    ⇒ 由退化性直接推出：g=0 时上下文通路对损失的梯度**恒为 0**
    ⇒ 「上下文能不能学」必须在**门打开之后**才测（见 `arch/test_ops_brain.py`）。

    参数
    ----
    d_model : 状态维度（= 本组通道数）
    gamma : 每步保留率（0 ⇒ 无历史；1 ⇒ 状态恒定）
    """

    def __init__(self, d_model: int, gamma: float = 0.9):
        super().__init__()
        self.d_model, self.gamma = d_model, gamma
        self.g = nn.Parameter(torch.zeros(1))      # 偏置强度（0 ⇒ 恒等）

    def context(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T) ⇒ 上下文状态 s: (B, D, T)（沿时间衰减累加）。

        ⚠️ **不能用「zeros_like + 逐位赋值」**：那样会切断 autograd 图
        （状态对输入没有梯度 ⇒ 整个算子学不到东西）⇒ 这里用 stack 保留图。
        """
        g = self.gamma
        acc = x[:, :, 0]
        states = [acc]
        for t in range(1, x.shape[-1]):
            acc = g * acc + (1.0 - g) * x[:, :, t]
            states.append(acc)
        return torch.stack(states, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T) ⇒ (B, D, T)，加性偏置后的信号。"""
        return x + self.g * self.context(x)

    @torch.no_grad()
    def state_decay(self, T: int, channel: int = 0) -> float:
        """**冲激**输入下，末帧状态相对首帧的比值 ⇒ 检验衰减常数是否按 γ 走。

        ⚠️ 必须是**单通道**冲激：若所有通道同时为 1，比值会失真。
        ⚠️ 常数输入**测不出衰减**（它是递归的不动点）⇒ 这里用冲激。
        """
        imp = torch.zeros(1, self.d_model, int(T))
        imp[:, channel, 0] = 1.0
        s = self.context(imp)
        s0 = s[:, channel, 0]
        denom = torch.where(s0.abs() < 1e-9, torch.full_like(s0, 1e-9), s0)
        return float((s[:, channel, -1] / denom).abs().mean())


class PredictiveInterface(nn.Module):
    """预测编码接口：**上行预测 / 下行误差** 分开（组内层间，如 V1→V2→V4→IT）。

    脑科学：预测编码（Rao & Ballard 1999, *Nature Neuroscience* 2:79-87；
    Friston 的自由能层级）——高层给出预测，低层给出**预测误差**。

    机制：p = W_up·x（上行预测）;  e = x − p（下行误差）
    ⇒ 两条路**物理上分开**（不混在一条线上）⇒ 误差路只带「预测没解释掉」的部分。

    初始化：**W_up = I（层间强预测先验）** ⇒ 训练初期误差路≈0 ⇒
    之后误差路里长出来的东西才是「新信息」；若初始化为 0，则一开始
    全部信息都在误差路上，预测路学什么无从判断。

    ⚠️ **必须等维**（d_in == d_out）：误差 e = x − p 要求 x 与 p 同维 ⇒
    组内层间（V1→V2→V4→IT）天然等维；跨组不等维的接口**不在这个算子里**
    （留给 exp90 的接口层）。
    """

    def __init__(self, d_in: int, d_out: int | None = None):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        if d_in != d_out:
            raise ValueError(
                f"PredictiveInterface 要求等维（d_in == d_out），收到 d_in={d_in}, d_out={d_out}；"
                "跨组不等维的接口不属于本算子")
        self.d_in, self.d_out = d_in, d_out
        self.W_up = nn.Parameter(torch.eye(d_in))    # 层间强预测先验
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
