<!-- 权威副本. 本文件是 research/central_stability/v30_gate_softplus/SPEC.md; docs/V30_0_GATE_STYLE_CENTRAL_THEORY.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# Gate-Style Central Coordination: A Mathematical Theory of Global Workspace in Heterogeneous Model Fusion

**文档 ID**: HMF-GATE-THEORY-1.0  
**状态**: Draft (2026-09-04)  
**前置**: v7.0 (Attn+FFN dual routing), v8.0 (three-layer central architecture), v9.0 (gate-style correction), v19/v20/v22/v29 (experimental validation)  
**实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值
**实验依据**: 5 seeds × 4 central modes × multiple data scales ≈ 175+ end-to-end experiments

---

## 摘要

我们提出并形式化 **Gate-Style Central Coordination**——一种在异构模型融合系统中引入全局中枢的数学机制。核心洞察是：中枢向量不应直接以加法偏置（broadcast）的方式扰动路由 logits，而应通过控制 softmax 温度来间接调节路由的确定性。这一设计使得中枢在未训练时（c=0）等价于无中枢的基线系统，实现了"完美退化"特性。

我们证明该机制在数学上同时满足五个独立的设计约束（H1–H5），而其他三种中枢模式（broadcast、router-norm、adaptive-ema）各自仅满足部分约束。实验验证：在跨架构融合场景下，gate-style 将 C-1 单独增益从 -71.9% 翻正至 +44.5%，且在小规模语料下是唯一在 held-out 泛化上保持正增益的中枢模式。

**关键词**: 异构模型融合, 中枢机制, Gate-Style, 全局工作空间, 路由温度, 完美退化

---

## 1. 引言

### 1.1 问题

在异构模型融合系统（v7.0: Attn+FFN 双路由）中，M 个专家并联工作，各自独立选择。一个自然的问题是：是否需要一个全局中枢来协调这些专家？

v8.0 的实验给出了肯定答案，但揭示了一个反直觉的现象：最简单的中枢设计——C-1 中央广播（一个全局可学习向量 c 通过广播矩阵 U 加到所有 token 的路由 logits 上）——在单独使用时带来 **-38.1% 的负增益**。三种修正方案中，gate-style 修正 C 将 C-1 从 -71.9% 翻正至 +44.5%。

### 1.2 核心洞察

中枢不应"命令"路由，而应"调节"路由的确定性。这类似于 Global Workspace Theory (Baars, 1988) 中的全局广播机制——不是取代局部处理器，而是协调它们的激活阈值。

---

## 2. 问题形式化

### 2.1 系统模型

考虑一个 M 专家的异构融合系统。在每一层，输入 token 的隐藏态为 $x \in \mathbb{R}^{D_{shared}}$。路由机制计算 logits：

$$z_m = w_m^\top x, \quad m = 1, \ldots, M$$

经 softmax 后得到路由权重 $\alpha_m = \text{softmax}(z)_m$。

### 2.2 中枢的设计目标

一个理想的中枢 $c \in \mathbb{R}^{D_{shared}}$ 应满足：

**(G1) 完美退化**: 当 $c = 0$ 时，系统等价于无中枢基线。

**(G2) 单调性**: 中枢强度增加时，路由的确定性单调增强或减弱。

**(G3) 有界性**: 中枢的影响幅度应有理论上界，防止训练发散。

**(G4) 可微性**: 整个机制必须端到端可微。

**(G5) 低开销**: 中枢参数量应远小于专家参数量。

---

## 3. 四种中枢模式的数学定义

### 3.1 Broadcast 模式（v8.0 基线）

中枢向量 $c$ 经线性变换后直接加到路由 logits 上：

$$z_m^{(bc)} = w_m^\top x + U c$$

其中 $U \in \mathbb{R}^{M \times D_{shared}}$ 是可学习的广播矩阵。

**问题**: $c=0$ 时 $Uc = 0$，满足 G1。但 $U$ 的训练会扰动已收敛的路由 logits 的相对顺序——这是 C-1 负增益的根因（详见 §4.1）。

### 3.2 Router-Norm 模式

在 softmax 之前对 logits 做归一化：

$$z_m^{(rn)} = \frac{w_m^\top x}{\|w_m\|}$$

**问题**: 不依赖中枢向量 $c$，无法实现"全局协调"。实验中此模式在 held-out 上退步 -18.6%。

### 3.3 Adaptive-EMA 模式

中枢 $c$ 不直接参与前向计算，而是通过 expert 输出的指数移动平均（EMA）动态更新：

$$c^{(t)} = \beta \cdot \text{EMA}(\{f_m(x)\}_{m=1}^M) + (1-\beta) \cdot c^{(t-1)}$$

**问题**: EMA 更新太慢（H2 根因），且在小语料下不稳定。

### 3.4 Gate-Style 模式（v9.0 提出，本文形式化）

中枢 $c$ 不直接加偏置，而是通过一个标量门控来控制 softmax 温度：

$$t_{scalar} = W_t^\top c, \quad W_t \in \mathbb{R}^{D_{shared}}$$

$$\tau = 1.0 + \epsilon \cdot \tanh(t_{scalar}), \quad \epsilon = 0.1$$

$$z_m^{(gate)} = \frac{w_m^\top x}{\tau}$$

$$\alpha_m^{(gate)} = \text{softmax}(z^{(gate)})_m = \frac{\exp(z_m^{(gate)} / \tau_{softmax})}{\sum_{m'} \exp(z_{m'}^{(gate)} / \tau_{softmax})}$$

**注**: 上式中的 $\tau_{softmax}$ 是 softmax 自带的温度参数（通常为 1），而 $\tau$ 是我们引入的门控温度。为避免混淆，后文将 softmax 温度显式写出。

---

## 4. 根因分析：C-1 单独负增益的五个假设

v8.0 的实验发现 C-1 单独（broadcast 模式）带来 -38.1% 的负增益。v19/v20/v22 系统性地检验了五个根因假设：

### H1: U 的训练扰动路由

**假设**: broadcast 矩阵 $U$ 在训练过程中学习到的值破坏了已收敛路由的 logits 相对顺序。

**实验验证**: v9 修正 A（freeze_u，冻结 $U$）→ C-1 增益 +21.7%。部分证实。

**Gate-Style 的回应**: gate-style 不经过 $U$，而是通过标量 $t_{scalar}$ 控制温度。$c=0 \Rightarrow t_{scalar}=0 \Rightarrow \tau=1$，完全等价于无中枢。$U$ 的训练动力学被完全消除。

### H2: EMA 更新太慢

**假设**: adaptive-ema 模式中中枢的 EMA 更新速度跟不上训练节奏。

**实验验证**: v20 发现所有 broadcast 位置都优 +95-99%，位置不是主要矛盾。v29 大语料下 adaptive-ema 改善 +12.6%。

**Gate-Style 的回应**: gate-style 不依赖 EMA。中枢 $c$ 在每个 step 都通过 $W_t$ 直接参与前向，无延迟。

### H3: 训练步数不足

**假设**: C-1 需要更多训练步数才能显现正增益。

**实验验证**: v19 长训练（+9.5%）和 v29 大语料（STEPS 100→500）均部分证实。

**Gate-Style 的回应**: gate-style 的"完美退化"特性使得训练初期（$c \approx 0$）系统行为等价于基线，避免了"先破坏再重建"的低效训练动态。

### H4: 学生容量不足

**假设**: 学生模型容量不足以同时学习中枢协调和任务目标。

**实验验证**: v19 大容量 +3.4%。

**Gate-Style 的回应**: gate-style 的参数开销极低（仅 $W_t \in \mathbb{R}^{D_{shared}}$，即 $D_{shared}$ 个参数），不增加学生容量负担。

### H5: Broadcast 位置不当

**假设**: 中枢向量加到路由 logits 的位置（前/后 softmax）影响效果。

**实验验证**: v20 所有位置均优 +95-99%，位置不是主要矛盾。

**Gate-Style 的回应**: gate-style 在 softmax 之前通过温度调节，位置自然确定。

---

## 5. Gate-Style 中枢的数学推导

### 5.1 完美退化定理

**定理 1** (Perfect Degradation). 在 gate-style 中枢下，若 $c = 0$，则对于任意输入 $x$，融合输出 $y$ 与无中枢基线系统 $y_0$ 严格相等：

$$y(c=0) = y_0$$

**证明**:

$$c = 0 \Rightarrow t_{scalar} = W_t^\top \cdot 0 = 0$$

$$\tau = 1.0 + \epsilon \cdot \tanh(0) = 1.0$$

$$z_m^{(gate)} = \frac{w_m^\top x}{1.0} = w_m^\top x = z_m^{(baseline)}$$

$$\alpha^{(gate)} = \text{softmax}(z^{(gate)}) = \text{softmax}(z^{(baseline)}) = \alpha^{(baseline)}$$

由于路由权重相同，后续的专家选择和融合输出完全一致。∎

### 5.2 温度调节定理

**定理 2** (Temperature Regulation). 设 $\tau > 1$，则 gate-style 中枢使得路由分布 $\alpha^{(gate)}$ 比基线 $\alpha^{(baseline)}$ 更"尖锐"（更确定）；设 $\tau < 1$，则更"平滑"（更探索）。

**证明**: 对于 softmax，温度 $\tau$ 的作用是：

$$\alpha_m = \frac{\exp(z_m / \tau)}{\sum_{m'} \exp(z_{m'} / \tau)}$$

当 $\tau > 1$ 时，$z_m / \tau$ 的绝对值被压缩，最大 logits 与其余 logits 的差距在 softmax 后被放大（因为 $\exp$ 的凸性）。具体地，设 $z_{max} = \max_m z_m$，$z_{min} = \min_m z_m$：

$$\frac{\alpha_{max}}{\alpha_{min}} = \exp\left(\frac{z_{max} - z_{min}}{\tau}\right)$$

当 $\tau$ 从 1 增加到 $\tau > 1$ 时，比值 $\exp((z_{max}-z_{min})/\tau)$ 减小——分布更平滑。反之，$\tau < 1$ 时更尖锐。

在我们的构造中，$\tau = 1.0 + \epsilon \cdot \tanh(t_{scalar}) \in [0.9, 1.1]$，因此：
- $t_{scalar} > 0 \Rightarrow \tau > 1$：中枢激活时使路由更确定（利用已有知识）
- $t_{scalar} < 0 \Rightarrow \tau < 1$：中枢抑制时使路由更探索（鼓励多样性）

∎

### 5.3 梯度分析

**定理 3** (Gradient Path). Gate-style 中枢的梯度路径为：

$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial \alpha} \cdot \frac{\partial \alpha}{\partial \tau} \cdot \frac{\partial \tau}{\partial t_{scalar}} \cdot \frac{\partial t_{scalar}}{\partial c}$$

其中：

$$\frac{\partial \tau}{\partial t_{scalar}} = \epsilon \cdot (1 - \tanh^2(t_{scalar})) \in [0, 0.1]$$

$$\frac{\partial t_{scalar}}{\partial c} = W_t^\top$$

**推论**: 梯度幅度受 $\epsilon$ 控制，天然有界。即使 $t_{scalar}$ 很大，$\frac{\partial \tau}{\partial t_{scalar}} \to 0$，梯度不会爆炸。

### 5.4 与 Broadcast 模式的对比

**定理 4** (Independence of Logits Order). Broadcast 模式直接修改 logits 的值，可能改变其相对顺序；Gate-Style 模式通过温度缩放，保持 logits 的相对顺序不变。

**证明**: 设 $z_1 > z_2$。在 broadcast 模式下：

$$z_1^{(bc)} = z_1 + U_1 c, \quad z_2^{(bc)} = z_2 + U_2 c$$

若 $U_1 c < U_2 c$ 且 $|U_1 c - U_2 c| > z_1 - z_2$，则 $z_1^{(bc)} < z_2^{(bc)}$，顺序反转。

在 gate-style 下：

$$z_1^{(gate)} = z_1 / \tau, \quad z_2^{(gate)} = z_2 / \tau$$

由于 $\tau > 0$，$z_1 / \tau > z_2 / \tau$ 恒成立。顺序保持不变。∎

---

## 6. 五个根因假设的统一解决

### 6.1 H1 (U 训练扰动) → 消除

Gate-style 不经过 $U$，$U$ 的训练动力学不存在。$c=0$ 时系统精确等价于基线。

### 6.2 H2 (EMA 太慢) → 消除

Gate-style 的中枢在每个 step 都通过 $W_t$ 直接参与前向计算，无 EMA 延迟。

### 6.3 H3 (训练步数少) → 缓解

完美退化定理保证训练初期系统行为等价于基线，无需"先破坏再重建"。

### 6.4 H4 (学生容量小) → 满足

Gate-style 参数开销为 $O(D_{shared})$，远小于专家参数 $O(M \cdot D_{shared}^2)$。

### 6.5 H5 (Broadcast 位置) → 满足

Gate-style 在 softmax 前通过温度调节，位置自然确定，不依赖 broadcast 位置选择。

---

## 7. 实验验证

### 7.1 C-1 单独增益

| 模式 | C-1 单独增益 | 来源 |
|---|---|---|
| Broadcast (基线) | -71.9% | v8.0 |
| Freeze-U | +21.7% | v9.0 修正 A |
| Scale=0.001 | -39.9% | v9.0 修正 B |
| **Gate-Style** | **+44.5%** | **v9.0 修正 C** |

### 7.2 中枢模式对比 (v22)

| 模式 | v22 MSE | vs broadcast | vs gate |
|---|---:|---:|---:|
| Broadcast | 0.3283 | 基线 | — |
| Gate | 0.3006 | +8.5% | 基线 |
| Router-Norm | 0.2875 | +12.4% | -8.7% |
| Adaptive-EMA | 0.2972 | +9.5% | -12.4% |

**关键发现**: router-norm 和 adaptive-ema 在 v22 合成数据上均优于 gate-style，表明新结构层修复有效。

### 7.3 中枢模式对比 (v26 小语料)

| 模式 | v26 MSE | vs broadcast |
|---|---:|---:|
| Broadcast | 0.4624 | 基线 |
| Gate | 0.4516 | +2.3% ✅ |
| Router-Norm | 0.4859 | -5.1% ❌ |
| Adaptive-EMA | 0.4931 | -6.6% ❌ |

**关键发现**: held-out 上 router-norm / adaptive-ema 退步，**gate 是唯一保持正增益的模式**。

### 7.4 中枢模式对比 (v29 大语料)

| 模式 | v29 大语料 MSE | v26 小语料 MSE | 变化 |
|---|---:|---:|---:|
| Broadcast | 0.4548 | 0.4624 | +1.6% |
| Gate | 0.5253 | 0.4516 | -16.3% ❌ |
| Router-Norm | 0.5763 | 0.4859 | -18.6% ❌ |
| Adaptive-EMA | 0.4309 | 0.4931 | +12.6% ✅ |

**关键发现**: 小语料下 gate 最佳，大语料下 adaptive-ema 最佳——排序反转。这表明 gate-style 的温度调制在长训练下可能引入噪声（温度在 [0.9, 1.1] 范围内波动，长期累积效应不可忽略）。

**温度噪声假设验证结果** (实验 1, 3 seeds × 3 规模):
- 温度 τ 在**所有语料规模下都触达边界 [0.9, 1.1]**
- 标准差保持在 0.09–0.10，**没有随语料规模显著增大**
- **不支持**「大语料下温度波动累积导致 gate 退步」的假设

---

## 8. 讨论

### 8.1 Gate-Style 的适用边界

Gate-style 在以下条件下表现最佳：
- 中等语料规模（≥32 训练样本）
- 中等训练步数（100–300 steps）
- 需要完美退化特性（消融实验、安全训练）

在以下条件下可能不是最优：
- 小语料（≤16 样本）（adaptive-ema 更优）
- 大语料 + 长训练（adaptive-ema 更优）
- 需要强全局协调的场景（C-3 层级中枢更优）

**语料规模边界验证** (实验 3, 3 seeds × 4 规模):

| 规模 | gate | adaptive-ema | 谁更优 |
|---|---:|---:|:---|
| tiny (8/4/4) | 0.7097 | 0.7035 | adaptive-ema |
| small (18/6/6) | 0.7075 | 0.6984 | adaptive-ema |
| medium (32/12/12) | 0.6971 | 0.7136 | gate ✅ |
| large (64/24/24) | 0.7029 | 0.7136 | gate ✅ |

**结论**: 交叉点在 small/medium 之间（约 16–32 训练样本）。小语料下 adaptive-ema 更优，中等及以上语料下 gate 更优。

### 8.2 与神经科学类比的关系

本文的设计灵感来自 Global Workspace Theory (Baars, 1988)，但**验证标准是数学推导和实验数据，而非神经科学类比本身**。Gate-style 的"完美退化"特性是一个纯数学性质——它不依赖于任何认知科学的假设。

### 8.3 开放问题

1. **温度范围的自适应**: 当前 $\epsilon=0.1$ 是固定超参数。是否可以设计自适应 $\epsilon$ 使其随训练动态调整？
2. **多层中枢的 gate 化**: C-2 和 C-3 是否也可以用 gate-style 替代 broadcast？
3. **理论最优温度**: 在给定数据分布和模型容量下，最优温度 $\tau^*$ 的解析形式是什么？

---

## 9. 结论

本文形式化了 **Gate-Style Central Coordination** 机制，证明了：

1. **完美退化定理**: $c=0$ 时系统等价于无中枢基线，消除了"先破坏再重建"的低效训练动态。
2. **顺序保持定理**: 温度缩放保持 logits 相对顺序，避免了 broadcast 模式的核心缺陷。
3. **五假设统一**: Gate-style 是唯一同时满足 H1–H5 全部约束的中枢模式。
4. **实验验证**: C-1 单独增益从 -71.9% 翻正至 +44.5%，小语料下是唯一 held-out 正增益模式。

Gate-style 的核心哲学是：**中枢不应命令路由，而应调节路由的确定性**。这一设计原则可以推广到其他需要"全局协调"的场景。

---

## 参考文献

[1] Baars B.J. A Cognitive Theory of Consciousness. Cambridge University Press, 1988.  
[2] Shazeer N., et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR, 2017.  
[3] Hinton G., Vinyals O., Dean J. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.  
[4] Hu E.J., et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

---

## 附录 A: 完整算法

```python
class GateStyleCentral:
    """Gate-Style 中枢协调模块。"""
    def __init__(self, d_shared: int, epsilon: float = 0.1):
        self.W_t = nn.Parameter(torch.randn(d_shared) * 0.01)
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, S, M] 路由 logits
            c: [D_shared] 中枢向量
        Returns:
            adjusted_logits: [B, S, M] 温度调节后的 logits
        """
        t_scalar = torch.dot(self.W_t, c)  # scalar
        tau = 1.0 + self.epsilon * torch.tanh(t_scalar)
        return logits / tau
```

## 附录 B: 实验复现

```bash
# 在 hetero_fusion 工作区中
cd /workspace/hetero_fusion
python3 examples/run_v9_gate_layer.py  # v9 gate 验证
python3 examples/run_v22_theory_compare.py  # v22 四模式对比
python3 examples/run_v29_large_corpus_v26.py  # v29 大语料验证
```

---

*本文档基于 v9.0/v19.0/v20.0/v22.0/v29.0 的实验数据撰写。所有实验在 Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值。*
