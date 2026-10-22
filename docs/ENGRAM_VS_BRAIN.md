# Engram（DeepSeek V4.1）与海马体的机制对照

> 本文对比 DeepSeek 的 **Engram 条件记忆模块**与神经科学的**海马体记忆索引理论**，
> 目的是判断：这一架构设计是否具有可检验的"类脑"依据，还是仅有修辞上的相似。
>
> 结论先行：**有一个真实的结构对应（线索触发的稀疏检索），但三个关键机制缺失**。
> 详见第三节。

---

## 一、Engram 是什么（基于官方实现文档）

来源：vLLM 官方文档 `features/engram`、论文 arXiv:2601.07372（ACL 2026）、
DeepSeek-V4.1-Flash 模型卡。

### 1.1 定位：**条件记忆**，而非条件计算

> "Where Mixture-of-Experts scales capacity through conditional computation,
> Engram scales it through conditional memory: parameters that are never
> computed over, only looked up, so capacity grows at O(1) per-token cost."

| | MoE | Engram |
|---|---|---|
| 扩容方式 | 更多专家（每个 token 激活少数几个）| 更大查找表（每个 token 查少数几行）|
| 参数是否被计算 | 是 | **否，只查表** |
| 单 token 成本 | O(激活参数量) | **O(1)** 查表 |

这是 DeepSeek 提出的"**第三条稀疏轴**"：除了 MoE 的专家稀疏，还有记忆查找稀疏。

### 1.2 两阶段机制

**阶段一：Retrieval（检索）**

1. **分词器压缩**：启动时把规范化后相同的 token 合并
   （NFKC、去重音、小写、压缩空白）→ `" The"`、`"the"`、`"THE"` 哈希到同一 id
2. **多头哈希**：对每个 n-gram 阶数 n，多个哈希头各自做乘法-XOR 哈希
   → 每个 (阶数, 头) 对索引表中**互不重叠**的素数大小桶区间
   → 检索到的行拼接成一个记忆向量

关键点：**索引仅由 token id 决定，在前向传播之前就已知**。
这正是它能 CPU offload 的原因。

**阶段二：Fusion（融合）**

```python
# 伪代码（基于文档描述）
memory = concat([table[h].lookup(ngram_hash(h, ctx)) for h in heads])
key = wkv(memory)
gate = sigmoid(dot(normalize(hidden), key))    # hidden 已聚合全局上下文
hidden = hidden + gate * value(memory)
```

**门控是核心**：

> "If the retrieved memory contradicts the current context, the gate tends
> toward zero and suppresses it."

即：检索到的记忆是**与上下文无关的先验**（可能有哈希碰撞、多义性），
用一个基于当前隐状态的 sigmoid 门决定采纳多少。

### 1.3 规模（V4.1-Flash 实测）

| 项 | 数值 |
|---|---|
| Engram 层数 | 2 层 |
| 每层行数 | ~3.84 亿 |
| 每行大小 | 256 字节 FP8 + ue8m0 块缩放 |
| **表总重** | **~200 GB** |
| 存放位置 | 默认 **pinned host memory**（CPU），UVA 直取 |

200 GB 的表、每 token 只查几行——这就是"参数容量与计算量解耦"的极致。

---

## 二、海马体的对应机制（神经科学一侧）

### 2.1 记忆索引理论（Teyler & DiScenna, 1986）

核心主张：**海马体不"存储"情景记忆本身，而是存储指向新皮层各区域的索引**。

```
情景体验 → 新皮层多个区域分布式编码
         → 海马体形成一个"索引"（指向这些区域的指针集合）
线索出现 → 海马体激活索引 → 新皮层各区域被重新激活 → 记忆重现
```

### 2.2 两个互补的计算

| 机制 | 解剖位置 | 功能 |
|---|---|---|
| **模式分离**（pattern separation）| 齿状回 DG | 把相似的输入映射到**不重叠**的稀疏表征 |
| **模式补全**（pattern completion）| CA3 递归环路 | 给部分线索 → 恢复**完整**记忆 |

DG 的稀疏编码是关键：稀疏 + 强投射到 CA3，使相似经历被区分开。
CA3 的递归吸引子动力学则负责从残缺线索恢复完整模式。

### 2.3 Engram 细胞（生物学实体，非模型概念）

Josselyn & Tonegawa 等人的工作：一组被学习过程激活的神经元集合，
后天重新激活这组细胞可唤起对应记忆。

**注意**：生物学的 "engram" 与 DeepSeek 的 "Engram" 模块**同名但不同物**。
前者是神经元集合，后者是哈希查找表。这个命名容易引起误解。

---

## 三、对照：一个真对应，三个缺失

### ✅ 对应 1：线索触发的稀疏检索（结构性）

| Engram 模块 | 海马体索引理论 |
|---|---|
| 前 n 个 token → 哈希 → 查表某几行 | 部分线索 → 激活索引 → 召回记忆 |
| 只查少数行（其余不碰）| 只有 engram 细胞集合被激活 |
| 索引由 token 决定，前向之前已知 | 索引由线索决定 |

**这是真实的类比**：二者都是"用小线索线性定位大容量存储中的一小部分"。
Engram 的 O(1) 查表与海马索引的"指针"概念在功能上确实同构。

### ❌ 缺失 1：没有模式分离

DG 的稀疏编码要求**相似输入映射到不重叠表征**。

Engram 的哈希恰恰相反：**相似输入被刻意映射到同一行**
（`" The"` / `"the"` / `"THE"` 哈希相同）。这是**模式合并**，不是分离。

设计意图也不同：Engram 想压榨语义密度，DG 想避免干扰。
**二者目标相反。**

### ❌ 缺失 2：没有模式补全

CA3 的递归吸引子能从残缺线索恢复完整模式——这是"记忆重建"。

Engram 的检索是**纯前馈的一次查表**：给定 n-gram，返回固定行，
没有迭代、没有吸引子动力学、不能从部分线索"演化"出完整模式。

它的"补全"完全依赖 Transformer 主干之后的注意力层。

### ❌ 缺失 3：记忆不可写

生物记忆的核心是**可塑性**：经验改变突触权重，新记忆被写入。

Engram 的表在推理时是**只读的静态查找表**。训练时写入，部署后冻结。
模型不能从新经验中在线更新记忆。

（这一点上 MoE 的专家也不可在线写——整个深度学习都有此局限。
但 Engram 把"记忆"显式独立出来，反而让这个缺失更刺眼。）

---

## 四、门控机制：一个值得注意的细节

Engram 的门控设计有一个**功能上很"生物"**的性质：

```python
gate = sigmoid(dot(normalize(hidden), key))
hidden = hidden + gate * value
```

检索到的记忆是**与上下文无关的先验**（context-independent prior），
而门由**已聚合全局上下文的隐状态**计算。若记忆与当前情境矛盾，门趋零。

这对应神经科学中的**情境依赖的检索**：海马体的召回受当前状态
（如 theta 相位、 neuromodulator 水平）调控，不是无条件回放。

**这是我认为 Engram 最有价值的类脑点**，比"查表"本身更有意义：
不是"有记忆表"，而是"**记忆的采纳受情境门控**"。

> **后续对照**：**MemTensor 的** MemT-1.0——另一家团队的另一条技术路线，
> 把同一个"门"挪到了**写侧**（写入强度 / 擦除 / 遗忘），
> 并补上了本文判为最致命缺失的"记忆可写"，另加**逐头遗忘**与**读-写隔离**。
> 见 [`MEMT_VS_HIPPOCAMPUS.md`](MEMT_VS_HIPPOCAMPUS.md)。

---

## 五、能否用一个实验检验？

假设：**如果 Engram 的门控真的在做"情境一致性筛选"，
那么人为破坏上下文（如打乱前序 token）应该使门控值系统性下降。**

这是一个可检验的预测：

| 实验 | 操作 | 预期（若假说成立）|
|---|---|---|
| E1 | 正常上下文下的平均 gate 值 | 基准 |
| E2 | 随机打乱前 n 个 token | gate 显著下降 |
| E3 | 保持 n-gram 但替换语义不相关的上下文 | gate 下降 |
| E4 | 对照组：只替换与记忆无关的远处 token | gate 不变 |

**难度**：V4.1-Flash 是 552B 模型，本仓库无算力运行。
可行的替代：用官方 Engram 仓库（`deepseek-ai/Engram`）里的小规模实现，
在合成任务上复现门控行为。

**这与本项目的关系**：OmniBrain 的域条件 SSM 也有类似结构——
域路由的 `φ = exp(tanh(offset) · c)` 可以看作一种"域一致性门"。
两者可以共享同一个实验框架。

---

## 六、结论

| 问题 | 答案 |
|---|---|
| Engram 是否"类脑"？ | **部分**。线索触发稀疏检索这一点是真实的 |
| 是否是海马体的实现？ | **不是**。缺模式分离、模式补全、可塑性 |
| 最有价值的类脑点？ | **情境门控**（记忆采纳受当前上下文调制），而非查表本身 |
| 命名问题 | "Engram" 在神经科学中特指神经元集合，此处指查找表，**易误导** |
| 可检验的预测 | 打乱上下文应降低 gate 值（E1–E4 设计）|

**一句话**：Engram 借用了"索引式检索"的**功能结构**，但没有借用海马体的
**计算机制**（分离/补全/可塑）。把它当作"类脑记忆"是过度解读；
把它当作"受情境门控的稀疏查表"则准确且有启发。

---

## 七、与 OmniBrain 项目的关系

| OmniBrain 的域条件 | Engram 的条件记忆 |
|---|---|
| 域标签 → 调制 SSM 状态转移 A | n-gram → 查表取记忆行 |
| `φ = exp(tanh(offset)·c)` 有界调制 | `gate = sigmoid(dot(hidden, key))` 门控 |
| 已验证：域信号存在（Phase 0, Δ +0.428）| 未公开门控行为的实证 |

**共性**：二者都是"条件化"——用某种索引（域标签 / n-gram）
调制主干的动力学或输入。

**可共享的实验框架**：
1. 构造"索引信号有效"与"索引信号随机"两组对照
2. 测主任务性能差
3. 测门控/调制系数的分布是否随输入变化

本项目的 Phase 1（域条件建模）就是这个框架的第一次实例化。

---

## 参考

| 主题 | 来源 |
|---|---|
| Engram 实现细节 | vLLM docs `features/engram`（2026-10）|
| Engram 论文 | arXiv:2601.07372 / ACL 2026 long 226 |
| 模型卡 | deepseek-ai/DeepSeek-V4.1-Flash |
| 官方实现 | github.com/deepseek-ai/Engram |
| 海马索引理论 | Teyler & DiScenna 1986, *Behavioral Neuroscience* |
| 模式分离/补全 | Rolls 2013, *Frontiers in Computational Neuroscience* |
| Engram 细胞 | Josselyn & Tominaga 2020, *Science* |
