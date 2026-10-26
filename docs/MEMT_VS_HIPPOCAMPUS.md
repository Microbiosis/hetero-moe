# MemT-1.0（MemTensor）与海马体的机制对照

> 本文对比 MemTensor 的 **MemT-1.0 记忆基础模型**与神经科学的**海马体记忆系统**，
> 目的是判断：这一架构是否具有可检验的"类脑"依据，还是仅有修辞上的相似。
>
> 结论先行：**有一个比 Engram 真实得多的对应——在本文对照的这些外部系统里，
> MemT-1.0 是第一个把"记忆可写、且允许被遗忘"做进计算图里的**。
> 它补上了 Engram 文档判定的最致命缺口（"记忆不可写"），
> 并且额外实现了 Engram 完全没有的**遗忘曲线**与**读-写隔离**。
> 但它仍然缺三样东西：稀疏正交的模式分离、递归式模式补全、状态依赖的可塑性调制。
> 详见第三、四节。

**归属说明**：本文对照的两个系统均属**第三方**——
DeepSeek 的 Engram（见 [`ENGRAM_VS_BRAIN.md`](ENGRAM_VS_BRAIN.md)）
与 **MemTensor** 的 MemT-1.0。
文中引用的少量源码片段仅用于标注机制证据，均注明了文件路径；
本文只做机制对照与批评，立场只针对机制本身。
本文提到的"这套比较 / 本系列"，指的是**本仓库的这组对照文档**
（`ENGRAM_VS_BRAIN.md` → `MEMT_VS_HIPPOCAMPUS.md`）。
两个被对照的系统分属不同团队、不同技术路线，只是在"给模型加一块记忆"这件事上目标相近。

**证据分级**（全文标注）：

| 标记 | 含义 |
|---|---|
| 【源码】 | 直接来自 `MemTensor/MemT-1.0` 开仓代码，可逐行核对 |
| 【README】 | 官方仓库 README 的自述声明，未经独立验证 |
| 【论文】 | 来自技术报告 arXiv:2610.26825（**本文未通读 PDF**，仅引用其标题/摘要级信息） |
| 【推论】 | 本文作者的推断，已标出推理链 |

**核对基线**：`MemTensor/MemT-1.0` @ `main`（tree sha `e75c13d090a6a9d34b29146d901bd84323185f24`），2026-10 取。

---

## 一、MemT-1.0 是什么

### 1.1 定位：记忆**原生**，而不是记忆外挂

> "MemT-1.0 is a new generation of memory foundation models that makes persistent,
> dynamically evolving memory an intrinsic part of model computation."
> ——【README】

关键主张有两个：

1. **持久状态恒定大小**：128K 历史下，相比完整 KV cache 存储降约 **66×**【README】
2. **两级记忆块**：一个 Native Memory Block 与主干注意力并行，含
   - **Local Memory Block**：动态多通道记忆矩阵，随交互步演化
   - **Hyper Memory Block**：可训练的记忆 q/k/v 投影，以及 **erase / write / forget / output** 四个门【README】

### 1.2 记忆状态的精确形状与读路径【源码】

状态是**固定大小的 FP32 张量**，与历史长度无关：

```python
# inference/memt_sglang/models/deepseek/config.py
@property
def state_shapes(self):
    return {
        layer_index: (1, self.num_heads, self.head_dim, self.head_dim)
        for layer_index in self.layers
    }
```

即：**每层、每头一个方阵 `S ∈ R^{head_dim × head_dim}`**，全模型固定。
DeepSeek-V4-Flash 侧汇总为 **336M 动态记忆状态**【README】，Nano 侧 **10M**。

**读路径是纯线性映射**【源码】：

```python
# inference/memt_sglang/models/deepseek/ops/read.py
state_matrix = states.reshape(count * heads, dim, states.shape[-1]).to(q.dtype)
result = torch.baddbmm(q.new_empty(()), q, state_matrix, beta=0, alpha=scale)
```

即 `read = q @ S · scale`，其中 `scale = head_dim ** -0.5`（对应配置项
`memory_read_scale: "k_inv_sqrt"`）。**读的时候不递推、不修改状态**。

### 1.3 写路径：GDN2 的误差修正递推【源码】

状态更新不是外积累加，而是 DeltaNet 式的**误差修正**形式。
Triton 内核里的两行是全部要害：

```python
# inference/memt_sglang/models/deepseek/ops/_triton/state.py
# v_new = u_wy − w_wy @ S
b_vn = b_u - tl.dot(b_w, b_h.to(b_w.dtype))
# S ← Diag(2^g_last) S + kgᵀ v_new
b_h = b_h * tl.exp2(b_gl.to(tl.float32))[:, None]
b_h = b_h + tl.dot(b_k, b_vn.to(b_k.dtype))
```

写成公式：

```
        ┌──────────────┐   ┌─────────────────────┐
   S ←  │ Diag(α) · S  │ + │ Kᵀ ( U − W S )      │
        └──────────────┘   └─────────────────────┘
          遗忘                误差修正写入
```

- `U`、`W` 由 `write_only_intra` 生成，即 WY 变换（把 token 级序列更新等价折叠成 tile 级矩阵乘）。
- `Kᵀ(U − WS)` = `U − W S` 是**当前状态下的预测误差**：先按现有记忆算预测 `WS`，再用真值 `U` 减去它。
  这意味着**新信息覆盖旧信息时会主动修正旧关联**，而不是盲目叠加。
- `α = 2^{g_last}`：tile 内的累积对数衰减。

### 1.4 四个门的精确定义【源码】

这是 MemT 最有信息量的部分。全部在
`inference/memt_sglang/models/deepseek/ops/compiled_tensor_ops.py`：

| 门 | 源码表达式 | 值域 | 含义 |
|---|---|---|---|
| **erase** | `erase_factor = (0.5 * F.logsigmoid(erase)).exp()` | `(0,1]` | 即 `√σ(erase)` |
| **write** | `write.sigmoid()` | `(0,1)` | 逐元素写入强度 β |
| **forget** | `log_decay = -decay_scale_log.exp() * softplus(decay + decay_bias)` | `(-∞,0]` **非正** | 逐头遗忘率 |
| **output** | `softcap * tanh(silu(g) / softcap)` | 有界 | 读出归一化的增益 |

三点必须说清楚：

1. **erase 是"稳定擦除门"**，不是随意加的衰减。它在 Q/K 归一化后作用在 K 上：

   ```python
   # prepare_stable_qk
   return l2norm(q_work).to(q.dtype), l2norm(k_work).to(k.dtype) * erase_factor
   ```

   因为 `k` 已被 L2 归一化，再乘 `√σ(erase)` ⇒ **`‖k‖² = σ(erase) ≤ 1`**。
   这保证 DeltaNet 删除项 `(I − βkkᵀ)` 的特征值不被放大，
   即状态更新**不会因为某次写入而指数发散**。配置项 `stable_gdn2_sqrt_erase_gate: True` 把它固化成了算法契约。

2. **forget 门非正**：`softplus > 0`、`exp(·) > 0`，所以数学上 `log_decay < 0`，
   于是 `α = exp(log_decay) ∈ (0,1)`，**只衰减、不增长**。
   （FP32 下 `softplus` 在极端负值处会下溢到 0，此时 `log_decay = 0`、`α = 1`，
   即"该元素不衰减"——仍然**不增长**，所以结论不变。）
   且衰减率是**逐头学习的**（`decay_scale_log` 形状 `[num_heads]`），
   逐元素调制（`decay_bias` 形状 `[value_dim] = heads × head_dim`）。
   ⇒ 一个模型里有**多个时间尺度的遗忘通道**。

3. **四个门分属不同分支**：`b_up`→erase、`w_up`→write、`f_down`→forget、`g_down`→output，
   由 `in_proj` 一次性投影出 `q,k,v` + 4 路门共 7 个量。

### 1.5 读-写隔离：回忆**不产生**学习【源码】

这是本文认为 MemT 最被低估的设计。

```python
# inference/memt_sglang/models/deepseek/layer.py
@torch.no_grad()
def packed(self, hidden_states, read_states, *, lengths, commit, write_states):
    ...
    if commit:
        read, next_state = token_memory_packed(query, key, v, log_decay, write,
                                               read_states, write_states, ...)
    else:
        read = snapshot_read(query, read_states, lengths, self.read_scale, ...)
        next_state = None
```

类文档字符串写得很直白：

> "All tokens read the incoming snapshot. A commit produces a separate final state;
> **a query returns no state update.** The scheduler controls publication and reset
> of that state."

执行器把这条纪律落到了运行时【源码】：

```python
# inference/memt_sglang/runtime/executor.py
for commit in (False, True):
    samples = [s for s in self.plan.samples
               if (s.request.operation == "commit") == commit and ...]
    ...
    if commit:
        self.pool.storage[layer].index_copy_(0, slots, final.unsqueeze(1))
```

⇒ **非 commit 的请求（普通生成）走 `snapshot_read` 分支，`next_state is None`，
已发布槽位一个字节都不会被改。** 只有显式的 `commit` 操作才产生新的状态。

### 1.6 事务语义：两槽位 MVCC【源码】

状态池的设计非常明确【源码】：

```python
# inference/memt_sglang/runtime/state.py
# One published slot per session and one candidate slot per transaction.
count = self.max_sessions + self.max_commits
```

控制面暴露了完整的动作集【源码，`runtime/control.py`】：
`create / begin / prepare / fence / publish / abort / status / memory_read /
kv_reset / reset / close / export / import`

于是每个会话的记忆生命周期是：

```mermaid
flowchart LR
    P["已发布槽位<br/>published slot<br/>（所有读取的唯一来源）"]
    T["事务候选槽位<br/>candidate slot<br/>（commit 在此演算）"]
    P -- "commit：从 published 出发演算" --> T
    T -- "publish：原子晋升" --> P
    T -- "abort：丢弃（槽位归还并清零）" --> X["丢弃"]
```

- `begin` 开事务，`publish` 晋升，`abort` 丢弃；
- `fence` 会中止该会话的**在途请求**（不改变已发布状态）；
- `import`（载入快照）**必须**在事务内进行（`"Import or pending writes require a transaction"`）；
- `finish(success=False)` 时不推进任何状态。

⇒ **读永远看到一致的已发布快照；写永远发生在私有副本上；发布是原子晋升。**
这是教科书式的多版本并发控制，而它恰好落在"记忆"这个词最难的地方。

### 1.7 快照：只存记忆，不存对话【源码】

```python
# inference/memt_sglang/runtime/snapshots.py
tensors = {f"memory.{layer}": data[slot].detach().cpu().contiguous()
           for layer, data in pool.storage.items()}
descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memt-state-")
save_file(tensors, temporary, metadata={"format": SNAPSHOT_FORMAT, "identity": identity})
```

- 格式 `memt-memory-v1`，safetensors，键名 `memory.<layer>`；
- 载入时校验 `identity`（模型/编码/格式身份）与 **FP32 + 有限值 + 精确形状**，
  不匹配即拒绝；
- 写入用临时文件 + `fsync` + `os.replace` 原子替换；自动分配的键用 `os.link`，
  保证"**自动分配的键绝不覆盖已有文件**"。

⇒ **快照里只有记忆矩阵，没有 transcript，没有 KV cache。**
README 的示例也确认：恢复会话时 `len(restored.messages) == 0`，但仍能回答
"我的猫叫什么名字、多大、喜欢吃什么"。

### 1.8 规模与许可

| 模型 | 主干 | 总参 | 激活参 | 可训练记忆参数 | 动态记忆状态 |
|---|---|---:|---:|---:|---:|
| MemT-1.0-Nano | Qwen3.6-35B-A3B | 36B | 3B | 451M | 10M |
| MemT-1.0-Flash | DeepSeek-V4-Flash | 288B | 17B | 3.61B | 336M |

其他【README】：11 个记忆基准上，Flash/Nano 在**被评测的参数化记忆方法**中排第 1/2；
但"**强文本记忆系统整体仍领先**，尤其在需要细节证据检索的长历史上"；
训练数据 34B tokens；`chunk_tokens = 2048`。

**许可（三条，别混）**：

| 对象 | 许可 | 商业使用 |
|---|---|---|
| 仓库软件（源码、SGLang 集成） | PolyForm Noncommercial 1.0.0 | **禁止** |
| 技术报告 / 论文材料 | CC BY-NC-SA 4.0 | 禁止（NC） |
| **MemT-1.0-Nano 权重** | **Apache-2.0**（HF 模型卡） | **允许** |
| 第三方组件（FLA） | MIT | 允许 |

README 还留了一句要紧的话："除非另有说明，**模型权重、数据集、基准资产、商标
与第三方材料不在上述许可的覆盖范围内**"——即"仓库 NC"**不等于**"权重 NC"，
但也不等于"权重一定 NC"，须以各次发布的条款为准。

> ⚠️ **本仓库只开源了推理侧**。仓库树中没有任何训练代码，
> README 提到的 "Mixture-of-Storage (MoS)" 与两阶段训练流程**在开源代码里没有对应实现**，
> 无法核验。本文所有机制结论只依赖推理侧代码，因而**不受此影响**；
> 但所有关于"它是怎么训出来的"的结论都只能标注为【README】或【论文】。

---

## 二、神经科学一侧：海马体做什么

### 2.1 三个理论，三个不同的关注点

海马体不是只有"索引理论"。把三个层次分清楚，后面的对照才不会失焦：

| 理论 | 主张 | 对工程设计的含义 |
|---|---|---|
| **索引理论**（Teyler & DiScenna 1986）| 海马不存情景本身，存指向新皮层的**索引** | 记忆 = 指针集合；召回 = 索引激活 → 皮层重激活 |
| **系统巩固理论**（Marr 1971；Frankland & Bontempi 2005）| 记忆经历**再巩固**，从海马依赖转为皮层依赖 | 需要一个"写→巩固→改写"的过程，而非一次写入 |
| **情节-语义转变**（McKenzie et al. 2017, *Science*）| 海马**模式分离失效也不阻止**记忆巩固 | 模式分离**不是**记忆的先决条件 |

第三条尤其重要：**它把"MemT 没有模式分离"从"致命缺陷"降级为"设计取舍"**。

### 2.2 突触可塑性：写侧的三个动作

生物学上的记忆写入发生在突触层面，至少有三个可分辨的动作：

| 生物学动作 | 分子/细胞层面 | 特征 |
|---|---|---|
| **LTP**（长时程增强）| 突触后 AMPA 受体插入 | 写入某条关联 |
| **LTD**（长时程抑制）| 受体内吞 | 削弱某条关联 |
| **遗忘 / 痕迹衰减** | 蛋白合成、突触修剪、trace decay | 旧关联随时间自行变弱 |

而**这三者都是"写侧"操作**。读取（recall）在海马是**被动的**：
检索由线索触发，痕迹本身不被改写；随后的再巩固（reconsolidation；
Hupbach 等 2008）是一个**独立的、需要触发的**过程——检索只是把记忆重新变得
"可塑"（labile），并不意味着它立刻被改写。

### 2.3 有限容量与遗忘曲线

海马体**不是**无损磁带。两条独立证据线：

- **容量**：Miller & MacKay (2006) 给出稀疏编码下的容量界；Fusi et al. (2005)
  讨论分布式存储与干扰。
- **遗忘曲线**：Wixted (2004, 2019) 的实证——遗忘是**幂律**而非指数，
  且**没有被提取就会加速**。
- **多时间尺度**：Migliore et al. (2016, *Nat Neurosci*) 与 McGrath et al. (2018, *eLife*)
  显示海马内部不同细胞群、不同上下文下的**时间常数不同**。

### 2.4 读-写分离：生物学上并不常见

工程上"读"和"写"分开是常规操作。生物学上却不然——大多数联想记忆模型
（Hopfield、DeltaNet、Mamba 的 SSM 状态）在**读的同时也在改状态**。
经典 Hopfield 网络有一个著名的病态：**按某顺序反复检索，会把检索顺序本身学进去**
（"记忆的干扰/遗忘"）。

海马不是这样的：**回忆不改写痕迹**，再巩固是一个被单独触发的、可能失败的操作。
MemT 恰好把这个分离做成了架构契约（§1.5–1.6）。

---

## 三、对照：三个真对应，三个仍缺

### ✅ 对应 1：固定容量的快权重 + 逐头遗忘（**最有价值的类脑点**）

| MemT【源码】 | 神经科学 |
|---|---|
| 状态 `S ∈ R^{d×d}`/头，**恒定大小**，与历史长度无关 | 海马容量有限，不是磁带 |
| `α = exp(log_decay) ∈ (0,1)`，**恒负**，逐头可学习 | 痕迹衰减 / 遗忘曲线 |
| `decay_scale_log` 逐头 ⇒ **多时间尺度通道** | 海马内不同细胞群时间常数不同（Migliore 2016; McGrath 2018） |
| 记忆是**状态**而非**轨迹** ⇒ 66× 存储降 | 压缩存储（分布式、皮层化） |

**这是与 Engram 最本质的差别。** Engram 是**只读**的静态查找表，
把"不遗忘"当作容量换性能的交易。MemT 反其道：**它承认记忆必须被遗忘，
并把遗忘做成了一个逐头学习的一等公民参数。**

Engram 文档最刺眼的那句"❌ 缺失 3：记忆不可写"，在 MemT 这里**不成立了**。

### ✅ 对应 2：门控在**写侧**，不在读侧

| | Engram | MemT | 生物学 |
|---|---|---|---|
| 门的位置 | **读侧**：检索到的记忆采纳多少 | **写侧**：写入强度 β、擦除 ε、遗忘 α | **写侧**：LTP/LTD/衰减 |
| 上下文的作用 | 决定"读不读" | 决定"写不写、写多少" | 决定"这一笔要不要记" |

Engram 文档把"读侧情境门控"评为最有价值的类脑点。MemT 把门挪到了写侧——
**这个位置更接近生物学**，因为生物记忆的"选择"发生在突触被写入的那一刻，
而不是在"要不要取用"的那一刻。

代价是：MemT **没有** Engram 那种显式的"当前上下文 vs 检索内容矛盾则弃用"机制。

### ✅ 对应 3：读-写隔离 ⇒ 回忆不污染记忆

| MemT【源码】 | 神经科学 |
|---|---|
| 非 commit 请求：`snapshot_read`，`next_state = None` | 回忆不改写痕迹 |
| 两槽位 MVCC：published / candidate，`publish` 才晋升 | 再巩固是**独立触发**、可能失败的过程 |
| `abort` 丢弃候选，`fence` 中止在途请求 | 巩固失败 ⇒ 痕迹保持原样 |
| 快照只含记忆，不含 transcript | 技能/习惯记忆独立于情景轨迹 |

**这是 MemT 独有的、Engram 完全不具备的结构。**
但要诚实地划出边界：**触发在 MemT 里是外部的**（agent 框架决定何时 commit），
**不是内生的**（生物学里检索本身会打开再巩固窗）。这是"对应 3"里唯一的折扣。

### ❌ 缺失 1：没有稀疏正交的模式分离

齿状回把相似输入映射到**不重叠**的稀疏表征【Rolls 2013】。
MemT 的状态是**稠密方阵**，"分离"靠的是 DeltaNet 的**误差修正**（覆盖而非叠加），
以及 erase 门保证的数值稳定——这是**功能上近似**，但**结构上不是**稀疏正交码。

**不过**：McKenzie et al. (2017, *Science*) 证明齿状回损伤后记忆巩固仍可进行，
所以这一项是**取舍**而非**硬伤**。真正的代价在别处：
稠密线性联想记忆的**干扰**（interference）会随写入次数增长，
而 DG 的稀疏码正是为压制干扰而存在的。

### ❌ 缺失 2：没有递归式模式补全

CA3 的递归吸引子动力学能从残缺线索**迭代地**恢复完整模式。
MemT 的读是**单次线性映射** `q @ S`——没有迭代、没有吸引子、没有"从部分线索演化出完整模式"。
补全能力完全外包给了主干 Transformer。

**这与 Engram 是同一个缺口。** MemT 把"写"做对了，"读"依然是最薄的一环。

### ❌ 缺失 3：没有状态依赖的可塑性调制

生物学里，**可塑性本身是被状态调节的**：多巴胺（纹状体）、乙酰胆碱（海马）、
去甲肾上腺素（前额叶）会改变"此刻写入的强度"。

MemT 的 `decay_scale_log` 是**学出来的常数**（逐头），`decay_bias` 也是常数。
没有"当前唤醒水平 / 情境 / 相位"这类**状态变量**去乘进写入强度。
⇒ MemT 的门是**上下文无关的**（对固定 token 而言），Engram 的读侧门是**上下文相关**的。

**有意思的是：两个系统恰好互补，各自缺对方有的那一半。**

---

## 四、把 Engram 和 MemT 放在同一把尺子上

| 维度 | Engram（DeepSeek V4.1） | MemT-1.0 | 神经科学理想型 |
|---|---|---|---|
| 记忆载体 | 哈希查找表，**只读** | FP32 快权重状态，**可写** | 突触权重，**可塑** |
| 容量 | 表总重 ~200 GB | 恒定大小，336M/10M 参数 | 有限但有组织 |
| 遗忘 | ❌ 无（表不衰减） | ✅ **逐头学习、恒负** | ✅ 痕迹衰减 + 多时间尺度 |
| 门位置 | 读侧（采纳门） | 写侧（erase/write/forget）+ 读侧 output | 写侧（LTP/LTD） |
| 模式分离 | ❌ 反模式分离（相似输入刻意同哈希） | ⚠️ 稠密 + 误差修正 | ✅ DG 稀疏正交 |
| 模式补全 | ❌ 单次查表 | ❌ 单次线性映射 | ✅ CA3 递归吸引子 |
| 读是否改写记忆 | —（表本来就只读） | ✅ **否**（query 无状态更新） | ✅ 回忆不改写痕迹 |
| 写入的事务性 | — | ✅ 两槽位 MVCC，publish/abort | ⚠️ 再巩固可失败但不隔离 |
| 快照能否脱离对话 | ✅ | ✅（只存 memory.*） | ✅ 技能独立于情景 |
| 记忆是否可从零训练 | ❌ 训练时冻结 | ✅ 451M/3.61B 可训练记忆参数 | — |

**一句话**：Engram 的类脑点在**读**，MemT 的类脑点在**写**；
而海马体两者都需要。MemT 是**这组对照里第一个把写侧做对**的系统。

---

## 五、能用实验检验吗？E1–E7

### 5.1 可行性：为什么这些预测现在真的可做

Engram 文档当时的结论是"V4.1-Flash 是 552B 模型，本仓库无算力运行"。
**MemT 的情况好得多**，原因是它的记忆块**小到可以整体移植**：

- 状态更新语义只有 3 行（`S ← αS + Kᵀ(U − WS)`）；
- 所有门都是**模块输出**，可直接取出做统计；
- 不需要跑主干就能测记忆块本身的行为。

⇒ **路线**：按 `DeepSeekMemoryBlock` 的语义写一个小规模忠实实现
（配置项 `FIXED_DEEPSEEK_MEMORY_CONFIG` 已经把算法选择全部钉死：
`gdn_version=stable_gdn2`、`memory_update_mode=token`、
`stable_gdn2_sqrt_erase_gate=True`、`tail_policy=commit`、
`memory_read_scale=k_inv_sqrt`、`memory_head_mode=mha`），
在合成任务上复现门控与遗忘行为。**这比跑一个 35B 模型便宜几个数量级。**

### 5.2 预测

| 实验 | 操作 | 预期（若假说成立）| 判据 |
|---|---|---|---|
| **E1** 写门选择性 | 喂入**高冗余/重复**内容 vs **新颖高熵**内容 | 冗余流上 `write` 门均值显著低，`‖ΔS‖` 接近 0；新颖流上高 | 门值分布 + 状态范数增量 |
| **E2** 遗忘曲线 | 写入一条关联后，在 N 个延迟步读回 | 读出量按 `exp(log_decay)` 衰减；**不同头的时间常数不同** | 各头读出量 vs 延迟的拟合时间常数 |
| **E3** 擦除门触发 | 反复写**同一个 key**，对比写**不同 key** | 同 key 时 erase 门上升、旧关联被覆盖；不同 key 时旧关联保留 | `‖S_after − S_before‖` + 旧关联读出量 |
| **E4** 读不写 | 纯查询（`commit=False`）前后对状态做逐位校验 | 状态**逐位相同**（`next_state is None`） | 状态张量哈希 |
| **E5** 事务隔离 | commit 到一半 `abort`；以及 `publish` 后 `import` 旧快照 | abort 后已发布状态不变；import 后旧快照内容原样恢复 | 发布槽位逐位校验 |
| **E6** 遗忘的**代价** | 关掉 forget 门（`log_decay → 0`）再跑 E2/E3 | 干扰上升：同类旧关联的错误读出显著增多 | 错误读出量 / 容量曲线 |
| **E7** 容量 vs 信噪比 | 图像数与重复数**各自**扫到 ≫200（test 与 train split 各做一遍）| 若图像数曲线在 200 后**继续上升** ⇒ 先前的平坦是评测集太小的假象，"低维映射"作废；若继续平掉且方差下降 ⇒ "低维"获独立支持 | R²(图像数) 与 R²(重复数) 两条外推曲线 |

**E6 是最有信息量的一条**：它把"遗忘"从一个工程特性变成一个**可证伪的权衡**——
遗忘不只是省空间，它在**换取干扰抑制**。如果关掉遗忘并不增加干扰，
那 MemT 的遗忘门就没有生物学对应物所说的功能，只是个正则化。

**E7 是唯一一条与 MemT 无关、纯本项目内部的预测**：它是对 §6.3 那条
"低维映射"解读的**证伪条件**，放在这里是为了标明它不属于机制检验，
而是**边界划定**。详见第六节。

### 5.3 与 ENGRAM 文档 E1–E4 的关系

两者**不冲突，是同一框架的两个轴**：

| | Engram 的门 | MemT 的门 |
|---|---|---|
| 位置 | 读侧 | 写侧 |
| E1 预测 | 打乱前序 token ⇒ gate 下降 | **打乱/替换新内容 ⇒ write/forget 门变化** |
| 测什么 | 采纳率（上下文一致性） | 写入率 + 遗忘率（选择性与时程） |

⇒ **可以共享同一个实验框架**：把"门控统计量"作为统一观测量，
先在 Engram 侧测采纳门、再在 MemT 侧测写入门与遗忘门，
检验"**情境一致性筛选**"与"**选择性写入 + 主动遗忘**"是不是同一个现象的两面。

---

## 六、与本项目（Hetero-MoE / OmniBrain）的关系

### 6.1 与 OmniBrain 域条件 SSM：同一个数学形式，不同的调制对象

| | OmniBrain 域条件 SSM | MemT 记忆块 |
|---|---|---|
| 门的形式 | `φ = exp(tanh(offset) · c)` | `α = exp(−exp(decay_scale_log) · softplus(decay + decay_bias))` |
| 数学形态 | **有界、恒负的乘性调制** | **有界、恒负的乘性调制** |
| 调制对象 | 状态转移矩阵的**相** | 状态**衰减率** |
| 条件来源 | **域标签 c**（外部条件） | **学出来的参数**（逐头） |

**形态上几乎同构**：两者都是"用一个有界门去乘进状态更新的速率"。
差别在于**条件从哪来**——OmniBrain 的门由外部域条件驱动，
MemT 的门由训练学出来（且逐头不同 ⇒ 多时间尺度）。

### 6.2 对 line 1「gate-style 中枢」的一个反向压力

本项目 line 1 的主张是**中央门式协调**（`docs/V30_0_GATE_STYLE_CENTRAL_THEORY.md`
→ 权威规范 `research/central_stability/v30_gate_softplus/SPEC.md`）。

MemT 提供的反例是：**门控调制可以在每个模块内部局部完成，而不需要中央协调。**
它的 erase/write/forget 三个门都是**局部的、逐头的、逐元素的**，
没有全局广播的门，也没有中央仲裁器。

⇒ **这对 line 1 的必要性论证是不利的。**
如果"门控"的主要收益（选择性写入 + 多时间尺度遗忘）可以在局部获得，
那么"必须有一个中央中枢来协调"就需要更强的论证。
**这是本文给本项目最重要的一条判断**：不是"MemT 支持我们"，
而是"MemT 让我们那条论证更难辩护"。

### 6.3 与 Tier 0 编码结果的对照：MemT 恰好补上 ridge 缺的两件

本项目的 ridge 编码 = **无门、无衰减、离线一次写死的线性联想记忆**。
它与 MemT 的记忆块是同一个算子家族，只差三样：

| | ridge 编码 | MemT 记忆块 |
|---|---|---|
| 写 | 全部训练样本都写 | **写门**选择性写 |
| 遗忘 | 无 | **逐头遗忘** |
| 读 | `X w`（无约束线性） | `q @ S`（归一化 + 门控输出） |

⇒ **MemT 是我们编码器在生物学方向上的一个"补全版"**：给同一个线性联想记忆
加上"选择性写入"和"主动遗忘"。

而 Tier 0 的标度律结果（`research/brain_domains/FINDINGS_TIER0_RSA.md` 第九节）正好是这条线索的**经验起点**：

> 重复数 ×40 ⇒ R² **×23.4**（+0.00089 → +0.02087，单调）
> 图像数 ×20 ⇒ R² 只 **×1.3**（+0.0136 → +0.0173，非单调）

**读法**：在 200 张图的范围内，**多给不同的图像几乎不提升泛化 R²**，
而多给重复次数提升一个数量级。这意味着"视觉特征 → 脑响应"这条映射
**内在维度很低**，少数几十个例子就能学出来。

**这对 MemT 的含义**：MemT 的有限容量状态**恰好要装的就是这种低维映射**——
它不是负担，而是必要条件。

**但必须标出一个反向约束（防止过度解读）**：

> ⚠️ **我们的饱和发生在"训练样本数"维度，MemT 的容量约束发生在"记忆槽位"维度，
> 两者不是一回事。** 我们的重复数-饱和（×40 ⇒ ×23.4）来自**单次观测的信噪比**，
> 而不是存储容量。因此**不能**用"海马体容量有限"来解释我们数据里的任何饱和现象。
>
> 混淆点也要点明：那条"图像数"曲线是在 **200 张图的 test split 内部**测的
> （训练池随 fold 增长、评测集固定），所以它测的是"**可学性是否饱和**"，
> 而不是"**存储是否够用**"。

⇒ **这恰好给出了一个可证伪的区分实验（E7）**：
在同一批图上，把**图像数**与**重复数**分别扫到远大于 200，
若图像数曲线在 200 之后**继续**上升（而不是继续平掉），
则先前的平坦是**评测集太小**的假象，"低维映射"的说法作废；
若继续平掉且方差下降，则"低维"得到独立支持。

**这是一个本项目现在就能做的实验**，而且它同时检验两个互不相干的命题。

### 6.4 可共享的实验框架

承 `docs/ENGRAM_VS_BRAIN.md` 第七节的框架，扩成三维：

1. **索引信号有效 vs 随机** → 测主干性能差（已有：Phase 0，Δ +0.428）
2. **门控/调制系数分布** → MemT 侧新增写入门、遗忘门两个观测量
3. ****状态动力学**（本项目尚缺）** → 门控不仅调制"写多少"，还调制"状态如何演化"。
   本项目此前的 τ（时间常数）研究线（`research/brain_domains/`）问的正是这个，
   而 `FINDINGS.md` 的结论是"域标签与原始活动冗余，该线已终止"。
   ⇒ **MemT 提示了这条线的第三条路**：不要问"域标签是否独立存在"，
   而要问"**门控调制是否改变了状态演化的时间结构**"。
   这是一个**不依赖域标签**的 τ 研究。

---

## 七、结论

| 问题 | 答案 |
|---|---|
| MemT-1.0 是否"类脑"？ | **比 Engram 更接近**，而且接近在**不同的轴**上 |
| 最接近海马体的是什么？ | **写侧**：可写的快权重 + 逐头学习的遗忘率 |
| 相对 Engram 补上了什么？ | **记忆可写**（Engram 三大缺失之三）、**遗忘曲线**、**读-写隔离** |
| 仍然缺什么？ | 稀疏正交的模式分离（**取舍，非硬伤**）、**递归式模式补全**、**状态依赖的可塑性调制** |
| 最有价值的类脑点？ | **写侧三门（erase / write / forget）+ 事务化的 read-写隔离**——"记忆可以被有选择地写入，并被有意地遗忘" |
| 命名/术语问题 | "记忆"在这里是**固定大小状态**，不是"存储的过往"；`memt\|session\|op` 协议里的 "memory" 更接近"工作状态" |
| 可检验的预测 | E1–E7 全部可在**小规模移植实现**上跑，无需 35B 模型 |
| 对本项目的影响 | ①对 line 1 中央门式协调构成**反向压力**；②给编码器补全方向；③给 τ 研究线一条**不依赖域标签**的新路 |

**一句话**：Engram 借用了海马体的**检索结构**；MemT 借用了海马体的**可塑性结构**
——可写、可衰减、且回忆不污染记忆。它仍然不是海马体（没有吸引子动力学，
没有稀疏正交码，可塑性不被状态调制），
但它是目前这组对照中**第一个把"记忆会变旧"当成机制而不是缺陷**的系统。

---

## 参考

| 主题 | 来源 |
|---|---|
| 记忆块实现 | `MemTensor/MemT-1.0` @ `main`，`inference/memt_sglang/models/deepseek/layer.py`（tree sha `e75c13d0`）|
| 门控精确定义 | `.../models/deepseek/ops/compiled_tensor_ops.py` |
| 状态递推 | `.../models/deepseek/ops/_triton/state.py`（`memt_gdn2_fwd_h`）|
| 读路径 | `.../models/deepseek/ops/read.py`（`snapshot_read`）|
| 算法契约 | `.../models/deepseek/config.py`（`FIXED_DEEPSEEK_MEMORY_CONFIG`）|
| 事务与状态池 | `.../runtime/control.py`、`.../runtime/executor.py`、`.../runtime/state.py`、`.../runtime/snapshots.py` |
| 官方自述 | 仓库 `README.md`（架构、规模、66×、基准排名、许可）|
| 技术报告 | arXiv:2610.26825《MemT-1.0: Towards Memory-Native Evolution》——**本文未通读 PDF** |
| 模型权重 | HF `MemTensor/MemT-1.0-Nano`（Apache-2.0）、`MemTensor/MemT-1.0-Flash` |
| 海马索引理论 | Teyler & DiScenna 1986, *Behavioral Neuroscience* 100(3):430–435 |
| 系统巩固 | Marr 1971；Frankland & Bontempi 2005, *Nat Rev Neurosci* 6:251–266 |
| 模式分离非必需 | McKenzie et al. 2017, *Science* 356(6341):eaal6827 |
| 模式分离/补全 | Rolls 2013, *Front Comput Neurosci* 7:63；Borzello et al. 2023, *Nat Rev Neurosci* 24(8):502–517 |
| 容量与干扰 | Miller & MacKay 2006, *J Comput Neurosci* 23(3):257–276；Fusi et al. 2005, *J Neurophysiol* 94(6):4300–4314 |
| 遗忘曲线 | Wixted 2004, *Psychol Learn Motiv* 43:19–59；Wixted 2019, *Psychol Sci* 30(4):416–423 |
| 多时间尺度 | Migliore et al. 2016, *Nat Neurosci* 19:1130–1142；McGrath et al. 2018, *eLife* 7:e39101 |
| 再巩固 | Hupbach, Hardt, Gomez & Nadel 2008, *Learn Mem* 15(8):574–579；Nadel, Hupbach, Hardt & Gomez 2008, *Handbook of Episodic Memory* 18:43–56 |
| 快权重 / 线性递归记忆 | Katharopoulos et al. 2020, ICML, arXiv:2006.16236；Schlag et al. 2021, NeurIPS, arXiv:2106.06294 |
| 本项目对照 | `docs/ENGRAM_VS_BRAIN.md`；`docs/V30_0_GATE_STYLE_CENTRAL_THEORY.md`；`research/brain_domains/FINDINGS_TIER0_RSA.md` |
