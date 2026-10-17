# OmniBrain：跨数据集的泛化脑域数据集

> 目标：把四个物种不同、模态不同、采样不同的神经数据集，投影到**统一张量布局**，
> 供域条件模型训练与跨数据集泛化评估。
>
> 本文档先固定 schema 与**哪些是事实、哪些是假设**，避免产出"看起来统一但不可信"的数据。

---

## 一、根本困难：四个数据集不能简单拼接

| 数据集 | 物种 | 信号 | 通道语义 | 时间基 | 域标签粒度 |
|---|---|---|---|---|---|
| Steinmetz | 小鼠 | spike counts | **神经元**（326–1240/会话）| 50 ms | 逐单元解剖区 |
| FlyWire | 果蝇 | 连接组 | 神经元（图节点）| **无时间** | 逐神经元解剖区 |
| THINGS-EEG | 人类 | EEG 电位 | **电极**（63 个）| 4 ms (250 Hz) | 电极→头皮区 |
| NLB 猕猴 | 猕猴 | spike counts | 神经元 | **未标注**（NLB 惯例 5 ms，待核）| 数据集级区域 |

**三个不可回避的语义断层**：

1. **通道语义不同** —— EEG 通道是"空间采样点"，spike 通道是"独立神经元"。
   两者投影到同一个 64 维空间**不保证可比**。
2. **物种不同** —— 小鼠 CA1 ≠ 人类海马 ≠ 果蝇蘑菇体。
   跨物种域映射是**假设**，不是事实。
3. **时间基不同** —— 50 ms / 4 ms / 未知。强行对齐会引入重采样伪影。

---

## 二、设计决策：分层 + 分视图，而非强行混合

### 2.1 不把 EEG 和 spike 混进同一个张量

采用**双视图（multi-view）**结构：

```
OmniBrain/
├── eeg_view/            # 电极空间：人类 EEG
│   └── windows (N, 64, 8)    # 64 通道 × 8 个 100ms bin
├── spike_view/          # 神经元空间：小鼠
│   └── windows (N, 64, 8)    # 64 个 PCA 成分 × 8 bin
├── spike_view_macaque/  # 神经元空间：猕猴（2026-10-17 新增）
│   └── windows (N, 64, 8)    # 同小鼠，但域是片段级
└── graph_view/          # 图空间：果蝇连接组
    └── graphs (M, 64)        # 每神经元的图嵌入，无时间维
```

**共享的东西只有两件**：`domain_l1` 标签、`window` 时间基。
模型可以学跨视图共享的域表征，但数据层**不假装两种信号同质**。

### 2.2 域标签用三层本体，只有 L1 跨数据集

```
L0  modality     : eeg | spike | connectome
L1  functional   : sensory | motor | memory | association | subcortical | other
L2  anatomical   : 原始名称（VISp / ME / CA1 / frontal ...）
```

- **训练只用 L1** —— 6 类，跨物种可勉强对齐（视觉就是视觉）
- **L2 仅供分析** —— 写清物种，禁止跨物种混用
- L1 映射表是**人工假设**，每一条都标注依据等级：
  - `strong`：文献共识（如 VISp → sensory）
  - `medium`：功能类比（如 CX → memory，属推理）
  - `weak`：仅按名称字面归类

### 2.3 时间对齐只覆盖两个数据集

| 数据集 | 原始 bin | 目标 100ms | 操作 | 是否采用 |
|---|---|---|---|---|
| Steinmetz | 50 ms | 2 bins 求和 | sum | ✅ |
| THINGS-EEG | 4 ms | 25 samples 均值 | mean | ✅ |
| **NLB 猕猴** | **20 ms** | **5 bins 求和** | **sum** | ✅ **2026-10-17 纳入** |
| FlyWire | 无时间 | — | — | ❌ 仅图视图 |

**猕猴的 bin 宽已核实**：NLB 官方 README 明确 "spike counts in 20 ms time bins"。

⚠️ **但域标签粒度不同，这是不可比的**：

| | 小鼠 spike_view | 猕猴 spike_view_macaque |
|---|---|---|
| 域来源 | 逐 unit 的解剖区（CA1/VISp/…）| 整个片段一个区域（Area2/DMFC/M1_PMd）|
| 域粒度 | **窗级**（随时变化）| **片段级常量**（整段不变）|
| 域条件建模 | 有信息可学 | **必然无增量**（模型只需记住片段）|

因此猕猴视图定位为**跨物种动力学对照**，不作域条件训练材料。

**顺带更正**：`area2-bump` 此前被标注为 "S1_M1 混合" 是错的。
NLB 官方文档明确它是 **Brodmann 区 2（躯体感觉皮层）**。

### 2.4 归一化分两级，都保留 A/B 开关

1. **per-channel z-score**（必需）—— 消除量纲与电极/单元阻抗差异
2. **per-domain 去均值**（默认关闭）—— 若开启会**抹掉域间均值差异**，
   使"域分类"退化为"域内形态分类"，需与随机标签对照一起评估

---

## 三、张量与元数据布局

```
omnibrain/
├── MANIFEST.json                  # 全局索引 + 资源统计 + 一致性状态
├── ontology.csv                   # L1/L2 映射表（含证据等级）
├── eeg_view/
│   ├── windows.npy                # (N, 64, 8) float16
│   ├── meta.parquet               # 每窗: subject, trial, t0, domain_l1, domain_l2
│   └── encoder.npz                # 电极→64 的投影矩阵
├── spike_view/
│   ├── windows_<session>.npy      # (n_win, 64, 8) float16
│   ├── meta_<session>.parquet
│   └── encoder_<session>.npz      # 逐 session 的 PCA（不同会话单元数不同）
└── graph_view/
    ├── embeds.npy                 # (M, 64) float32
    └── meta.parquet
```

### 样本与域标签的定义

**一个样本 = 一个时间窗内的多通道活动 + 该窗的活动加权主导域。**

为什么不用"通道自身的域"：一个时间窗内所有通道同时活动，不存在"这个窗只属于一个电极"。
因此域标签按**活动强度加权投票**：

```
domain_score[d] = Σ_{ch ∈ d} |z_ch|      对窗内每个 bin 求平均后累加
domain_l1       = argmax_d domain_score[d]
domain_conf     = max(score) / Σ(score)   # 记录置信度, 低于阈值可丢弃
```

`domain_conf` 一并输出，便于事后按置信度筛选（默认保留全部，仅标注）。

**窗口定义（关键修正）**：统一的是 **bin 宽 = 100 ms**，不是窗长。

原因：THINGS-EEG 单个 trial 只有 250 samples @ 250 Hz = **1.0 s = 10 bins**。
若强行要求 32 bins (3.2 s)，EEG 侧将无窗可用，只能拼接跨 trial 片段 —— 那会
把 trial 边界当连续信号，引入伪结构。因此：

| 视图 | 可用长度 | 窗口 | 步长 | 每 trial/段 窗数 |
|---|---|---|---|---|
| eeg_view | 10 bins | **8 bins (0.8 s)** | 4 | 1 |
| spike_view | 数百~上千 bins | **8 bins (0.8 s)** | 4 | 数十~数百 |

窗长 8 bins 是两者交集，且足够覆盖 EEG 的主要诱发成分（0.8 s）。
**这是取舍，不是对齐** —— 文档明确记录。

---

## 四、必须随附的三项对照（否则结论无意义）

1. **随机标签对照** —— 打乱 `domain_l1` 重训，若性能相当则域信号是假的
2. **数据集来源对照** —— 只训单个数据集 vs 混合，量化"跨数据集迁移"到底有没有
3. **shuffle 通道对照** —— 打乱通道顺序（破坏空间结构，保留统计量），
   检验模型是否真的用了空间结构

---

## 五、构建顺序（分阶段，每阶段可独立验收）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P1 | spike_view（Steinmetz）| 窗口数、域分布、z-score 后均值≈0 标准差≈1 |
| P2 | eeg_view（THINGS-EEG）| 同上 + 通道投影可逆性检查 |
| P3 | graph_view（FlyWire）| 图嵌入维度、域分布 |
| P4 | 合并 + MANIFEST | 跨视图域标签一致，无 NaN |
| P5 | 三个对照实验脚本 | 各自可跑出数字 |

**本实现覆盖 P1–P4**，P5 的对照脚本单独提供。

---

## 六、诚实声明（写入 MANIFEST）

```json
{
  "known_limitations": [
    "EEG 与 spike 分属不同视图，未假设二者同质，跨视图迁移是研究对象而非前提",
    "跨物种 L1 映射为人工标注，证据等级见 ontology.csv",
    "猕猴数据集未纳入时序视图（上游 bin 宽未核实）",
    "per-domain 去均值默认关闭；开启会削弱域可分性"
  ]
}
```
