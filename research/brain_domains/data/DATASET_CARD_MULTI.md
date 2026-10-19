---
license: cc-by-4.0
tags:
  - eeg
  - electrophysiology
  - connectome
  - neuroscience
  - multi-species
  - time-series
size_categories:
  - 1M<n<10M
---
> **状态**：这份四源 100 ms 数据集已下线。五个来源的 bin 宽为 50 / 20 / 20 / 4 ms，
> 拼成同一张量需要向最粗看齐，等于对细粒度来源做不可逆降采样。现只发布
> [`illusion-brain/primate-spike-20ms`](https://huggingface.co/datasets/illusion-brain/primate-spike-20ms)
> （猕猴与人类皮层内，均为 20 ms，无需重采样）。本文件保留为构建记录。


# OmniBrain-Multi

1,409,850 个神经活动时间窗，来自 4 个来源、3 个物种，统一为 `(64, 8)` 张量，
附逐窗标签列。

另有 139,255 个果蝇连接组节点嵌入，单独存放。

这些标签**不构成层级结构**。实测见下文「标签的实测性质」。

## 来源

| source | 物种 | 模态 | 窗口数 | 原始数据 |
|---|---|---:|---:|---|
| `mouse_steinmetz` | *Mus musculus* | 胞外放电 | 314,332 | dandi:000017 |
| `macaque_nlb` | *Macaca mulatta* | 胞外放电 | 187,067 | dandi:000127 / 000130 / 000070 |
| `human_intracortical` | *Homo sapiens* | 皮层内记录 | 86,851 | FALCON H1/H2（dandi:000950 / 000954）|
| `human_eeg` | *Homo sapiens* | 头皮 EEG | 821,600 | THINGS-EEG2（OSF 3jk45）|
| `flywire` | *Drosophila melanogaster* | 连接组 | 139,255 节点 | FlyWire |

各来源的许可证不同，逐项见 [LICENSE.md](LICENSE.md)。

## 处理

四份时间序列数据走同一套流程；果蝇不同。

### 1. 时间重采样到 100 ms

| 来源 | 上游 bin | 操作 |
|---|---|---|
| `mouse_steinmetz` | 50 ms | 相邻 2 bin 求和 |
| `macaque_nlb` | 20 ms | 相邻 5 bin 求和 |
| `human_intracortical` | 20 ms | 相邻 5 bin 求和 |
| `human_eeg` | 4 ms（250 Hz） | 25 个采样取均值 |

放电计数可加，故求和；电位不可加，故取均值。

### 2. 归一化

逐 unit（或逐电极）沿时间轴 z-score，`μ` 与 `σ` 由该 unit 全部时间点算出。
EEG 按电极在 trial × 时间上合并计算。

### 3. 投影到 64 维

| 来源 | 方法 | 可逆 |
|---|---|---|
| `mouse_steinmetz` | 逐 session PCA（对 z-score 后的 unit×bin 矩阵做 SVD），取前 64 个时间主成分 | 否 |
| `macaque_nlb` | 逐记录片段 PCA，同上 | 否 |
| `human_intracortical` | 逐记录片段 PCA，同上 | 否 |
| `human_eeg` | 63 电极补一个零通道 | 是（去掉末通道） |
| `flywire` | 对称归一化邻接矩阵上做图信号传播，α=0.85，迭代 8 次，64 个随机初始信号 | 否 |

unit 数不足 64 时零填充到 64。各 session / 片段的投影矩阵互不相同，
因此第 k 个成分在不同来源之间没有对应关系。

`human_intracortical` 每个片段是一个 FALCON 记录块（H1 或 H2），
投影在块内完成。

头皮 EEG 的 z-score 在 (被试, split) 组内计算，覆盖该组全部试次
（train 66,160 / test 16,000），不是在抽样子集上算的。

### 4. 切窗

窗长 8 bin（0.8 s），步长 4 bin（50% 重叠）。末尾不足一窗的部分丢弃。

### 5. 存储

窗口转为 float16，按来源顺序纵向拼接。索引表行序与 `timeseries.npy`
第 0 维一致。图视图单独存为 `graph.npy`（float32）。

### 6. 标签生成

`domain_l1` 由窗内活动算出，不是上游标注：对窗内每个通道取 `|z|` 的均值，
按预定义的通道 → 功能域映射累加，除以该域的通道数，取最大者为该窗的域。

小鼠的通道 → 域映射来自 Allen CCF 区名前缀；猕猴与人类侵入式来自记录区；
EEG 来自电极布置。

`region` 与 `region_raw`：小鼠取 session 内 unit 数最多的区（session 级），
猕猴与人类取上游 `area` 列（窗级）。

### 7. 不可还原

经 PCA 与时间聚合后，窗口无法反推回原始放电序列或电压。
`human_eeg` 的补零通道可直接去掉；其余来源的投影不可逆。

## 文件

| 文件 | 形状 | 内容 |
|---|---|---|
| `timeseries.npy` | (1409850, 64, 8) float16 | 时间窗，1.44 GB |
| `graph.npy` | (139255, 64) float32 | 连接组节点嵌入，34 MB |
| `index_timeseries.parquet` | 1,409,850 行 | 逐窗标签 |
| `index_graph.parquet` | 139,255 行 | 逐节点标签 |
| `labels.json` | — | 各标签的取值与计数 |
| `group_sizes.csv` | — | source × domain 样本数 |
| `region_map.csv` | — | 每个区域原值到 UBERON 的映射 |
| `shared_nodes.csv` | — | 三个跨物种节点的窗口数 |
| `region_groups.npz` | — | 三个节点的行号数组 |

## 标签的实测性质

5 个标签列的实际结构**不是层级**。审计方法与结果如下。

### 结构：偏序集，不是链

对每对标签检验细化关系（α 的每个等价类是否完整落在 β 的某个类内）：

```
        modality      species
             \        /
              source
                 |
               region

        domain_l1  ← 与所有属性交叉
```

覆盖关系只有三条：`region < source < modality`、`region < source < species`。

`species` 与 `modality` **互不细化**——人类同时有头皮 EEG 与皮层内记录，
所以二者不能串联成链。最长链长度为 0（按细→粗排序的严格定义）。

### 投影算子：domain 与所有属性都不可交换

若两个划分嵌套，其投影算子应满足 Πa Πb = Πb Πa。实测最大偏差：

| 属性对 | max\|diff\| | |
|---|---:|---|
| source × species | 0.00e+00 | 可交换 |
| source × modality | 0.00e+00 | 可交换 |
| source × region | 0.00e+00 | 可交换 |
| species × region | 0.00e+00 | 可交换 |
| modality × region | 0.00e+00 | 可交换 |
| source × domain | 3.07e-02 | 不可交换 |
| species × modality | 4.11e-02 | 不可交换 |
| species × domain | 3.40e-02 | 不可交换 |
| modality × domain | 1.33e-02 | 不可交换 |
| region × domain | 4.26e-02 | 不可交换 |

`domain` 与全部属性都不可交换，与整个偏序集正交。

### 方差分解：语义标签合计解释 0.05%

沿唯一成立的链 `region ⊂ source ⊂ species`，在 120,000 个采样窗上做
精确正交分解：

| 层 | 方差占比 |
|---|---:|
| `Π_region` | 0.03% |
| `Π_source − Π_region` | 0.02% |
| `Π_species − Π_source` | 0.00% |
| **`I − Π_species`（残差）** | **100.0%** |

合计 1.0005（float32 累积误差）。

**所有语义标签加起来只解释 0.05% 的方差。**

因此这些标签适合用于：分组统计、域适应实验的设计变量、数据溯源。
不适合用于：作为模型的条件输入、当作数据的内在组织维度。

### 笛卡儿积：91.2% 为空

理论组合 `4 × 2 × 34 × 6 = 1632`，实际存在 143 个。

空位来自结构约束，不是缺数据：`human_eeg` 无解剖区、`flywire` 无时间窗、
小鼠数据无 `area` 列。

### 区域

`region_kind` 标明 `region` 属于哪类对象。

`index_timeseries.parquet`（1,409,850 行）：

| `region_kind` | 行数 | 内容 |
|---|---:|---|
| `anatomy` | 588,250 | 小鼠 Allen CCF 缩写、猕猴记录区、人 `M1` |
| `none` | 821,600 | 头皮 EEG，无焦点位置 |

`index_graph.parquet`（139,255 行）：`neuropil` 80,209、`cell_type` 27,089、`region` 105、`none` 31,852。

小鼠取 session 主导区（`region_level = session_majority`，314,332 行），
猕猴与人类逐窗（`window_area`，273,918 行）。

`human_intracortical` 的原值是 FALCON 任务编号（H1 抓取、H2 手写），
两套 Utah array 均位于手部运动皮层，统一记为 `M1`。

跨物种同源节点：

| 节点 | UBERON | 小鼠 | 猕猴 | 人 | 合计 |
|---|---|---:|---:|---:|---:|
| 初级运动皮层 | `UBERON:0001384` | 9,298 | 168,048 | 86,851 | 264,197 |
| 前额叶 | `UBERON:0000451` | 16,520 | 13,292 | — | 29,812 |
| 初级体感皮层 | `UBERON:0008933` | 6,901 | 5,727 | — | 12,628 |

27 个 UBERON 标识里，25 个只出现在单一物种的样本里。果蝇的 FBbt 条目与脊椎动物
无同源节点，`uberon_id` 为空。

复合标签（`M1_PMd`）的 `uberon_id` 写成 `UBERON:0001384|UBERON:0016634`，
两段都属于 `shared_node = M1`。

`region_groups.npz` 存这三个节点的行号（int32，对应 `index_timeseries` 行号）。

## 索引列

| 列 | 说明 |
|---|---|
| `source` | 4 个来源之一 |
| `species` | 学名 |
| `modality` | 记录方式 |
| `group_key` | 来源内的分组键（session / file / subject）|
| `window_idx` | 来源内序号，从 0 连续到该来源行数减一 |
| `domain_l1` | 6 类功能域之一 |
| `region` | 区域标签：小鼠 Allen 缩写、猕猴记录区、人 `M1`；EEG 为空 |
| `region_raw` | 上游原值，含 FALCON 任务编号 `h1_intracortical` / `h2_intracortical` |
| `region_kind` | `anatomy` / `neuropil` / `cell_type` / `region` / `none` |
| `structure_id` | Allen `MBA:` 或 FlyBase `FBbt:` 标识 |
| `uberon_id` | UBERON 标识，复合标签以 `\|` 并列 |
| `shared_node` | `M1` / `S1` / `PFC`，其余为空 |
| `mapping_status` | `exact` / `exact_single_species` / `rollup` / `coarse` / `compound` / `task_code_reinterpreted` / `uberon_missing` / `no_vertebrate_homolog` / `fbbt_unresolved` / `unassigned` |
| `region_level` | region 的粒度来源 |
| `area` | 上游原始区名（仅猕猴/人类）|
| `n_units` | 记录单元数（仅猕猴/人类）|
| `split` | `train` / `test`，仅头皮 EEG 有值 |
| `image_idx` | 刺激图在 THINGS 中的序号，仅头皮 EEG 有值 |
| `rep` | 该图的第几次重复，仅头皮 EEG 有值 |

索引行与 `timeseries.npy` 第 0 维一一对应。

## 加载

```python
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/omnibrain-multi"

W = np.load(hf_hub_download(REPO, "timeseries.npy", repo_type="dataset"),
            mmap_mode="r")                          # (1409850, 64, 8) float16
idx = pd.read_parquet(hf_hub_download(REPO, "index_timeseries.parquet",
                                      repo_type="dataset"))

# 取单一来源
sub = idx[idx.source == "mouse_steinmetz"]
x = W[sub.index.to_numpy()]                         # (n, 64, 8)

# 按功能域分组
for dom, g in idx.groupby("domain_l1"):
    print(dom, len(g))
```

## 64 维不是共同坐标系

每个来源各自做主成分分析投影到 64 维。小鼠的第 3 成分与人类的第 3 成分
不对应同一个生物实体。

因此：

- 可以把全部样本送进同一模型训练（形状一致）
- 可以用 `source` 做分组统计、域适应、多任务学习
- 不能声称这是"同一空间中的向量"
- 不能直接比较不同来源的第 k 个成分
- 不能把 `source` / `species` / `region` 当作数据的内在层次

跨源真正可比较的是无量纲统计量（自相关衰减、谱斜率、有效维度），
不是原始通道值。

## 各来源的统计特性

抽样 200 窗的统计（float32 还原后）：

| source | 均值 | 标准差 | 绝对值最大 |
|---|---:|---:|---:|
| `mouse_steinmetz` | +0.084 | 2.071 | 25.0 |
| `macaque_nlb` | −0.001 | 0.895 | 8.2 |
| `human_intracortical` | −0.006 | 1.253 | 8.4 |
| `human_eeg` | −0.000 | 0.991 | 59.7 |

小鼠的方差最大，因为其 PCA 前 64 成分解释方差比例较低（0.24–0.43），
残差更大。

## 在这份数据上做过的分析

| 结论 | 数值 | 状态 |
|---|---|---|
| **语义标签合计解释的方差** | **0.05%**（残差 100.0%）| 否定"分层" |
| 功能域标签与活动信号冗余 | 域 vs 随机 −0.04% | 否定 |
| 跨物种无守恒的动力学常数 | 峰度为 PCA 伪影 | 否定 |
| 超额曲线的深度依赖形状跨物种一致 | r = 0.75–0.99（6/8 配置）| 部分成立 |
| 时间常数 τ 可测但不预测行为 | ρ = +0.013, n=502 | 否定 |

## 限制

- 各来源的 64 维成分语义不同，不构成共同坐标系
- 标签不构成层级，且只解释 0.05% 的方差
- 采样率与空间规模差异大（小鼠 205–1240 units，人类 63 电极）
- 小鼠的 region 是 session 级，不是逐窗标注
- `macaque_nlb` 的 `M1_PMd` 跨 M1 与 PMd，无法在行级拆分
- `SCig`、`APN` 两个小鼠区（23,823 行）缺 UBERON 映射
- `human_eeg` 无解剖区
- 头皮 EEG 由 250 Hz 降到 100 ms bin（bin 内 25 采样取均值），且只保留刺激后 0–800 ms
- 方差分解的 0.05% 是在 588,250 个非 EEG 窗上算的：EEG 无 region，不在这条链上
- 图视图无时间维，单独存为 `graph.npy`
- 数据来自不同任务与实验室，未做行为对齐
- 91.2% 的标签组合为空

## 构建

从上游原始数据到本数据集共四步，均在 HF Space 内运行：

| 步骤 | 脚本 | 输入 | 输出 |
|---|---|---|---|
| 1 | `build_omnibrain.py` | Steinmetz、THINGS-EEG、FlyWire 原始文件 | `steinmetz/`、`graph_view/`、`eeg_view/` |
| 2 | `build_macaque_view.py` | NLB 的 `spikes_*.npz`（20 ms） | `spike_view_macaque/` |
| 3 | `build_human_invasive_view.py` | FALCON 的 `spike_counts` | `spike_view_human/` |
| 4 | `build_multi.py` | 上述四个视图 | 本数据集的主体文件 |
| 5 | `build_region_map.py` | 本数据集索引表 + Allen CCF + UBERON bridge | 区域映射列与三个附加文件 |

```bash
python research/brain_domains/data/build_omnibrain.py /domains /domains/omnibrain
python research/brain_domains/data/build_macaque_view.py
python research/brain_domains/data/build_human_invasive_view.py
python research/brain_domains/data/build_multi.py
python research/brain_domains/data/build_region_map.py
```

中间产物（各视图的 `windows_*.npz`、`meta.parquet`、PCA 编码器）保存在
bucket `illusion-brain/brain-domains`，未随本数据集发布。

代码仓库：`Microbiosis/hetero-moe`

## 引用

各来源的原始文献与许可证见 [LICENSE.md](LICENSE.md)。
使用本数据集时，请引用所使用的每个来源的原始工作。
