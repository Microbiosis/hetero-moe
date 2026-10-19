---
license: cc-by-4.0
tags:
  - electrophysiology
  - neuroscience
  - spike-counts
  - multi-species
  - time-series
size_categories:
  - 1M<n<10M
---

# Primate Spike 20 ms

猕猴与人类皮层内记录的放电计数，两种来源的上游 bin 宽都是 20 ms，未做时间重采样。

| 文件 | 形状 | 内容 |
|---|---|---|
| `counts.npy` | (5483479, 192) uint8 | 原始放电计数，1.05 GB |
| `counts_index.parquet` | 175 行 | 每个片段的边界、单元数、区域 |
| `timeseries.npy` | (273918, 64, 40) float16 | 派生窗口，1.40 GB |
| `index_timeseries.parquet` | 273,918 行 | 派生窗口的标签 |

`counts.npy` 是未压缩的原始计数：没有 PCA、没有切窗、没有重叠。
片段在行方向首尾相接，边界由 `counts_index.parquet` 的 `bin_start`/`bin_end` 给出。
超出 `n_units` 的列恒为 0。

`timeseries.npy` 是从它派生的一个视图（逐片段 PCA 到 64 维、800 ms 窗、50% 重叠），
方便直接训练。**建议优先用 `counts.npy`**：窗长、步长、PCA 秩都变成参数，
而不是写死在文件里。

## 来源

| source | 物种 | 片段 | 单元 | bin 数 | 时长 | 原始数据 |
|---|---|---:|---:|---:|---:|---|
| `macaque_nlb` | *Macaca mulatta* | 72 | 40–192 | 3,743,510 | 0.3–67 min | dandi:000070 / 000127 / 000130 |
| `human_intracortical` | *Homo sapiens* | 103 | 176–192 | 1,739,969 | 0.4–28 min | dandi:000950 / 000954 |

猕猴的 72 段来自三个记录集：`churchland`（67 段，M1 与 PMd，中心外伸手）、
`dmfc-rsg`（3 段，DMFC，计时）、`area2-bump`（2 段，S1 区 2，触碰扰动）。
人类的 103 段来自 FALCON H1（40 段，七自由度抓取）与 H2（63 段，手写），
两套 Utah array 均植入手部运动皮层。

各来源均为 CC BY 4.0，逐项见 [LICENSE.md](LICENSE.md)。

## 处理

`counts.npy` 只做两件事：把不等长序列对齐成矩阵（补零到该片段的最大长度），
把 float16 计数转为 uint8（全库最大计数为 20）。**不做 z-score、不做 PCA。**

`timeseries.npy` 在此基础上：

1. 逐单元沿片段做 z-score
2. 逐片段 PCA 到 `k = min(64, n_units−1, n_bins−1)` 个成分
3. 切 40 bin（800 ms）窗，步长 20 bin（50% 重叠）
4. 转 float16

## 索引列

`counts_index.parquet`（175 行）：

| 列 | 说明 |
|---|---|
| `segment_idx` | 0–174 |
| `source` | `macaque_nlb` / `human_intracortical` |
| `species` | 学名 |
| `modality` | 均为 `extracellular_spike` |
| `group_key` | 记录 session 标识 |
| `file` | 上游文件名 |
| `bin_start` | 该片段在 `counts.npy` 中的起始行 |
| `bin_end` | 结束行（不含） |
| `n_bins_20ms` | 该片段总 bin 数 |
| `n_units` | 有效单元数（其余列为补零） |
| `region_raw` | 上游记录区代码 |
| `region` | 统一后的区名 |
| `region_kind` | 均为 `anatomy` |
| `shared_node` | `M1` / `PFC` / `S1` |
| `mapping_status` | `exact` / `coarse` / `compound` / `task_code_reinterpreted` |
| `region_note` | 该标签的来源说明 |
| `domain_l1` | 6 类功能域之一 |

`index_timeseries.parquet`（273,918 行）在以上基础上加 `window_idx`、
`win_in_segment`、`t0_bin20`、`n_bins_20ms`、`explained_var`。

两个索引的 `segment_idx` 一致。

## 区域

| 区域 | bin 数 | 片段 | 来源 |
|---|---:|---:|---|
| `M1` | 5,098,949 | 170 | 猕猴 `churchland` + 人类全部 |
| `PFC` | 265,932 | 3 | 猕猴 `dmfc-rsg` |
| `S1` | 114,598 | 2 | 猕猴 `area2-bump` |

两个物种都覆盖的只有 `M1`。

`churchland` 的 `M1_PMd` 跨初级运动皮层与背侧前运动皮层，上游数据无法在单元级
拆分，故整段记为 `M1` 并标 `mapping_status = compound`。

`area2-bump` 的标签依据 dandi:000127 的
`general/extracellular_ephys/electrodes:location`，值为 `S1 Area 2`。

## 加载

```python
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/primate-spike-20ms"

C = np.load(hf_hub_download(REPO, "counts.npy", repo_type="dataset"),
            mmap_mode="r")                       # (5483479, 192) uint8
ci = pd.read_parquet(hf_hub_download(REPO, "counts_index.parquet",
                                     repo_type="dataset"))

seg = ci[ci.segment_idx == 0].iloc[0]
x = C[seg.bin_start:seg.bin_end, :seg.n_units]   # (n_bins, n_units)

# 同一片段在窗口版中的行号
idx = pd.read_parquet(hf_hub_download(REPO, "index_timeseries.parquet",
                                      repo_type="dataset"))
w = idx[idx.segment_idx == 0]
```

## ACF 不是指数，没有单一时间常数

在完整片段（20 s–67 min，最长 12 万 bin）上算 `R(τ) = tr C(τ)/tr C(0)`，
逐片段做单指数拟合：

| 量 | 中位数 | p10–p90 |
|---|---:|---|
| `R(1)`，即 20 ms | 0.359 | −0.017–0.950 |
| `R(2)`，即 40 ms | **0.412** | 0.260–0.930 |
| 单指数拟合的 T | 1356 ms | 181–8883 |
| 幂律指数 β | 0.371 | 0.207–0.595 |

两个要点：

1. **`R(2) > R(1)`**，曲线在第一个 bin 处凹陷再回升，不是单调下降。
   单指数拟合对非单调曲线没有意义。
2. **同一条曲线换拟合区间，T 的中位极差是 2.6 倍，p90 是 13.9 倍。**

凹陷的来源是 `tr C(τ)` 把两类东西加在一起：

| 成分 | `R(1)` 中位数 | 含义 |
|---|---:|---|
| 对角项 `Σ_i⟨x_i x_i(τ)⟩` | 0.144 | 单单元自相关，含不应期 |
| **非对角项 `Σ_{i≠j}⟨x_i x_j(τ)⟩`** | **0.706** | 群体耦合 |

去掉对角项后，曲线单调的比例从 38.3% 升到 41.7%，仍不足一半。

幂律指数 β = 0.371 即 `R(τ) ∝ τ^(−0.37)`，**没有特征时间尺度**。
任何「τ」都是从这条幂律上摘下来的一个点，数值取决于在哪个区间读。

## 可跨片段比较的量

每个片段有自己的 PCA 基，片段之间没有对应关系。因此逐成分统计
（第 k 个成分的 ACF、谱、能量）只能在同一片段内比较。

跨片段有意义的量必须对正交混合不变：

| 量 | 构造 |
|---|---|
| 成分谱 | `C = xᵀx / T` 的特征值 |
| 有效维度 | 谱的参与率 |
| **trace-ACF** | `tr C(τ) / tr C(0)` |
| **cross-ACF** | `Σ_{i≠j}⟨x_i x_j(τ)⟩ / Σ_{i≠j}⟨x_i x_j(0)⟩` |

`tr(Qᵀ C(τ) Q) = tr C(τ)`，所以前三个与基的选取无关。
cross-ACF 更进一步，去掉了单单元的不应期。

**不要用逐成分 ACF 的中位数。** top-1 特征值占 30.9%（猕猴）、6.0%（人类），
逐成分中位数会取到噪声或单成分。实测在 72 个猕猴片段上，这个做法给出
τ = 6.9 ms，而正交不变量给出 271.8 ms，差 39 倍。
检验代码在 `Microbiosis/hetero-moe` 的 `research/brain_domains/lib/tau.py`。

## 时间常数不可逐窗辨识

窗长 800 ms（40 bin），而整段拟合的名义 T 是 1356 ms，所以每窗只装 0.6 个。
逐窗估计得到的值中位数 3.52 ms，**小于一个 bin（4 ms）**。

可辨识下限：至少 5–10 个候选时间尺度。要做到这一点必须用整段
（`counts.npy`），而不是 800 ms 窗（`timeseries.npy`）。

## 已核查的性质

| 检查 | 结果 |
|---|---|
| τ 随分析窗口长度的收敛 | 1000 bin(20 s) 后比值 0.948–1.000，稳定 |
| τ 的分裂信度（前半 vs 后半） | Pearson r = 0.824，Spearman ρ = 0.863，n = 175 |
| 幅度校正后的物种差异 | 14.6× → 4.8×；72.3% 的表面差异来自信噪比 |
| 幂律指数 β | 猕猴 0.338，人类 0.434，p = 0.0016 |

## 限制

- 人类仅来自 2 名被试（FALCON H1、H2），癫痫监测植入，不是健康人群代表样本
- 猕猴只有三个记录区，`PFC` 实际只覆盖 `DMFC` 一块
- `M1_PMd` 无法拆分为 M1 与 PMd
- 每个片段的 PCA 基不同，不构成共同坐标系
- 任务不同：猕猴为中心外伸手与计时，人类为抓取与手写
- 逐片段 z-score 假设一个片段是一个平稳态；伸手任务有基线/准备/运动/奖励多个阶段
- 人类片段中位 2 min，猕猴中位 16 min，可估计的时间尺度上限不同
- 计数已二值化到 uint8，但全库最大值为 20，无截断

## 引用

各来源的原始文献见 [LICENSE.md](LICENSE.md)。
