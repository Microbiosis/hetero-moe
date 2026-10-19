---
license: cc-by-4.0
tags:
  - electrophysiology
  - neuroscience
  - spike-counts
  - multi-species
  - time-series
size_categories:
  - 100K<n<1M
---

# Primate Spike 20 ms

273,918 个神经活动时间窗，来自猕猴与人类皮层内记录。两个来源的上游 bin 宽
都是 20 ms，未做时间重采样。每个窗为 `(64, 40)` — 64 个主成分 × 40 个
20 ms bin，即 800 ms。

## 来源

| source | 物种 | 片段数 | 单元数 | 窗口数 | 原始数据 |
|---|---|---:|---:|---:|---|
| `macaque_nlb` | *Macaca mulatta* | 72 | 40–192 | 187,067 | dandi:000070 / 000127 / 000130 |
| `human_intracortical` | *Homo sapiens* | 103 | 176–192 | 86,851 | dandi:000950 / 000954 |

猕猴的 72 段来自三个记录集：`churchland`（67 段，M1 与 PMd）、`dmfc-rsg`（3 段）、
`area2-bump`（2 段）。人类的 103 段来自 FALCON H1（40 段，七自由度抓取）与
H2（63 段，手写），两套 Utah array 均植入手部运动皮层。

各来源均为 CC BY 4.0，逐项见 [LICENSE.md](LICENSE.md)。

## 处理

两个来源完全相同，没有任何一步只在其中一个来源上执行。

1. **补齐**：每个片段的单元为不等长序列，补零成 `(n_units, max_len)`。
2. **z-score**：逐单元沿整个片段的时间轴，减去均值除以标准差。
3. **投影到 64 维**：对 `(n_units, n_bins)` 的 z-score 矩阵求单元间协方差
   `C = zc zcᵀ`，`eigh` 分解后取前 `k = min(64, n_units−1, n_bins−1)` 个特征向量，
   `proj = zcᵀ U`。单元数不足 64 时零填充到 64。投影在片段内完成，
   片段之间的 64 维基互不相同。
4. **切窗**：40 bin（800 ms），步长 20 bin（50% 重叠），末尾不足一窗丢弃。
5. **存储**：float16。

片段的平均解释方差：猕猴 0.655，人类 0.413。

窗口内的活动经过 PCA 与 z-score，**无法反推回原始放电序列**。

## 文件

| 文件 | 形状 | 内容 |
|---|---|---|
| `timeseries.npy` | (273918, 64, 40) float16 | 时间窗，1.40 GB |
| `index_timeseries.parquet` | 273,918 行 | 逐窗标签 |

## 索引列

| 列 | 说明 |
|---|---|
| `source` | `macaque_nlb` / `human_intracortical` |
| `species` | 学名 |
| `modality` | 均为 `extracellular_spike` |
| `group_key` | 记录 session 标识 |
| `segment_idx` | 0–174，片段全局编号 |
| `file` | 上游文件名 |
| `window_idx` | 来源内序号，从 0 连续到该来源行数减一 |
| `win_in_segment` | 片段内窗序号 |
| `t0_bin20` | 窗起点，单位为 20 ms bin |
| `n_bins_20ms` | 该片段总 bin 数 |
| `n_units` | 该片段单元数 |
| `explained_var` | 该片段前 64 个成分解释的方差占比 |
| `region_raw` | 上游记录区代码 |
| `region` | 统一后的区名：`M1` / `PFC` / `S1` |
| `region_kind` | 均为 `anatomy` |
| `shared_node` | `M1` / `PFC` / `S1` |
| `mapping_status` | `exact` / `coarse` / `compound` / `task_code_reinterpreted` |
| `region_note` | 该标签的来源说明 |
| `domain_l1` | 6 类功能域之一 |

索引行与 `timeseries.npy` 第 0 维一一对应。

## 区域

| 区域 | 窗口数 | 来源 |
|---|---:|---|
| `M1` | 254,899 | 猕猴 `churchland` 168,048 + 人类 86,851 |
| `PFC` | 13,292 | 猕猴 `dmfc-rsg` |
| `S1` | 5,727 | 猕猴 `area2-bump` |

两个物种都覆盖的只有 `M1`。`PFC` 与 `S1` 仅来自猕猴。

`churchland` 的 `M1_PMd` 跨初级运动皮层与背侧前运动皮层，上游数据无法在单元级
拆分，故整段记为 `M1` 并标 `mapping_status = compound`。

## 加载

```python
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/primate-spike-20ms"

W = np.load(hf_hub_download(REPO, "timeseries.npy", repo_type="dataset"),
            mmap_mode="r")                       # (273918, 64, 40) float16
idx = pd.read_parquet(hf_hub_download(REPO, "index_timeseries.parquet",
                                      repo_type="dataset"))

# 两个物种在同一区域上的窗口
mac = idx.index[(idx.shared_node == "M1") & (idx.source == "macaque_nlb")].to_numpy()
hum = idx.index[(idx.shared_node == "M1") & (idx.source == "human_intracortical")].to_numpy()
x_mac = W[mac]          # (168048, 64, 40)
x_hum = W[hum]          # ( 86851, 64, 40)
```

## 可跨窗比较的量

每个片段有自己的 64 维 PCA 基，片段之间没有对应关系。因此逐成分统计
（第 k 个成分的 ACF、谱、能量）只能在同一片段内比较。

跨片段有意义的量必须对正交混合不变：

| 量 | 构造 |
|---|---|
| 成分谱 | `C = xᵀx / T` 的特征值 |
| 有效维度 | 谱的参与率 |
| **trace-ACF** | `R(τ) = tr C(τ) / tr C(0)` |
| 窗矩阵奇异值 | 同上 |

`tr C(τ)/tr C(0)` 是按方差加权的平均自相关：对任意正交 `Q`，
`tr(Qᵀ C(τ) Q) = tr C(τ)`，所以它与基的选取无关。

**不要用逐成分 ACF 的中位数。** 本数据的 top-1 特征值占 43.8%，
其余成分接近独立噪声，逐成分中位数取到的是噪声底：在 72 个片段上，
其中位数给出 τ = 6.9 ms，而不变量给出 271.8 ms，差 39 倍。
检验方法见 `Microbiosis/hetero-moe` 的 `research/brain_domains/lib/tau.py`。

## 时间常数不可逐窗辨识

窗长 800 ms（40 bin），而片段级不变量 τ 中位数 271.8 ms，
所以每窗只装 **2.9 个 τ**。估 τ 一般需要 5–10 个。

20 ms 栅格上多数单元的放电计数接近独立（ACF(1) 的逐单元中位数 0.049，
Fano factor 中位 1.04–1.37）。因此：

- 片段级（数千 bin）可以估出不变量 τ
- 逐窗（40 bin）不能；`runs/tau-fixed` 的逐试次结果中位 3.52 ms，小于一个 bin

## 64 维不是共同坐标系

每个片段各自做 PCA。猕猴第 3 个成分与人类第 3 个成分不对应同一个生物实体。

因此可以把全部窗送进同一模型（形状一致），也可以用 `source` 做分组统计，
但不能声称这是同一空间中的向量，也不能直接比较不同来源的第 k 个成分。

跨源可比较的是无量纲量：自相关衰减、谱斜率、有效维度。

## 未发布的 100 ms 版本

同一批 175 个片段也曾构建为 100 ms 版本（每窗 8 bin），与小鼠 50 ms、头皮 EEG 4 ms
一起拼成过四源数据集。那份数据已下线，因为把 20 ms 与 4 ms 的数据压到 100 ms
是对细粒度来源的不可逆降采样。

| | 100 ms 版本（已下线） | 本数据集 |
|---|---|---|
| bin 宽 | 100 ms | 20 ms |
| 每窗 bin 数 | 8 | 40 |
| PCA 输入 | 5 bin 求和后的 100 ms 序列 | 原生 20 ms 序列 |
| 每窗时长 | 800 ms | 800 ms |

时间跨度相同，PCA 输入不同，因此两者的 64 维基不同，不能逐元素对照。
本数据集保留了 5 倍的时序细节。重建脚本在 `Microbiosis/hetero-moe` 的
`research/brain_domains/data/` 下。

## 限制

- 只有两个物种，且人类仅来自 2 名被试（FALCON H1、H2），不是健康人群代表样本
- 猕猴只有三个记录区，`PFC` 实际只覆盖 `DMFC` 一块
- `M1_PMd` 无法拆分为 M1 与 PMd
- 每个片段的 64 维基不同，不构成共同坐标系
- 采样率虽统一为 20 ms，但任务不同：猕猴为中心外伸手与计时，人类为抓取与手写
- 所有窗均经过 z-score，片段间的绝对放电率差异已被移除

## 引用

各来源的原始文献见 [LICENSE.md](LICENSE.md)。
