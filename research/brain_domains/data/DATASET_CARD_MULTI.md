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
  - 100K<n<1M
---

# OmniBrain-Multi

620,250 个神经活动时间窗，来自 4 个来源、3 个物种、2 种记录方式，
统一为 `(64, 8)` 张量，带五级分层索引。

另有 139,255 个果蝇连接组节点嵌入，单独存放。

## 来源

| source | 物种 | 模态 | 窗口数 | 原始数据 |
|---|---|---:|---:|---|
| `mouse_steinmetz` | *Mus musculus* | 胞外放电 | 314,332 | dandi:000070 |
| `macaque_nlb` | *Macaca mulatta* | 胞外放电 | 187,067 | dandi:000127 / 000130 / 000070 |
| `human_intracortical` | *Homo sapiens* | 皮层内记录 | 86,851 | FALCON H1/H2（dandi:000950 / 000954）|
| `human_eeg` | *Homo sapiens* | 头皮 EEG | 32,000 | THINGS-EEG |
| `flywire` | *Drosophila melanogaster* | 连接组 | 139,255 节点 | FlyWire |

各来源的许可证不同，逐项见 [LICENSE.md](LICENSE.md)。

## 文件

| 文件 | 形状 | 内容 |
|---|---|---|
| `timeseries.npy` | (620250, 64, 8) float16 | 时间窗，0.59 GB |
| `graph.npy` | (139255, 64) float32 | 连接组节点嵌入，34 MB |
| `index_timeseries.parquet` | 620,250 行 | 逐窗分层标签 |
| `index_graph.parquet` | 139,255 行 | 逐节点标签 |
| `hierarchy.json` | — | 层级定义与各组统计 |
| `group_sizes.csv` | — | source × domain 样本数 |

## 分层

```
L0  样本      620,250 个时间窗（每窗 8 个 100ms bin）
L1  来源      4 个 + 1 个图源
L2  物种      Mus musculus / Macaca mulatta / Homo sapiens
              （+ Drosophila 仅图视图）
L3  模态      extracellular_spike / scalp_eeg / connectome
L4  解剖区    34 个（小鼠 314,332 + 猕猴 187,067 + 人类侵入式 86,851）
L5  功能域    sensory / motor / memory / association / subcortical / other
```

### L4 的来源不同

| source | region 来源 | 粒度 |
|---|---|---|
| `mouse_steinmetz` | `steinmetz/units.parquet` 按 session 取主导区 | session 级 |
| `macaque_nlb` | 上游 `area` 列 | 窗级 |
| `human_intracortical` | 上游 `area` 列 | 窗级 |
| `human_eeg` | 无 | — |

`index_timeseries.parquet` 的 `region_level` 列标明是哪种：
`session_majority`（314,332）或 `window_area`（273,918）。

## 索引列

| 列 | 说明 |
|---|---|
| `source` | 4 个来源之一 |
| `species` | 学名 |
| `modality` | 记录方式 |
| `group_key` | 来源内的分组键（session / file / subject）|
| `window_idx` | 组内窗口序号 |
| `domain_l1` | 6 类功能域之一 |
| `region` | 解剖区（EEG 为空）|
| `region_level` | region 的粒度来源 |
| `area` | 上游原始区名（仅猕猴/人类）|
| `n_units` | 记录单元数（仅猕猴/人类）|

索引行与 `timeseries.npy` 第 0 维一一对应。

## 加载

```python
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/omnibrain-multi"

W = np.load(hf_hub_download(REPO, "timeseries.npy", repo_type="dataset"),
            mmap_mode="r")                          # (620250, 64, 8) float16
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
- 可以用 `source` 做分层分析、域适应、多任务学习
- 不能声称这是"同一空间中的向量"
- 不能直接比较不同来源的第 k 个成分

跨源真正可比较的是无量纲统计量（自相关衰减、谱斜率、有效维度），
不是原始通道值。

## 各来源的统计特性

抽样 200 窗的统计（float32 还原后）：

| source | 均值 | 标准差 | 绝对值最大 |
|---|---:|---:|---:|
| `mouse_steinmetz` | +0.084 | 2.071 | 25.0 |
| `macaque_nlb` | −0.001 | 0.895 | 8.2 |
| `human_intracortical` | −0.006 | 1.253 | 8.4 |
| `human_eeg` | +0.004 | 0.911 | 18.7 |

小鼠的方差最大，因为其 PCA 前 64 成分解释方差比例较低（0.24–0.43），
残差更大。

## 在这份数据上做过的分析

| 结论 | 数值 | 状态 |
|---|---|---|
| 功能域标签与活动信号冗余 | 域 vs 随机 −0.04% | 否定 |
| 跨物种无守恒的动力学常数 | 峰度为 PCA 伪影 | 否定 |
| 超额曲线的深度依赖形状跨物种一致 | r = 0.75–0.99（6/8 配置）| 部分成立 |
| 时间常数 τ 可测但不预测行为 | ρ = +0.013, n=502 | 否定 |

## 限制

- 各来源的 64 维成分语义不同，不构成共同坐标系
- 采样率与空间规模差异大（小鼠 205–1240 units，人类 63 电极）
- 小鼠的 region 是 session 级，不是逐窗标注
- `human_eeg` 无解剖区
- 图视图无时间维，单独存为 `graph.npy`
- 数据来自不同任务与实验室，未做行为对齐

## 构建

```bash
python research/brain_domains/data/build_multi.py
```

代码仓库：`Microbiosis/hetero-moe`

## 引用

各来源的原始文献与许可证见 [LICENSE.md](LICENSE.md)。
使用本数据集时，请引用所使用的每个来源的原始工作。
