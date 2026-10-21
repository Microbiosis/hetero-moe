---
license: cc-by-4.0
language:
- en
tags:
- neuroscience
- electrophysiology
- eeg
- connectome
- spike-trains
- mouse
- macaque
- human
- brain-regions
pretty_name: Brain Domains — 域标注神经数据集
task_categories:
- other
configs:
- config_name: slice_catalog
  data_files:
  - split: train
    path: framework/slice_catalog.parquet
---

# 脑域数据集

域标注神经数据，供「脑功能域作为架构约束」的研究使用。14 个成员，来自 4 个上游数据集。

**规模**：922 个文件 / 30.14 GiB。数据保留原始形状（不补零、不共享单元轴），跨成员比较只在
L2 不变量层进行。

## 先读哪里

| 想知道 | 打开 |
|---|---|
| 有哪些表、各多少行 | `framework/framework.json`（schema 1.3，35 张表 / 2 份文档） |
| 口径、主要结果、限制 | `framework/docs/METHODS.md` |
| 怎么按区 / 成员检索切片 | `framework/docs/RETRIEVAL.md` + `framework/slice_catalog.parquet` |
| 每个上游固定到什么版本 | `framework/sources.parquet`（含 `pin_kind` / `pin_value` / `pin_status`） |

## 成员

| 成员 | 物种 | 模态 | 规模 |
|---|---|---|---|
| `steinmetz` | *Mus musculus* | 胞外放电 | 39 session / 24,029 单元 / 50 ms |
| `primate_macaque_nlb` | *Macaca mulatta* | 胞外放电 | 72 片段 / 20 ms / PMd·M1 各 96 电极 |
| `primate_dmfc_rsg` | *Macaca mulatta* | 胞外放电 | 2 文件 / 94 单元 |
| `primate_human_eeg` | *Homo sapiens* | 头皮 EEG | 31 文件 / 63 通道 / 4 ms |
| `omnibrain4d_eeg_test` / `_train` | *Homo sapiens* | 头皮 EEG | 200×80×63×250 / 16540×4×63×250 |
| `omnibrain_spike_view` / `_macaque` / `_human` | 三物种 | 胞外放电 | 39 / 72 / 103 文件 |
| `omnibrain_eeg_view` | *Homo sapiens* | 头皮 EEG | 32,000 窗 |
| `omnibrain_graph_view` | *Drosophila* | 连接组 | 139,255 × 64 |
| `flywire` | *Drosophila* | 连接组 | 107,265 × 22 |
| `omnibrain_multi` | 混合 | 混合 | 1,409,850 × 64 × 8 |
| `counts_v4_ragged` | 混合 | 胞外放电 | 175 段 / 998,102,320 元素 |

## 目录

```
framework/      框架层：索引、来源 pin、L2 不变量、L3 检索目录、文档
steinmetz/      小鼠尖峰（39 session）
primate/        猕猴片段
omnibrain/      多视图（spike / eeg / graph）
omnibrain-multi/ 跨成员对齐后的窗口张量
omnibrain4d/    EEG 四维张量
counts-v4/      ragged 计数
flywire/        果蝇连接组
hierarchy/      结构层级
```

## 已知限制

1. 神经元数按 session 变化（326–1,240），跨 session 建模需对齐策略。
2. Steinmetz 为 50 ms 分箱，不适合毫秒级动态；FlyWire 无时间维度。
3. 域在 session 间分布不均（每 session 2–9 个域），LODO 需先取跨 session 交集。
4. 人类颅内（FALCON）NWB **没有电极表也没有 units 表**——`preproc/h2_preproc.py` 全文无
   `add_electrode` / `add_unit`，只写 20 ms 分箱的 `binned_spikes`，那批“单元”实为矩阵列号，
   且无解剖定位（区间标签记作 `unknown`）。
5. 猕猴“单元”可能是电极级群体信号：上游是 96 电极 × 2 阵列，论文写的是 146 / 218 个
   single isolations（single 与 multi-unit 混合）。
6. FlyWire 的区标签与连接组特征同源，用连接组预测脑区接近同义反复；`node_features` 已分
   safe（6 列）与 leaky（12 列）两组，结论必须用 safe 组 + 随机标签对照得出。
7. PR（participation ratio）不能跨成员比较——成员间单元轴不共享、窗长与预处置不同。

## 本数据集的许可

**CC-BY-4.0**。再加工部分（区域归并、切分提取、特征聚合、不变量计算）由 illusion-brain 完成，
沿用上游许可。使用时请同时署名下列上游数据集。

## 上游来源与署名

| 上游 | 用于 | 许可 | 引用 |
|---|---|---|---|
| Steinmetz et al. 2019 | `steinmetz`、`omnibrain_spike_view` | CC-BY-4.0 | *Nature* 576:266–273, [10.1038/s41586-019-1787-x](https://doi.org/10.1038/s41586-019-1787-x) |
| Churchland et al. 2012 | `primate_macaque_nlb` | CC-BY-4.0 | *Nature* 487:51–56, [10.1038/nature11129](https://doi.org/10.1038/nature11129)（DANDI:000070） |
| Area2_Bump | `primate_macaque_nlb` | CC-BY-4.0 | [10.7554/eLife.48198](https://doi.org/10.7554/eLife.48198)（DANDI:000127） |
| DMFC_RSG (Sohn & Jazayeri) | `primate_dmfc_rsg` | CC-BY-4.0 | [10.1016/j.neuron.2019.06.012](https://doi.org/10.1016/j.neuron.2019.06.012)（DANDI:000130） |
| FALCON H1 / H2 | `omnibrain_multi`（人类颅内） | CC-BY-4.0 | DANDI:000950、DANDI:000954 |
| THINGS-EEG2 | `primate_human_eeg`、`omnibrain_eeg_view`、`omnibrain4d` | CC-BY-4.0 | [10.1016/j.neuroimage.2022.119754](https://doi.org/10.1016/j.neuroimage.2022.119754)（OSF 3jk45） |
| FlyWire FAFB v783 | `flywire`、`omnibrain_graph_view` | CC-BY-4.0 | *Nature* 2024, [10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y) |
| Allen Mouse Brain StructureGraph | 区身份映射 | Allen Terms of Use | api.brain-map.org structure_graph_download/1.json |
| MBAO | 区身份映射 | CC-BY-4.0 | github.com/brain-bican/mouse_brain_atlas_ontology |
| UBERON | 区身份映射（全名 fallback） | CC-BY-3.0 | github.com/obophenotype/uberon |

逐条版本固定（版本号 / commit / 内容哈希）见 `framework/sources.parquet`。

## 引用

```bibtex
@dataset{illusion_brain_domains_2026,
  title  = {Brain Domains: a domain-annotated neural dataset with an L0--L4 processing ladder},
  author = {illusion-brain},
  year   = {2026},
  note   = {14 members from 4 upstream datasets; CC-BY-4.0}
}
```
