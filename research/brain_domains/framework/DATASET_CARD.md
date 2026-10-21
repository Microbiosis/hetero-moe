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

14 个成员的域标注神经数据：小鼠与猕猴的胞外放电、人类头皮 EEG、果蝇连接组。
全部 CC-BY-4.0。表格是 parquet，张量是 npz。

| | |
|---|---|
| 文件 | 538 |
| 体积 | 30.13 GiB |
| 版本 | `v1.0` |

```bash
hf download illusion-brain/brain-domains --repo-type dataset --revision v1.0
```

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
framework/  索引、来源、L2 不变量、L3 检索目录、方法说明
steinmetz/  小鼠尖峰（39 session）
primate/    猕猴片段
omnibrain/  多视图（spike / eeg / graph）
omnibrain-multi/  跨成员对齐的窗口张量
omnibrain4d/      EEG 四维张量
counts-v4/  ragged 计数
flywire/    果蝇连接组
hierarchy/  结构层级
```

`framework/framework.json` 是总索引：34 张表、各自的行数与用途、以及该先读哪个文件。
口径与结果见 `framework/docs/METHODS.md`，检索用法见 `framework/docs/RETRIEVAL.md`。

## 限制

1. 神经元数按 session 变化（326–1,240），跨 session 建模需对齐策略。
2. Steinmetz 为 50 ms 分箱，不适合毫秒级动态；FlyWire 无时间维度。
3. 域在 session 间分布不均（每 session 2–9 个域），跨 session 学习需先取交集。
4. 人类颅内（FALCON）没有电极表：NWB 里只有 20 ms 分箱的 `binned_spikes`，
   所谓“单元”是矩阵列号，且无解剖定位（区间标签记作 `unknown`）。
5. 猕猴“单元”可能是电极级群体信号（上游为 96 电极 × 2 阵列）。
6. FlyWire 的区标签与连接组特征同源，用连接组预测脑区接近同义反复；
   `node_features` 已分 safe / leaky 两组，结论须用 safe 组加随机标签对照。
7. PR（participation ratio）不能跨成员比较：单元轴不共享、窗长与预处置不同。

## 许可与署名

本数据集 CC-BY-4.0。使用时请同时署名下列上游。

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
| UBERON | 区身份映射 | CC-BY-3.0 | github.com/obophenotype/uberon |

逐条版本固定（版本号 / commit / 内容哈希）见 `framework/sources.parquet`。

## 引用

本数据集：

```bibtex
@misc{illusionbrain2026braindomains,
  title        = {Brain Domains: a domain-annotated neural dataset},
  author       = {{illusion-brain}},
  year         = {2026},
  howpublished = {Hugging Face Datasets},
  url          = {https://huggingface.co/datasets/illusion-brain/brain-domains},
  note         = {CC-BY-4.0; 14 members from 4 upstream datasets}
}
```

同时须引用上游：7 篇论文 + 3 个本体/图集，见上方「许可与署名」表。
