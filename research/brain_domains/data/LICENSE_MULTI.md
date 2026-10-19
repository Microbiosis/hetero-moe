# 许可证与署名

本数据集及其全部 5 个来源均为 CC BY 4.0。

## 各来源

| source | 窗口数 | 许可证 | 原始数据 |
|---|---:|---|---|
| `mouse_steinmetz` | 314,332 | CC BY 4.0 | [dandi:000017](https://dandiarchive.org/dandiset/000017) |
| `macaque_nlb` | 187,067 | CC BY 4.0 | [dandi:000127](https://dandiarchive.org/dandiset/000127)、[000130](https://dandiarchive.org/dandiset/000130)、[000070](https://dandiarchive.org/dandiset/000070) |
| `human_intracortical` | 86,851 | CC BY 4.0 | [dandi:000950](https://dandiarchive.org/dandiset/000950)、[000954](https://dandiarchive.org/dandiset/000954) |
| `human_eeg` | 821,600 | CC BY 4.0 | [OSF 3jk45](https://osf.io/3jk45/) |
| `flywire` | 139,255 节点 | CC BY 4.0 | [FlyWire](https://flywire.ai/) |

DANDI 的许可证字段于 2026-10-19 通过 `api.dandiarchive.org/api/dandisets/<id>/versions/draft/` 查询：

```
dandi:000017 → ['spdx:CC-BY-4.0']  Distributed coding of choice, action and engagement across the mouse brain  (House mouse)
dandi:000070 → ['spdx:CC-BY-4.0']  Neural population dynamics during reaching                                (Rhesus monkey)
dandi:000127 → ['spdx:CC-BY-4.0']  Area2_Bump: macaque somatosensory area 2 spiking activity               (Rhesus monkey)
dandi:000130 → ['spdx:CC-BY-4.0']  DMFC_RSG: macaque dorsomedial frontal cortex spiking activity           (Rhesus monkey)
dandi:000950 → ['spdx:CC-BY-4.0']  FALCON Benchmark H2: Human Handwriting iBCI                              (Homo sapiens)
dandi:000954 → ['spdx:CC-BY-4.0']  FALCON Benchmark H1: Human 7DoF Reach and Grasp Motor BCI               (Homo sapiens)
```

## 上游数据的改动

| source | 上游形式 | 本数据集所做处理 |
|---|---|---|
| `mouse_steinmetz` | 50 ms 分箱的放电计数 | 2 bin 求和至 100 ms；逐 unit z-score；逐 session PCA 至 64 维；切 8-bin 窗，步长 4 bin |
| `macaque_nlb` | 20 ms 分箱的放电计数 | 5 bin 求和至 100 ms；逐 unit z-score；逐记录片段 PCA 至 64 维；切 8-bin 窗，步长 4 bin |
| `human_intracortical` | 20 ms 分箱的放电计数 | 同上 |
| `human_eeg` | 250 Hz 电位 | 25 采样均值至 100 ms；逐电极 z-score；63 通道补零至 64；切 8-bin 窗，步长 4 bin |
| `flywire` | 神经元间连接 | 对称归一化邻接上的图信号传播（α=0.85，8 次迭代，64 个随机初始信号）；无时间维 |

四份时间序列数据转为 float16 后纵向拼接，`timeseries.npy` 第 0 维与
`index_timeseries.parquet` 行序一一对应。图视图单独存为 `graph.npy`（float32）。

`human_eeg` 覆盖 THINGS-EEG2 的全部 821,600 试次（10 被试 × 200 测试图 × 80 重复
+ 10 被试 × 16,540 训练图 × 4 重复），每试次取刺激后 0–800 ms 的 8 个 100 ms bin。

两部分索引表另加了区域映射列（`region_raw`、`region_kind`、`structure_id`、
`uberon_id`、`shared_node`、`mapping_status`）。映射依据为 Allen 小鼠 CCF
结构表（`api.brain-map.org`）与 UBERON 的 MBA、FBbt bridge 文件，
`human_intracortical` 的两个 FALCON 任务编号按植入位置记为 M1。
`region_map.csv` 列出每一行的对应关系。

窗口内的活动经过 PCA 与时间聚合，**无法反推回原始放电序列或电压**。
`human_eeg` 的补零通道可直接去除；其余来源的投影不可逆。

## 引用

```bibtex
@article{steinmetz2019distributed,
  title={Distributed coding of choice, action and engagement across the mouse brain},
  author={Steinmetz, Nicholas A and Zatka-Haas, Peter and Carandini, Matteo and Harris, Kenneth D},
  journal={Nature}, volume={576}, pages={266--273}, year={2019},
  doi={10.1038/s41586-019-1787-x}
}

@inproceedings{pei2021neural,
  title={Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity},
  author={Pei, Felix and Ye, Joel and others},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2021}
}

@article{karpowicz2024fewshot,
  title={Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark},
  author={Karpowicz, Brianna M and Ye, Joel and others},
  journal={NeurIPS Datasets and Benchmarks}, year={2024}
}

@article{gifford2022large,
  title={A large and rich EEG dataset for modeling human visual object recognition},
  author={Gifford, Alessandro T and Dwivedi, Kartik and Roig, Gemma and Cichy, Radoslaw M},
  journal={NeuroImage}, volume={264}, pages={119754}, year={2022},
  doi={10.1016/j.neuroimage.2022.119754}
}

@article{dorkenwald2023flywire,
  title={Neuronal wiring diagram of an adult brain},
  author={Dorkenwald, Sven and others},
  journal={Nature}, volume={634}, pages={124--138}, year={2024},
  doi={10.1038/s41586-024-07558-y}
}
```

`macaque_nlb` 的三份记录见上表所列 DANDI 条目。
