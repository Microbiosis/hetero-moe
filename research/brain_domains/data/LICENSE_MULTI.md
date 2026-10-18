# 许可证与署名

本数据集的 5 个来源全部为 **CC BY 4.0**，无混入其他条款的数据。

## 逐来源

| source | 窗口数 | 许可证 | 原始来源 | 核实方式 |
|---|---:|---|---|---|
| `mouse_steinmetz` | 314,332 | CC BY 4.0 | [dandi:000070](https://dandiarchive.org/dandiset/000070) | DANDI API，2026-10-18 |
| `macaque_nlb` | 187,067 | CC BY 4.0 | [dandi:000127](https://dandiarchive.org/dandiset/000127) + [000130](https://dandiarchive.org/dandiset/000130) + [000070](https://dandiarchive.org/dandiset/000070) | 同上 |
| `human_intracortical` | 86,851 | CC BY 4.0 | [dandi:000950](https://dandiarchive.org/dandiset/000950)（FALCON H2）+ [000954](https://dandiarchive.org/dandiset/000954)（FALCON H1） | 同上 |
| `human_eeg` | 32,000 | CC BY 4.0 | [OSF 3jk45](https://osf.io/3jk45/) | OSF 项目页 |
| `flywire` | 139,255 | CC BY 4.0 | [FlyWire](https://flywire.ai/) | 上游说明 |

### 核实记录

2026-10-18 通过 DANDI API 查询 `versions/draft` 的 `license` 字段：

```
dandi:000070 → ['spdx:CC-BY-4.0']  Neural population dynamics during reaching
dandi:000127 → ['spdx:CC-BY-4.0']  Area2_Bump: macaque somatosensory area 2 spiking activity
dandi:000130 → ['spdx:CC-BY-4.0']  DMFC_RSG: macaque dorsomedial frontal cortex spiking activity
dandi:000950 → ['spdx:CC-BY-4.0']  FALCON Benchmark H2: Human Handwriting iBCI
dandi:000954 → ['spdx:CC-BY-4.0']  FALCON Benchmark H1: Human 7DoF Reach and Grasp
```

## 各来源的改动

| source | 上游格式 | 本数据集的改动 |
|---|---|---|
| `mouse_steinmetz` | 50 ms 分箱的 spike 计数 | 50→100 ms 求和；逐 session z-score + PCA 到 64 维；切 8-bin 零重叠窗 |
| `macaque_nlb` | 20 ms 分箱的 spike 计数 | 20→100 ms 求和；逐片段 z-score + PCA；切窗 |
| `human_intracortical` | 20 ms 分箱的 spike 计数 | 同上 |
| `human_eeg` | 250 Hz 电位，已白化 | 4→100 ms 均值；63 通道零填充至 64；切窗 |
| `flywire` | 连接组邻接 | 图信号传播得节点嵌入（64 维），无时间维 |

**窗口内的活动经过 PCA 与时间聚合，不能反推回原始电压或放电序列。**

## 本数据集的聚合

拼接与索引构建不改变各来源数值，只做两件事：

1. 4 个 `(n, 64, 8)` 数组纵向拼接为 `(620250, 64, 8)`
2. 生成统一的分层索引表

## 使用要求

按 CC BY 4.0：

- 保留署名（引用所使用来源的原始文献）
- 注明修改（投影到 64 维并切窗，而非原始信号）
- 标明许可证
- 不附加额外限制

## 引用

```bibtex
@article{steinmetz2019distributed,
  title={Distributed coding of choice, action and engagement across the mouse brain},
  author={Steinmetz, Nicholas A and Zatka-Haas, Peter and Carandini, Matteo and Harris, Kenneth D},
  journal={Nature}, volume={576}, pages={266--273}, year={2019}
}

@inproceedings{pei2021neural,
  title={Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity},
  author={Pei, Felix and Ye, Joel and others},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2021}
}

@article{gifford2022large,
  title={A large and rich EEG dataset for modeling human visual object recognition},
  author={Gifford, Alessandro T and Dwivedi, Kartik and Roig, Gemma and Cichy, Radoslaw M},
  journal={NeuroImage}, volume={264}, pages={119754}, year={2022}
}

@article{dorkenwald2023flywire,
  title={Neuronal wiring diagram of an adult brain},
  author={Dorkenwald, Sven and others},
  journal={Nature}, year={2023}
}
```

`human_intracortical` 的原始文献见 FALCON Benchmark 论文（dandi:000950 / 000954 页面）。
