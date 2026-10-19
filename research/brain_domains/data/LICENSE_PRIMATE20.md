# 许可证与署名

本数据集的两个来源均为 CC BY 4.0。

| source | 窗口数 | 许可证 | 原始数据 |
|---|---:|---|---|
| `macaque_nlb` | 187,067 | CC BY 4.0 | [dandi:000070](https://dandiarchive.org/dandiset/000070)、[000127](https://dandiarchive.org/dandiset/000127)、[000130](https://dandiarchive.org/dandiset/000130) |
| `human_intracortical` | 86,851 | CC BY 4.0 | [dandi:000950](https://dandiarchive.org/dandiset/000950)、[000954](https://dandiarchive.org/dandiset/000954) |

许可证字段于 2026-10-19 通过 `api.dandiarchive.org/api/dandisets/<id>/versions/draft/` 查询：

```
dandi:000070 → ['spdx:CC-BY-4.0']  Neural population dynamics during reaching            (Rhesus monkey)
dandi:000127 → ['spdx:CC-BY-4.0']  Area2_Bump: macaque somatosensory area 2              (Rhesus monkey)
dandi:000130 → ['spdx:CC-BY-4.0']  DMFC_RSG: macaque dorsomedial frontal cortex          (Rhesus monkey)
dandi:000950 → ['spdx:CC-BY-4.0']  FALCON Benchmark H2: Human Handwriting iBCI            (Homo sapiens)
dandi:000954 → ['spdx:CC-BY-4.0']  FALCON Benchmark H1: Human 7DoF Reach and Grasp BCI   (Homo sapiens)
```

## 上游数据的改动

| source | 上游形式 | 本数据集所做处理 |
|---|---|---|
| `macaque_nlb` | 20 ms 分箱的放电计数，每片段一个 `(n_units, n_bins)` 数组 | 逐单元 z-score；逐片段 PCA 至 64 维；切 40-bin 窗，步长 20 bin |
| `human_intracortical` | 20 ms 分箱的放电计数，每片段每单元一个变长数组 | 补零成矩阵；逐单元 z-score；逐片段 PCA 至 64 维；切 40-bin 窗，步长 20 bin |

未做时间重采样：两个来源的上游 bin 宽都是 20 ms。

窗口内的活动经 z-score 与 PCA 后，无法反推回原始放电序列。

## 区域标签的依据

`region` 不是上游字段，由各数据集对应的 NWB 文件读出：

```
dandi:000127  general/extracellular_ephys/electrodes:location → b"S1 Area 2"
dandi:000130  general/extracellular_ephys/electrodes:location → b"DMFC"
dandi:000070  源论文：M1 与 PMd 同时记录
dandi:000950  FALCON H2，手部运动皮层 Utah array
dandi:000954  FALCON H1，手部运动皮层 Utah array
```

`Area2` 归入 `UBERON:0008933`（初级体感皮层），`DMFC` 归入 `UBERON:0000451`
（前额叶），`M1`/`M1_PMd` 归入 `UBERON:0001384`（初级运动皮层）。

## 引用

```bibtex
@article{pei2021neural,
  title={Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity},
  author={Pei, Felix and Ye, Joel and others},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2021}
}

@article{karpowicz2024fewshot,
  title={Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark},
  author={Karpowicz, Brianna M and Ye, Joel and others},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2024}
}

@article{churchland2012neural,
  title={Neural population dynamics during reaching},
  author={Churchland, Mark M and Cunningham, John P and Kaufman, Matthew T and others},
  journal={Nature}, volume={487}, pages={51--56}, year={2012}
}
```
