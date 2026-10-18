---
license: cc-by-4.0
tags:
  - eeg
  - neuroscience
  - things-eeg
  - brain-response
  - time-series
size_categories:
  - 100K<n<1M
---

# THINGS-EEG 4D

[THINGS-EEG](https://doi.org/10.1016/j.neuroimage.2022.119754) 的 EEG 数据，
按 `(image, repetition, channel, time)` 四轴组织，附 5 个频带的功率。

10 名被试，test 与 train 两个 split。

## 来源

|  |  |
|---|---|
| 原始数据 | [OSF 3jk45](https://osf.io/3jk45/)，CC BY 4.0 |
| 直接上游 | [Haitao999/things-eeg](https://huggingface.co/datasets/Haitao999/things-eeg) |
| 上游子集 | `Preprocessed_data_250Hz_whiten` |

相对上游的改动：`.pt` 解包为 `.npz`，按被试重排轴顺序，新增频带功率。
EEG 数值未改（上游已将 float32 压为 float16）。

## 结构

|  | test | train |
|---|---:|---:|
| eeg | (200, 80, 63, 250) | (16540, 4, 63, 250) |
| band | (200, 80, 63, 5) | (16540, 4, 63, 5) |
| 试次 | 160,000 | 661,600 |
| 体积 | 4.5 GB | 20.2 GB |

两个 split 尺寸不同来自 THINGS-EEG 的设计：训练集图像多、重复少，
测试集图像少、重复多。因此 test 适合按重复平均，train 适合跨图像泛化。

### 轴

| 轴 | 含义 |
|---|---|
| image | THINGS 图像条件 |
| repetition | 同一图像的重复呈现 |
| channel | EEG 电极，10-20 系统，63 通道 |
| time | 刺激后 0–1000 ms，250 Hz |

### 频带

delta (1–4)、theta (4–8)、alpha (8–13)、beta (13–30)、gamma (30–45) Hz。
Hann 窗后取 rFFT，频带内取功率均值。绝对功率，未做 1/f 校正。

## meta.parquet

| 列 | 说明 |
|---|---|
| subject | sub-01 … sub-10 |
| image_idx | 对应 `eeg[image_idx]` |
| rep | 对应 `eeg[image_idx, rep]` |
| label | 类别编号 |
| img | 图像在 THINGS 中的相对路径 |
| text | 概念名，如 `aardvark` |

## 加载

```python
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/things-eeg-4d"
eeg = np.load(hf_hub_download(REPO, "test/sub-01.npz", repo_type="dataset"))["eeg"]
```

EEG 未做基线校正。常用做法是减去刺激前 48 ms 的均值：

```python
x = eeg.astype(np.float32)
x -= x[:, :, :, :12].mean(axis=3, keepdims=True)
```

按重复切分（test）：

```python
Xtr = eeg[:, :40].mean(axis=1)
Xte = eeg[:, 40:].mean(axis=1)
```

## 在这份数据上测过的结果

| 任务 | 设置 | 结果 |
|---|---|---|
| 200 类解码（test） | 40 重复训练 / 40 测试 | 15–50× 随机 |
| 100 类解码（train，同图像） | 8 张图 → 新重复 | 4.1× 随机 |
| 100 类解码（train，新图像） | 8 张图 → 另 2 张图 | 2.4× 随机 |
| 打乱标签对照 | | 0.8× |
| 时间常数 τ（test） | ACF 全曲线拟合 | 147–262 ms |

15–50× 与 4.1× 的差别主要来自训练样本数（每类 40 个 vs 8 个），
不全是同图像与新图像的差别。

## 限制

- 上游已白化，不是原始电压
- 无行为数据，passive viewing 范式
- train 每张图只有 4 次重复，单试次估计不可靠
- band 为绝对功率，跨频带比较需先归一化

## 许可证

CC BY 4.0。使用时保留署名、注明修改、标明许可证、不附加额外限制。

## 引用

```bibtex
@article{gifford2022large,
  title={A large and rich EEG dataset for modeling human visual object recognition},
  author={Gifford, Alessandro T and Dwivedi, Kartik and Roig, Gemma and Cichy, Radoslaw M},
  journal={NeuroImage},
  volume={264},
  pages={119754},
  year={2022},
  doi={10.1016/j.neuroimage.2022.119754}
}
```
