---
license: cc-by-4.0
tags:
  - eeg
  - neuroscience
  - things-eeg
  - brain-response
  - time-series
  - tensor
size_categories:
  - 100K<n<1M
---

# THINGS-EEG 4D — 按原始四轴组织的派生数据集

把 [THINGS-EEG](https://doi.org/10.1016/j.neuroimage.2022.119754) 按它原本的
四个轴重新组织：`image × repetition × channel × time`，并派生频带功率。

**这是一份派生数据集，不是原始发布。** 上游链条与修改项见下。

---

## 一、许可证与派生链条

### 链条

```
① OSF 3jk45（原始发布）
     Gifford, Dwivedi, Roig, Cichy (2022)
     License: CC-BY 4.0  ← https://osf.io/3jk45/
        │
        │  修改：EEG 由 float32 压缩为 float16（有损），其余不变
        ↓
② HF 镜像 Haitao999/things-eeg
     （该仓库未标注 license 字段）
        │
        │  修改：解包 PyTorch .pt → 按被试重组为 .npz
        │        张量轴由 upstream 布局重排为 (image, repetition, channel, time)
        │        新增派生量：5 个频带的功率
        ↓
③ 本数据集
     License: CC-BY 4.0
```

### CC-BY 4.0 要求的四项，逐项对应

| 要求 | 本数据集的落实 |
|---|---|
| **署名** | 见下方「引文」与「上游」两节 |
| **标明修改** | 见上链条第 ②③ 步，以及下方「具体修改项」|
| **保留许可证声明** | 本文件顶部 `license: cc-by-4.0`；全文见 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **不增加额外限制** | 本数据集未附加任何使用限制 |

### 具体修改项（相对于 ② 镜像）

| 修改 | 说明 | 是否有损 |
|---|---|---|
| 容器格式 | `.pt`（PyTorch pickle+zip）→ `.npz`（NumPy）| 无损 |
| 轴顺序 | 重排为 `(image, repetition, channel, time)` | 无损 |
| 组织方式 | 按上游 split 的原始文件拆成逐被试 | 无损 |
| **数值精度** | float16 → float16（保持）| 无损（精度丢失发生在 ①→②）|
| **新增 band** | 5 频带功率（Hann 窗 + rFFT）| **新增派生量** |

**关于「数值是否改变」的准确表述**：

- 相对于 ② 镜像：**数值未变**，只改容器与轴顺序
- 相对于 ① 原始：**数值已变**（①→② 做了 float32→float16 有损压缩）
- `band` 是全新派生量，① 与 ② 都没有

### 免责

原始数据集与本派生数据集**均不提供任何担保**（CC-BY 4.0 Section 5）。

---

## 二、上游

| 项 | 值 |
|---|---|
| 原始发布 | [OSF 3jk45](https://osf.io/3jk45/) |
| 论文 | Gifford et al. (2022), *NeuroImage* 264:119754 |
| 本数据集的上游 | [Haitao999/things-eeg](https://huggingface.co/datasets/Haitao999/things-eeg) |
| 上游子集 | `Preprocessed_data_250Hz_whiten`（250 Hz 采样，已白化）|

---

## 三、内容

### 两个 split 的轴尺寸不同

| | `test/` | `train/` |
|---|---:|---:|
| `eeg` 形状 | (200, **80**, 63, 250) | (**16,540**, **4**, 63, 250) |
| 图像数 | 200 | 16,540 |
| 重复数 | 80 | 4 |
| 每被试试次 | 16,000 | 66,160 |
| 10 被试合计 | 160,000 | 661,600 |
| 试次间信度（奇偶 split-half）| +0.072 | +0.005 |
| 体积 | 4.5 GB | 20.2 GB |

尺寸不同来自上游设计：

- 训练集：图像多、重复少
- 测试集：图像少、重复多

### 文件

```
test/
├── sub-01.npz … sub-10.npz          eeg  float16
├── sub-01_band.npz … sub-10_band.npz band float32
├── meta.parquet                      160,000 行
└── axis_info.json

train/
├── sub-01.npz … sub-10.npz          eeg  float16
├── sub-01_band.npz … sub-10_band.npz band float32
├── meta.parquet                      661,600 行
└── axis_info.json
```

### 四个原始轴

| 轴 | 含义 | 单位 |
|---|---|---|
| `image` | THINGS 图像条件 | 索引 |
| `repetition` | 同一图像的重复呈现 | 试次序号 |
| `channel` | EEG 电极（10-20 系统）| 电极名见 `ch_names` |
| `time` | 刺激后时间窗 | 250 Hz × 4 ms = 0–1000 ms |

### 派生轴：frequency

| 索引 | 名称 | 范围 | 计算 |
|---:|---|---|---|
| 0 | delta | 1–4 Hz | Hann 窗 + rFFT，频带内功率均值 |
| 1 | theta | 4–8 Hz | 同上 |
| 2 | alpha | 8–13 Hz | 同上 |
| 3 | beta | 13–30 Hz | 同上 |
| 4 | gamma | 30–45 Hz | 同上 |

**`band` 是绝对功率，未做 1/f 校正。**
跨频带比较前应先转相对功率（各频带除以总功率）。

---

## 四、meta.parquet

| 列 | 类型 | 说明 |
|---|---|---|
| `subject` | str | `sub-01` … `sub-10` |
| `image_idx` | int | 对应 `eeg[image_idx]` |
| `rep` | int | 对应 `eeg[image_idx, rep]` |
| `label` | int | 整数类别标签 |
| `img` | str | 图像在 THINGS 中的相对路径 |
| `text` | str | 概念名（英文字符串，如 `aardvark`）|

`text` 是**类别标签的字符串形式**，不是自然语言语料。
本数据集因此不标注 `language` 字段。

---

## 五、加载

```python
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/things-eeg-4d"

eeg = np.load(hf_hub_download(REPO, "test/sub-01.npz", repo_type="dataset"))["eeg"]
meta = pd.read_parquet(hf_hub_download(REPO, "test/meta.parquet", repo_type="dataset"))
```

### 基线校正

`eeg` 未做基线校正。常用做法是减去 0–48 ms（前 12 个采样点）的均值：

```python
x = eeg.astype(np.float32)
base = x[:, :, :, :12].mean(axis=3, keepdims=True)
x = x - base
```

### 按 repetition 切分（test）

```python
n_tr = 40
Xtr = eeg[:, :n_tr].mean(axis=1)   # (200, 63, 250)
Xte = eeg[:, n_tr:].mean(axis=1)
```

### 按图像切分（train，跨图像泛化）

同概念 10 张图中取 8 张训练、2 张测试，用 `meta.text` 作概念键。

---

## 六、在这份数据上测过的结果

以下数字来自配套仓库 `Microbiosis/hetero-moe`，可复现。

| 任务 | 配置 | 结果 |
|---|---|---|
| 100 类解码（test）| 40 重复训练 / 40 测试 | 15–50× 随机（200 类时）|
| 100 类解码（train，同图像）| 8 图 × rep0 → rep1 | 4.1× 随机 |
| 100 类解码（train，新图像）| 8 图 × rep0 → 图 9,10 | 2.4× 随机 |
| 打乱标签基线 | | 0.8× |
| 时间常数 τ（test）| ACF 全曲线指数拟合 | 147–262 ms |

**注意**：15–50× 与 4.1× 的差距主要来自**训练样本数**（40/类 vs 8/类），
不全是「同图像 vs 新图像」的差别。新图像条件下为 2.4×。

---

## 七、已核查的性质

| 性质 | 方法 | test | train |
|---|---|---|---|
| 全部元素 `isfinite` | 逐文件检查 | 通过 | 通过 |
| 零值占比 | 全元素统计 | 0.00% | 0.00% |
| `label` 沿 repetition 恒定 | 逐被试比对 | 10/10 | 10/10 |
| 同图像不同 rep 是否不同 | 抽查 | 不同 | 不同 |
| 不同图像是否不同 | 抽查 | 不同 | 不同 |
| 奇偶试次 split-half 相关 | 50 图抽样 | +0.072 | +0.005 |

`train` 信度接近 0 是 4 次重复的直接后果。

---

## 八、已知限制

1. **已白化** —— 上游 `whiten` 版本，不是原始电压值。
2. **float16 有损** —— 精度损失发生在上游 ①→②，本数据集未再损失。
3. **无行为数据** —— THINGS-EEG 是 passive viewing 范式。
4. **train 重复仅 4 次** —— 无法支撑可靠的单试次估计。
5. **被试 10 人** —— 公开样本，非特定人群。
6. **`band` 为绝对功率** —— 跨频带比较须先归一化。
7. **仅 EEG** —— 不含其他模态或其他物种。

---

## 九、构建脚本

配套仓库 `Microbiosis/hetero-moe`：

- `research/brain_domains/data/build_4d_eeg.py`（test）
- `research/brain_domains/data/build_4d_eeg_train.py`（train）

---

## 十、引文

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
