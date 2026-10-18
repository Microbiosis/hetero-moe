---
license: cc-by-4.0
language:
  - en
tags:
  - eeg
  - neuroscience
  - things-eeg
  - brain
  - vision
  - tensor
size_categories:
  - 100K<n<1M
task_categories:
  - feature-extraction
  - image-classification
pretty_name: OmniBrain 4D (THINGS-EEG)
---

# OmniBrain 4D — THINGS-EEG 四轴完整视图

把 THINGS-EEG 重组成**四轴完整保留**的结构：`image × repetition × channel × time`，
外加派生的**频率轴**。

供需要**试次结构、时间进程、频带分解**的工作直接使用。
不是把 EEG 压平成 `(n_trial, n_channel)` 的二维表。

---

## ⚠️ 先读这一条：train 与 test 的轴结构完全不同

| | `test/` | `train/` |
|---|---:|---:|
| 形状 | (200, **80**, 63, 250) | **(16,540, 4, 63, 250)** |
| 图像数 | 200 | **16,540** |
| 重复数 | 80 | **4** |
| 每被试试次 | 16,000 | 66,160 |
| 10 被试总计 | 160,000 | **661,600** |
| 试次间信度 | **+0.072** | **+0.005** |
| 磁盘 | 4.5 GB | 20.2 GB |

**这是 THINGS-EEG 的原始设计，不是错误：**

- 训练集：**图像多（16,540）、重复少（4）** → 用于学跨图像泛化
- 测试集：**图像少（200）、重复多（80）** → 用于稳定评估

### 直接后果

| 用途 | 用哪个 split |
|---|---|
| 按 repetition 平均降噪 | `test/`（80 次有效）|
| 单试次分析 | 都难（train 只有 4 次，信度 0.005）|
| **跨图像概念泛化** | **`train/`（16,540 张图）** |
| 强调试次的稳定评估 | `test/` |

---

## 来源

| 项 | 值 |
|---|---|
| 上游 | [THINGS-EEG](https://doi.org/10.1016/j.neuroimage.2022.119754) |
| 上游子集 | `Preprocessed_data_250Hz_whiten`（250 Hz，已白化）|
| 许可 | **CC-BY-4.0** |
| 处理 | 解包 PyTorch `.pt` → 按被试重组为 `.npz`；派生频带功率 |

**数值未改变。** 频带功率是新派生量（见下）。

---

## 结构

```
test/
├── sub-01.npz          eeg  (200, 80, 63, 250) float16  ≈446 MB
├── sub-01_band.npz     band (200, 80, 63, 5)   float32  ≈18 MB
├── ...                 ×10 被试
├── meta.parquet        160,000 行
└── axis_info.json

train/
├── sub-01.npz          eeg  (16540, 4, 63, 250) float16  ≈1.94 GB
├── sub-01_band.npz     band (16540, 4, 63, 5)   float32  ≈0.08 GB
├── ...                 ×10 被试
├── meta.parquet        661,600 行
└── axis_info.json
```

**总计 24.7 GB / 821,600 个四维样本。**

---

## 五轴定义

| 轴 | test | train | 含义 |
|---|---:|---:|---|
| `image` | 200 | **16,540** | 图像条件（THINGS）|
| `repetition` | 80 | **4** | 同一图像的重复呈现 |
| `channel` | 63 | 63 | EEG 电极 |
| `time` | 250 | 250 | 刺激后 0–1000 ms（250 Hz × 4 ms）|
| `frequency` | 5 | 5 | 频带功率（派生）|

### 频带

| 索引 | 名称 | 范围 |
|---:|---|---|
| 0 | delta | 1–4 Hz |
| 1 | theta | 4–8 Hz |
| 2 | alpha | 8–13 Hz |
| 3 | beta | 13–30 Hz |
| 4 | gamma | 30–45 Hz |

`band` 为**绝对功率**（Hann 窗 + rFFT，频带内平均）。
跨频带差数个数量级（1/f），比较时须先转相对功率。

---

## meta.parquet

| 列 | 类型 | 说明 |
|---|---|---|
| `subject` | str | `sub-01` … `sub-10` |
| `image_idx` | int | 对应 `eeg[image_idx]` |
| `rep` | int | 对应 `eeg[image_idx, rep]` |
| `label` | int | 类别标签 |
| `img` | str | 图像相对路径 |
| `text` | str | 概念名（如 `aardvark`）|

- `test/`: 160,000 行（10 × 200 × 80）
- `train/`: 661,600 行（10 × 16,540 × 4）
- **均无缺失值**

`text` 是**概念键**：train 中 1654 个概念，每概念 10 张不同图像，
可直接用于"同概念不同图像"的泛化实验。

---

## 加载

```python
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/omnibrain-4d"

# 下载单个被试（446 MB）
f = hf_hub_download(REPO, "test/sub-01.npz", repo_type="dataset")
eeg = np.load(f)["eeg"]          # (200, 80, 63, 250) float16

# meta
m = pd.read_parquet(hf_hub_download(REPO, "test/meta.parquet", repo_type="dataset"))
```

### 典型用法

```python
# ---- test: 按 repetition 切分（同类不跨折）
n_tr = 40
Xtr = eeg[:, :n_tr].mean(axis=1)   # (200, 63, 250)
Xte = eeg[:, n_tr:].mean(axis=1)

# ---- train: 按图像切分（跨图像泛化）
# 概念 c 的 10 张图中取 8 训练 / 2 测试
```

**注意**：`eeg.npz` 未做基线校正。分析前建议减 0–50 ms 基线：

```python
base = eeg[:, :, :, :12].mean(axis=3, keepdims=True)  # 0-48 ms
x = eeg.astype(np.float32) - base
```

---

## 已验证的性质

| 性质 | 检验 | test | train |
|---|---|---|---|
| 数值完整性 | 全元素 `isfinite` | ✅ | ✅ |
| 零值占比 | | 0.00% | 0.00% |
| `label` 沿 repetition 恒定 | 逐被试 | 10/10 | 10/10 |
| 同图像不同 rep 数组不同 | 抽查 | ✅ | ✅ |
| 不同图像数组不同 | 抽查 | ✅ | ✅ |
| **试次间信度** | 奇偶 split-half | **+0.072** | **+0.005** |

`train` 信度接近 0 是 **4 次重复的必然结果**，不是构建错误。

---

## 基准结果（本数据集上测过）

| 任务 | 配置 | 结果 |
|---|---|---|
| 200 类解码（test）| 40 rep 训练 / 40 测试 | **15–50× 随机** |
| 100 类解码（train，同图像）| 8 图 × rep0 → rep1 | **4.1× 随机** |
| **100 类解码（train，新图像）** | 8 图 × rep0 → 图 9,10 | **2.4× 随机** |
| 打乱标签基线 | | 0.8× |
| 时间常数 τ（test）| ACF 全曲线拟合 | 147–262 ms（5/5 通过分辨率检验）|

**关键**：从 15–50× 降到 4.1×，主因是训练样本从 40 降到 8；
再降到 2.4× 才是"见新图像"的代价（1.74 倍）。

---

## 已知限制

1. **已白化** —— 上游 `whiten` 版本，不是原始电压。
2. **无行为数据** —— THINGS-EEG 是被动观看范式。
3. **float16 存储** —— 原始 float32 会占 2 倍；如需全精度须回上游。
4. **train 重复只有 4 次** —— 无法做可靠的单试次估计。
5. **被试 10 人** —— 标准公开样本，非特定人群。

---

## 复现

构建脚本见配套仓库 `Microbiosis/hetero-moe`：

- `research/brain_domains/data/build_4d_eeg.py`（test）
- `research/brain_domains/data/build_4d_eeg_train.py`（train）

---

## 引文

使用本数据请同时引用 THINGS-EEG：

```bibtex
@article{gifford2022things,
  title={A large and rich EEG dataset for modeling human visual object recognition},
  author={Gifford, Alessandro T and Dwivedi, Kartik and Roig, Gemma and Cichy, Radoslaw M},
  journal={NeuroImage},
  volume={264},
  pages={119754},
  year={2022},
  doi={10.1016/j.neuroimage.2022.119754}
}
```
