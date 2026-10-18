# OmniBrain 4D — THINGS-EEG 训练集

EEG 数据的四轴完整保留版本：`image × repetition × channel × time`。

**这是 train split。** test split 见 `omnibrain4d/eeg/`。

---

## ⚠️ 先读这一条：train 与 test 的轴结构完全不同

| | test | **train** |
|---|---:|---:|
| 形状 | (200, **80**, 63, 250) | **(16,540, 4, 63, 250)** |
| 图像数 | 200 | **16,540** |
| 重复数 | 80 | **4** |
| 每被试试次 | 16,000 | 66,160 |
| 10 被试总计 | 160,000 | **661,600** |
| 磁盘 | 4.5 GB | **20.2 GB** |

**这不是数据错误，是 THINGS-EEG 的原始设计**：

- 训练集：**图像多（16,540）、重复少（4）** —— 用于学习泛化
- 测试集：**图像少（200）、重复多（80）** —— 用于稳定评估

### 直接后果：两个 split 不能混用同一个试次平均策略

| 用途 | test | train |
|---|---|---|
| 按 repetition 平均降噪 | ✅ 80 次，信度 +0.072 | ❌ **只有 4 次，信度 +0.005** |
| 单试次解码 | 难（噪声大）| 极难 |
| 学跨图像泛化 | ❌ 只有 200 张 | ✅ **16,540 张** |

**train 集不适合做「试次平均」类的分析**（4 次平均几乎没有降噪效果）。

---

## 来源与许可

| 项 | 值 |
|---|---|
| 上游数据集 | THINGS-EEG |
| 上游子集 | `Preprocessed_data_250Hz_whiten/<sub>/train.pt` |
| 许可 | CC-BY-4.0 |
| 处理 | 解包 `.pt` → 按被试重组为 `.npz`；派生频带功率。**未改变数值。** |

**若使用本数据，请引用 THINGS-EEG 原始文献**（见文末）。

---

## 结构

```
omnibrain4d/eeg_train/
├── sub-01.npz          eeg  (16540, 4, 63, 250) float16  ≈1.94 GB
├── sub-01_band.npz     band (16540, 4, 63, 5)   float32  ≈0.08 GB
├── ...
├── sub-10.npz
├── sub-10_band.npz
├── meta.parquet        661,600 行
└── axis_info.json
```

**总量**：19.4 GB（raw）+ 0.8 GB（band）= **20.2 GB**。

---

## 五轴定义

| 轴 | 长度 | 含义 |
|---|---:|---|
| `image` | **16,540** | 图像条件（THINGS 训练集）|
| `repetition` | **4** | 同一图像的重复呈现 |
| `channel` | 63 | EEG 电极 |
| `time` | 250 | 刺激后 0–1000 ms（250 Hz × 4 ms）|
| `frequency` | 5 | 频带功率（派生）|

频带：delta(1–4) / theta(4–8) / alpha(8–13) / beta(13–30) / gamma(30–45) Hz。

`band` 为**绝对功率**（Hann 窗 + rFFT，各频带内平均），跨频带可比性差（1/f），
比较时须先转相对功率。

---

## meta.parquet

| 列 | 类型 | 说明 |
|---|---|---|
| `subject` | str | `sub-01` … `sub-10` |
| `image_idx` | int | 0–16539 |
| `rep` | int | 0–3 |
| `label` | int | 类别标签 |
| `img` | str | 图像文件名 |
| `text` | str | 文本描述 |

661,600 行 = 10 被试 × 16,540 × 4。**无缺失值。**

---

## 加载

```python
import numpy as np, pandas as pd

BASE = "hf://buckets/illusion-brain/brain-domains/omnibrain4d/eeg_train"
eeg = np.load(f"{BASE}/sub-01.npz")["eeg"]        # (16540, 4, 63, 250) float16
meta = pd.read_parquet(f"{BASE}/meta.parquet")    # 661,600 行

# 单试次
x = eeg[1234, 2]          # (63, 250)
```

### 典型用法（跨图像泛化）

```python
# 按图像划分（train/test split 内部再切分）
n_tr = 14000
Xtr = eeg[:n_tr].mean(axis=1)    # (14000, 63*250) —— 4 次平均
Xte = eeg[n_tr:].mean(axis=1)
```

---

## 已验证的性质

| 性质 | 检验 | 结果 |
|---|---|---|
| 10 个 raw 文件 | 形状与 dtype | 全部 `(16540,4,63,250)` float16 |
| 数值完整性 | `isfinite` | 全部通过 |
| 零值占比 | 全元素 | **0.00%**（无补零污染）|
| meta 完整性 | 逐列空值 | 全 0 |
| `label` 沿 rep 恒定 | 逐被试 | **10/10 为 True** |
| **试次间信度** | 奇偶 split-half（50 抽样）| **−0.036 ~ +0.047，中位 +0.005** |

**信度接近 0 是数据结构的必然结果**：4 次重复，split-half 每半只有 2 次平均，
无法压制单试次噪声。这不是构建错误。

---

## 已知限制

1. **重复仅 4 次** —— 无法做可靠的单试次估计或试次平均降噪。
2. **已白化** —— 上游 `whiten` 版本，不是原始电压。
3. **无行为数据** —— THINGS-EEG 是被动观看范式。
4. **float16 存储** —— 原始 float32 会占 2 倍；如需全精度须回到上游。
5. **图像集庞大（16,540）** —— 做全量 200 类式解码会非常慢；建议先做跨图像泛化
   或降采样子集。

---

## 复现

```bash
python research/brain_domains/data/build_4d_eeg_train.py
```

需要：`/training` 卷挂载（含 `things-eeg/.../train.pt`）、`torch`、约 25 GB 空闲磁盘。
Space 上 8 vCPU 实测 **6.5 分钟**完成 10 个被试。

后台运行（避免请求超时）：

```bash
python scripts/launch_bg.py build_4d_eeg_train.py
python scripts/check_log.py build_4d_eeg_train.py
```

---

## 引文

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
