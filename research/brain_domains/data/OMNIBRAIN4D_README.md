# THINGS-EEG 4D — 按原始四轴组织的派生数据集（test split）

EEG 数据的**四轴完整保留**版本：`image × repetition × channel × time`。
供需要试次结构、时间进程、频带分解的工作直接使用。

---

## 来源与许可

| 项 | 值 |
|---|---|
| 上游数据集 | THINGS-EEG（THINGS 图像集 + 10 名被试的 EEG 记录）|
| 上游子集 | `Preprocessed_data_250Hz_whiten`（250 Hz 采样，已白化）|
| 许可 | CC-BY-4.0 |
| 处理 | 解包 `.pt` → 重组为按被试的 `.npz`；派生频带功率 |
| 修改声明 | 原数据为 PyTorch `.pt` 字典；本版本转为 NumPy `.npz`，未改变数值。频带功率为新派生量（见下）。|

**若使用本数据，请同时引用 THINGS-EEG 原始文献。**
原始数据中未改变的部分保留其原始许可（CC-BY-4.0）。

---

## 结构

```
omnibrain4d/eeg/
├── sub-01.npz          eeg  (200, 80, 63, 250) float16  ≈446 MB
├── sub-01_band.npz     band (200, 80, 63, 5)   float32  ≈18 MB
├── ...
├── sub-10.npz
├── sub-10_band.npz
├── meta.parquet        160,000 行：逐 (image, repetition) 的标签
└── axis_info.json      五轴定义 + 逐被试检查记录
```

**总量**：4.35 GB（raw）+ 0.17 GB（band），10 被试 × 16,000 试次 = **160,000 个四维样本**。

---

## 五轴定义

| 轴 | 长度 | 含义 | 类型 |
|---|---:|---|---|
| `image` | 200 | 图像条件（THINGS 测试集）| 类别 |
| `repetition` | 80 | 同一图像的重复呈现 | 试次 |
| `channel` | 63 | EEG 电极（10-20 系统）| 空间 |
| `time` | 250 | 刺激后 0–1000 ms（250 Hz × 4 ms）| 时间 |
| `frequency` | 5 | 频带功率（派生）| 频谱 |

### 频带划分

| 索引 | 名称 | 范围 |
|---:|---|---|
| 0 | delta | 1–4 Hz |
| 1 | theta | 4–8 Hz |
| 2 | alpha | 8–13 Hz |
| 3 | beta | 13–30 Hz |
| 4 | gamma | 30–45 Hz |

`band` 的计算：对 time 轴做 Hann 窗 + rFFT，取各频带内的平均功率。
**数值为绝对功率（未归一化）**，跨频带差数个数量级（1/f 特性），
若要比较频带间差异须先转为相对功率。

---

## meta.parquet

| 列 | 类型 | 说明 |
|---|---|---|
| `subject` | str | `sub-01` … `sub-10` |
| `image_idx` | int | 0–199，对应 `eeg[image_idx]` |
| `rep` | int | 0–79，对应 `eeg[image_idx, rep]` |
| `label` | int | 类别标签（与 `image_idx` 相同，上游格式如此）|
| `img` | str | 图像文件名 |
| `text` | str | 图像对应的文本描述 |

160,000 行 = 10 被试 × 200 图像 × 80 重复。**无缺失值。**

---

## 加载

```python
import numpy as np, pandas as pd

BASE = "hf://buckets/illusion-brain/brain-domains/omnibrain4d/eeg"

eeg = np.load(f"{BASE}/sub-01.npz")["eeg"]        # (200, 80, 63, 250) float16
band = np.load(f"{BASE}/sub-01_band.npz")["band"] # (200, 80, 63, 5)  float32
meta = pd.read_parquet(f"{BASE}/meta.parquet")    # 160,000 行

# 单被试的某个试次
x = eeg[42, 7]            # (63, 250)  图像 42 的第 8 次重复
```

### 典型用法

```python
# 按 repetition 切分训练/测试（同类不跨折，避免泄漏）
n_tr = 40
train = eeg[:, :n_tr]     # (200, 40, 63, 250)
test  = eeg[:, n_tr:]     # (200, 40, 63, 250)

# 时间窗平均后做解码
win = eeg[:, :, :, 12:125].mean(axis=3)   # 50–500 ms → (200, 80, 63)
```

---

## 已验证的性质

以下数值均可在 `axis_info.json` 中逐被试查到。

| 性质 | 检验方式 | 结果 |
|---|---|---|
| `label` 沿 repetition 恒定 | 逐被试比对 | **10/10 为 True**（确认轴语义）|
| 试次间信度 | 奇偶试次 split-half 相关（去平均响应后）| 0.052–0.132，中位 0.072 |
| 频带的图像间变异 | 逐频带能量的变异系数 | 0.06–0.33，中位 0.123 |
| 数值完整性 | 全部元素 `isfinite` | 通过 |
| 重复独立性 | 同图像不同 rep 的数组不同 | 通过 |
| 图像独立性 | 不同图像数组不同 | 通过 |

**试次间信度低（≈0.07）说明单试次 EEG 以噪声为主。**
80 次重复主要用于**平均降噪**，不适合做单试次预测。

---

## 已知限制

1. **本文件只覆盖 test split**（200 张图）。train split（16,540 张 × 4 重复）
   见同仓库的 `eeg_train/` 或数据集仓库的 `train/`。
2. **已白化**。上游提供的 `whiten` 版本做过白化处理，不是原始电压。
   若需要原始信号，须回到上游未白化数据。
3. **没有行为数据**。THINGS-EEG 主要是被动观看范式，无逐试次行为标签。
4. **被试是标准公开样本**（10 人），非特定人群。
5. **`band` 是绝对功率**，未经 1/f 校正。

---

## 与现有分析的关系

在本仓库的 `research/brain_domains/` 中，以下结论基于此数据集：

| 结论 | 数值 | 文档 |
|---|---|---|
| 200 类解码 | 7.6–24.8%（15–50× 随机）| `FINDINGS_4D.md` |
| 时间常数 τ | 202 ms（p10–p90 149–251，5/5 通过分辨率检验，正交不变）| `FINDINGS_TAU_GAUGE.md` |
| τ 与解码无关 | ρ = −0.10（n=10）| `FINDINGS_TAU_NO_FUNCTION.md` |
| 四轴均不可替代 | 消融损失 82–97% | `FINDINGS_4D.md` |

---

## 复现构建

```bash
python research/brain_domains/data/build_4d_eeg.py /out/omnibrain4d/eeg
```

脚本读上游 `/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-XX/test.pt`，
无需本地下载（在 HF Space 或挂载了 `illusion-brain/training-data` 桶的环境中运行）。

---

## 引文

原始数据请引用 THINGS-EEG：

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
