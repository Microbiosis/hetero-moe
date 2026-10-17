# 脑域数据集 v1（brain-domains bucket）

> 从桶内原始神经数据构建的**域标注数据集**，供域条件建模与跨域实验使用。
> 构建器：`build_domain_dataset.py` · 加载器：`load_domain_dataset.py`
> 存储：[`hf://buckets/illusion-brain/brain-domains`](https://huggingface.co/buckets/illusion-brain/brain-domains)（282 MB）

数据集**不入本仓库**（体积 + 上游许可），只入桶。本目录保存构建与加载代码、以及这份卡片。

---

## 一、包含什么

### A. `steinmetz/` — 小鼠尖峰 × 脑区（主数据集）

| 项 | 数值 |
|---|---|
| 来源 | 桶内 `spikes/steinmetz-spikes/`（Steinmetz 2019, Neuropixels）|
| session 数 | 39 |
| 神经元（unit）数 | **24,029**（原始 27,212，剔除未标注的 `root` 后）|
| 时间分辨率 | 50 ms 分箱 |
| 每 session 时长 | 45–68 分钟（52,955–81,668 bins）|
| 功能域 | **13** 个 |

**域分布**（按 unit 数）：

| 域 | units | 域 | units |
|---|---:|---|---:|
| thalamus | 4,571 | hypothalamus | 1,146 |
| midbrain | 3,282 | striatum | 1,006 |
| hippocampus | 2,744 | septal | 778 |
| visual | 2,652 | somatosensory | 623 |
| other | 2,551 | brainstem | 276 |
| prefrontal | 2,507 | auditory | 113 |
| motor | 1,780 | | |

**每个 session 可用域数**：min 2 / median 5 / max 9 —— 跨域实验需按 session 取交集。

**文件**：
```
steinmetz/
├── spikes_session_000.npz … _038.npz   # 每 session 一个 (n_units, n_bins) float32
├── units.parquet                        # session, unit_idx, orig_unit_idx, region,
│                                        # functional_domain, domain_id
└── splits.json                          # 逐 session 的 train_end / val_end / n_bins
```

**切分来自原始元数据的 `split_boundaries`**（时间切分，非随机划分）——这正好满足本项目的样本外纪律：验证集在时间上晚于训练集，不存在未来信息泄漏。

### B. `flywire/` — 果蝇连接组 × 脑区（对照数据集）

| 项 | 数值 |
|---|---|
| 来源 | 桶内 `connectome/flywire-connectome/` |
| 神经元 | **107,265**（有 region 标注的；原始 139,255）|
| 突触边 | 15,091,983 |
| 脑区 | 24 |
| 功能域 | **8** |

**域分布**：visual 80,383（75%，严重不平衡）/ olfactory 11,741 / memory 5,177 /
central_complex 2,869 / somatosensory 2,991 / ascending 2,362 / other 1,411 / descending 331

**文件**：
```
flywire/
├── node_features.parquet   # 逐神经元: 6 个 safe 特征 + 12 个 leaky 特征 + region + domain_id
└── feature_sets.json       # safe/leaky 分组 + 泄漏警告
```

---

## 二、⚠️ 两个数据集的可信度不同，用法也不同

| | Steinmetz | FlyWire |
|---|---|---|
| **标签与特征的关系** | 脑区标签**独立于**尖峰活动（标签来自解剖定位）| 脑区标签与连接组特征**同源** |
| **泄漏风险** | 低 | **高** —— 见下 |
| **推荐用途** | 域条件建模的主实验 | 仅作对照 / 结构分析 |
| **样本不平衡** | 中等（最大/最小 = 40×）| 严重（visual 占 75%）|

### FlyWire 的泄漏问题（README 陷阱 #2 的延伸）

神经元在哪个脑区，决定了它和谁连接、用什么递质——所以「用连接组特征预测脑区」接近同义反复。
本数据集把特征分成两组：

- **safe（6 个）**：`in_degree` / `out_degree` / `in_synapses` / `out_synapses` / `in_syn_per_edge` / `out_syn_per_edge`
  —— 粗粒度连接统计，泄漏较弱
- **leaky（12 个）**：按 6 种神经递质加权的突触总量 —— 编码了伙伴类型，即脑区身份

**任何「域结构有用」的结论必须用 safe 组 + 随机标签对照得出**（陷阱 #4 协议）。
用 leaky 组得到的漂亮数字没有意义。

---

## 三、怎么用

```python
# 编程
import sys; sys.path.insert(0, "research/brain_domains/data")
from load_domain_dataset import DomainDataset

ds = DomainDataset("/path/to/brain-domains")
for session, spikes, units, split in ds.iter_sessions():
    tr = slice(0, split["train_end"])
    va = slice(split["train_end"], split["val_end"])
    te = slice(split["val_end"], split["n_bins"])
    # spikes: (n_units, n_bins) float32, 已按 1/8 缩放
    # units["domain_id"]: 每个 unit 的域标签
    break
```

```bash
# 自检（会逐项验证对齐/映射/切分/值域，任何一项失败即非零退出）
python research/brain_domains/data/load_domain_dataset.py /path/to/brain-domains
```

## 四、可支撑的实验

| 实验 | 用哪个 | 说明 |
|---|---|---|
| **域条件群体预测** | Steinmetz | 用 t 时刻全体 unit 活动预测 t+1，模型额外接 domain_id |
| **跨域泛化**（leave-one-domain-out）| Steinmetz | 13 个域全部 ≥100 units，可做 LODO |
| **域分类 sanity check** | Steinmetz | 由活动模式预测所属域 —— 应先验证「活动可携带域信息」 |
| 连接组→脑区（**须用 safe 特征**）| FlyWire | 仅作结构侧对照 |
| 跨物种域结构比较 | 两者 | 必须显式标注物种（陷阱 #3）|

## 五、已知限制

1. **神经元数按 session 变化**（326–1,240），跨 session 联合建模需对齐策略或按 session 训练。
2. **域在 session 间分布不均**（每 session 只有 2–9 个域），LODO 需先筛出跨 session 都有的域。
3. **Steinmetz 是 50 ms 分箱**，不适合毫秒级时间动态研究（那需要原始采样率数据）。
4. **FlyWire 无时间维度**，只是静态连接组。
5. 两个数据集都**不含行为标签**（Steinmetz 的 `metadata.json` 只有脑区与时长），
   因此不能直接做「域 × 行为」联合分析。

## 六、构建复现

```bash
# 在 HF Jobs 上（CPU 即可，约 3 分钟）
hf jobs run --flavor cpu-upgrade \
  -v hf://buckets/illusion-brain/training-data:/bucket:ro \
  -v hf://buckets/illusion-brain/brain-domains:/out:rw \
  -v ./research/brain_domains/data:/scripts:ro \
  python:3.12 \
  bash -c "pip install --quiet pandas pyarrow numpy && mkdir -p /tmp/work && \
    cp /bucket/spikes/steinmetz-spikes/{metadata.json,session_*.npy} /tmp/work/ && \
    cp /bucket/connectome/flywire-connectome/{nodes,connections}.parquet /tmp/work/ && \
    python /scripts/build_domain_dataset.py /tmp/work /out"
```
