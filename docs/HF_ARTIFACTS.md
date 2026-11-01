# HuggingFace 产物索引

> 统一入口：[**Hetero-MoE 研究档案** Collection](https://huggingface.co/collections/redmingwei/hetero-moe-6ae6b67d79b0423227575c99)
> 更新时间：2026-11-01

本仓库的权重、数据、实验产物都放在 HuggingFace。本文档说明**哪个是哪个**，
避免同名仓库混淆。

## 一、从哪开始

| 想做什么 | 用哪个 | 说明 |
|---|---|---|
| **直接跑模型** | [`HeteroMoEProjSpace-v2`](https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2) | 唯一可 `from_pretrained` 加载的完整模型，645.19M / 2.6 GB |
| **复算测量数字** | [`backbone-slices`](https://huggingface.co/redmingwei/backbone-slices) + [`hetero-moe-cache`](https://huggingface.co/datasets/redmingwei/hetero-moe-cache) | 权重切片 295 MB（私有）/ 隐藏状态缓存 4.46 GB |
| **看结论** | [v62 SPEC](aligner/../research/aligner/v62_dimension_projection_space/SPEC.md) | 规范 / 声明 / 指标 / 测试都在子包里 |
| **验证别人能不能用** | `research/aligner/v62_dimension_projection_space/cleanroom.py` | 干净 clone + 只从 Hub 下载，8 步 |

## 二、命名陷阱（三个极像的仓库）

| 仓库 | 大小 | 是什么 | 能直接用吗 |
|---|---|---|---|
| **`HeteroMoEProjSpace-v2`** | 2.6 GB | **完整模型**：三骨干 + 投影空间 + 融合层 | ✅ `trust_remote_code=True` |
| `hetero-moe-projspace-v2` | 214 MB | v62 训练后的 `state_dict`，**不含骨干** | ❌ 需自建模块再灌权重 |
| `hetero-moe-projspace` | 3.3 MB | v48–v61 的**分析产物**（秩扫描 / Q/O 诊断 json） | ❌ 只是读数 |

## 三、完整清单

### ① 主产物

| 仓库 | 类型 | 大小 | 内容 |
|---|---|---|---|
| `HeteroMoEProjSpace-v2` | model | 2.6 GB | v62 完整模型，645,190,802 参数 |
| `backbone-slices` | model（私有） | 295 MB | 三骨干 layer0–3 × 6 功能槽权重切片 |
| `hetero-moe-cache` | dataset | 4.46 GB | ag_news/food101 真实隐藏状态 |

### ② 历史完整模型（**不可直接加载**）

| 仓库 | 大小 | 问题 |
|---|---|---|
| `HeteroMoE-v1` | 2.4 GB | 无 `auto_map`、无建模代码 → `trust_remote_code` 失效 |
| `HeteroMoE-v1-tuned` | 2.4 GB | 同上 |

`save_pretrained` 只保存配置与权重，**不保存 Python 文件**。这是 v62 净室测试
发现并修复的问题（v62 有 `modeling_hetero_moe_projspace.py` + `auto_map`）。

### ③ 训练实验记录（失败与混合结果）

| 仓库 | 大小 | 结果 |
|---|---|---|
| `hetero-moe-router` | 14.6 MB | 专家冻结：loss 52.97→13.09，但探针停在 0.390/0.296（随机 0.25） |
| `hetero-moe-router-unfrozen` | 14.6 MB | 专家解冻：task loss 0.112，负载塌缩 `[0.295,0.814,0.890,0.001]` |
| `hetero-moe-shared-routed` | 17.7 MB | 融合成功（余弦 0.029→0.107），路由仍塌 |

三者都以裸 `state_dict` 保存，**不是** `from_pretrained` 格式。
`report.json` 里的 loss 曲线**不能**当性能指标引用。

### ④ 分析产物（v48–v61）

| 仓库 | 大小 | 关键读数 |
|---|---|---|
| `hetero-moe-projspace` | 3.3 MB | 秩扫描：CKA 全程 0.998，保真度 24 维 0.165 / 256 维 0.9998；Q/O 诊断：两两 0.987 / 三模型 0.677 |
| `hetero-moe-projspace-v2` | 214 MB | v62 训练曲线（1500 步，最佳 val_acc 0.9016） |
| `hetero-moe-dimspace` | <0.1 MB | `D_shared` 扫描对照 |
| `hetero-moe-align` | <0.1 MB | raw/orth/one/per 四种对齐：成对 0.99999 vs 共享 0.56 |

### ⑤ 展示产物

| 仓库 | 大小 | 内容 |
|---|---|---|
| `structure-film-public` | 101 MB | 机制可视化视频 + 族谱 3D 素材 |
| `structure-film-job`（私有） | 106 MB | 《谱系》110 秒长片渲染工程 |

## 四、Space 与 Bucket

| 名称 | 类型 | 状态 | 用途 |
|---|---|---|---|
| `redmingwei/hetero-moe` | Space (docker) | ⏸ **PAUSED** | 跑实验（cpu-upgrade）。v58/v59/v60/v61 都在这里跑过 |
| `redmingwei/hetero-moe-jobs` | Bucket | — | 各代实验的原始产物桶，SPEC 里引用的 `v58-area2/` 等路径都在这里 |
| `redmingwei/IllusionBrain` | Space (gradio) | ⚠️ **RUNTIME_ERROR** | 与 hetero-moe 无关的另一个项目；错误是 `hf-mount failed` |
| `redmingwei/seance` | Space (streamlit) | ▶ RUNNING | 与 hetero-moe 无关 |
| `redmingwei/brain-domains-scripts` | dataset（私有） | **空仓库** | 只有 `.gitattributes`，无实际内容 |

## 五、被代码引用的仓库

只有三个仓库被代码硬编码引用：

| 仓库 | 引用处 |
|---|---|
| `HeteroMoEProjSpace-v2` | SPEC / README / cleanroom.py (7 处) |
| `backbone-slices` | `measure.py` (5 处) |
| `hetero-moe-cache` | 训练脚本 (1 处) |

其余 9 个仓库**没有代码引用**，只作为历史归档存在。
改它们的名字或删掉不会破坏任何脚本。

## 六、已知的可复核性缺口

| 项 | 状况 |
|---|---|
| v56 的单侧探针 0.727/0.648 | 只在 `v56_capacity_allocation/SPEC.md §4`，**未落盘 metrics.json** |
| `expected_y` 跨版本复现 | 5.21.0 生成，5.16.1 下 Δ=7.9e-05；要 bit 级一致需 pin 版本 |
| v48–v61 的 Space 运行记录 | 在 bucket 里，不在仓库；bucket 没有对外索引 |
