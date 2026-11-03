# HuggingFace 产物索引

> 统一入口：[**Hetero-MoE 研究档案** Collection](https://huggingface.co/collections/redmingwei/hetero-moe-6ae6b67d79b0423227575c99)
> 更新时间：2026-11-01（仓库合并后）
> **11 model / 7941.6 MB → 4 model / 3155.3 MB（省 4.79 GB，减 60%）**

本仓库的权重、数据、实验产物都放在 HuggingFace。本文档说明**哪个是哪个**，
避免同名仓库混淆。

## 一、从哪开始

| 想做什么 | 用哪个 | 说明 |
|---|---|---|
| **直接跑模型** | [`HeteroMoEProjSpace-v2`](https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2) | 唯一可 `from_pretrained` 加载的完整模型，645.19M / 2.6 GB |
| **复算测量数字** | [`backbone-slices`](https://huggingface.co/redmingwei/backbone-slices) + [`hetero-moe-cache`](https://huggingface.co/datasets/redmingwei/hetero-moe-cache) | 权重切片 295 MB（私有）/ 隐藏状态缓存 4.46 GB |
| **看结论** | [v62 SPEC](aligner/../research/aligner/v62_dimension_projection_space/SPEC.md) | 规范 / 声明 / 指标 / 测试都在子包里 |
| **验证别人能不能用** | `research/aligner/v62_dimension_projection_space/cleanroom.py` | 干净 clone + 只从 Hub 下载，8 步 |

## 二、命名陷阱（已彻底解决）

合并前有三个名字极像的仓库，看名字会直接选错。现已收敛成两个清晰角色：

| 原来 | 现在在哪 |
|---|---|
| `HeteroMoEProjSpace-v2` (2.6 GB) | **主仓**（唯一可加载）`HeteroMoEProjSpace-v2` |
| `hetero-moe-projspace-v2` (214 MB) | → `HeteroMoEProjSpace-v2/checkpoints/v62.3/` |
| `hetero-moe-projspace` (3.3 MB) | **分析仓**，吸收 `dimspace/` `align/` `v33_reference/` |

一个能跑模型，一个存读数。`HeteroMoE-v1` 已删除（见 §三「已删除的仓库」）。

## 二·五、当前版本状态（v62.3）

| 项 | 值 |
|---|---|
| 当前版本 | **v62.3**（`checkpoints/v62.3/`） |
| held-out 准确率 | **0.9076**（随机 0.25） |
| 专家负载 / 均衡 | [0.350, 0.527, 0.572, 0.551] / **0.612** |
| 槽门控熵 | **0.647**（软分布） |
| 槽门控 EMA（v62.4） | step0 全 0.25 → step1500 [Q 0.172, K 0.268, V 0.254, O 0.306] —— **四槽都在正常分工** |

⚠️ **v62.0 / v62.1 / v62.2 的端到端数字与槽门控结论一律作废。**
那些版本的 `proj_space.router` 是 lazy 创建的（既没被 `load_state_dict` 加载上，
也从未进入 optimizer），槽门控全由一个**随机初始化**的 router 决定 ——
v62.0 报的「均衡 0.8774」尤其误导，它统计的正是这个从不参与计算的模块。

---

## 三、完整清单（4 model + 3 dataset）

### ① 主产物

| 仓库 | 大小 | 内容 |
|---|---|---|
| **`HeteroMoEProjSpace-v2`** | **2.81 GB** | **v62.3** 完整模型（唯一可 `from_pretrained` 加载）+ `checkpoints/v62.3/`（训练 state_dict 214 MB + 报告）|
| `backbone-slices`（私有） | 295 MB | 三骨干 layer0–3 × 6 功能槽权重切片 |
| `hetero-moe-cache`（dataset） | 4.46 GB | ag_news/food101 真实隐藏状态 |

### ② 训练实验记录（失败与混合结果）

| 仓库 | 大小 | 内容 |
|---|---|---|
| `hetero-moe-router` | 47 MB | 三合一：根目录(冻结) / `unfrozen/` / `shared-routed/` |

| 子目录 | task loss | 专家负载 | 判定 |
|---|---|---|---|
| 根目录（冻结） | 52.97 → 13.09 | — | 路由没学（探针 0.390/0.296） |
| `unfrozen/` | 52.97 → **0.112** | `[0.295,0.814,0.890,0.001]` | **塌缩** |
| `shared-routed/` | 52.97 → 0.10 | `[0.001,0.999,0.9997,0.0001]` | 融合成功，路由塌 |

### ③ 分析产物（v48–v61）

| 仓库 | 大小 | 内容 |
|---|---|---|
| `hetero-moe-projspace` | 5 MB | 五合一：根目录(秩扫描/QO 诊断) / `dimspace/` / `align/` / `v33_reference/` |

| 子目录 | 关键读数 |
|---|---|
| 根目录 | 秩扫描：CKA 全程 0.998，保真度 24 维 0.165 / 256 维 0.9998；Q/O 诊断：两两 0.987 / 三模型 0.677 |
| `dimspace/` | `D_shared` 扫描：固定的数不可能同时适配秩 3 和秩 30 |
| `align/` | raw/orth/one/per 四种对齐：成对 0.99999 vs 共享 0.56；正交化与 raw 逐位相同 |
| `v33_reference/` | v33 结构的基准输出（**结论已被推翻，仅作对照证据**） |

### ④ 展示产物

| 仓库 | 大小 | 内容 |
|---|---|---|
| `structure-film-public` | 105 MB | **v62 机制可视化**（`v62_mechanism.mp4`，8 幕 / 60 秒 / 数字全部来自真实前向）+ 旧版机制可视化 + 族谱 3D 素材 |
| `structure-film-job`（私有） | 106 MB | 《谱系》110 秒长片渲染工程 |

### 已删除的仓库（及原因）

| 仓库 | 原大小 | 处理 | 原因 |
|---|---|---|---|
| `brain-domains-scripts` | 空 | 删除 | 只有 `.gitattributes`，无内容 |
| `HeteroMoE-v1-tuned` | 2.38 GB | → 随 `HeteroMoE-v1` 一并删除，训练记录转存 `v33_reference/train_report.json` | 与 `HeteroMoE-v1` 99.9995% 逐位相同（只有 3076 个参数不同）；基座删除后 13 KB delta 失去意义 |
| `hetero-moe-dimspace` | <0.1 MB | → `hetero-moe-projspace/dimspace/` | 并入分析仓 |
| `hetero-moe-align` | <0.1 MB | → `hetero-moe-projspace/align/` | 并入分析仓 |
| `hetero-moe-router-unfrozen` | 14.6 MB | → `hetero-moe-router/unfrozen/` | 同族实验合并 |
| `hetero-moe-shared-routed` | 17.7 MB | → `hetero-moe-router/shared-routed/` | 同族实验合并 |
| `hetero-moe-projspace-v2` | 214 MB | → `HeteroMoEProjSpace-v2/checkpoints/v62.3/` | 中间产物，归属主仓 |
| **`HeteroMoE-v1`** | **2.41 GB** | **删除**，证据转存 `hetero-moe-projspace/v33_reference/` | 见下 |

#### 为什么删 `HeteroMoE-v1`

四条同时成立：

| 项 | 状况 |
|---|---|
| 代码引用 | **零**。`grep -r "HeteroMoE-v1" --include="*.py"` 全仓库为空 |
| 可否加载 | **不可**。`architectures: null`，无 `auto_map`，无建模代码 |
| 可否重建 | **可以**。`P_m` 是 `nn.init.orthogonal_`，专家是 `torch.randn*0.02`，只需 seed；v48 的 A 臂就是这么本地重建的 |
| 结论对不对 | **不对**（见下） |

v33 的「融合优于单专家」结论被四代审计推翻：

- **v50**：融合输出对**源专家身份**的可解码性 **1.000**，对目标类别只有 **0.279**（随机 0.125）
  → 该空间里路由的任务是平凡的「认出这是哪个底座的」
- **v52**：冻结随机臂下 A/B/D 三臂 held-out **逐位相同**，`fuse 7.18` vs 平凡基线 `0.125`
  → **模型输出与目标无关，它根本没在做这个任务**
- **v53**：三套独立实现同向确认

所以这是一个**装载着错误结论的结构**。权重不值得存，但它的**输出读数**是「v33 为什么错」
的直接证据，因此保留 1.7 MB 到 `hetero-moe-projspace/v33_reference/`：

```
example_inputs.npz      1.2 MB   v33 的基准输入
expected_y.npy          0.5 MB   该结构的实际输出（不是正确答案）
expected_alpha.npy      7.3 KB   该结构的路由权重
train_report.json       2.6 KB   3076 参数微调记录 + before/after 路由负载
```

`train_report.json` 记录的路由塌缩：微调后 Qwen 的专家负载从 `[0.04, 0.67, 0.50, 0.79]`
变成 `[0.0, 1.0, 0.04, 0.0]`，熵从 0.482 掉到 ≈ 0。**更集中，不是更均衡。**

## 四、Space 与 Bucket

| 名称 | 类型 | 状态 | 用途 |
|---|---|---|---|
| `redmingwei/hetero-moe` | Space (docker) | ⏸ PAUSED（按需） | 跑实验（cpu-upgrade）。v58–v61 都在这里跑过 |
| `redmingwei/hetero-moe-jobs` | Bucket（私有） | — | **各代实验原始产物桶**，见下表 |
| `redmingwei/jobs-artifacts` | Bucket | — | 通用作业产物 |
| `redmingwei/IllusionBrain` | Space (gradio) | ▶ RUNNING | 另一项目（EEG 4D 训练），与 hetero-moe 无关 |
| `redmingwei/seance` | Space (streamlit) | ▶ RUNNING | 与 hetero-moe 无关 |

### Bucket `hetero-moe-jobs` 完整路径

44 个前缀，共约 1.5 MB。`<prefix>/metrics.json` 是对应代的原始读数，
`<prefix>t-*` 是单元测试运行的产物。

| 代 | 前缀 | 说明 |
|---|---|---|
| v48 | `v48-2583ed45/` | 16 文件，含 `REPRODUCE.md` |
| v49 | `v49-v49_artifacts/` | 18 文件 |
| v50 | `v50-v50_artifacts/` | 8 文件 |
| v51 | `v51-v51_artifacts/` | 7 文件 |
| v52 | `v52-v52_artifacts/` | 15 文件 |
| v53 | `v53-v53_artifacts/cells/` | 11 文件 |
| v54 | `v54-a32eda6-cpu8/`（主）、`v54-86e0f4e/`、`v54-93e7b0d-t4/` | 主运行 42 文件 |
| v56 | `v56-556f137/` `v56-556f137b/` `v56-556f137c/` `v56-556f137d/` | 四次重跑 |
| v57 | `v57-adaptive/` | 2 文件 |
| v58 | `v58-area4/`（**引用此**）、`v58-area/` `v58-area2/` `v58-area3/`、`v58cal-1/`、`v58t-1..3/` | v58 共 7 个前缀 |
| v59 | `v59-area2/`（**引用此**）、`v59-area1/`、`v59t-1..8/` | 测试 8 轮 |
| v60 | `v60-area1/`（**引用此**）、`v60-d1-1/`、`v60t-1..5/` | `v60-d1-1/` 是补充分析 |
| v61 | `v61-run2/`（**引用此**）、`v61-diag1/`、`v61t-1..4/` | `v61-diag1/` 是 0.188 诊断 |

> ⚠️ 注意：`v58_entanglement_area/FINDINGS.md` 写的是 `v58-area2`，
> 但 `v58-area4/` 才是最后一次全量重跑（commit `a93ecec`，1269s）。
> `results/metrics.json` 引用的是 `area4`。

读取示例：

```python
import json, urllib.request
url = "https://huggingface.co/api/buckets/redmingwei/hetero-moe-jobs/tree"
# 带 token；返回 [{"type":"file","path":"v61-run2/metrics.json",...}]
```

## 五、被代码引用的仓库

只有三个仓库被代码硬编码引用：

| 仓库 | 引用处 |
|---|---|
| `HeteroMoEProjSpace-v2` | SPEC / README / cleanroom.py |
| `backbone-slices` | `measure.py` |
| `hetero-moe-cache` | 训练脚本 |

三个名字与路径合并后**均未变**，所有脚本不受影响。
其余 4 个仓库（`hetero-moe-projspace` / `hetero-moe-router` / 两个 film）**没有代码引用**。

## 六、已知的可复核性缺口

| 项 | 状况 |
|---|---|
| v56 的单侧探针 0.727/0.648 | 只在 `v56_capacity_allocation/SPEC.md §4`，**未落盘 metrics.json** |
| `expected_y` 跨版本复现 | 5.21.0 生成，5.16.1 下 Δ=7.9e-05；要 bit 级一致需 pin 版本 |
| v48–v61 的 Space 运行记录 | 在 bucket 里，路径已在 §四 列全；bucket 本身无公开网页索引 |
