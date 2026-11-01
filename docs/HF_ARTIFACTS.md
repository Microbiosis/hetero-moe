# HuggingFace 产物索引

> 统一入口：[**Hetero-MoE 研究档案** Collection](https://huggingface.co/collections/redmingwei/hetero-moe-6ae6b67d79b0423227575c99)
> 更新时间：2026-11-01（仓库合并后）
> **合并前 11 model / 7941.6 MB → 合并后 5 model / 5560.6 MB（省 2.38 GB）**

本仓库的权重、数据、实验产物都放在 HuggingFace。本文档说明**哪个是哪个**，
避免同名仓库混淆。

## 一、从哪开始

| 想做什么 | 用哪个 | 说明 |
|---|---|---|
| **直接跑模型** | [`HeteroMoEProjSpace-v2`](https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2) | 唯一可 `from_pretrained` 加载的完整模型，645.19M / 2.6 GB |
| **复算测量数字** | [`backbone-slices`](https://huggingface.co/redmingwei/backbone-slices) + [`hetero-moe-cache`](https://huggingface.co/datasets/redmingwei/hetero-moe-cache) | 权重切片 295 MB（私有）/ 隐藏状态缓存 4.46 GB |
| **看结论** | [v62 SPEC](aligner/../research/aligner/v62_dimension_projection_space/SPEC.md) | 规范 / 声明 / 指标 / 测试都在子包里 |
| **验证别人能不能用** | `research/aligner/v62_dimension_projection_space/cleanroom.py` | 干净 clone + 只从 Hub 下载，8 步 |

## 二、命名陷阱（已解决：合并后只剩一个 2.8 GB 的大仓）

合并前有三个名字极像的仓库，看名字会直接选错。现已合并到主仓库：

| 原来 | 现在在哪 |
|---|---|
| `HeteroMoEProjSpace-v2` (2.6 GB) | **仍是主仓**，`HeteroMoEProjSpace-v2` |
| `hetero-moe-projspace-v2` (214 MB) | → `HeteroMoEProjSpace-v2/checkpoints/v62/` |
| `hetero-moe-projspace` (3.3 MB) | **仍是分析仓**，吸收了 `dimspace/` `align/` |

## 三、完整清单（合并后：5 model + 3 dataset）

### ① 主产物

| 仓库 | 大小 | 内容 |
|---|---|---|
| **`HeteroMoEProjSpace-v2`** | **2.81 GB** | v62 完整模型 + `checkpoints/v62/`（训练 state_dict 214 MB + 曲线） |
| `backbone-slices`（私有） | 295 MB | 三骨干 layer0–3 × 6 功能槽权重切片 |
| `hetero-moe-cache`（dataset） | 4.46 GB | ag_news/food101 真实隐藏状态 |

### ② 历史模型（**不可直接加载**）

| 仓库 | 大小 | 内容 |
|---|---|---|
| `HeteroMoE-v1` | 2.41 GB | v33 结构 595.2M + `tuned/`（v1-tuned 全部差异 **13 KB**） |

> **v1-tuned 的合并是本轮最大的一笔。** 逐张量比对发现：`HeteroMoE-v1-tuned` 的
> 2.38 GB 权重与本仓 **99.9995% 逐位相同**，真正变化的只有 3076 个数
> （`W_router` + 每专家 γ/β/α，与 `trainable_params: 3076` 吻合）。
> 冻结区（591M 骨干 + `P_m` + 专家）逐位未变。
> 现只保留 13 KB delta + 重建脚本，**省 2.38 GB**。

### ③ 训练实验记录（失败与混合结果）

| 仓库 | 大小 | 内容 |
|---|---|---|
| `hetero-moe-router` | 47 MB | 三合一：根目录(冻结) / `unfrozen/` / `shared-routed/` |

| 子目录 | task loss | 专家负载 | 判定 |
|---|---|---|---|
| 根目录（冻结） | 52.97 → 13.09 | — | 路由没学（探针 0.390/0.296） |
| `unfrozen/` | 52.97 → **0.112** | `[0.295,0.814,0.890,0.001]` | **塌缩** |
| `shared-routed/` | 52.97 → 0.10 | `[0.001,0.999,0.9997,0.0001]` | 融合成功，路由塌 |

### ④ 分析产物（v48–v61）

| 仓库 | 大小 | 内容 |
|---|---|---|
| `hetero-moe-projspace` | 3.3 MB | 四合一：根目录(秩扫描/QO 诊断) / `dimspace/` / `align/` |

| 子目录 | 关键读数 |
|---|---|
| 根目录 | 秩扫描：CKA 全程 0.998，保真度 24 维 0.165 / 256 维 0.9998；Q/O 诊断：两两 0.987 / 三模型 0.677 |
| `dimspace/` | `D_shared` 扫描：固定的数不可能同时适配秩 3 和秩 30 |
| `align/` | raw/orth/one/per 四种对齐：成对 0.99999 vs 共享 0.56；正交化与 raw 逐位相同 |

### ⑤ 展示产物

| 仓库 | 大小 | 内容 |
|---|---|---|
| `structure-film-public` | 101 MB | 机制可视化视频 + 族谱 3D 素材 |
| `structure-film-job`（私有） | 106 MB | 《谱系》110 秒长片渲染工程 |

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
| `HeteroMoEProjSpace-v2` | SPEC / README / cleanroom.py (7 处) |
| `backbone-slices` | `measure.py` (5 处) |
| `hetero-moe-cache` | 训练脚本 (1 处) |

其余仓库**没有代码引用**，只作为历史归档存在。
改它们的名字或删掉不会破坏任何脚本。

## 六、已知的可复核性缺口

| 项 | 状况 |
|---|---|
| v56 的单侧探针 0.727/0.648 | 只在 `v56_capacity_allocation/SPEC.md §4`，**未落盘 metrics.json** |
| `expected_y` 跨版本复现 | 5.21.0 生成，5.16.1 下 Δ=7.9e-05；要 bit 级一致需 pin 版本 |
| v48–v61 的 Space 运行记录 | 在 bucket 里，路径已在 §四 列全；bucket 本身无公开网页索引 |
