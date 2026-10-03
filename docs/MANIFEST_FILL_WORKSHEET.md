# MANIFEST 人工填写工作表

> 本文件由 `tools/manifest_worksheet.py` 生成, **不含任何自动写入**。
> 下面的"候选证据"是检索结果与行号, 供人工核对; 请勿未经核对直接采信。

待填包数: **29**

## 字段语义

| 字段 | 含义 |
|---|---|
| `generation.parent` | 本代在代际链中的父版本号 (如 16.0)。注意别把研究线级别的父版本当包级 |
| `generation.one_mechanism_added` | 本代唯一新增的机制 (设计原则: 每代只加一个) |
| `claim.metric` | 指标名, 通常是 fuse_mse |
| `claim.value` | 本代在该指标上的数值 (MSE 越小越好) |
| `claim.baseline` | 对照基线, 形如 {"ref": "...", "value": 数字} |
| `claim.delta_pct` | 相对基线的变化百分比, 正=变好(若指标越小越好) |
| `claim.seeds` | 该结论的随机种子数 |
| `protocol.eval_set` | train_distribution | held_out —— 评估输入与训练分布是否同源 |
| `acceptance.criterion` | 什么条件算成功/失败 (一句话可判定) |
| `acceptance.outcome` | pass | fail | invalidated |
| `leakage_audit.status` | clean | invalidated | not_applicable | pending —— 移走直接目标信号后是否仍成立 |

### `v10_embedding` — 10.0

- 路径: `research/aligner/v10_embedding`
- 研究线: `aligner`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| 成员表 | `research/aligner/README.md:12` | \| v10 \| attn aligner (跨底座共享 K/V 注意力) \| `aligner.SemanticAligner`, `CrossArchAttnAligner`, `AlignedFusionLayer` \| |
| SPEC | `research/aligner/v10_embedding/SPEC.md:19` | v10.0 提出**语义对齐器 (SemanticAligner)**: 跨底座共享 K/V 注意力, 让异构底座的"同一概念"在共享空间自然对齐。 |
| README | `research/aligner/README.md:20` | v10  attn aligner      fuse 0.34   (-70%) |

### `v13_mixture` — 13.0

- 路径: `research/aligner/v13_mixture`
- 研究线: `aligner`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/aligner/v13_mixture/SPEC.md:9` | > **核心结论**: **学生蒸馏后 (0.3164) 比教师 (0.3302) 还好 4.2%** — 嵌合成功 |
| 成员表 | `research/aligner/README.md:13` | \| v13 \| MixtureAligner (跨底座 + 跨专家注意力) + 蒸馏入口 \| `aligner.MixtureAligner`, `student.StudentModel`, `distill.distill_loss` \| |
| SPEC | `research/aligner/v13_mixture/SPEC.md:66` | 3 变体 × 5 seeds = 15 run: |

### `v15_nested` — 15.0

- 路径: `research/aligner/v15_nested`
- 研究线: `aligner`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| 成员表 | `research/aligner/README.md:14` | \| v15 \| 嵌套对齐器 (MoM: mixture of mixtures) — 天花板验证 \| `aligner.MoMAligner` \| |
| SPEC | `research/aligner/v15_nested/SPEC.md:15` | v13 MixtureAligner 端到端优于 v10 CrossArchAttnAligner (fuse 0.34 → 0.26, 23%). |
| README | `research/aligner/README.md:22` | v15  MoM (嵌套)        fuse 0.26   (-77%)   ← 无进一步提升 |

### `v19_c1_analysis` — 19.0

- 路径: `research/central_diagnostics/v19_c1_analysis`
- 研究线: `central_diagnostics`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| 成员表 | `research/central_diagnostics/README.md:12` | \| v19 \| 5 假设对照实验 (H1-H5) \| `diagnostics.DiagnosticRunner`, `CapacityComparison`, `LongTrainingComparison` \| |
| SPEC | `research/central_diagnostics/v19_c1_analysis/SPEC.md:15` | v8.1 已有部分诊断 (H1: U 训练扰动, H2: EMA 太慢). v9.0 修正 C 让 C-1 工作. 但根因分析不完整: |
| README | `research/central_diagnostics/README.md:21` | \| H3: 训练步数太少 \| v19 验证 \| 长训练 (500 步): +9.5% \| |

### `v20_c1_position` — 20.0

- 路径: `research/central_diagnostics/v20_c1_position`
- 研究线: `central_diagnostics`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| 成员表 | `research/central_diagnostics/README.md:13` | \| v20 \| H5 (位置) 验证 — broadcast 加到 attn vs ffn vs both \| `position.CentralBroadcaster`, `BroadcastPosition` \| |
| SPEC | `research/central_diagnostics/v20_c1_position/SPEC.md:18` | - v8.0 默认: ffn 路由 (z_ffn = W_ffn·x + U@c) |
| SPEC | `research/central_diagnostics/v20_c1_position/SPEC.md:26` | FFN    = "ffn"      # 加到 ffn 路由 (v8.0 默认) |

### `v22_central_theory` — 22.0

- 路径: `research/central_stability/v22_central_theory`
- 研究线: `central_stability`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/central_stability/v22_central_theory/SPEC.md:6` | > **日期**: 2026-08-30 (v22.0) → 2026-09-03 (v22.0 完整实现, 修复简化层不稳定) |
| SPEC | `research/central_stability/v22_central_theory/SPEC.md:7` | > **状态**: ✅✅ Verified (单测 14/14 PASS, 端到端 4 mode × 5 seeds 全跑通) |
| SPEC | `research/central_stability/v22_central_theory/SPEC.md:8` | > **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |

### `v30_gate_softplus` — 30.0

- 路径: `research/central_stability/v30_gate_softplus`
- 研究线: `central_stability`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/central_stability/v30_gate_softplus/SPEC.md:6` | **文档 ID**: HMF-GATE-THEORY-1.0 |
| SPEC | `research/central_stability/v30_gate_softplus/SPEC.md:8` | **前置**: v7.0 (Attn+FFN dual routing), v8.0 (three-layer central architecture), v9.0 (gate-style correction), v19/v20/v22/v29 (experimental validation) |
| SPEC | `research/central_stability/v30_gate_softplus/SPEC.md:9` | **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |

### `v31_gate_gradient_scale` — 31.0

- 路径: `research/central_stability/v31_gate_gradient_scale`
- 研究线: `central_stability`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| 成员表 | `research/central_stability/README.md:14` | \| v31 \| tanh 前梯度重缩放 \| `workspace.GradientScaledGateStyleWorkspace` \| |
| README | `research/central_stability/README.md:23` | \| adaptive-ema \| decay = sigmoid(α·‖c‖) \| 0.2972 \| **0.3094** \| 0.4931 (退步) \| **0.4309** \| |
| README | `research/central_stability/README.md:36` | - v31 gradient-scale: v9=0.6927 → **v31=0.6795 (-1.9%)** ✅ 改善 |

### `v25_corpus_central` — 25.0

- 路径: `research/corpus_scale/v25_corpus_central`
- 研究线: `corpus_scale`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/corpus_scale/v25_corpus_central/SPEC.md:7` | > **状态**: ✅✅ Verified (单测 10/10 PASS, 端到端 4 mode × 5 seeds 全跑通) |
| SPEC | `research/corpus_scale/v25_corpus_central/SPEC.md:8` | > **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |
| SPEC | `research/corpus_scale/v25_corpus_central/SPEC.md:11` | >   - `python3 examples/run_v25_corpus_central.py` (端到端 4 mode × 5 seeds, 训练集 + held-out) |

### `v26_real_corpus` — 26.0

- 路径: `research/corpus_scale/v26_real_corpus`
- 研究线: `corpus_scale`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/corpus_scale/v26_real_corpus/SPEC.md:7` | > **状态**: ✅✅ Verified (单测 11/11 PASS, 端到端 4 mode × 5 seeds 全跑通) |
| SPEC | `research/corpus_scale/v26_real_corpus/SPEC.md:8` | > **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |
| SPEC | `research/corpus_scale/v26_real_corpus/SPEC.md:11` | >   - `python3 examples/run_v26_real_corpus.py` (端到端 4 mode × 5 seeds) |

### `v27_router_norm_fix` — 27.0

- 路径: `research/corpus_scale/v27_router_norm_fix`
- 研究线: `corpus_scale`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/corpus_scale/v27_router_norm_fix/SPEC.md:8` | > **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |
| SPEC | `research/corpus_scale/v27_router_norm_fix/SPEC.md:11` | >   - `python3 examples/run_v27_router_norm_fix.py` (端到端 5 patch × 4 mode × 5 seeds = 80 run) |
| SPEC | `research/corpus_scale/v27_router_norm_fix/SPEC.md:12` | > **核心结论**: **4 个补丁均让 held-out 退步加剧** — **负面发现**: |

### `v28_architectural_fix` — 28.0

- 路径: `research/corpus_scale/v28_architectural_fix`
- 研究线: `corpus_scale`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/corpus_scale/v28_architectural_fix/SPEC.md:7` | > **状态**: ✅✅ Verified (单测 16/16 PASS, 端到端 11 modes × 5 seeds = 55 run 全跑通) |
| SPEC | `research/corpus_scale/v28_architectural_fix/SPEC.md:8` | > **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |
| SPEC | `research/corpus_scale/v28_architectural_fix/SPEC.md:12` | > **核心结论**: **7 个架构变体全部让 held-out 退步加剧** — **第二个负面发现**: |

### `v29_large_corpus` — 29.0

- 路径: `research/corpus_scale/v29_large_corpus`
- 研究线: `corpus_scale`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/corpus_scale/v29_large_corpus/SPEC.md:8` | > **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值 |
| SPEC | `research/corpus_scale/v29_large_corpus/SPEC.md:11` | >   - `python3 examples/run_v29_large_corpus_v26.py` (4 modes × 5 seeds = 20 run) |
| SPEC | `research/corpus_scale/v29_large_corpus/SPEC.md:12` | >   - `python3 examples/run_v29_large_corpus_v27.py` (5 patch × 4 mode × 5 seeds = 100 run) |

### `v21_multimodal` — 21.0

- 路径: `research/multimodal/v21_multimodal`
- 研究线: `multimodal`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/multimodal/v21_multimodal/SPEC.md:9` | > **核心结论**: **多模态学生蒸馏后 (0.1578) 比 v13 mixture 教师 (0.3190) 还优 50.5%** |
| 成员表 | `research/multimodal/README.md:12` | \| v21 \| BERT-text + ViT-image + 模态路由器 + LoRA (蒸馏自 v13 mixture 教师) \| `student.MultiModalStudent`, `LoRAHiddenAdapter` \| |
| 成员表 | `research/multimodal/README.md:20` | \| v21 \| 多模态蒸馏 \| fuse 0.16, **比 v13 mixture 教师超 50.5%** \| 多模态可蒸馏 \| |

### `v23_vqa` — 23.0

- 路径: `research/multimodal/v23_vqa`
- 研究线: `multimodal`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/multimodal/v23_vqa/SPEC.md:7` | > **状态**: ✅ Verified (单测 8/8 PASS, 端到端 3/3 seeds) |
| 成员表 | `research/multimodal/README.md:13` | \| v23 \| Mini-VQA 4 类 toy 任务 \| `dataset.MiniVQADataset`, `generate_synthetic_vqa_sample` \| |
| SPEC | `research/multimodal/v23_vqa/SPEC.md:20` | - **MiniVQADataset**: 内置 16 张合成图片 + 16 个问题 + 16 个答案 (4 类: yes/no/red/blue) |

### `v24_clip` — 24.0

- 路径: `research/multimodal/v24_clip`
- 研究线: `multimodal`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/multimodal/v24_clip/SPEC.md:7` | > **状态**: ✅✅ Verified (单测 8/8 PASS + 端到端 3 mode × 3 seeds 跑通) |
| 成员表 | `research/multimodal/README.md:14` | \| v24 \| CLIP 预训练教师 + 真实多模态数据 \| `teacher.CLIPTeacher`, `corpus.MiniImageTextCorpus` \| |
| SPEC | `research/multimodal/v24_clip/SPEC.md:17` | v23 + v23b 失败: 24 训练样本, 合成图像, 学生从零学习 image-text 对齐. |

### `v7_attention` — 7.0

- 路径: `research/routing_evolution/v7_attention`
- 研究线: `routing_evolution`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/routing_evolution/v7_attention/SPEC.md:6` | > **版本**: v7.0 \| **状态**: ✅ Verified (2026-08-26) \| **前置**: v6.0 Verified |
| SPEC | `research/routing_evolution/v7_attention/SPEC.md:9` | >   - `bash run.sh v7a` (跨架构: TinyBERT + TinyLlama + ViT-tiny, 5 seeds) |
| SPEC | `research/routing_evolution/v7_attention/SPEC.md:10` | >   - `bash run.sh v7b` (同架构多任务: 3× TinyBERT, 5 seeds) |

### `v8_central` — 8.0

- 路径: `research/routing_evolution/v8_central`
- 研究线: `routing_evolution`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/routing_evolution/v8_central/SPEC.md:6` | > **版本**: v8.0 \| **状态**: ✅ Verified (2026-08-26) \| **前置**: v7.0 Verified |
| SPEC | `research/routing_evolution/v8_central/SPEC.md:10` | >   - `bash run.sh v8` (4 mode × 5 seeds 对照实验) |
| 成员表 | `research/routing_evolution/README.md:13` | \| v8 \| 全局中枢三路线 (C-1 broadcast / C-2 协同 / C-3 层级) \| `central.CentralWorkspace` (历史, 现已迁移), `fusion.CentralAugmentedFusionLayer`, `layer.HierarchicalCentralLayer` \| |

### `v9_gate_central` — 9.0

- 路径: `research/routing_evolution/v9_gate_central`
- 研究线: `routing_evolution`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/routing_evolution/v9_gate_central/SPEC.md:7` | > **状态**: ✅✅ Verified (单测 12/12 PASS, 端到端 3 mode × 5 seeds) |
| 成员表 | `research/routing_evolution/README.md:14` | \| v9 \| gate-style 中枢修正 C (v8 C-1 失败的根因修复) \| `layer.GateStyleWorkspace`, `workspace.make_workspace` \| |
| SPEC | `research/routing_evolution/v9_gate_central/SPEC.md:15` | v8.0 时代, 三种中枢实现 + 三种修正散落在 [`v8_central/central.py`](../v8_central/central.py) 的 `CentralWorkspace` 的 `mode` / `freeze_u` / `init_scale` 参数里, 历史实验脚本 `run_v9_fix_a.py` / `run_v9_fix_bc.py` 又各自维护一份变体配置。问 |

### `v33_scale_hetero` — 33.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v33_scale_hetero`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v33_scale_hetero/SPEC.md:6` | > **版本**: v33.0 \| **状态**: ✅ Verified (2026-09-07, 原快照) \| **前置**: v6.0 Verified |
| SPEC | `research/scale_heterogeneous/v33_scale_hetero/SPEC.md:7` | > **归属路线**: 规模异构+潜意识联想 (原 V6 衍生路线; 由外部快照 v7.0 重编号而来) |
| SPEC | `research/scale_heterogeneous/v33_scale_hetero/SPEC.md:8` | > **重编号对照**: 快照 v7.0 → 本项目 v33.0 (本项目 v7.0 已占用: Attn+FFN 双路由) |

### `v34_dmn_subconscious` — 34.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v34_dmn_subconscious`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v34_dmn_subconscious/SPEC.md:6` | > **版本**: v34.0 \| **状态**: ✅ Verified (2026-09-08, 原快照) \| **前置**: v33.0 Verified |
| SPEC | `research/scale_heterogeneous/v34_dmn_subconscious/SPEC.md:7` | > **归属路线**: 规模异构+潜意识联想 (原 V6 衍生路线; 由外部快照 v8.0 重编号而来) |
| SPEC | `research/scale_heterogeneous/v34_dmn_subconscious/SPEC.md:8` | > **重编号对照**: 快照 v8.0 → 本项目 v34.0 (本项目 v8.0 已占用: 全局中枢 Global Workspace) |

### `v35_container_entries` — 35.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v35_container_entries`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v35_container_entries/SPEC.md:6` | > **版本**: v35.0-α \| **状态**: **Negative Result (负结果, 边界已界定)** (2026-09-08, 原快照) |
| SPEC | `research/scale_heterogeneous/v35_container_entries/SPEC.md:7` | > **归属路线**: 大模型容器 + 异质入口 (原 V6 衍生路线; 由外部快照 v9.0-α 重编号而来) |
| SPEC | `research/scale_heterogeneous/v35_container_entries/SPEC.md:8` | > **重编号对照**: 快照 v9.0-α → 本项目 v35.0-α (本项目 v9.0 已占用: gate-style 中枢) |

### `v36_alpha2_knn` — 36.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v36_alpha2_knn`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v36_alpha2_knn/SPEC.md:6` | > **版本**: v36.0 \| **状态**: ✅ Verified (原快照 v9.0-α2) |
| SPEC | `research/scale_heterogeneous/v36_alpha2_knn/SPEC.md:7` | > **归属路线**: 大模型容器 + 异质入口 (外部快照 v9.0-α2 → 本项目 v36.0) |
| SPEC | `research/scale_heterogeneous/v36_alpha2_knn/SPEC.md:8` | > **前置**: v35.0-α (基线, 含已知塌缩 bug; 见 §1) |

### `v37_b_selftrained_entry` — 37.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v37_b_selftrained_entry`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v37_b_selftrained_entry/SPEC.md:6` | > **版本**: v37.0 \| **状态**: ✅ Verified (原快照 v9.0-β) |
| SPEC | `research/scale_heterogeneous/v37_b_selftrained_entry/SPEC.md:7` | > **归属路线**: 大模型容器 + 异质入口 (外部快照 v9.0-β → 本项目 v37.0) |
| SPEC | `research/scale_heterogeneous/v37_b_selftrained_entry/SPEC.md:8` | > **前置**: v36.0 (α2 修复后的逐样本入口基线) |

### `v38_g_adult_boundary` — 38.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v38_g_adult_boundary`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v38_g_adult_boundary/SPEC.md:6` | > **版本**: v38.0-α \| **状态**: ⚠ Boundary Result (2026-09-15; 原快照 v9.0-γ) — **结论经 v39 γ2 判定为架构伪边界, 见 §4 前注** |
| SPEC | `research/scale_heterogeneous/v38_g_adult_boundary/SPEC.md:7` | > **归属路线**: 大模型容器 + 异质入口 (外部快照 v9.0-γ → 本项目 v38.0-α) |
| SPEC | `research/scale_heterogeneous/v38_g_adult_boundary/SPEC.md:8` | > **前置**: v37.0 (β 在合成任务上成立) |

### `v39_g2_interaction` — 39.0  `[author-only: 不在版本库]`

- 路径: `research/scale_heterogeneous/v39_g2_interaction`
- 研究线: `scale_heterogeneous`
- 待填 10 项: `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `research/scale_heterogeneous/v39_g2_interaction/SPEC.md:6` | > **版本**: v39.0 \| **状态**: ✅ Verified (架构修正已实测确认; 原快照 v9.0-γ2) |
| SPEC | `research/scale_heterogeneous/v39_g2_interaction/SPEC.md:7` | > **归属路线**: 大模型容器 + 异质入口 (外部快照 v9.0-γ2 → 本项目 v39.0) |
| SPEC | `research/scale_heterogeneous/v39_g2_interaction/SPEC.md:8` | > **前置**: v38.0-α (γ, 其"三臂打平"结论经本版判定为**架构人工产物**) |

### `v12_compose` — 12.0  `[author-only: 不在版本库]`

- 路径: `archive/negative_findings/v12_compose`
- 研究线: `archive/negative_findings`
- 待填 11 项: `generation.one_mechanism_added`, `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

**候选证据(需核对)**

| 来源 | 位置 | 原文 |
|---|---|---|
| SPEC | `archive/negative_findings/v12_compose/SPEC.md:18` | - v9: C-1 单独从 -71.9% 翻正到 +44.5% |
| SPEC | `archive/negative_findings/v12_compose/SPEC.md:27` | \| v9-linear-gate \| P_m 线性 \| gate-style \| v9.0 修正 C (基线) \| |
| SPEC | `archive/negative_findings/v12_compose/SPEC.md:36` | v9-linear-gate \|    1.1625 \|       +0.0% |

### `v23b_hard_vqa` — 23.0  `[author-only: 不在版本库]`

- 路径: `archive/negative_findings/v23b_hard_vqa`
- 研究线: `archive/negative_findings`
- 待填 11 项: `generation.one_mechanism_added`, `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

_未检索到含数字的候选证据, 需手工查证_。

### `v5_alpha` — 5.0  `[author-only: 不在版本库]`

- 路径: `archive/pre_baseline/v5_alpha`
- 研究线: `archive/pre_baseline`
- 待填 11 项: `generation.one_mechanism_added`, `generation.parent`, `claim.metric`, `claim.value`, `claim.baseline`, `claim.delta_pct`, `claim.seeds`, `protocol.eval_set`, `acceptance.criterion`, `acceptance.outcome`, `leakage_audit.status`

_未检索到含数字的候选证据, 需手工查证_。
