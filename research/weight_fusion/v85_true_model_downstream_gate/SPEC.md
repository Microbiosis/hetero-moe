# V85.0 — 真预训练异构底座 (`Qwen2.5-0.5B` + `BERT-L4`) 与真实 `AG_News` 语料下游防泄漏门禁合龙 (`hetero_moe v5.1`)

## 1. 背景与定位

在 `v84_inherent_flaw_remediation` 中，我们完成了 `v4–v83` 全历史方法论缺陷的实证审计、`hetero_moe v5.0` 的五项机制修复，以及统一防泄漏下游门禁 `DownstreamEvalGate` 的构建。然而，`v84` 仍遗留一项最核心的方法论断层：
- `v84` 的 `Exp 4`（下游分类）与 `Exp 5`（下游序列 LM）使用的是**人造 64 锚点高斯合成表征**，而非真实预训练模型与真实文本语料；
- 与此同时，主线在 `v77–v83` 使用真实 HuggingFace 预训练模型（`Qwen/Qwen2.5-0.5B-Instruct` + `google/bert_uncased_L-4_H-256_A-4`）时，一直停留在层边界残差补丁与回声探针 $R^2$ 上，从未接入 `hetero_moe` 主包或通过 `DownstreamEvalGate` 的真实分类与语言建模 PPL 门禁。

`v85_true_model_downstream_gate` 从本项目**「异构预训练模型表征与权重的几何对齐、稀疏门控融合与脑启发局部模块化接口研究平台」**的定位出发，完成真预训练底座、真实文本语料、`hetero_moe v5.1` 核心包与幻脑线去坍缩算子（`DeCollapsedLocalInterface`）的全面合龙。

---

## 2. 核心机制与方法论突破

### 2.1 100% 真实预训练异构底座与预训练 FFN 权重装载
- **因果 GQA 解码器底座**：`Qwen/Qwen2.5-0.5B-Instruct`（24 层全模型，$D_{\text{qwen}}=896$，BPE 151k 词表，SwiGLU FFN）。
- **双向 MHA 编码器底座**：`google/bert_uncased_L-4_H-256_A-4`（4 层全模型，$D_{\text{bert}}=256$，WordPiece 30k 词表，GELU FFN）。
- **真预训练 FFN 切片 + `v5.1` 内部子层 LoRA**：`PretrainedBERTFFNExpert` 与 `PretrainedQwenSwiGLUExpert` 直接装载各自预训练 FFN 权重切片（`intermediate.dense / output.dense` 与 `gate_proj / up_proj / down_proj`），并通过 `ExpertSpec(..., internal_lora=True)` 在冻结原权重的条件下对内部非线性子层直接挂载低秩适应器。

### 2.2 文档级零重叠三集划分 (`SplitGuard`) 与可辨识性守卫 (`IdentifiabilityGuard`)
- 使用真实 `AG_News` 新闻语料（128 篇文档，4 类：`World` / `Sports` / `Business` / `Sci-Tech`）。
- **先在文档级**执行 `SplitGuard.make_splits(128, 0.50, 0.20, seed=42)`（训练 64 篇 / 验证 26 篇 / 测试 38 篇，文档交集严格为 0）。
- 每篇新闻提取 3 个真实子视角（全文、标题与首句导语、正文主体细节，共 384 个样本），同一文档的所有视角 100% 归属同一集合，彻底杜绝跨集文本泄漏。
- 所有子空间降维与任务锚定判别对齐矩阵仅在训练集 `tr_idx` 上拟合，自由度严格通过 `IdentifiabilityGuard.check(..., max_dof_per_sample=1.0)`。

### 2.3 攻克真模型三大物理墙（与幻脑线 `DeCollapsedLocalInterface` 数学同构）
1. **Massive Activation 常数主轴剥离与 RMS 锁相**：
   - 实测 `Qwen2.5-0.5B` 在第 3–20 层出现显著 Massive Activations（`L3 max=816.0`、`L4 max=1664.0`，而 `BERT L3 max=11.66`，相差 >70×，且跨样本常数偏置方向一致性 >0.95）。
   - 这与幻脑线 `exp102–106` 发现的跨样本常数方向坍缩（`consistency = 0.9905`）和宿主范数失配（`76.4 vs 0.1`）在数学上完全同构。
   - 通过训练集去常数均值中心化 + 安全秩主子空间 + 训练集 RMS 锁相（以及 `DeCollapsedLocalInterface` 的正交公模剥离 + 二阶重中心化 + 宿主范数锁相），彻底消除巨幅常数激活对路由器和共享空间的淹没。
2. **训练集锚定共享判别槽对齐（超越 `v79` 无监督对齐盲区）**：
   - `v79` (`ALIGN_SOURCE_NOT_FOUND`) 已证伪无监督跨架构对齐（仅对齐低秩共模噪声而抹除互补判别方向）。
   - `v85` 在训练集干净样本上将各底座的安全秩主成分投向统一的共享任务锚点坐标系，使异构编码器与解码器在同一语义槽空间内可通约。
3. **零人工域 ID 的内生跨层自洽度路由**：
   - 路由器完全不接收任何人工域标签或位置索引，而是直接读取各底座自身观测张量的**中心化跨层余弦自洽度** $\cos(\tilde{h}_{\text{early}}, \tilde{h}_{\text{late}})$ 与槽标准差置信度：当某底座在特定视角或上下文窗口信道退化时，其层间自洽度自然趋近 0，路由器结合 `gate_rank=4` 低秩向量门控自动将权重切换至高自洽度底座；当双底座均高自洽时则执行互补融合。

---

## 3. 三大预登记实验与门禁判据

| 实验 | 任务与评测门禁 | 对比基线与消融臂 | 通过判据 |
|---|---|---|---|
| **Exp 1** | 真实 `AG_News` 新闻分类 (`DownstreamEvalGate.evaluate_classification`) | `bert_l4_frozen_probe`、`qwen_05b_frozen_probe`、`naive_unnormalized_massive_qwen_bert`、`static_unrouted_equal_average`、`bert_l4_single_internal_lora_nofusion`、`qwen_05b_single_internal_lora_nofusion`、多数类零参数基线、标签洗牌阴性对照 | `verdict == "PASS"`：`v5.1` 准确率严格高于全部 6 个单底座/静态平均/未锁相基线，且 2000 次配对 Bootstrap 95% CI 下界 `ci_low > 0` |
| **Exp 2** | 真实 `AG_News` 序列语言建模 PPL/NLL (`DownstreamEvalGate.evaluate_lm`) | `bert_l4_causal_prefix_probe`、`qwen_05b_causal_prefix_probe`、`static_unrouted_equal_avg_lm`、`bert_l4_single_internal_lora_lm`、`qwen_05b_single_internal_lora_lm`、均匀分布零参数基线、阴性对照 | `verdict == "PASS"`：`v5.1` 序列 NLL/PPL 严格低于全部 5 个基线，且序列级配对 Bootstrap 95% CI 下界 `ci_low > 0` |
| **Exp 3** | `hetero_moe v5.1` 五项工程升级与主线-幻脑线算子同构审计 | 内部非线性子层 LoRA 非零梯度、训练期 `train_capacity_factor` 容量截断 STE、权重+激活双端 INT4 量化、`TensorParallelSwiGLU` 4 分片等价性、Phase 1→2 Adam 动量保留、`DeCollapsedLocalInterface` 处理 Qwen Massive Activations | `all_v5_1_checks_passed == True` |
