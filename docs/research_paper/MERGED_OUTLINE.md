# 合并稿大纲 — Heterogeneous Reconstruction

> 状态：待确认（2026-10-11）
> 合并来源：`microfusion.tex`（+`microfusion_zh.tex`）与 `paper.tex`
> 产物：新文件 `reconstruct.tex` / `reconstruct_zh.tex`（原两篇保留不动，可回滚）

## 0. 统一中心论点

冻结的异构预训练底座可以被**重构**为可用系统，分两步：先在**子层**用共享稀疏路由 + 语义对齐器把多个底座融合，
再把集成分解（蒸馏）为**单一独立学生**。贯穿全栈的一条方法论主线是
**样本外纪律**——项目里相当一部分 in-sample 增益（中枢模式、语料规模边界、长训练、修复方案）
在留出评估下塌陷，而另一些（对齐器阶梯、attention 通路、规模异构融合）存活。
本稿的贡献因此分为「机制层」与「证据层」两半：机制层给出可融合的架构与可容许的协调设计；
证据层用统一样本外协议标出哪些结论是真的。

## 1. 标题与定位

- 英文：*Reconstructing Frozen Heterogeneous Pretrained Models: Sublayer Fusion, Coordinated Routing, and Chimera Distillation*
- 中文：*重构冻结的异构预训练模型：子层融合、协调式路由与嵌合蒸馏*
- 版式：沿用 `neurips_2025.sty`（投稿稿式）；合并后篇幅显著增长，需决定目标venue（见第 6 节待决项）

## 2. 章节结构

### 1 Introduction
- 1.1 问题：异构冻结底座的组合，三条既有路线各自的不足
- 1.2 两个重建步骤（子层融合 → 嵌合蒸馏）+ 中间的共享路由
- 1.3 方法论主线：样本外纪律
- 1.4 贡献（**紧凑清单，不再用 C1–C4 标签**，改用具名发现）

### 2 Related Work
- MoE 路由；异构基础模型协作（**异质性谱系图**，沿用 microfusion 版）；权重合并与 upcycling；
  PEFT 与适配器组合；知识蒸馏；温度缩放与全局协调
- GWT 明确降级为「设计灵感，不产生可检验预测」

### 3 Method I — Heterogeneous sublayer fusion
- 3.1 Problem setup（冻结底座、共享空间、原生编码）
- 3.2 融合层（稀疏 Top-K 路由 / SwiGLU 专家 / INT4 伪量化 / 正门控残差融合）
- 3.3 两阶段训练协议
- 3.4 对齐器设计空间（**去结果化**，只讲四种设计）
- 3.5 双通路路由（attention 可路由）

### 4 Method II — Coordinating the shared router
- 4.1 四种中枢模式（broadcast / router-norm / adaptive-EMA / gate-style）
- 4.2 设计约束 G1–G5 与三条定理（完美退化、保序、唯一性）
- （原 paper.tex §4.3「五个假设」占位段**删除**）

### 5 Method III — Reconstructing an independent model
- 5.1 学生架构（TinyBERT + LoRA）
- 5.2 嵌合蒸馏目标（0.7 教师 + 0.3 目标）
- 5.3 多模态学生
- 5.4 评估纪律（为什么全部学生评估改用留出输入）

### 6 Experiments
- 6.1 Setup and honest scope（430 测试；CPU 沙箱）
- 6.2 子层融合增益与路径分解 — v6/v7 + **v46（通路主张样本外成立 +82.7%/+70.6%）**
- 6.3 对齐器阶梯与天花板 — v10/v13/v14/v15 + **v41 容量对照（阶梯成立，主发现）**
- 6.4 中枢模式：in-sample 与样本外 — v22 + **v43（+44.5%→+4.0%）** + **v44（router-norm 双口径最优，主表不成立）**
- 6.5 语料规模边界未在样本外确立 — v29 + **v45**
- 6.6 tanh 饱和失效模式与两种修复 — v29 诊断 + **v47 修正（修复恢复梯度流，但对融合 MSE 无差别）**
- 6.7 嵌合蒸馏：表观裕度、泄漏分解与多教师 — v16/v18/v21/v32/v42 + **v40 分派归因**
- 6.8 任务级评估：一条边界 — v23 VQA（双读法）

### 7 Discussion
- 7.1 四条设计原则（标注为领域性启发法）
- 7.2 反复出现的规律「更多不等于更好」
- 7.3 **样本外纪律：哪些 in-sample 结论存活、哪些塌陷**（本稿的方法论主张）
- 7.4 数据规模（原 C3，降级为一个段落）

### 8 Limitations / 9 Conclusion
- 规模是首要局限；INT4 非部署核；VQA 未决

### 附录
- A 超参数表（沿用 microfusion）
- B 证明（沿用 paper.tex 附录，三条定理）
- C 样本外审计协议（v47 口径汇总表）

## 3. 关键内容改写（相对原文的实质变动）

| # | 位置 | 变动 |
|---|---|---|
| 1 | 标题/摘要/引言 | 从「micro-fusion + 蒸馏」扩为「重构」三步；新增样本外纪律主线 |
| 2 | 原 paper.tex 摘要/引言/讨论/结论 | **删除**「gate-style 最优」「语料规模边界成立」；改为「G1–G5 可容许性（理论）+ 样本外排序反转（实证）」 |
| 3 | 原 paper.tex §5.2 已修正段 | 升格为 §6.4/§6.5 的正文叙事，不再是「贴在表下的更正」 |
| 4 | microfusion §5.3 蒸馏 | 由「宣称→撤回」改为**正向叙事**：以 v32/v42 泄漏分解为结论，机制证据在**学生间对比** |
| 5 | microfusion §5.3 多教师 | 补 v40：加权三教师退回是**形式伤害**而非增益；教师侧 13% headroom 传不到学生 |
| 6 | 全文 C1–C4 | 删除标签式复述，改具名发现（对齐器天花板 / 门控稀疏路由 / G1–G5 / 嵌合蒸馏 / 样本外纪律） |
| 7 | 测试口径 | 全文统一 430（消除 383） |
| 8 | 版本/实验计数 | 统一为 47 代 / 430 测试（消除「29 版 / 175+」） |

## 4. 暂不纳入（保留为后续工作或姊妹产物）

- 研究线 8 规模异构范式 v33–v39 的**完整**展开（仅以一段 + 表格摘要引用其样本外读数：v33 +26.7%、v34 +6.4%、v35 −11.1%）
- v33–v39 的专有辅助模型细节
- v47 逐包 16 项明细（收进附录 C 汇总）

## 5. 交付

- `reconstruct.tex` → `reconstruct.pdf`（xelatex/pdflatex）
- `reconstruct_zh.tex` → `reconstruct_zh.pdf`（xelatex + ctex）
- 合并后的 `reconstruct.bib`（microfusion 37 条 + paper.tex 10 条，去重）

## 6. 待决项

1. **目标 venue**：合并后远超 NeurIPS 9 页正文限制。是投期刊（长文），还是把审计细节全部下沉附录以压到会议篇幅？
2. 新文件名是否用 `reconstruct.*`（可改）。
