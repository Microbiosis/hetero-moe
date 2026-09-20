# research/ — 8 条研究线总览

本项目不是线性研究。每个 `research/<line>/` 目录是一条独立研究线, 回答一个具体研究问题。

## 8 条研究线

| # | 研究线 | 研究问题 | 主要版本 | 关键结论 |
|---|---|---|---|---|
| 1 | [routing_evolution](./routing_evolution/) | 异构底座之间路由机制如何演化? | v7→v8→v9 | gate-style 修正 +88%; 中枢"必须有结构" |
| 2 | [central_diagnostics](./central_diagnostics/) | 为什么 v8 C-1 单独会反向增益? 5 个根因假设 | v19→v20 | H1 U 训练扰动是主因; H5 位置不关键 |
| 3 | [central_stability](./central_stability/) | 除 v9 gate-style 还有哪些结构修复? | v22→v30→v31 | 4 种中枢模式; v30 softplus +3.0%; v31 grad-scale +1.9% |
| 4 | [aligner](./aligner/) | 异构底座怎么对齐语义空间? | v10→v13→v15 | mixture > attn > linear; 嵌套对齐器有天花板 |
| 5 | [distillation](./distillation/) | 能不能蒸出真正独立的单底座? | v16→v17→v18 | 单教师蒸馏可行 (+47.5%); 多教师反而退步 |
| 6 | [multimodal](./multimodal/) | 多模态能超单模态? | v21→v23→v24 | BERT+ViT 蒸馏可; toy 不可行 |
| 7 | [corpus_scale](./corpus_scale/) | 中枢机制在真实语料下还稳吗? | v25→v26→v27→v28→v29 | 最优模式依赖数据规模 |
| 8 | [scale_heterogeneous](./scale_heterogeneous/) | "1大+N小" + "大模型容器"独立范式 | v33→v34→v35→v36→v37→v38→v39 | 容器范式与路由范式是两条路径 |

## 关键交叉复用

```
research/routing_evolution/v9 (gate-style)
    ├── 复用: research/central_diagnostics/ (根因诊断)
    ├── 复用: research/central_stability/ (4 mode + 变体)
    │       ↓
    └── 复用: research/corpus_scale/ (在真实语料下重测)
                  ↑ 复用 research/distillation/v18 (MiniCorpus)

research/aligner/v13 (mixture + 蒸馏入口)
    ├── 复用: research/distillation/ (学生 + 蒸馏)
    │       ↓
    └── 复用: research/multimodal/ (BERT+ViT 学生)

research/scale_heterogeneous/ (独立范式)
    ├── v34 ← v33
    ├── v37 ← v36
    └── v38/v39 ← v36 + v37
```

## 最重要的 5 个发现 (跨线总结)

1. **mixture > attn > linear** 严格成立 (research/aligner/)
2. **gate-style 中枢是最稳定的单点修复** +88.1% (research/routing_evolution/v9)
3. **单教师蒸馏可行 +47.5%, 多教师反而退步** (research/distillation/)
4. **最优中枢模式依赖数据规模** (research/corpus_scale/v29)
5. **路由融合 vs 大模型容器是两条独立路径** (research/scale_heterogeneous/)

## 最重要的 5 个负结果 (跨线总结)

1. **v8 C-1 broadcast 单独启用反向增益** (research/routing_evolution/)
2. **v12 组合 v9+v10 无叠加** (archive/negative_findings/v12_compose/)
3. **v17 多教师蒸馏全部退步** (research/distillation/)
4. **v23b 8 类陷阱 toy 全部 = 随机** (archive/negative_findings/v23b_hard_vqa/)
5. **v27+v28 中枢修正 patch/架构变体全部失败** (research/corpus_scale/)