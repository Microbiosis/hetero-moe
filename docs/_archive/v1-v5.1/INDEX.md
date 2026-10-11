# v1.0 → v5.1 历史规范档案

> **目录**: `docs/_archive/v1-v5.1/`  
> **来源**: 异构微观融合系统早期规范文档 (2026-08-21 至 2026-08-24)  
> **状态**: 已归档, 不再修订

---

## 时间线

```
v1.0 (草案)        2026-08-21  初始算法草案, 存在已知致命缺陷
   ↓
v2.0 (修正草案)    2026-08-21  核心算法方向正确, Triton 实现仍不可运行
   ↓
v3.0 (系统规范)    2026-08-21  算法逻辑闭环, 可作为数学规范基准
   ↓
v4.0 (Draft)       2026-08-21  整合全部修正项, 接近冻结, 参考实现存在层间传递 Bug
   ↓
v4.0 (Final)       2026-08-21  ✅ 已冻结, 工程基准开发手册 (本文档集核心)
   ↓
v4.0 (Spec Reconstructed)  2026-08-21  跨 AI 对话重构的简化版
   ↓
v4.0 (Four Model Analysis)  v4.0 验收维度分析 (4 模型 BERT/Llama/ViT/Mixtral)
   ↓
v5.0-α (Design)    2026-08-?    路由粒度泛化 (sequence/embedding)
v5.0-α (Verification)         实验验证
v5.0-α (Errata)    2026-08-?    V1/V2 阻断级 bug 修正
   ↓
v5.1 (Granularity) 2026-08-24   路由粒度匹配损失粒度 (本研究最高产出)
v5.1 (Final Conclusion Audit)  起点问题最终回答
```

## 文件清单

| # | 文件 | 大小 | 作用 |
|---|---|---:|---|
| 01 | `01_v1.0_draft.md` | 5767 B | 初始算法草案 |
| 02 | `02_v2.0_revised.md` | 4839 B | 修正草案 |
| 03 | `03_v3.0_system_spec.md` | 3513 B | 系统规范 |
| 04 | **`04_v4.0_final.md`** | 13722 B | **v4.0 冻结规范 (核心)** |
| 05 | `05_v4.0_spec_reconstructed.md` | 1848 B | v4.0 简化重构版 |
| 06 | `06_v4.0_four_model_analysis.md` | 1315 B | v4.0 验收维度分析 |
| 07 | `07_v5.0_alpha_design.md` | 1345 B | v5.0-α 设计 |
| 08 | `08_v5.0_alpha_verification.md` | 992 B | v5.0-α 实验验证 |
| 09 | `09_v5.0_alpha_errata.md` | 978 B | v5.0-α V1/V2 修正 |
| 10 | `10_v5.1_spec_granularity.md` | 1508 B | v5.1 粒度匹配规范 |
| 11 | `11_v5.1_final_conclusion_audit.md` | 1949 B | v5.1 最终审计 |

## 与正式 docs/ 关系

正式 docs/ 已有文档（V5_1_SPEC.md, V5_ALPHA_*.md, V6_0_SPEC.md 等）是从这里**演化**而来。
本 archive 是**演化前的原始快照**，用于：

1. **审计追溯**：为何某规范是这样设计的？
2. **草稿对比**：draft → final 期间改了什么？
3. **Errata 历史**：V1/V2 bug 修正记录

## 现行项目主线

- `docs/V4_0_FINAL.md` ← v4.0 冻结规范速览 (本 archive 04 的指针)
- `docs/V5_1_SPEC.md` ← v5.1 正式规范
- `docs/V5_ALPHA_VERIFICATION.md` + `_RESULTS.md` ← v5.0-α 验证
- `docs/V6_0_SPEC.md` ← v6.0 跨架构族规范
- ... (v7-v24 后续规范)

---

**归档说明**: 这些文档是 2026-08-21 至 2026-08-24 期间的**研究快照**，代表当时的认知。
代码与现行 docs/ 可能与本 archive 有差异（详见代码事实 + README 当前版本演进表）。
