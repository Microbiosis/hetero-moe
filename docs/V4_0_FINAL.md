# V4.0 — 异构微观融合系统实现规范 (Final)

> **版本**: v4.0 Final (冻结规范)  
> **日期**: 2026-08-21  
> **状态**: ✅ 冻结（项目基础规范）  
> **来源**: 完整版在 [`_archive/v1-v5.1/04_v4.0_final.md`](_archive/v1-v5.1/04_v4.0_final.md)

---

## 与本仓库其他文档的关系

V4.0 是**整套工程的数学/工程基准**，所有后续 v5-v24 都基于此规范：

| 后续版本 | 相对 v4.0 的改动 |
|---|---|
| **v5.0-α** | 路由粒度泛化 (sequence/embedding)，mask 广播 |
| **v5.1** | 三粒度算子/损失/验收 (per-token / per-span / per-sequence) |
| **v6.0** | 跨架构族融合 (移除 v4.0 "M 个底座架构必须一致" 约束) |
| **v7.0** | + Attn + FFN 双路由（v4.0 只 FFN）|
| **v8.0** | + 三层中枢 C-1/C-2/C-3 |
| **v9.0** | + C-1 gate-style 修补 |
| ... | (后续版本均以 v4.0 为基线) |

## V4.0 核心要点（速览）

1. **架构前提**：M 个同构 Transformer 底座，仅 FFN 异构，Attention 共享冻结
2. **张量约束**：D 严格一致，共享 Tokenizer，KV Cache 隔离
3. **前向/反向**：分布对齐 (γ/β) + FakeQuant-INT4 (STE, bits=4, group_size=128) + 稀疏路由 (Top-K STE)
4. **训练分相**：Phase 1（前 10%）均匀路由 + 冻结 W_router，Phase 2（后 90%）解锁 Top-K 路由 + L_balance + L_z
5. **学习率隔离**：Router 1e-4 / 适配器 1e-2 / α_m 1e-3
6. **α_m 初始化**：Uniform(0.01, 0.1)（避免 1e-4 级死锁）

## 完整规范

请阅读 [`_archive/v1-v5.1/04_v4.0_final.md`](_archive/v1-v5.1/04_v4.0_final.md)（13722 字节，8 节，含数学推导）。

## V4.0 隐藏假设（v5.0 揭示）

- **A1**: Token 级均匀可分解性
- **A2**: 单一训练目标 (next-token prediction)
- **A3**: 同构架构族（仅权重异构）

A1/A2/A3 在 v5.0/v6.0 都被挑战过：v5.0 引入 sequence 粒度破坏 A1，v6.0 引入跨架构破坏 A3。
