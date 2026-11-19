# V71.0 — 四槽 / 三模态独立性审计

> 本文件是指针。权威规范在
> [`research/aligner/v71_slot_audit/SPEC.md`](../research/aligner/v71_slot_audit/SPEC.md)

**一句话**：借鉴 neuro-physical 的 Dim II / Dim III 方法论，对四槽 canonical 做表示独立性与三模态梯度子空间独立性审计。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v71_slot_audit/SPEC.md` |
| 机器可读声明 | `research/aligner/v71_slot_audit/MANIFEST.json` |
| 指标快照 | `research/aligner/v71_slot_audit/results/metrics.json` |
| 权威测试 | `research/aligner/v71_slot_audit/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
