# V69.0 — 异构 P_m 维度调解

> 本文件是指针。权威规范在
> [`research/aligner/v69_hetero_pm/SPEC.md`](../research/aligner/v69_hetero_pm/SPEC.md)

**一句话**：异构 P_m——大模型降维、小模型先局部提炼再稠密升维并加门控，替代统一压到 256 维的冻结随机正交。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v69_hetero_pm/SPEC.md` |
| 机器可读声明 | `research/aligner/v69_hetero_pm/MANIFEST.json` |
| 指标快照 | `research/aligner/v69_hetero_pm/results/metrics.json` |
| 权威测试 | `research/aligner/v69_hetero_pm/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
