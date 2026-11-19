# V68.0 — 架构图 + zero-OOM SSM 四分支 + 单模态消融

> 本文件是指针。权威规范在
> [`research/aligner/v68_arch_ablation/SPEC.md`](../research/aligner/v68_arch_ablation/SPEC.md)

**一句话**：恢复四分支并配 zero-OOM SSM（autograd checkpointing + 状态 tile，d_state=16 时显存仅 4.6MB），同时交付三模态异构融合架构图与单模态消融开关。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v68_arch_ablation/SPEC.md` |
| 机器可读声明 | `research/aligner/v68_arch_ablation/MANIFEST.json` |
| 指标快照 | `research/aligner/v68_arch_ablation/results/metrics.json` |
| 权威测试 | `research/aligner/v68_arch_ablation/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
