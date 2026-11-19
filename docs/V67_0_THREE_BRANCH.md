# V67.0 — 去掉 SSM 分支：3 异构 proposer

> 本文件是指针。权威规范在
> [`research/aligner/v67_three_branch/SPEC.md`](../research/aligner/v67_three_branch/SPEC.md)

**一句话**：去掉 SSM 分支，改为 Attention / Conv / SwiGLU 三种异构——避开 SSM 状态维度与显存、并行扫描之间的三角冲突。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v67_three_branch/SPEC.md` |
| 机器可读声明 | `research/aligner/v67_three_branch/MANIFEST.json` |
| 指标快照 | `research/aligner/v67_three_branch/results/metrics.json` |
| 权威测试 | `research/aligner/v67_three_branch/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
