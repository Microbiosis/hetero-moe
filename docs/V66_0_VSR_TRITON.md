# V66.0 — VSR 真视觉任务 + MoA Triton 融合内核库

> 本文件是指针。权威规范在
> [`research/aligner/v66_vsr_triton/SPEC.md`](../research/aligner/v66_vsr_triton/SPEC.md)

**一句话**：一代两件事：VSR 真视觉空间关系任务（caption 只断言空间关系、必须看图判断真假），以及四个 MoA Triton 融合内核附 numpy 位级验证器。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v66_vsr_triton/SPEC.md` |
| 机器可读声明 | `research/aligner/v66_vsr_triton/MANIFEST.json` |
| 指标快照 | `research/aligner/v66_vsr_triton/results/metrics.json` |
| 权威测试 | `research/aligner/v66_vsr_triton/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
