# V64.0 — MoA 异构 proposer 融合层

> 本文件是指针。权威规范在
> [`research/aligner/v64_moa_proposer/SPEC.md`](../research/aligner/v64_moa_proposer/SPEC.md)

**一句话**：v62 的融合层是同构的（4 个专家全是 SwiGLU FFN），v64 把它换成真正的 MoA 异构 proposer——多种归纳偏置分支并行提案、稀疏路由聚合，接口与 v62 完全兼容。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v64_moa_proposer/SPEC.md` |
| 机器可读声明 | `research/aligner/v64_moa_proposer/MANIFEST.json` |
| 指标快照 | `research/aligner/v64_moa_proposer/results/metrics.json` |
| 权威测试 | `research/aligner/v64_moa_proposer/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
